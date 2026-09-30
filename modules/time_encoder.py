# coding=utf-8
"""
TimeEncoder: 时间元数据编码器
==========================================
参考 GT-Loc (ICCV 2025) 的时间编码器设计，将无人机视频对应的拍摄时间
编码为 512 维特征向量，与 CLIP 文本/视觉特征在同一语义空间内进行对比学习。

核心引用：
  GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space
  David G. Shatwell et al., ICCV 2025

架构（来自 GT-Loc）：
  - 角度时间表示 (Angular Time Representation)：
        month/day → 季节角 theta ∈ [-π, π)
        hour/minute/second → 时段角 phi ∈ [-π, π)
  - 3 个不同尺度 (σ=1, 16, 256) 的 TimeEncoderCapsule：
        GaussianEncoding(input=2, encoded=256) → Linear(512,1024) → ReLU
        → Linear(1024,1024) → ReLU → Linear(1024,1024) → ReLU → Linear(1024,512)
  - 三路求和 → L2 归一化 → 512 维时间特征

输出维度与 CLIP ViT-B/32 文本/视觉编码器一致（512D），无需投影层。

输入格式: (B, 5) Tensor — [month, day, hour, minute, second]
    - month: 1–12
    - day: 1–31
    - hour: 0–23（本地时间）
    - minute: 0–59
    - second: 0–59
"""

import math
import os
import sys

import torch
import torch.nn as nn

# ── 将本地 geo-clip 包路径加入 sys.path，复用其 GaussianEncoding ──────────────
_GEO_CLIP_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'geo-clip')
)
if _GEO_CLIP_DIR not in sys.path:
    sys.path.insert(0, _GEO_CLIP_DIR)

from geoclip.model.rff import GaussianEncoding  # noqa: E402


# ── 时间转换常数（来自 GT-Loc） ─────────────────────────────────────────────────
MINUTE_TO_HOUR = 1 / 60
SECOND_TO_HOUR = MINUTE_TO_HOUR / 60

DAY_TO_MONTH = 1 / (365 / 12)
HOUR_TO_MONTH = DAY_TO_MONTH / 24
MINUTE_TO_MONTH = MINUTE_TO_HOUR * HOUR_TO_MONTH
SECOND_TO_MONTH = SECOND_TO_HOUR * HOUR_TO_MONTH


def angular_time_representation(T: torch.Tensor) -> torch.Tensor:
    """将 [month, day, hour, minute, second] 转换为循环角度表示 [theta, phi]。

    Args:
        T: (B, 5) — [month, day, hour, minute, second]

    Returns:
        (B, 2) — [theta(季节角), phi(时段角)]，范围 [-π, π)
    """
    month, day, hour, minute, second = torch.chunk(T.float(), 5, dim=1)

    # 连续化月份（含日偏移）
    month_d = (month - 1) + (day - 1) * DAY_TO_MONTH

    # 连续化小时（含分秒偏移）
    hour_d = hour + minute * MINUTE_TO_HOUR + second * SECOND_TO_HOUR

    # 映射到 [-π, +π) 的角度（循环表示，消除边界不连续）
    theta = 2 * math.pi * month_d / 12 - math.pi   # 季节角（年周期）
    phi = 2 * math.pi * hour_d / 24 - math.pi      # 时段角（日周期）

    return torch.cat((theta, phi), dim=1)  # (B, 2)


class TimeEncoderCapsule(nn.Module):
    """单尺度时间编码胶囊（对应特定 σ 的感受野）。"""

    def __init__(self, sigma: float, embedding_dim: int = 512):
        super(TimeEncoderCapsule, self).__init__()
        rff_encoding = GaussianEncoding(
            sigma=sigma, input_size=2, encoded_size=embedding_dim // 2
        )
        self.sigma = sigma
        self.capsule = nn.Sequential(
            rff_encoding,
            nn.Linear(embedding_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(1024, embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.capsule(x)
        x = self.head(x)
        return x


class TimeEncoder(nn.Module):
    """多尺度时间编码器（参考 GT-Loc）。

    将 [month, day, hour, minute, second] 编码为 512D 特征向量。

    Args:
        sigma: 各 Capsule 的高斯核标准差列表，默认 [1, 16, 256]
        embedding_dim: 输出特征维度，默认 512
        simple_mlp: 是否使用简化MLP版本（消融对照），默认 False
    """

    def __init__(self, sigma=None, embedding_dim: int = 512, simple_mlp: bool = False):
        super(TimeEncoder, self).__init__()
        if sigma is None:
            sigma = [2 ** 0, 2 ** 4, 2 ** 8]
        self.sigma = sigma
        self.n = len(self.sigma)
        self.embedding_dim = embedding_dim
        self.simple_mlp = simple_mlp

        if simple_mlp:
            # 简化版本：直接将原始[月,日,时,分,秒]作为5D向量输入MLP
            self.mlp = nn.Sequential(
                nn.Linear(5, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, embedding_dim),
            )
        else:
            # 完整版本：使用角度表示+RFF
            for i, s in enumerate(self.sigma):
                self.add_module(
                    'TimeEnc' + str(i),
                    TimeEncoderCapsule(sigma=s, embedding_dim=embedding_dim)
                )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: (B, 5) — [month, day, hour, minute, second]

        Returns:
            (B, embedding_dim) — 时间特征（未归一化）
        """
        if self.simple_mlp:
            # 简化版本：直接使用原始时间向量
            return self.mlp(time.float())
        else:
            # 完整版本：使用角度表示+RFF
            time = angular_time_representation(time)
            time_features = torch.zeros(
                time.shape[0], self.embedding_dim, device=time.device
            )
            for i in range(self.n):
                time_features += self._modules['TimeEnc' + str(i)](time)
            return time_features


class GeoTimeEncoder(nn.Module):
    """
    时间元数据编码器，封装 TimeEncoder。

    Args:
        from_pretrained_path (str or None):
            不为 None → 从 GT-Loc checkpoint 中提取时间编码器权重
            None → 随机初始化
        sigma (str or None):
            时间编码器高斯核标准差列表，逗号分隔的字符串
            None → 默认 [1, 16, 256]
        simple_mlp (bool):
            True → 仅使用MLP（消融对照，不使用角度表示+RFF）
            False → 使用角度表示+RFF（默认）

    输入格式: (B, 5) Tensor — [month, day, hour, minute, second]
    输出格式: (B, 512) 已 L2 归一化的单位向量
    """

    def __init__(self, from_pretrained_path: str = None, sigma: str = None, simple_mlp: bool = False):
        super().__init__()
        
        # 解析sigma参数
        if sigma is None:
            sigma_list = [2 ** 0, 2 ** 4, 2 ** 8]
        else:
            sigma_list = [float(s.strip()) for s in sigma.split(',')]
        
        self.simple_mlp = simple_mlp
        self.time_encoder = TimeEncoder(
            sigma=sigma_list,
            embedding_dim=512,
            simple_mlp=simple_mlp,
        )
        # 兼容从 shell 传入的字符串 "None"/"none"/""
        if from_pretrained_path is not None and str(from_pretrained_path).strip().lower() not in ("none", ""):
            self._load_pretrained(from_pretrained_path)

    def _load_pretrained(self, ckpt_path: str):
        """从 GT-Loc 完整 checkpoint 中提取 time_encoder 权重。"""
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # GT-Loc checkpoint 中 time_encoder 的 key 前缀为 'time_encoder.'
        prefix = 'time_encoder.'
        time_state = {
            k[len(prefix):]: v for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        if time_state:
            self.time_encoder.load_state_dict(time_state, strict=False)

    def forward(self, time_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_coords: (B, 5) — [month, day, hour, minute, second]

        Returns:
            time_features: (B, 512) — L2 归一化时间特征向量
        """
        time_features = self.time_encoder(time_coords)  # (B, 512)
        time_features = time_features / (
            time_features.norm(dim=-1, keepdim=True) + 1e-8
        )
        return time_features  # (B, 512) 单位向量
