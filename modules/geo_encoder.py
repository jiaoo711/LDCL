# coding=utf-8
"""
GeoLocationEncoder: GPS 经纬度坐标编码器
==========================================
封装本地 geo-clip 的 LocationEncoder，将无人机视频对应的 GPS 经纬度坐标
编码为 512 维特征向量，与 CLIP 文本/视觉特征在同一语义空间内进行对比学习。

核心引用：
  GeoCLIP: Clip-Inspired Alignment between Locations and Images
           for Effective Worldwide Geo-localization, NeurIPS 2023

架构（来自 Geo-CLIP 原论文）：
  - 等面积地球投影 (Equal-Earth Projection)：(lat, lon) → 2D 坐标
  - 3 个不同尺度 (σ=1, 16, 256) 的 LocationEncoderCapsule：
        GaussianEncoding(input=2, encoded=256) → Linear(512,1024) → ReLU ×3 → Linear(1024,512)
  - 三路求和 → 512 维 GPS 特征

输出维度与 CLIP ViT-B/32 文本/视觉编码器一致（512D），无需投影层。
"""

import os
import sys
import torch
import torch.nn as nn

# ── 将本地 geo-clip 包路径加入 sys.path，免去 pip 安装 ──────────────────────
_GEO_CLIP_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'geo-clip')
)
if _GEO_CLIP_DIR not in sys.path:
    sys.path.insert(0, _GEO_CLIP_DIR)

from geoclip.model.location_encoder import LocationEncoder   # noqa: E402


class GeoLocationEncoder(nn.Module):
    """
    GPS 经纬度坐标编码器，封装 Geo-CLIP LocationEncoder。

    Args:
        from_pretrained (bool):
            True  → 加载 Geo-CLIP 预训练权重（推荐，收敛更快）
            False → 随机初始化（消融实验：验证预训练权重的贡献）

    输入格式: (B, 2) Tensor，列顺序为 [longitude, latitude]
              与 video_gps.json 约定一致 → 内部自动转换为 Geo-CLIP 期望的 [lat, lon]

    输出格式: (B, 512) 已 L2 归一化的单位向量
              与 CLIP 文本/视觉特征空间一致，可直接计算余弦相似度
    """

    def __init__(self, from_pretrained: bool = True):
        super().__init__()
        self.location_encoder = LocationEncoder(
            sigma=[2 ** 0, 2 ** 4, 2 ** 8],
            from_pretrained=from_pretrained,
        )

    def forward(self, gps_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gps_coords: (B, 2) — [longitude, latitude]，与 video_gps.json 格式一致

        Returns:
            gps_features: (B, 512) — L2 归一化 GPS 特征向量
        """
        # video_gps.json 约定 [lon, lat]；Geo-CLIP 的 equal_earth_projection 期望 [lat, lon]
        lat_lon = gps_coords[:, [1, 0]].float()          # (B, 2) 列顺序交换
        gps_features = self.location_encoder(lat_lon)    # (B, 512)
        gps_features = gps_features / (
            gps_features.norm(dim=-1, keepdim=True) + 1e-8
        )
        return gps_features                              # (B, 512) 单位向量
