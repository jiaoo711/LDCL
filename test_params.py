#!/usr/bin/env python
# coding=utf-8
"""测试 params.py 是否正确包含新增的时间编码器参数"""

import sys
sys.path.insert(0, '.')

from params import get_args

# 模拟命令行参数
test_args = [
    '--do_train', '0',
    '--output_dir', './test_output',
    '--time_encoder_sigma', '1,256',
    '--time_encoder_simple_mlp', '1',
]

try:
    args = get_args()
    print("✓ 参数解析成功!")
    print(f"  time_encoder_sigma: {args.time_encoder_sigma}")
    print(f"  time_encoder_simple_mlp: {args.time_encoder_simple_mlp}")
except Exception as e:
    print(f"✗ 参数解析失败: {e}")
    sys.exit(1)
