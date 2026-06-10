"""
验证 staryolon 模型能否正确构建和前向传播。
"""
import sys
import os

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO

YAML_PATH = 'ultralytics/cfg/models/staryolon.yaml'

# === Test 1: Build with DetectionModel ===
print("=== Build with DetectionModel ===")
model = DetectionModel(YAML_PATH, ch=3, nc=80)
print(f'Strides: {model.stride}')
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

# Forward pass
x = torch.randn(1, 3, 640, 640)
y = model(x)
print(f'Output type: {type(y)}')
if isinstance(y, (list, tuple)):
    print(f'Output length: {len(y)}')
    for i, yi in enumerate(y):
        print(f'  [{i}] shape: {yi.shape}')
print("DetectionModel test PASSED\n")

# === Test 2: Build with YOLO ===
print("=== Build with YOLO ===")
model2 = YOLO(YAML_PATH)
print(f'Model type: {type(model2)}')
print(f'Params: {sum(p.numel() for p in model2.model.parameters()):,}')
print("YOLO test PASSED\n")

print("All tests passed!")
