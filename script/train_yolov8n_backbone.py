"""
YOLOv8n 训练脚本 — 冻结 head，只训练 backbone。
加载 coco 预训练权重，head 部分冻结，backbone 从头训练。
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path
import torch

from ultralytics import YOLO

# ── 配置 ──
DATA_PATH = os.path.join(PROJECT_ROOT, 'res/dataset/dataset_cfg/coco512.yaml')
PRETRAINED = os.path.join(PROJECT_ROOT, 'res/model/yolov8n.pt')

BATCH = 8
IMGSZ = 640
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    print("=" * 60)
    print("Building YOLOv8n from pretrained...")
    print("=" * 60)

    # 从预训练权重构建模型
    model = YOLO(PRETRAINED)

    print(f"\nTotal params: {sum(p.numel() for p in model.model.parameters()):,}")

    # ── 训练：冻结 head（model.10 ~ model.22），训练 backbone（model.0 ~ model.9） ──
    # YOLOv8n: 0-9 backbone, 10-22 head
    head_layers = list(range(10, 23))  # [10, 11, ..., 22]

    print("\n" + "=" * 60)
    print("Training backbone only (head frozen with pretrained weights)")
    print("=" * 60)

    model.train(
        data=DATA_PATH,
        epochs=100,
        batch=BATCH,
        imgsz=IMGSZ,
        device=DEVICE,

        # 冻结 head 层
        freeze=head_layers,

        optimizer='AdamW',
        lr0=1e-3,
        weight_decay=5e-4,

        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,

        save=True,
        save_period=10,
        workers=4,
        project=os.path.join(PROJECT_ROOT, 'runs/train'),
        name='yolov8n-backbone-train',
        exist_ok=True,
        pretrained=False,
        verbose=True,

        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,
        lrf=0.01,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
