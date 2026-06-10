"""
YOLOv8n — 只训练 backbone 第8层(C2f)，其余层加载预训练权重并冻结。
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
from ultralytics import YOLO

# ── 配置 ──
YAML_PATH = os.path.join(PROJECT_ROOT, 'ultralytics/cfg/models/yolov8nmod.yaml')
DATA_PATH = os.path.join(PROJECT_ROOT, 'res/dataset/dataset_cfg/coco512.yaml')
PRETRAINED = os.path.join(PROJECT_ROOT, 'res/model/yolov8n.pt')

BATCH = 16
IMG_SIZE = 640
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    print("=" * 60)
    print("Building YOLOv8n from YAML...")
    print("=" * 60)

    model: YOLO = YOLO(YAML_PATH)

    # ── 加载预训练权重（跳过 model.8） ──
    if os.path.exists(PRETRAINED):
        print(f"\nLoading from {PRETRAINED} (skipping model.8) ...")
        ckpt = torch.load(PRETRAINED, map_location='cpu', weights_only=False)
        src_sd = ckpt.get('model', ckpt)
        if isinstance(src_sd, torch.nn.Module):
            src_sd = src_sd.state_dict()

        target_sd = model.model.state_dict()
        filtered = {}
        skip_prefix = 'model.8.'
        for k, v in src_sd.items():
            if k.startswith(skip_prefix):
                continue  # 跳过 layer 8，让它随机初始化
            if k in target_sd and target_sd[k].shape == v.shape:
                filtered[k] = v

        model.model.load_state_dict(filtered, strict=False)
        loaded = len(filtered)
        total = len(target_sd)
        print(f"  Loaded {loaded}/{total} params. Layer 8 randomly initialized.")

        # ⚠️ 关键修复：设置 ckpt 使其为真值
        # 否则 model.train() 内部 get_model(weights=None) 会丢弃已加载的权重
        model.ckpt = {"epoch": 0}
    else:
        print(f"\nNo pretrained weights found.")

    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable (before freeze): {trainable_params:,}")

    # ── 训练：冻结所有层，只训练 model.8 ──
    FREEZE = [i for i in range(23) if i != 8]  # 0-22 去掉 8

    print("\n" + "=" * 60)
    print(f"Training only layer 8 (C2f). Freezing layers: {FREEZE}")
    print("=" * 60)

    model.train(
        data=DATA_PATH,
        epochs=50,
        batch=BATCH,
        imgsz=IMG_SIZE,
        device=DEVICE,
        freeze=FREEZE,

        optimizer='AdamW',
        lr0=1e-3,
        lrf=0.01,
        weight_decay=5e-4,

        save=True,
        save_period=10,
        workers=4,
        project=os.path.join(PROJECT_ROOT, 'runs/train'),
        name='yolov8nmod-newStar',
        exist_ok=True,
        verbose=True,
        pretrained=False,

        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
