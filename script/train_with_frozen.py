"""
使用层映射表加载预训练权重，冻结已映射层，训练未映射层。
支持 --resume 从已保存的训练目录恢复训练。

映射文件位于 ./res/mapping/{pretrained}_{model}.json，格式:
  {"pretrained_layer_idx": "target_layer_idx", ...}

只会加载映射文件中列出的层，未列出层随机初始化。
已映射（加载了预训练权重）的层被冻结，未映射的层参与训练。

用法（正常训练）:
  conda run -n yolov8 python script/train_with_frozen.py \
    --pt yolov8n --model yolov8nmod --dataset coco512

用法（恢复训练）:
  conda run -n yolov8 python script/train_with_frozen.py \
    --resume staryolon-100 --epochs 100 --lr 1e-4
"""
import sys
import os
import json
import argparse
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
from ultralytics import YOLO
from ultralytics.utils import yaml_load


def load_mapping(mapping_path):
    """加载层映射表，返回 {src_idx: tgt_idx} 均为 int。"""
    with open(mapping_path) as f:
        raw = json.load(f)

    mapping = {}
    for k, v in raw.items():
        sk = int(k.replace("model.", "")) if "model." in k else int(k)
        tv = int(v.replace("model.", "")) if "model." in v else int(v)
        mapping[sk] = tv
    return mapping


def build_loaded_state_dict(src_sd, target_sd, mapping):
    """
    根据映射表从 src_sd 提取参数到 target_sd 的 key 空间。
    返回 (loaded_dict, mismatches, not_founds)。
    """
    loaded = {}
    mismatches = []
    not_founds = []

    for src_idx, tgt_idx in mapping.items():
        src_prefix = f"model.{src_idx}."
        tgt_prefix = f"model.{tgt_idx}."
        for k, v in src_sd.items():
            if k.startswith(src_prefix):
                tgt_k = tgt_prefix + k[len(src_prefix):]
                if tgt_k in target_sd:
                    if target_sd[tgt_k].shape == v.shape:
                        loaded[tgt_k] = v
                    else:
                        mismatches.append((k, tgt_k, v.shape, target_sd[tgt_k].shape))
                else:
                    not_founds.append((k, tgt_k))
    return loaded, mismatches, not_founds

def main():
    parser = argparse.ArgumentParser(description="Train with layer mapping")
    parser.add_argument("--pt", type=str, default="yolov8n", help="预训练模型名（正常训练时使用）")
    parser.add_argument("--model", type=str, default="yolov8nmod", help="目标模型名（正常训练时使用）")
    parser.add_argument("--dataset", type=str, default="coco512", help="数据集名")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--cos_lr", action="store_true", default=True,
                        help="使用余弦退火，否则使用线性退火")
    parser.add_argument("--lrf", type=float, default=1,
                        help="最终学习率 = lr * lrf")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output", type=str, default="",
                        help="runs 中输出文件夹名（正常训练时使用）")
    parser.add_argument("--fraction", type=float, default=1.,
                        help="采样前N%进行训练")
    parser.add_argument("--save_period", type=int, default=10,
                        help="每 N 个 epoch 保存一次权重")
    parser.add_argument("--resume", type=str, default="",
                        help="从已保存的训练文件夹路径（相对于 runs/train/）中继续训练")

    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==================== 恢复训练模式 ====================
    if args.resume:
        resume_dir = os.path.join(PROJECT_ROOT, "runs/train", args.resume)
        weights_path = os.path.join(resume_dir, "weights", "last.pt")
        args_path = os.path.join(resume_dir, "args.yaml")

        if not os.path.exists(weights_path):
            print(f"[Error] Resume weights not found: {weights_path}")
            sys.exit(1)
        if not os.path.exists(args_path):
            print(f"[Error] args.yaml not found: {args_path}")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"Resuming training from {resume_dir}")
        print(f"{'='*60}\n")

        # 1. 加载原始训练参数
        resume_args = yaml_load(args_path)
        print(f"  Loaded {len(resume_args)} saved args")

        # 2. 删除恢复训练不用的键
        for key in ('resume', 'model', 'save_dir', 'mode'):
            resume_args.pop(key, None)

        # 3. 确保保存到原目录
        resume_args['project'] = os.path.join(PROJECT_ROOT, "runs/train")
        resume_args['name'] = args.output or f"{resume_args['name'].split('-')[0]}-{int(time.time())}"
        resume_args['exist_ok'] = True

        # 4. 检测 CLI 显式指定的参数
        cli_specified = set()
        for token in sys.argv[1:]:
            if token.startswith('--'):
                name = token[2:].split('=')[0].replace('-', '_')
                cli_specified.add(name)

        # 5. 构建覆盖参数：CLI → resume_args（pt 不传递）
        override_args = {}
        cli_dict = vars(args)
        for key in cli_specified:
            # 这些参数在 resume_args 中不覆盖
            if key in ('resume', 'pt', 'model'):
                continue
            # dataset 参数需要特殊处理
            if key == 'dataset':
                val = cli_dict[key]
                if val:
                    override_args['data'] = os.path.join(
                        PROJECT_ROOT, f"res/dataset/dataset_cfg/{val}.yaml")
            # lr 参数需要特殊处理
            elif key == 'lr':
                override_args['lr0'] = cli_dict[key]
            # 直接覆盖
            elif key in cli_dict:
                override_args[key] = cli_dict[key]

        # 6. 合并：CLI 覆盖原始参数
        resume_args.update(override_args)

        if override_args:
            print(f"  CLI overrides: {override_args}")

        # 7. 加载模型权重
        model = YOLO(weights_path)

        # 8. 恢复训练
        model.train(**resume_args)
        print("\nResume training completed!")
        return

    # ==================== 正常训练模式 ====================
    # 构建路径
    mapping_path = os.path.join(PROJECT_ROOT, f"res/mapping/{args.pt}_{args.model}.json")
    pretrained_path = os.path.join(PROJECT_ROOT, f"res/model/{args.pt}.pt")
    model_yaml = os.path.join(PROJECT_ROOT, f"ultralytics/cfg/models/{args.model}.yaml")
    data_yaml = os.path.join(PROJECT_ROOT, f"res/dataset/dataset_cfg/{args.dataset}.yaml")

    output_name = args.output or f"{args.model}-{int(time.time())}"
    print(f"Output name: {output_name}")
    print(f"LR schedule: {'cosine' if args.cos_lr else 'linear'} (lrf={args.lrf})")

    # 1. 加载映射表
    if not os.path.exists(mapping_path):
        print(f"[Error] Mapping file not found: {mapping_path}")
        sys.exit(1)
    print(f"\nLoading mapping from {mapping_path} ...")
    mapping = load_mapping(mapping_path)
    print(f"  {len(mapping)} entries:")
    for s, t in sorted(mapping.items()):
        print(f"    model.{s} -> model.{t}")

    # 2. 构建目标模型
    if not os.path.exists(model_yaml):
        print(f"[Error] Model YAML not found: {model_yaml}")
        sys.exit(1)
    print(f"\nBuilding model from {model_yaml} ...")
    model = YOLO(model_yaml)
    total_layers = len(model.model.model)
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"  Layers: {total_layers}, Params: {total_params:,}")

    # 3. 加载预训练权重（根据映射表）
    if not os.path.exists(pretrained_path):
        print(f"[Error] Pretrained weights not found: {pretrained_path}")
        sys.exit(1)
    print(f"\nLoading pretrained from {pretrained_path} ...")
    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    src_sd = ckpt.get("model", ckpt)
    if isinstance(src_sd, torch.nn.Module):
        src_sd = src_sd.state_dict()

    target_sd = model.model.state_dict()
    loaded_dict, mismatches, not_founds = build_loaded_state_dict(src_sd, target_sd, mapping)

    model.model.load_state_dict(loaded_dict, strict=False)
    print(f"  Loaded {len(loaded_dict)} / {len(target_sd)} parameter tensors")

    if mismatches:
        print(f"  Shape mismatches ({len(mismatches)}):")
        for s, t, sh, th in mismatches[:5]:
            print(f"    {s} -> {t}:  {list(sh)} vs {list(th)}")
    if not_founds:
        print(f"  Keys not found in target ({len(not_founds)}):")
        for s, t in not_founds[:5]:
            print(f"    {s} -> {t}  (target key '{t}' missing)")

    # 避免 model.train() 丢弃已加载的权重
    model.ckpt = {"epoch": 0}

    # 4. 确定冻结策略：已映射层冻结，未映射层训练
    mapped_tgt_idxs = set(mapping.values())
    freeze_list = sorted(mapped_tgt_idxs)
    trainable_idxs = [i for i in range(total_layers) if i not in mapped_tgt_idxs]

    # 计算可训练参数量
    learnable_params = sum(
        p.numel() for n, p in model.model.named_parameters()
        if not any(f"model.{i}." in n for i in freeze_list)
    )

    print(f"\n{'='*60}")
    print(f"Freeze strategy: mapped layers frozen, unmapped layers trained")
    print(f"  Frozen  layers ({len(freeze_list)}): {freeze_list}")
    print(f"  Trainable layers ({len(trainable_idxs)}): {trainable_idxs}")
    print(f"  Trainable params: {learnable_params:,} ({learnable_params/total_params*100:.2f}%)")
    print(f"{'='*60}\n")

    # 5. 开始训练
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        device=device,
        freeze=freeze_list,
        optimizer="AdamW",
        lr0=args.lr,
        lrf=args.lrf,
        weight_decay=5e-4,
        save=True,
        save_period=args.save_period,
        workers=args.workers,
        amp=True,
        project=os.path.join(PROJECT_ROOT, "runs/train"),
        name=output_name,
        exist_ok=True,
        verbose=True,
        pretrained=False,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=args.cos_lr,
        fraction=args.fraction,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()