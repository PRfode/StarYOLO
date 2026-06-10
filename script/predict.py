"""
使用 YOLOv8n 在 coco512 上测试 mAP。
用法:
  conda activate yolov8
  cd G:/gitOutside/StarYOLO
  python script/predict.py --model yolov8n --dataset coco512
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLOv8n inference / val on coco")
    parser.add_argument("--dataset", type=str, default="coco512", help="数据集名")
    parser.add_argument("--model", type=str, default="yolov8n", help="模型文件名")
    parser.add_argument("--batch", type=int, default=16, help="批大小")
    parser.add_argument("--conf", type=float, default=0.001, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--show_dtc", action="store_true", help="逐图输出检测结果")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ===== 模型路径 =====
    model_path = os.path.join(PROJECT_ROOT, f'res/model/{args.model}.pt')
    data_yaml = os.path.join(PROJECT_ROOT, f'res/dataset/dataset_cfg/{args.dataset}.yaml')

    # ===== 加载模型 =====
    print(f"Loading model from {model_path} ...")
    model = YOLO(model_path)
    total = sum(p.numel() for p in model.model.parameters())
    print(f"Model params: {total:,}")

    # ===== val: 计算 mAP =====
    print(f"\nValidating on {args.dataset} (batch={args.batch}) ...")
    results = model.val(
        data=data_yaml,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=device,
        plots=False,
        save_json=False,
        verbose=True,
    )

    # # ===== 输出 mAP / mAP50 / mAP75 =====
    # metrics = results.results_dict
    # print(f"\n=== Overall Metrics ===")
    # print(f"  mAP50:    {metrics.get('metrics/mAP50', 0):.3f}")
    # print(f"  mAP75:    {metrics.get('metrics/mAP75', 0):.3f}")
    # print(f"  mAP:      {metrics.get('metrics/mAP50-95', 0):.3f}")

    # ===== 逐图推理输出 =====
    if args.show_dtc:
        print("\n--- Per-image detections ---")
        pred_results = model.predict(
            data=data_yaml,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou,
            device=device,
            verbose=False,
        )
        for result in pred_results:
            fname = Path(result.path).name
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    print(f"{fname}: {cls_name} {conf:.3f} [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")


if __name__ == "__main__":
    main()
