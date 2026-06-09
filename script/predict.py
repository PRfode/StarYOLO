# 使用 torch 加载 YOLOv8n 模型，支持 COCO128 / COCO2017 数据集进行预测
import sys
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse

from dataset import COCO128Dataset, COCO2017Dataset, collate_fn


def non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45):
    """对模型原始输出做 NMS（支持批量）"""
    from torchvision.ops import nms

    device = preds.device
    batch_detections = []

    for pred in preds:
        pred = pred.permute(1, 0)  # [8400, 84]
        boxes = pred[:, :4]
        scores, cls_id = pred[:, 4:].max(dim=1)

        mask = scores > conf_thres
        boxes, scores, cls_id = boxes[mask], scores[mask], cls_id[mask]

        if boxes.shape[0] == 0:
            batch_detections.append(torch.zeros((0, 6), device=device))
            continue

        # cxcywh -> xyxy
        boxes_xyxy = torch.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        keep = nms(boxes_xyxy, scores, iou_thres)
        boxes_xyxy, scores, cls_id = boxes_xyxy[keep], scores[keep], cls_id[keep]

        dets = torch.cat([boxes_xyxy, scores.unsqueeze(1), cls_id.unsqueeze(1).float()], dim=1)
        batch_detections.append(dets)

    return batch_detections


def scale_boxes(boxes, orig_shape, img_size=640):
    """将预测框坐标从模型输出空间缩放回原图尺寸"""
    h, w = orig_shape
    r = img_size / max(h, w)
    new_w, new_h = int(round(w * r)), int(round(h * r))

    dw = img_size - new_w
    dh = img_size - new_h

    boxes[:, [0, 2]] -= dw / 2
    boxes[:, [1, 3]] -= dh / 2
    boxes[:, [0, 2]] *= (w / new_w)
    boxes[:, [1, 3]] *= (h / new_h)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)

    return boxes


def build_dataloader(cfg, project_root, which, batch_size=16):
    """根据 YAML 配置构建 DataLoader"""
    dataset_root = project_root / cfg["path"]
    names = cfg["names"]

    # 判断类型
    # 从 yaml 中取 split，可能是 "images/train2017"，只需末尾部分作标注文件名
    split = cfg.get("val", cfg.get("train", "train2017"))
    split_name = Path(split).name  # "train2017" / "val2017"
    img_dir = dataset_root / split  # yaml 值已包含 images/ 前缀

    if which == 'coco2017' or which == 'coco512':
        # COCO2017 风格：JSON 标注
        ann_dir = dataset_root / "annotations"
        ann_file = ann_dir / f"instances_{split_name}.json"
        print(f"Dataset type: COCO2017 (JSON annotation)")
        dataset = COCO2017Dataset(img_dir, ann_file, img_size=640)
    elif which == 'coco128':
        # COCO128 风格：YOLO .txt 标签
        print(f"Dataset type: COCO128 (YOLO .txt labels)")
        dataset = COCO128Dataset(img_dir, img_size=640)
    else:
        raise ValueError(f"Unsupported dataset type: {which}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
    print(f"  Images: {len(dataset)}  |  Batch size: {batch_size}")
    return dataloader, names


def main():
    parser = argparse.ArgumentParser(description="YOLOv8n torch inference")
    parser.add_argument("--dataset", type=str, default="coco128",
                        help="数据集名（将从 dataset_cfg/ 加载）")
    parser.add_argument("--batch", type=int, default=16, help="批大小")
    # parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    # parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    project_root = Path("./")

    # ===== 1. 加载 YAML 配置 =====
    cfg_path = project_root / "res/dataset/dataset_cfg" / f'{args.dataset}.yaml'
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"Config: {cfg_path.name}")

    # ===== 2. 加载模型 =====
    model_path = project_root / "res/model/yolov8n.pt"
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = ckpt["model"].to(device)
    model.float().eval()
    print(f"Model: YOLOv8n ({sum(p.numel() for p in model.parameters()):,} params)")

    # ===== 3. 构建 DataLoader =====
    dataloader, names = build_dataloader(cfg, project_root, args.dataset, args.batch)

    # ===== 4. 推理 =====
    total_detections = 0
    with torch.no_grad():
        for batch in dataloader:
            images, filenames, shapes = batch[:3]
            images = images.to(device)

            preds = model(images)
            raw = preds[0] if isinstance(preds, (list, tuple)) else preds
            results = non_max_suppression(raw)

            for dets, fname, shape in zip(results, filenames, shapes):
                total_detections += dets.shape[0]
                if dets.shape[0] > 0:
                    boxes = scale_boxes(dets[:, :4].clone(), shape)
                    for box, score, cls_id in zip(boxes, dets[:, 4], dets[:, 5]):
                        cls_name = names[int(cls_id.item())]
                        print(f"{fname}: {cls_name} {score:.3f} "
                              f"[{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")

    print(f"\nTotal detections: {total_detections}")


if __name__ == "__main__":
    main()
