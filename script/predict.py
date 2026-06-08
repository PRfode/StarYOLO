# 使用 torch 加载 YOLOv8n 模型和 COCO128 数据集进行预测
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np

from dataset import COCO128Dataset, collate_fn


def non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45):
    """对模型原始输出做 NMS（支持批量）"""
    from torchvision.ops import nms

    device = preds.device
    batch_detections = []

    for pred in preds:  # pred shape: [84, 8400]
        pred = pred.permute(1, 0)  # [8400, 84]
        boxes = pred[:, :4]
        scores, cls_id = pred[:, 4:].max(dim=1)

        # 置信度过滤
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

        # 按类别分组做 NMS
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

    # 去除填充偏移
    boxes[:, [0, 2]] -= dw / 2
    boxes[:, [1, 3]] -= dh / 2

    # 缩放到原图
    boxes[:, [0, 2]] *= (w / new_w)
    boxes[:, [1, 3]] *= (h / new_h)

    # 裁剪到图像边界
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)

    return boxes


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    project_root = Path('./')

    # ===== 1. 从 dataset_cfg 读取 YAML 配置 =====
    cfg_path = project_root / 'res/dataset/dataset_cfg/coco128.yaml'
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 解析数据集路径：path + train 组成完整图片目录
    dataset_root = project_root / cfg["path"]
    img_dir = dataset_root / cfg["train"]
    names = cfg["names"]  # dict {0: "person", 1: "bicycle", ...}
    print(f"Config loaded: {cfg_path.name} ({len(names)} classes)")

    # ===== 2. 使用 torch 加载 YOLOv8n 模型 =====
    model_path = project_root / "res/model/yolov8n.pt"
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = ckpt["model"].to(device)
    model.float().eval()
    print(f"Model loaded: YOLOv8n ({sum(p.numel() for p in model.parameters()):,} params)")

    # ===== 3. 使用 torch DataLoader 加载数据集 =====
    dataset = COCO128Dataset(img_dir, img_size=640)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
    print(f"Dataset loaded: {len(dataset)} images from {img_dir}")

    # ===== 4. 推理 =====
    total_detections = 0
    with torch.no_grad():
        for images, filenames, shapes in dataloader:
            images = images.to(device)

            # 前向传播
            preds = model(images)

            # NMS 后处理
            raw = preds[0] if isinstance(preds, (list, tuple)) else preds
            results = non_max_suppression(raw)

            # 输出预测结果
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
