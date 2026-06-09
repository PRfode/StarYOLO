# 使用 torch 加载 YOLOv8n 模型，支持 COCO128 / COCO2017 数据集进行预测
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import numpy as np

from dataset import COCO128Dataset, COCO2017Dataset, collate_fn
from ultralytics import YOLO

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

    split = cfg.get("val", cfg.get("train", "train2017"))
    split_name = Path(split).name
    img_dir = dataset_root / split

    if which == 'coco2017' or which == 'coco512':
        ann_dir = dataset_root / "annotations"
        ann_file = ann_dir / f"instances_{split_name}.json"
        print(f"Dataset type: COCO2017 (JSON annotation)")
        dataset = COCO2017Dataset(img_dir, ann_file, img_size=640)
    elif which == 'coco128':
        print(f"Dataset type: COCO128 (YOLO .txt labels)")
        dataset = COCO128Dataset(img_dir, img_size=640)
    else:
        raise ValueError(f"Unsupported dataset type: {which}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
    print(f"  Images: {len(dataset)}  |  Batch size: {batch_size}")
    dataloader.dataset.num_classes = len(names)
    return dataloader, names


def compute_map(all_preds, all_gts, num_classes=80):
    """计算 mAP 指标。

    Args:
        all_preds: list of [N_i, 6] = (x1, y1, x2, y2, conf, cls) 在原图绝对坐标
        all_gts:   list of [M_i, 5] = (cls, x1, y1, x2, y2) 在原图绝对坐标
        num_classes: 类别数 (COCO=80)

    Returns:
        map_50_95, map_50, map_75
    """
    from ultralytics.utils.metrics import ap_per_class, box_iou

    device = all_preds[0].device if len(all_preds) > 0 else "cpu"
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = len(iouv)

    # 展平所有 predictions 和 ground truths
    pred_list = []   # [x1, y1, x2, y2, conf, cls]
    gt_list = []     # [x1, y1, x2, y2, cls]
    gt_cls_list = [] # cls for each GT

    for preds, gts in zip(all_preds, all_gts):
        pred_list.append(preds)
        for i in range(gts.shape[0]):
            gt_list.append(gts[i, 1:5])  # x1,y1,x2,y2
            gt_cls_list.append(gts[i, 0])  # cls

    if not pred_list:
        return 0.0, 0.0, 0.0

    preds = torch.cat(pred_list, dim=0) if pred_list else torch.zeros(0, 6, device=device)
    # targets: [M, 5] = (img_id, cls, x1, y1, x2, y2)
    targets = []
    for img_id, (preds_i, gts_i) in enumerate(zip(all_preds, all_gts)):
        for i in range(gts_i.shape[0]):
            targets.append(torch.tensor([img_id, gts_i[i, 0], *gts_i[i, 1:5]], device=device))
    targets = torch.stack(targets) if targets else torch.zeros(0, 6, device=device)

    if preds.shape[0] == 0 or targets.shape[0] == 0:
        return 0.0, 0.0, 0.0

    # 按置信度排序
    sort_idx = preds[:, 4].argsort(descending=True)
    preds = preds[sort_idx]

    # 按图片和类别做匹配
    nl, npr = targets.shape[0], preds.shape[0]
    tp = torch.zeros(npr, niou, dtype=torch.bool, device=device)

    # 按 image_id 分组 targets
    gt_by_img = {}
    for t in targets:
        img_id = int(t[0].item())
        if img_id not in gt_by_img:
            gt_by_img[img_id] = []
        gt_by_img[img_id].append(t[1:])  # (cls, x1, y1, x2, y2)

    # 按 image_id 分组 predictions
    pred_by_img = {}
    for p in preds:
        # We need img_id for each prediction. Since predictions are in order of images,
        # we need to track img_id per prediction.
        pass

    # Simpler approach: use per-image matching
    # Since predictions and targets are already per-image in all_preds/all_gts,
    # let's concatenate with image_id
    all_preds_with_img = []
    all_gts_with_img = []
    for img_id in range(len(all_preds)):
        p = all_preds[img_id]
        g = all_gts[img_id]
        if p.shape[0] > 0:
            img_col = torch.full((p.shape[0], 1), img_id, device=device)
            all_preds_with_img.append(torch.cat([img_col, p], dim=1))  # (img_id, x1,y1,x2,y2,conf,cls)
        if g.shape[0] > 0:
            img_col = torch.full((g.shape[0], 1), img_id, device=device)
            all_gts_with_img.append(torch.cat([img_col, g], dim=1))  # (img_id, cls, x1,y1,x2,y2)

    if not all_preds_with_img:
        return 0.0, 0.0, 0.0

    preds_all = torch.cat(all_preds_with_img, dim=0)
    gts_all = torch.cat(all_gts_with_img, dim=0) if all_gts_with_img else torch.zeros(0, 6, device=device)

    # 按置信度排序 predictions
    sort_idx = preds_all[:, 5].argsort(descending=True)  # sort by conf (index 5)
    preds_all = preds_all[sort_idx]

    npr = preds_all.shape[0]
    nl = gts_all.shape[0]
    tp = torch.zeros(npr, niou, dtype=torch.bool, device=device)
    fp = torch.zeros(npr, niou, dtype=torch.bool, device=device)

    # 每组 ground truth 的匹配状态
    gt_matched = torch.zeros(nl, niou, dtype=torch.bool, device=device)

    # 按 image_id 索引 ground truths
    gt_indices_by_img = {}
    for i, gt in enumerate(gts_all):
        img_id = int(gt[0].item())
        if img_id not in gt_indices_by_img:
            gt_indices_by_img[img_id] = []
        gt_indices_by_img[img_id].append(i)

    # 每个 prediction 与同图片的 GT 匹配
    for i, pred in enumerate(preds_all):
        img_id = int(pred[0].item())
        pred_cls = int(pred[6].item())
        pred_box = pred[1:5].unsqueeze(0)  # (1, 4) in xyxy

        # 找同一张图片中同一类别的 GT
        candidate_gt_idxs = gt_indices_by_img.get(img_id, [])
        best_iou = torch.zeros(niou, device=device)
        best_gt_idx = -1

        for gt_idx in candidate_gt_idxs:
            gt = gts_all[gt_idx]
            if int(gt[1].item()) != pred_cls:
                continue  # 类别不匹配
            gt_box = gt[2:6].unsqueeze(0)  # (1, 4) in xyxy
            iou = box_iou(pred_box, gt_box).squeeze(0)  # scalar

            if iou > best_iou[0]:
                best_iou = iou.expand(niou)  # same IoU for all thresholds
                best_gt_idx = gt_idx

        if best_gt_idx >= 0:
            for j in range(niou):
                if best_iou[j] >= iouv[j] and not gt_matched[best_gt_idx, j]:
                    tp[i, j] = True
                    gt_matched[best_gt_idx, j] = True
                else:
                    fp[i, j] = True
        else:
            fp[i] = True

    # 调用 ultralytics 的 ap_per_class
    tp_np = tp.cpu().numpy()
    conf_np = preds_all[:, 5].cpu().numpy()
    pred_cls_np = preds_all[:, 6].cpu().numpy().astype(int)
    target_cls_np = gts_all[:, 1].cpu().numpy().astype(int)

    ret = ap_per_class(tp_np, conf_np, pred_cls_np, target_cls_np)
    ap = ret[5]  # shape: [num_classes, 10] (10 IoU thresholds: 0.5~0.95)

    map_50_95 = ap.mean()
    map_50 = ap[:, 0].mean()
    map_75 = ap[:, 5].mean() if ap.shape[1] > 5 else 0.0

    return float(map_50_95), float(map_50), float(map_75)


ULTRALYTICS_PATH = Path('./ultralytics')

def main():
    parser = argparse.ArgumentParser(description="YOLOv8n torch inference")
    parser.add_argument("--dataset", type=str, default="coco128",
                        help="数据集名（将从 dataset_cfg/ 加载）")
    parser.add_argument("--batch", type=int, default=16, help="批大小")
    parser.add_argument("--model", type=str, default="yolov8n",
                        help="模型文件名（将从 res/model/ 加载）")
    parser.add_argument("--show_dtc", action="store_true", help="打印每一张图片的所有检测结果")
    parser.add_argument("--no-res", action="store_false", dest="show_res", default=True,
                        help="跳过性能测试 (mAP)")
    parser.add_argument("--conf", type=float, default=0.001, help="置信度阈值（mAP 评估时推荐低阈值）")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
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
    model_path = project_root / "res/model" / f'{args.model}.pt'
    model_cfg_path = ULTRALYTICS_PATH / f'models/{args.model}.yaml'
    pretrained = YOLO(str(model_cfg_path)).load(str(model_path))
    model = pretrained.model.float().eval()
    print(f"Model: {args.model} ({sum(p.numel() for p in model.parameters()):,} params)")

    # ===== 3. 构建 DataLoader =====
    dataloader, names = build_dataloader(cfg, project_root, args.dataset, args.batch)
    num_classes = len(names)

    # ===== 4. 推理 =====
    all_preds = []  # 收集每张图的预测 (用于 mAP)
    all_gts = []    # 收集每张图的 GT (用于 mAP)
    total_detections = 0

    with torch.no_grad():
        for batch in dataloader:
            images, filenames, shapes = batch[:3]
            labels = batch[3] if len(batch) > 3 else None
            images = images.to(device)

            # YOLO 返回 Results 列表，自带 NMS，框已在原图坐标 (xyxy)
            results = pretrained(images, conf=args.conf, iou=args.iou, verbose=False)

            for i, (result, fname, shape) in enumerate(zip(results, filenames, shapes)):
                if result.boxes is not None:
                    dets = result.boxes.data.clone()  # [N, 6] = (x1, y1, x2, y2, conf, cls) 在 640x640 空间
                    # 将框从 640x640 空间缩放到原图坐标
                    scale_boxes(dets[:, :4], shape)
                else:
                    dets = torch.zeros((0, 6), device=device)
                total_detections += dets.shape[0]

                # 从数据集获取 GT
                if labels is not None:
                    gt_yolo = labels[i]  # [M, 5] = (cls, cx, cy, w, h) 归一化
                    gt_abs = torch.zeros(gt_yolo.shape[0], 5, device=device)
                    if gt_yolo.shape[0] > 0:
                        h_img, w_img = shape
                        cx, cy, w_n, h_n = gt_yolo[:, 1], gt_yolo[:, 2], gt_yolo[:, 3], gt_yolo[:, 4]
                        gt_abs[:, 0] = gt_yolo[:, 0]  # cls
                        gt_abs[:, 1] = (cx - w_n / 2) * w_img  # x1
                        gt_abs[:, 2] = (cy - h_n / 2) * h_img  # y1
                        gt_abs[:, 3] = (cx + w_n / 2) * w_img  # x2
                        gt_abs[:, 4] = (cy + h_n / 2) * h_img  # y2
                else:
                    gt_abs = torch.zeros(0, 5, device=device)

                if args.show_dtc and dets.shape[0] > 0:
                    for box in dets:
                        cls_name = names[int(box[5].item())]
                        print(f"{fname}: {cls_name} {box[4]:.3f} "
                              f"[{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")

                all_preds.append(dets.cpu())
                all_gts.append(gt_abs.cpu())

    print(f"\nTotal detections: {total_detections}")

    # ===== 5. 计算 mAP (如果 show_res) =====
    if args.show_res:
        print("\n--- Performance ---")
        map_50_95, map_50, map_75 = compute_map(all_preds, all_gts, num_classes)
        print(f"APval (mAP@0.5:0.95): {map_50_95:.3f}")
        print(f"APval50 (mAP@0.5):    {map_50:.3f}")
        print(f"APval75 (mAP@0.75):   {map_75:.3f}")


if __name__ == "__main__":
    main()
