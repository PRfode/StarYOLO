import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np


def _load_coco_json(ann_file):
    """加载 COCO JSON 标注，返回 images 信息列表和 image_id -> annotations 的映射"""
    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 构建 category_id -> 0~79 索引的映射（按 id 排序）
    cats_sorted = sorted(data["categories"], key=lambda x: x["id"])
    cat_id_to_idx = {cat["id"]: i for i, cat in enumerate(cats_sorted)}

    # 按 image_id 整理 annotations，过滤 iscrowd
    anns_by_img = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        img_id = ann["image_id"]
        if img_id not in anns_by_img:
            anns_by_img[img_id] = []
        anns_by_img[img_id].append(ann)

    # 建立 image_id -> image info 的字典
    img_info_by_id = {}
    for img_info in data["images"]:
        img_info_by_id[img_info["id"]] = img_info

    return data["images"], anns_by_img, img_info_by_id, cat_id_to_idx


class COCO2017Dataset(Dataset):
    """使用 torch Dataset 加载 COCO2017 数据集（读取 JSON 标注）"""

    def __init__(self, img_dir, ann_file, img_size=640):
        self.img_dir = Path(img_dir)
        self.img_size = img_size

        images, self.anns_by_img, img_info_by_id, cat_id_to_idx = _load_coco_json(ann_file)

        # 只保留磁盘上确实存在的图片
        self.valid_images = []
        for img_info in images:
            img_path = self.img_dir / img_info["file_name"]
            if img_path.exists():
                self.valid_images.append(img_info)
            # 可选：如果图片多可以跳过校验以加速

        self.cat_id_to_idx = cat_id_to_idx
        print(f"  COCO2017: {len(self.valid_images)} images, "
              f"{sum(len(v) for v in self.anns_by_img.values())} annotations, "
              f"{len(cat_id_to_idx)} classes")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_info = self.valid_images[idx]
        img_path = self.img_dir / img_info["file_name"]

        # 读取图片
        img = cv2.imread(str(img_path))
        orig_h, orig_w = img.shape[:2]

        # letterbox 预处理
        img = self._letterbox(img)

        # HWC -> CHW, BGR -> RGB, 归一化
        img = img[..., ::-1].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        # 构建标签: [N, 5] (cls_id, x_center, y_center, width, height) 归一化
        labels = []
        for ann in self.anns_by_img.get(img_info["id"], []):
            x, y, w, h = ann["bbox"]  # COCO 格式: [左上x, 左上y, 宽, 高] (像素)
            cls_idx = self.cat_id_to_idx[ann["category_id"]]
            # 转为 YOLO 格式: [cx, cy, w, h] 归一化
            cx = (x + w / 2) / orig_w
            cy = (y + h / 2) / orig_h
            w_norm = w / orig_w
            h_norm = h / orig_h
            labels.append([cls_idx, cx, cy, w_norm, h_norm])

        labels_tensor = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros(0, 5)

        return torch.from_numpy(img), img_info["file_name"], (orig_h, orig_w), labels_tensor

    def _letterbox(self, img):
        h, w = img.shape[:2]
        r = self.img_size / max(h, w)
        new_w, new_h = int(round(w * r)), int(round(h * r))

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        dw = self.img_size - new_w
        dh = self.img_size - new_h
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2

        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img
