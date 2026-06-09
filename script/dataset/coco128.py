import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np


def collate_fn(batch):
    """自定义 collate：保留 shape 为原始 tuple，避免 default_collate 拆包"""
    imgs = torch.stack([item[0] for item in batch], 0)
    names = [item[1] for item in batch]
    shapes = [item[2] for item in batch]
    # 如果包含标签则也一并保留
    labels = [item[3] for item in batch] if len(batch[0]) > 3 else None
    if labels is not None:
        return imgs, names, shapes, labels
    return imgs, names, shapes


class COCO128Dataset(Dataset):
    """使用 torch Dataset 加载 COCO128 数据集（YOLO .txt 标签格式）"""

    def __init__(self, img_dir, img_size=640):
        self.img_dir = Path(img_dir)
        self.img_paths = sorted(self.img_dir.glob("*.jpg"))
        self.img_size = img_size

        # 标签目录：images/train2017 → labels/train2017
        img_parent = self.img_dir.parent  # images/
        label_parent = img_parent.parent / "labels"  # ../labels/
        split_name = self.img_dir.name   # train2017
        self.label_dir = label_parent / split_name if label_parent.exists() else None

    def __len__(self):
        return len(self.img_paths)

    def _load_label(self, img_path):
        """加载 YOLO 格式 .txt 标签，返回 [N, 5] = (cls, cx, cy, w, h) 归一化"""
        if self.label_dir is None:
            return torch.zeros(0, 5)
        txt_path = self.label_dir / img_path.name.replace(".jpg", ".txt")
        if not txt_path.exists():
            return torch.zeros(0, 5)
        boxes = []
        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls_id = float(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                boxes.append([cls_id, cx, cy, w, h])
        return torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 5)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(str(img_path))
        orig_shape = img.shape[:2]

        labels = self._load_label(img_path)

        img = self._letterbox(img)

        # HWC -> CHW, BGR -> RGB, 归一化
        img = img[..., ::-1].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        return torch.from_numpy(img), img_path.name, orig_shape, labels

    def _letterbox(self, img):
        """等比例缩放 + 边缘填充至 self.img_size x self.img_size"""
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
