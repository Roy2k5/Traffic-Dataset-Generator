"""
detection_dataset.py
====================
PyTorch Dataset for Object Detection (Faster R-CNN / raw YOLO tensors).

Data layout expected:
    output/train/
        rgb/    XXXX.png   — 1637×991 RGB images
        yolo/   XXXX.txt   — YOLO format labels

Split mapping (sequential, no shuffle):
    train : index 0000–0899  (900 images)
    val   : index 0900–0999  (100 images)
    test  : index 1000–1199  (200 images)

Classes:
    YOLO 0 = car  → Faster R-CNN label 1
    YOLO 1 = tree → Faster R-CNN label 2
    YOLO 2 = lamppost → Faster R-CNN label 3
    (label 0 is reserved for background in Faster R-CNN)
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T

# ── Constants ────────────────────────────────────────────────────────────────
IMG_W, IMG_H = 1637, 991

SPLIT_RANGES = {
    'train': (0,    900),
    'val':   (900,  1000),
    'test':  (1000, 1200),
}

CLASS_NAMES = ['__background__', 'car', 'tree', 'lamppost']   # index = label id


# ── Collate function (required by Faster R-CNN DataLoader) ───────────────────
def collate_fn(batch):
    """Handles variable-size targets (different number of boxes per image)."""
    return tuple(zip(*batch))


# ── Dataset ──────────────────────────────────────────────────────────────────
class TrafficDetectionDataset(Dataset):
    """
    Returns:
        mode='fasterrcnn' :
            image  — FloatTensor (3, H, W) in [0, 1], NOT normalised
                     (Faster R-CNN normalises internally via GeneralizedRCNN)
            target — dict with keys:
                       'boxes'  : FloatTensor (N, 4)  [x1, y1, x2, y2] pixels
                       'labels' : LongTensor  (N,)    1-indexed (0 = background)
                       'image_id': LongTensor (1,)
        mode='yolo' :
            image  — FloatTensor (3, H, W) normalised with ImageNet stats
            target — FloatTensor (N, 5)   [class_id, cx, cy, w, h] normalised
    """

    def __init__(self, root_dir: str, split: str = 'train',
                 mode: str = 'fasterrcnn', augment: bool = True):
        """
        Args:
            root_dir : path to ``output/train`` directory
            split    : 'train' | 'val' | 'test'
            mode     : 'fasterrcnn' | 'yolo'
            augment  : apply random augmentation (train split only)
        """
        assert split in SPLIT_RANGES, f"split must be one of {list(SPLIT_RANGES)}"
        assert mode  in ('fasterrcnn', 'yolo')

        self.root_dir = root_dir
        self.rgb_dir  = os.path.join(root_dir, 'rgb')
        self.yolo_dir = os.path.join(root_dir, 'yolo')
        self.mode     = mode
        self.augment  = (augment and split == 'train')

        start, end    = SPLIT_RANGES[split]
        self.indices  = list(range(start, end))

        # ImageNet normalisation (used in yolo mode)
        self._normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                      std =[0.229, 0.224, 0.225])

    # ── helpers ──────────────────────────────────────────────────────────────

    def _load_image(self, idx: int) -> Image.Image:
        path = os.path.join(self.rgb_dir, f'{idx:04d}.png')
        return Image.open(path).convert('RGB')

    def _load_yolo(self, idx: int):
        """Return list of (class_id, cx, cy, w, h) tuples; empty if no labels."""
        path = os.path.join(self.yolo_dir, f'{idx:04d}.txt')
        labels = []
        if os.path.isfile(path):
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        labels.append((cls, cx, cy, bw, bh))
        return labels

    def _yolo_to_xyxy(self, labels, img_w=IMG_W, img_h=IMG_H):
        """Convert YOLO normalised format to absolute pixel [x1,y1,x2,y2]."""
        boxes, ids = [], []
        for cls, cx, cy, bw, bh in labels:
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(img_w), x2), min(float(img_h), y2)
            if x2 - x1 > 1 and y2 - y1 > 1:          # skip degenerate boxes
                boxes.append([x1, y1, x2, y2])
                ids.append(cls + 1)                   # +1: 0 reserved for BG
        return boxes, ids

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        img_pil  = self._load_image(real_idx)
        labels   = self._load_yolo(real_idx)

        # ── Faster R-CNN mode ─────────────────────────────────────────────────
        if self.mode == 'fasterrcnn':
            img_w, img_h = img_pil.size          # (W, H)

            # Optional augmentation: horizontal flip only (safe for detection)
            if self.augment and torch.rand(1).item() < 0.5:
                img_pil = TF.hflip(img_pil)
                flipped = []
                for cls, cx, cy, bw, bh in labels:
                    flipped.append((cls, 1.0 - cx, cy, bw, bh))
                labels = flipped

            img_tensor = TF.to_tensor(img_pil)       # (3,H,W) float [0,1]

            boxes, ids = self._yolo_to_xyxy(labels, img_w, img_h)

            if boxes:
                boxes_t  = torch.as_tensor(boxes, dtype=torch.float32)
                labels_t = torch.as_tensor(ids,   dtype=torch.int64)
            else:
                boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
                labels_t = torch.zeros((0,),   dtype=torch.int64)

            target = {
                'boxes'   : boxes_t,
                'labels'  : labels_t,
                'image_id': torch.tensor([real_idx], dtype=torch.int64),
            }
            return img_tensor, target

        # ── YOLO / raw mode ───────────────────────────────────────────────────
        else:
            img_tensor = TF.to_tensor(img_pil)
            img_tensor = self._normalize(img_tensor)

            if labels:
                target = torch.tensor(
                    [[cls, cx, cy, bw, bh] for cls, cx, cy, bw, bh in labels],
                    dtype=torch.float32
                )
            else:
                target = torch.zeros((0, 5), dtype=torch.float32)

            return img_tensor, target
