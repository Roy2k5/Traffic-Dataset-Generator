"""
segmentation_dataset.py
=======================
PyTorch Dataset for Semantic Segmentation (UNet / DeepLabV3).

Mask encoding:
    Masks are RGB instance-colormap images produced by the renderer.
    They are pre-converted to semantic label maps (.npy) by:
        python src/dataloader/precompute_masks.py   ← run this ONCE first

    Semantic label values:
        0 = background
        1 = car       (YOLO class 0)
        2 = tree      (YOLO class 1)
        3 = lamppost  (YOLO class 2)

Data layout:
    output/train/
        rgb/             XXXX.png   — 1637×991 RGB images
        semantic_mask/   XXXX.npy   — (991, 1637) uint8 semantic labels

Split mapping (sequential, no shuffle):
    train : 0000–0899  (900 images)
    val   : 0900–0999  (100 images)
    test  : 1000–1199  (200 images)
"""

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T

# ── Constants ────────────────────────────────────────────────────────────────
NUM_SEG_CLASSES = 4          # background + car + tree + lamppost
CLASS_NAMES     = ['background', 'car', 'tree', 'lamppost']

SPLIT_RANGES = {
    'train': (0,    900),
    'val':   (900,  1000),
    'test':  (1000, 1200),
}

# ImageNet normalisation (DeepLabV3 backbone trained on ImageNet)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Generic normalisation (UNet — centred around 0.5)
_GENERIC_MEAN  = [0.5, 0.5, 0.5]
_GENERIC_STD   = [0.5, 0.5, 0.5]


class TrafficSegDataset(Dataset):
    """
    Returns:
        image  — FloatTensor (3, img_size[1], img_size[0])  normalised
        mask   — LongTensor  (img_size[1], img_size[0])     semantic labels [0..3]

    Args:
        root_dir            : path to ``output/train``
        split               : 'train' | 'val' | 'test'
        img_size            : (width, height) tuple, default (512, 512)
        normalize_for_deeplab: True  → ImageNet mean/std
                              False → generic [0.5, 0.5, 0.5] mean/std
        augment             : random horizontal flip (train only)
    """

    def __init__(self, root_dir: str, split: str = 'train',
                 img_size: tuple = (512, 512),
                 normalize_for_deeplab: bool = False,
                 augment: bool = True):
        assert split in SPLIT_RANGES, f"split must be one of {list(SPLIT_RANGES)}"

        self.root_dir   = root_dir
        self.rgb_dir    = os.path.join(root_dir, 'rgb')
        self.mask_dir   = os.path.join(root_dir, 'semantic_mask')
        self.img_size   = img_size               # (W, H)
        self.augment    = (augment and split == 'train')

        # Check pre-computed masks exist
        if not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(
                f"Semantic mask directory not found: {self.mask_dir}\n"
                "Please run:  python src/dataloader/precompute_masks.py"
            )

        mean = _IMAGENET_MEAN if normalize_for_deeplab else _GENERIC_MEAN
        std  = _IMAGENET_STD  if normalize_for_deeplab else _GENERIC_STD
        self._normalize = T.Normalize(mean=mean, std=std)

        start, end      = SPLIT_RANGES[split]
        self.indices    = list(range(start, end))

    # ── helpers ──────────────────────────────────────────────────────────────

    def _load_image(self, idx: int) -> Image.Image:
        path = os.path.join(self.rgb_dir, f'{idx:04d}.png')
        return Image.open(path).convert('RGB')

    def _load_mask(self, idx: int) -> np.ndarray:
        path = os.path.join(self.mask_dir, f'{idx:04d}.npy')
        return np.load(path)                     # (H, W) uint8

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]

        img_pil  = self._load_image(real_idx)
        mask_np  = self._load_mask(real_idx)     # (H, W) uint8

        # ── Resize ───────────────────────────────────────────────────────────
        img_pil  = img_pil.resize(self.img_size, Image.BILINEAR)
        mask_pil = Image.fromarray(mask_np, mode='L')
        mask_pil = mask_pil.resize(self.img_size, Image.NEAREST)   # no interpolation for labels

        # ── Augmentation (train only) ─────────────────────────────────────────
        if self.augment and torch.rand(1).item() < 0.5:
            img_pil  = TF.hflip(img_pil)
            mask_pil = TF.hflip(mask_pil)

        # ── Convert to tensors ────────────────────────────────────────────────
        img_tensor  = TF.to_tensor(img_pil)             # (3, H, W) float [0,1]
        img_tensor  = self._normalize(img_tensor)

        mask_tensor = torch.from_numpy(
            np.array(mask_pil, dtype=np.int64)
        )                                                # (H, W) long

        return img_tensor, mask_tensor
