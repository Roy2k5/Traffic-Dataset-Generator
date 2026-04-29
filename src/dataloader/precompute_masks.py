"""
precompute_masks.py
====================
Chạy một lần để chuyển đổi instance mask RGB → semantic label map (.npy).

Usage (từ thư mục gốc dự án):
    python src/dataloader/precompute_masks.py

Output: output/train/semantic_mask/XXXX.npy
  - shape (991, 1637) uint8
  - values: 0=background, 1=car, 2=tree, 3=lamppost
"""

import os
import sys
import colorsys
import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = os.path.join(os.path.dirname(__file__), '..', '..')
MASK_DIR   = os.path.join(ROOT, 'output', 'train', 'mask')
YOLO_DIR   = os.path.join(ROOT, 'output', 'train', 'yolo')
OUT_DIR    = os.path.join(ROOT, 'output', 'train', 'semantic_mask')
IMG_W, IMG_H = 1637, 991
TOTAL       = 1200

# ── Semantic class mapping ───────────────────────────────────────────────────
# YOLO class_id (0,1,2) + 1 → semantic label (1,2,3)
# 0 = background
CLASS_NAMES = {0: 'background', 1: 'car', 2: 'tree', 3: 'lamppost'}


def build_semantic_mask(mask_rgb: np.ndarray, yolo_path: str,
                        img_w: int = IMG_W, img_h: int = IMG_H) -> np.ndarray:
    """
    Convert an RGB instance-colormap mask to a semantic label map.

    Args:
        mask_rgb : ndarray (H, W, 3) uint8 — raw RGB mask from renderer
        yolo_path: path to the corresponding YOLO .txt label file
        img_w, img_h: original image resolution

    Returns:
        ndarray (H, W) uint8 with values:
          0 = background
          1 = car  (YOLO class 0)
          2 = tree (YOLO class 1)
          3 = lamppost (YOLO class 2)
    """
    semantic = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)

    if not os.path.isfile(yolo_path):
        return semantic  # no objects → all background

    with open(yolo_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    seen_colors: dict = {}  # dominant_color → semantic_label

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cls        = int(parts[0])              # YOLO class id (0,1,2)
        cx, cy, bw, bh = map(float, parts[1:])

        # Convert normalized bbox → pixel
        x1 = max(0,        int((cx - bw / 2) * img_w))
        y1 = max(0,        int((cy - bh / 2) * img_h))
        x2 = min(img_w,    int((cx + bw / 2) * img_w))
        y2 = min(img_h,    int((cy + bh / 2) * img_h))

        patch = mask_rgb[y1:y2, x1:x2]         # (patch_H, patch_W, 3)
        if patch.size == 0:
            continue

        # Exclude background (pure black)
        is_bg    = np.all(patch == 0, axis=2)
        fg_pixels = patch[~is_bg]               # (N, 3)
        if len(fg_pixels) == 0:
            continue

        # Dominant colour in the foreground region
        uniq, counts   = np.unique(fg_pixels.reshape(-1, 3), axis=0, return_counts=True)
        dom_color      = tuple(int(v) for v in uniq[counts.argmax()])

        if dom_color in seen_colors:
            # Colour already assigned — use stored label (handles duplicates)
            label = seen_colors[dom_color]
        else:
            label = cls + 1              # 0→1(car), 1→2(tree), 2→3(lamppost)
            seen_colors[dom_color] = label

        # Paint all pixels matching this colour with the semantic label
        match = np.all(mask_rgb == np.array(dom_color, dtype=np.uint8), axis=2)
        semantic[match] = label

    return semantic


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    skipped = 0

    for idx in tqdm(range(TOTAL), desc='Precomputing semantic masks'):
        out_path  = os.path.join(OUT_DIR, f'{idx:04d}.npy')
        if os.path.isfile(out_path):
            skipped += 1
            continue

        mask_path = os.path.join(MASK_DIR, f'{idx:04d}.png')
        yolo_path = os.path.join(YOLO_DIR, f'{idx:04d}.txt')

        mask_rgb  = np.array(Image.open(mask_path).convert('RGB'))
        semantic  = build_semantic_mask(mask_rgb, yolo_path)

        np.save(out_path, semantic)

    total_processed = TOTAL - skipped
    print(f'\nDone. Processed {total_processed} masks, skipped {skipped} existing.')
    print(f'Output dir: {os.path.abspath(OUT_DIR)}')

    # Quick sanity check on first mask
    sample = np.load(os.path.join(OUT_DIR, '0000.npy'))
    print(f'Sample 0000.npy  shape={sample.shape}  unique={np.unique(sample)}')


if __name__ == '__main__':
    main()
