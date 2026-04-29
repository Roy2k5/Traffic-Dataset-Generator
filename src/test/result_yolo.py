"""
result_yolo.py
==============
Visualize YOLOv5 detection results on 5 sample test images.

For each image, draws bounding boxes with class labels and confidence scores.
Uses the ultralytics predict API.

Output saved to: src/test/results_yolov5/

Run from project root:
    python src/test/result_yolo.py
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT    = os.path.join(ROOT, 'output', 'train')
BEST_WEIGHTS = os.path.join(ROOT, 'src', 'logs', 'yolov5', 'run', 'weights', 'best.pt')
OUT_DIR      = os.path.join(ROOT, 'src', 'test', 'results_yolov5')
CONF_THRESHOLD = 0.5

SAMPLE_INDICES = [1000, 1030, 1060, 1100, 1150]

# YOLO class ids: 0=car, 1=tree, 2=lamppost
CLASS_NAMES = {0: 'car', 1: 'tree', 2: 'lamppost'}
CLASS_COLORS = {
    0: '#DC143C',    # car — crimson
    1: '#00B400',    # tree — green
    2: '#FFD700',    # lamppost — gold
}


def load_gt_boxes(idx):
    """Load ground-truth YOLO labels and convert to pixel [x1,y1,x2,y2]."""
    yolo_path = os.path.join(DATA_ROOT, 'yolo', f'{idx:04d}.txt')
    if not os.path.isfile(yolo_path):
        return [], []

    img_path = os.path.join(DATA_ROOT, 'rgb', f'{idx:04d}.png')
    img = Image.open(img_path)
    w, h = img.size

    boxes, labels = [], []
    with open(yolo_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            boxes.append([x1, y1, x2, y2])
            labels.append(cls)
    return boxes, labels


def draw_boxes(ax, boxes, labels, scores=None, is_gt=False):
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        color = CLASS_COLORS.get(label, '#FFFFFF')
        lw = 2 if is_gt else 2.5

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=lw, edgecolor=color, facecolor='none',
            linestyle='--' if is_gt else '-'
        )
        ax.add_patch(rect)

        name = CLASS_NAMES.get(label, f'cls{label}')
        if scores is not None:
            text = f'{name} {scores[i]:.2f}'
        else:
            text = f'{name} (GT)'

        ax.text(x1, y1 - 4, text,
                fontsize=8, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor=color, alpha=0.8, edgecolor='none'))


def main():
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    if not os.path.isfile(BEST_WEIGHTS):
        print(f'ERROR: Best weights not found: {BEST_WEIGHTS}')
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    model = YOLO(BEST_WEIGHTS)
    print(f'Loaded YOLOv5 weights: {BEST_WEIGHTS}')

    for idx in SAMPLE_INDICES:
        img_path = os.path.join(DATA_ROOT, 'rgb', f'{idx:04d}.png')
        img_pil  = Image.open(img_path).convert('RGB')
        img_np   = np.array(img_pil)

        # Inference
        results = model.predict(
            source=img_path, conf=CONF_THRESHOLD,
            device=device, verbose=False
        )
        result = results[0]

        pred_boxes  = result.boxes.xyxy.cpu().numpy()     # (N, 4)
        pred_labels = result.boxes.cls.cpu().numpy().astype(int)  # (N,)
        pred_scores = result.boxes.conf.cpu().numpy()     # (N,)

        # Ground truth
        gt_boxes, gt_labels = load_gt_boxes(idx)

        # ── Plot ─────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f'YOLOv5 Detection — Image {idx:04d}',
                     fontsize=16, y=1.02)

        # Left: Ground truth
        axes[0].imshow(img_np)
        axes[0].set_title(f'Ground Truth ({len(gt_boxes)} objects)')
        axes[0].axis('off')
        if gt_boxes:
            draw_boxes(axes[0], gt_boxes, gt_labels, is_gt=True)

        # Right: Predictions
        axes[1].imshow(img_np)
        axes[1].set_title(f'Predictions ({len(pred_boxes)} detections, thr={CONF_THRESHOLD})')
        axes[1].axis('off')
        if len(pred_boxes) > 0:
            draw_boxes(axes[1], pred_boxes, pred_labels, pred_scores)

        # Legend
        legend_patches = [
            patches.Patch(color=c, label=CLASS_NAMES[k])
            for k, c in CLASS_COLORS.items()
        ]
        fig.legend(handles=legend_patches, loc='lower center',
                   ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))

        save_path = os.path.join(OUT_DIR, f'{idx:04d}.png')
        fig.savefig(save_path, bbox_inches='tight', dpi=150, pad_inches=0.3)
        plt.close(fig)
        print(f'  Saved: {save_path}')

    print(f'\nAll results saved to: {OUT_DIR}')


if __name__ == '__main__':
    main()
