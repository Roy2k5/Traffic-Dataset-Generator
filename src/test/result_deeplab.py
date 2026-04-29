"""
result_deeplab.py
=================
Visualize DeepLabV3 segmentation results on 5 sample test images.

For each image, saves a side-by-side figure:
    [Original RGB] | [Ground-Truth Mask] | [Predicted Mask]

Output saved to: src/test/results_deeplabv3/

Run from project root:
    python src/test/result_deeplab.py
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from dataloader.segmentation_dataset import NUM_SEG_CLASSES

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT   = os.path.join(ROOT, 'output', 'train')
CKPT_PATH   = os.path.join(ROOT, 'src', 'checkpoint', 'deeplabv3', 'best.pth')
OUT_DIR     = os.path.join(ROOT, 'src', 'test', 'results_deeplabv3')
IMG_SIZE    = (256, 256)   # must match train_deeplabv3.py

SAMPLE_INDICES = [1000, 1030, 1060, 1100, 1150]

CLASS_NAMES  = ['background', 'car', 'tree', 'lamppost']
CLASS_COLORS = np.array([
    [0,   0,   0],     # background
    [220, 20,  60],    # car — crimson
    [0,   180, 0],     # tree — green
    [255, 215, 0],     # lamppost — gold
], dtype=np.uint8)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_model(ckpt_path, device):
    model = deeplabv3_resnet50(weights=None, aux_loss=True)
    model.classifier[4]     = nn.Conv2d(256, NUM_SEG_CLASSES, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, NUM_SEG_CLASSES, kernel_size=1)

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    epoch = ckpt.get('epoch', '?')
    miou  = ckpt.get('val_miou', float('nan'))
    print(f'Loaded DeepLabV3 checkpoint: epoch={epoch}, val_mIoU={miou:.4f}')
    model.to(device).eval()
    return model


def mask_to_rgb(mask_np, colors):
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c, color in enumerate(colors):
        rgb[mask_np == c] = color
    return rgb


def make_legend(class_names, class_colors):
    patches = []
    for name, color in zip(class_names, class_colors):
        patches.append(mpatches.Patch(
            color=np.array(color) / 255.0, label=name
        ))
    return patches


@torch.no_grad()
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if not os.path.isfile(CKPT_PATH):
        print(f'ERROR: Checkpoint not found: {CKPT_PATH}')
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    model     = build_model(CKPT_PATH, device)
    normalize = T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    sem_dir = os.path.join(DATA_ROOT, 'semantic_mask')
    has_gt  = os.path.isdir(sem_dir)

    legend_patches = make_legend(CLASS_NAMES, CLASS_COLORS)

    for idx in SAMPLE_INDICES:
        img_path = os.path.join(DATA_ROOT, 'rgb', f'{idx:04d}.png')
        img_pil  = Image.open(img_path).convert('RGB')

        img_resized = img_pil.resize(IMG_SIZE, Image.BILINEAR)
        img_tensor  = TF.to_tensor(img_resized)
        img_tensor  = normalize(img_tensor)
        inp         = img_tensor.unsqueeze(0).to(device)

        out    = model(inp)
        logits = out['out']                                    # (1, C, H, W)
        pred   = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        pred_rgb = mask_to_rgb(pred, CLASS_COLORS)

        if has_gt:
            gt_mask = np.load(os.path.join(sem_dir, f'{idx:04d}.npy'))
            gt_pil  = Image.fromarray(gt_mask, mode='L').resize(IMG_SIZE, Image.NEAREST)
            gt_rgb  = mask_to_rgb(np.array(gt_pil), CLASS_COLORS)
            ncols   = 3
        else:
            ncols = 2

        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
        fig.suptitle(f'DeepLabV3 Segmentation — Image {idx:04d}',
                     fontsize=16, y=1.02)

        axes[0].imshow(img_pil.resize(IMG_SIZE, Image.BILINEAR))
        axes[0].set_title('Original RGB')
        axes[0].axis('off')

        col = 1
        if has_gt:
            axes[col].imshow(gt_rgb)
            axes[col].set_title('Ground Truth')
            axes[col].axis('off')
            col += 1

        axes[col].imshow(pred_rgb)
        axes[col].set_title('Prediction')
        axes[col].axis('off')

        fig.legend(handles=legend_patches, loc='lower center',
                   ncol=len(CLASS_NAMES), fontsize=10,
                   bbox_to_anchor=(0.5, -0.02))

        save_path = os.path.join(OUT_DIR, f'{idx:04d}.png')
        fig.savefig(save_path, bbox_inches='tight', dpi=150, pad_inches=0.3)
        plt.close(fig)
        print(f'  Saved: {save_path}')

    print(f'\nAll results saved to: {OUT_DIR}')


if __name__ == '__main__':
    main()
