"""
test_deeplabv3.py
=================
Evaluate DeepLabV3+ on the test split (images 1000–1199).

Metrics:
    - Per-class IoU  (background, car, tree, lamppost)
    - Mean IoU
    - Pixel Accuracy

Output written to: src/test/deeplabv3_results.txt

Run from project root:
    python src/test/test_deeplabv3.py
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from dataloader.segmentation_dataset import TrafficSegDataset, NUM_SEG_CLASSES

# ── Paths & Config ────────────────────────────────────────────────────────────
DATA_ROOT   = os.path.join(ROOT, 'output', 'train')
CKPT_PATH   = os.path.join(ROOT, 'src', 'checkpoint', 'deeplabv3', 'best.pth')
RESULT_PATH = os.path.join(ROOT, 'src', 'test', 'deeplabv3_results.txt')
IMG_SIZE    = (512, 512)
CLASS_NAMES = ['background', 'car', 'tree', 'lamppost']


def build_model(num_classes, ckpt_path, device):
    model = deeplabv3_resnet50(weights=None, aux_loss=True)
    model.classifier[4]     = nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    trained_epoch = ckpt.get('epoch', '?')
    val_miou      = ckpt.get('val_miou', float('nan'))
    print(f'Loaded checkpoint: epoch={trained_epoch}, val_mIoU={val_miou:.4f}')
    model.to(device).eval()
    return model


def compute_iou_acc(preds, targets, num_classes):
    iou_list = []
    for c in range(num_classes):
        pred_c   = (preds == c)
        target_c = (targets == c)
        inter    = (pred_c & target_c).sum().item()
        union    = (pred_c | target_c).sum().item()
        iou_list.append(inter / (union + 1e-8))
    pixel_acc = (preds == targets).sum().item() / (targets.numel() + 1e-8)
    return iou_list, pixel_acc


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    all_iou   = [0.0] * num_classes
    all_acc   = 0.0
    n_batches = 0
    infer_ms  = []

    for images, masks in tqdm(loader, desc='Testing'):
        images = images.to(device)

        t0  = time.time()
        out = model(images)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.time() - t0) / len(images) * 1000
        infer_ms.append(elapsed)

        logits = out['out']
        preds  = logits.argmax(dim=1).cpu()
        iou, acc = compute_iou_acc(preds, masks, num_classes)
        for c in range(num_classes):
            all_iou[c] += iou[c]
        all_acc   += acc
        n_batches += 1

    avg_iou  = [v / max(n_batches, 1) for v in all_iou]
    mean_iou = float(np.mean(avg_iou))
    avg_acc  = all_acc / max(n_batches, 1)
    avg_ms   = float(np.mean(infer_ms))
    return avg_iou, mean_iou, avg_acc, avg_ms


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if not os.path.isfile(CKPT_PATH):
        print(f'ERROR: Checkpoint not found: {CKPT_PATH}')
        sys.exit(1)

    model = build_model(NUM_SEG_CLASSES, CKPT_PATH, device)

    test_ds = TrafficSegDataset(DATA_ROOT, split='test',
                                img_size=IMG_SIZE,
                                normalize_for_deeplab=True,
                                augment=False)
    loader  = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2)
    print(f'Test images: {len(test_ds)}')

    avg_iou, mean_iou, avg_acc, avg_ms = evaluate(
        model, loader, device, NUM_SEG_CLASSES
    )

    lines = [
        '=== DeepLabV3 Segmentation Test Results ===',
        f'Dataset: {len(test_ds)} images (index 1000-1199)',
        f'Checkpoint: {CKPT_PATH}',
        '',
        'Per-class IoU:',
    ]
    for c, name in enumerate(CLASS_NAMES):
        lines.append(f'  {name:<12}: {avg_iou[c]*100:.2f}%')
    lines += [
        '',
        f'Mean IoU      : {mean_iou*100:.2f}%',
        f'Pixel Accuracy: {avg_acc*100:.2f}%',
        '',
        f'Inference time: {avg_ms:.1f} ms/image',
    ]

    report = '\n'.join(lines)
    print('\n' + report)

    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    print(f'\nResults saved to: {RESULT_PATH}')


if __name__ == '__main__':
    main()
