"""
test_fasterrcnn.py
==================
Evaluate Faster R-CNN on the test split (images 1000–1199).

Metrics:
    - Per-class AP@0.5
    - mAP@0.5
    - mAP@0.5:0.95
    - Average inference time per image

Output written to: src/test/fasterrcnn_results.txt

Run from project root:
    python src/test/test_fasterrcnn.py
"""

import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from dataloader.detection_dataset import TrafficDetectionDataset, collate_fn

# ── Paths & Config ────────────────────────────────────────────────────────────
DATA_ROOT   = os.path.join(ROOT, 'output', 'train')
CKPT_PATH   = os.path.join(ROOT, 'src', 'checkpoint', 'fasterrcnn', 'best.pth')
RESULT_PATH = os.path.join(ROOT, 'src', 'test', 'fasterrcnn_results.txt')
NUM_CLASSES = 4          # background + car + tree + lamppost
CLASS_NAMES = ['car', 'tree', 'lamppost']   # without background for reporting


def build_model(num_classes, ckpt_path, device):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    trained_epoch = ckpt.get('epoch', '?')
    val_map50     = ckpt.get('val_map50', float('nan'))
    print(f'Loaded checkpoint: epoch={trained_epoch}, val_mAP@0.5={val_map50:.4f}')
    model.to(device).eval()
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    metric      = MeanAveragePrecision(iou_type='bbox', class_metrics=True)
    infer_times = []

    for images, targets in tqdm(loader, desc='Testing'):
        images = [img.to(device) for img in images]

        t0      = time.time()
        outputs = model(images)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed = (time.time() - t0) / len(images) * 1000   # ms per image
        infer_times.append(elapsed)

        preds, gts = [], []
        for out, tgt in zip(outputs, targets):
            preds.append({
                'boxes' : out['boxes'].cpu(),
                'scores': out['scores'].cpu(),
                'labels': out['labels'].cpu(),
            })
            gts.append({
                'boxes' : tgt['boxes'],
                'labels': tgt['labels'],
            })
        metric.update(preds, gts)

    result   = metric.compute()
    map50    = float(result['map_50'].item())
    map_5095 = float(result['map'].item())

    # per-class AP@0.5
    per_class_ap50 = []
    if 'map_per_class' in result:
        # torchmetrics returns per-class map at threshold 0.5 via map_50_per_class
        for v in result.get('mar_100_per_class', []):
            per_class_ap50.append(float(v))
    # Use ap_class_all from map_per_class if available
    if 'map_per_class' in result:
        per_class_ap50 = [float(v) for v in result['map_per_class']]

    avg_infer = float(np.mean(infer_times))
    return map50, map_5095, per_class_ap50, avg_infer


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if not os.path.isfile(CKPT_PATH):
        print(f'ERROR: Checkpoint not found: {CKPT_PATH}')
        sys.exit(1)

    model = build_model(NUM_CLASSES, CKPT_PATH, device)

    test_ds = TrafficDetectionDataset(DATA_ROOT, split='test',
                                      mode='fasterrcnn', augment=False)
    loader  = DataLoader(test_ds, batch_size=4, shuffle=False,
                         num_workers=2, collate_fn=collate_fn)
    print(f'Test images: {len(test_ds)}')

    map50, map_5095, per_class_ap50, avg_infer = evaluate(model, loader, device)

    # ── Format output ─────────────────────────────────────────────────────────
    lines = [
        '=== Faster R-CNN Test Results ===',
        f'Dataset: {len(test_ds)} images (index 1000-1199)',
        f'Checkpoint: {CKPT_PATH}',
        '',
        'Per-class AP@0.5:',
    ]
    if per_class_ap50 and len(per_class_ap50) >= len(CLASS_NAMES):
        for i, name in enumerate(CLASS_NAMES):
            lines.append(f'  {name:<12}: {per_class_ap50[i]*100:.2f}%')
    else:
        lines.append('  (per-class AP not available — check torchmetrics version)')

    lines += [
        '',
        f'mAP@0.5     : {map50*100:.2f}%',
        f'mAP@0.5:0.95: {map_5095*100:.2f}%',
        '',
        f'Inference time: {avg_infer:.1f} ms/image',
    ]

    report = '\n'.join(lines)
    print('\n' + report)

    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    print(f'\nResults saved to: {RESULT_PATH}')


if __name__ == '__main__':
    main()
