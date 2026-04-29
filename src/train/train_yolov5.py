"""
train_yolov5.py
===============
Train YOLOv5s on the synthetic traffic dataset via ultralytics API.

Steps:
    1. Build output/yolov5_data/ with the exact folder structure ultralytics expects.
    2. Write data.yaml.
    3. Call model.train().

Config:
    model     = YOLOv5s (yolov5s.pt pretrained)
    epochs    = 50
    imgsz     = 640
    batch     = 16
    lr0       = 0.01

Checkpoints are saved automatically by ultralytics inside:
    src/logs/yolov5/run/weights/  (best.pt, last.pt, epoch*.pt every save_period)

Run from project root:
    python src/train/train_yolov5.py
"""

import os
import sys
import shutil
import random
import yaml

import torch
import numpy as np
from ultralytics import YOLO

# Ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Hyperparameters ───────────────────────────────────────────────────────────
DATA_ROOT    = os.path.join(ROOT, 'output', 'train')
YOLO_DATA    = os.path.join(ROOT, 'output', 'yolov5_data')
LOG_DIR      = os.path.join(ROOT, 'src', 'logs', 'yolov5')

EPOCHS       = 50
IMGSZ        = 640
BATCH        = 16
LR0          = 0.01
SAVE_PERIOD  = 5        # save checkpoint every N epochs

SPLIT_RANGES = {        # (start_inclusive, end_exclusive)
    'train': (0,    900),
    'val':   (900,  1000),
    # test  : (1000, 1200) — evaluated separately in test_yolov5.py
}

CLASS_NAMES = {0: 'car', 1: 'tree', 2: 'lamppost'}


# ── Build directory structure ─────────────────────────────────────────────────
def setup_yolo_dirs(force: bool = False):
    """Copy images and labels into the ultralytics-standard layout."""
    for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        os.makedirs(os.path.join(YOLO_DATA, sub), exist_ok=True)

    rgb_src  = os.path.join(DATA_ROOT, 'rgb')
    yolo_src = os.path.join(DATA_ROOT, 'yolo')

    print('Setting up YOLOv5 data directory …')
    for split, (start, end) in SPLIT_RANGES.items():
        for idx in range(start, end):
            img_src  = os.path.join(rgb_src,  f'{idx:04d}.png')
            lbl_src  = os.path.join(yolo_src, f'{idx:04d}.txt')

            img_dst  = os.path.join(YOLO_DATA, 'images',  split, f'{idx:04d}.png')
            lbl_dst  = os.path.join(YOLO_DATA, 'labels',  split, f'{idx:04d}.txt')

            if force or not os.path.isfile(img_dst):
                shutil.copy2(img_src, img_dst)
            if force or not os.path.isfile(lbl_dst):
                if os.path.isfile(lbl_src):
                    shutil.copy2(lbl_src, lbl_dst)
                else:
                    # Create empty label file (image with no objects)
                    open(lbl_dst, 'w').close()

    # Write data.yaml
    yaml_path = os.path.join(YOLO_DATA, 'data.yaml')
    data_yaml = {
        'path'  : os.path.abspath(YOLO_DATA),
        'train' : 'images/train',
        'val'   : 'images/val',
        'nc'    : len(CLASS_NAMES),
        'names' : [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))],
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f'  data.yaml written → {yaml_path}')
    return yaml_path


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    yaml_path = setup_yolo_dirs(force=False)

    model = YOLO('yolov5s.pt')      # downloads pretrained weights automatically

    results = model.train(
        data        = yaml_path,
        epochs      = EPOCHS,
        imgsz       = IMGSZ,
        batch       = BATCH,
        lr0         = LR0,
        device      = device,
        project     = LOG_DIR,
        name        = 'run',
        exist_ok    = True,         # allow reuse of 'run' dir
        save        = True,
        save_period = SAVE_PERIOD,
        seed        = SEED,
        verbose     = True,
    )

    print('\nTraining complete.')
    best_weights = os.path.join(LOG_DIR, 'run', 'weights', 'best.pt')
    if os.path.isfile(best_weights):
        print(f'Best weights: {best_weights}')
    else:
        print('(Best weights path may differ — check src/logs/yolov5/run/weights/)')

    return results


if __name__ == '__main__':
    train()
