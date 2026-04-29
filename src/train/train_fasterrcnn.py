"""
train_fasterrcnn.py
===================
Train Faster R-CNN (ResNet-50 FPN backbone) on the synthetic traffic dataset.

Classes:
    background (0) | car (1) | tree (2) | lamppost (3)

Config:
    epochs        = 50
    batch_size    = 4   (reduce to 2 if OOM)
    lr            = 0.005
    momentum      = 0.9
    weight_decay  = 0.0005
    lr_step_size  = 10
    lr_gamma      = 0.5

Run from project root:
    python src/train/train_fasterrcnn.py
"""

import os
import sys
import random
import time
import torch
import numpy as np

# Ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from dataloader.detection_dataset import TrafficDetectionDataset, collate_fn

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Hyperparameters ───────────────────────────────────────────────────────────
DATA_ROOT    = os.path.join(ROOT, 'output', 'train')
CKPT_DIR     = os.path.join(ROOT, 'src', 'checkpoint', 'fasterrcnn')
LOG_DIR      = os.path.join(ROOT, 'src', 'logs', 'fasterrcnn')

NUM_CLASSES  = 4        # background + car + tree + lamppost
EPOCHS       = 50
BATCH_SIZE   = 4
LR           = 0.005
MOMENTUM     = 0.9
WEIGHT_DECAY = 0.0005
STEP_SIZE    = 10
GAMMA        = 0.5
SAVE_EVERY   = 5        # save checkpoint every N epochs

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes: int):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ── Validation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type='bbox')

    for images, targets in tqdm(loader, desc='  Val', leave=False):
        images  = [img.to(device) for img in images]
        outputs = model(images)

        preds = []
        gts   = []
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

    result = metric.compute()
    map50  = float(result['map_50'].item())
    map_   = float(result['map'].item())
    return map50, map_


# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = TrafficDetectionDataset(DATA_ROOT, split='train',
                                       mode='fasterrcnn', augment=True)
    val_ds   = TrafficDetectionDataset(DATA_ROOT, split='val',
                                       mode='fasterrcnn', augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, collate_fn=collate_fn,
                              pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False,
                              num_workers=2, collate_fn=collate_fn,
                              pin_memory=(device.type == 'cuda'))

    print(f'Train: {len(train_ds)} | Val: {len(val_ds)}')

    # ── Model ──────────────────────────────────────────────────────────────────
    model     = build_model(NUM_CLASSES).to(device)
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=STEP_SIZE,
                                                 gamma=GAMMA)
    writer    = SummaryWriter(log_dir=LOG_DIR)
    best_map  = -1.0

    # ── Epoch loop ─────────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for images, targets in tqdm(train_loader,
                                    desc=f'Epoch {epoch:02d}/{EPOCHS} [train]',
                                    leave=False):
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses    = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - t0

        # ── Validation ────────────────────────────────────────────────────────
        map50, map_ = evaluate(model, val_loader, device)
        current_lr  = scheduler.get_last_lr()[0]

        print(f'Epoch {epoch:02d}/{EPOCHS} | '
              f'Loss={avg_loss:.4f} | '
              f'mAP@0.5={map50:.4f} | '
              f'mAP@0.5:0.95={map_:.4f} | '
              f'LR={current_lr:.6f} | '
              f'Time={elapsed:.1f}s')

        writer.add_scalar('Loss/train',      avg_loss, epoch)
        writer.add_scalar('mAP/val_50',      map50,    epoch)
        writer.add_scalar('mAP/val_50_95',   map_,     epoch)
        writer.add_scalar('LR',              current_lr, epoch)

        # ── Checkpoint every N epochs ─────────────────────────────────────────
        if epoch % SAVE_EVERY == 0:
            ckpt = os.path.join(CKPT_DIR, f'epoch_{epoch:03d}.pth')
            torch.save({
                'epoch'              : epoch,
                'model_state_dict'   : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_map50'          : map50,
                'val_map'            : map_,
            }, ckpt)
            print(f'  ↳ Checkpoint saved: {ckpt}')

        # ── Best model ────────────────────────────────────────────────────────
        if map50 > best_map:
            best_map = map50
            ckpt     = os.path.join(CKPT_DIR, 'best.pth')
            torch.save({
                'epoch'              : epoch,
                'model_state_dict'   : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_map50'          : map50,
                'val_map'            : map_,
            }, ckpt)
            print(f'  ↳ Best model saved (mAP@0.5={best_map:.4f})')

    writer.close()
    print(f'\nTraining complete. Best mAP@0.5 = {best_map:.4f}')
    print(f'Best checkpoint: {os.path.join(CKPT_DIR, "best.pth")}')


if __name__ == '__main__':
    train()
