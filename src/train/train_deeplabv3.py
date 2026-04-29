"""
train_deeplabv3.py
==================
Train DeepLabV3+ (ResNet-50 backbone) for semantic segmentation.

Classes:
    0 = background | 1 = car | 2 = tree | 3 = lamppost

Config:
    epochs          = 50
    batch_size      = 8
    lr_backbone     = 1e-4   (lower for pretrained backbone)
    lr_head         = 1e-3   (higher for randomly initialised head)
    img_size        = (512, 512)
    loss            = CrossEntropyLoss(main) + 0.4 * CrossEntropyLoss(aux)
    normalisation   = ImageNet mean/std (required by ResNet backbone)
    scheduler       = ReduceLROnPlateau (patience=5, factor=0.5, max mode)
    save_every      = 5 epochs + best model (by val mIoU)

Prerequisites:
    Run precompute_masks.py first.

Run from project root:
    python src/train/train_deeplabv3.py
"""

import os
import sys
import random
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from dataloader.segmentation_dataset import TrafficSegDataset, NUM_SEG_CLASSES

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Hyperparameters ───────────────────────────────────────────────────────────
DATA_ROOT = os.path.join(ROOT, "output", "train")
CKPT_DIR = os.path.join(ROOT, "src", "checkpoint", "deeplabv3")
LOG_DIR = os.path.join(ROOT, "src", "logs", "deeplabv3")

EPOCHS = 50
BATCH_SIZE = 8
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
AUX_WEIGHT = 0.4
IMG_SIZE = (256, 256)
SAVE_EVERY = 5

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

CLASS_NAMES = ["background", "car", "tree", "lamppost"]


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes: int):
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    # Replace final conv layers with num_classes channels
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


# ── Differential parameter groups ─────────────────────────────────────────────
def get_param_groups(model):
    head_params = list(model.classifier.parameters()) + list(
        model.aux_classifier.parameters()
    )
    head_ids = set(id(p) for p in head_params)
    backbone_params = [
        p for p in model.parameters() if id(p) not in head_ids and p.requires_grad
    ]
    return [
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD},
    ]


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_iou_acc(preds, targets, num_classes):
    iou_list = []
    for c in range(num_classes):
        pred_c = preds == c
        target_c = targets == c
        inter = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        iou_list.append(inter / (union + 1e-8))
    pixel_acc = (preds == targets).sum().item() / (targets.numel() + 1e-8)
    return iou_list, pixel_acc


# ── Validation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_iou = [0.0] * num_classes
    all_acc = 0.0
    n_batches = 0

    for images, masks in tqdm(loader, desc="  Val", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        out = model(images)
        logits = out["out"]
        aux = out["aux"]
        loss = criterion(logits, masks) + AUX_WEIGHT * criterion(aux, masks)
        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        iou, acc = compute_iou_acc(preds.cpu(), masks.cpu(), num_classes)
        for c in range(num_classes):
            all_iou[c] += iou[c]
        all_acc += acc
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_iou = [v / max(n_batches, 1) for v in all_iou]
    mean_iou = float(np.mean(avg_iou))
    avg_acc = all_acc / max(n_batches, 1)
    return avg_loss, avg_iou, mean_iou, avg_acc


# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = TrafficSegDataset(
        DATA_ROOT,
        split="train",
        img_size=IMG_SIZE,
        normalize_for_deeplab=True,
        augment=True,
    )
    val_ds = TrafficSegDataset(
        DATA_ROOT,
        split="val",
        img_size=IMG_SIZE,
        normalize_for_deeplab=True,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"Image size: {IMG_SIZE} | Classes: {NUM_SEG_CLASSES}")

    model = build_model(NUM_SEG_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(get_param_groups(model))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )
    writer = SummaryWriter(log_dir=LOG_DIR)
    best_miou = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for images, masks in tqdm(
            train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS} [train]", leave=False
        ):
            images = images.to(device)
            masks = masks.to(device)

            out = model(images)
            logits = out["out"]
            aux = out["aux"]
            loss = criterion(logits, masks) + AUX_WEIGHT * criterion(aux, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        val_loss, iou_list, mean_iou, pixel_acc = evaluate(
            model, val_loader, criterion, device, NUM_SEG_CLASSES
        )
        scheduler.step(mean_iou)
        current_lr_bb = optimizer.param_groups[0]["lr"]
        current_lr_hd = optimizer.param_groups[1]["lr"]

        iou_str = "  ".join(
            f"{CLASS_NAMES[c]}={iou_list[c]:.3f}" for c in range(NUM_SEG_CLASSES)
        )
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"TrainLoss={avg_loss:.4f} | "
            f"ValLoss={val_loss:.4f} | "
            f"mIoU={mean_iou:.4f} | "
            f"Acc={pixel_acc:.4f} | "
            f"LR_bb={current_lr_bb:.2e} | "
            f"LR_hd={current_lr_hd:.2e} | "
            f"Time={elapsed:.1f}s"
        )
        print(f"  IoU: {iou_str}")

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Metric/mIoU", mean_iou, epoch)
        writer.add_scalar("Metric/PixAcc", pixel_acc, epoch)
        writer.add_scalar("LR/backbone", current_lr_bb, epoch)
        writer.add_scalar("LR/head", current_lr_hd, epoch)
        for c, name in enumerate(CLASS_NAMES):
            writer.add_scalar(f"IoU/{name}", iou_list[c], epoch)

        if epoch % SAVE_EVERY == 0:
            ckpt = os.path.join(CKPT_DIR, f"epoch_{epoch:03d}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_miou": mean_iou,
                    "val_loss": val_loss,
                },
                ckpt,
            )
            print(f"  ↳ Checkpoint: {ckpt}")

        if mean_iou > best_miou:
            best_miou = mean_iou
            ckpt = os.path.join(CKPT_DIR, "best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_miou": mean_iou,
                    "val_loss": val_loss,
                },
                ckpt,
            )
            print(f"  ↳ Best model saved (mIoU={best_miou:.4f})")

    writer.close()
    print(f"\nTraining complete. Best mIoU = {best_miou:.4f}")
    print(f'Best checkpoint: {os.path.join(CKPT_DIR, "best.pth")}')


if __name__ == "__main__":
    train()
