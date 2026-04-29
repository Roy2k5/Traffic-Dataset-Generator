"""
train_unet.py
=============
Train UNet for semantic segmentation on the synthetic traffic dataset.

Architecture: src/model/unet.py (UNet with 3-input channels, 4-class output)

Classes:
    0 = background | 1 = car | 2 = tree | 3 = lamppost

Config:
    epochs        = 50
    batch_size    = 4    (reduced from 8 to avoid OOM; use 2 if still OOM)
    lr            = 1e-4   (Adam)
    img_size      = (256, 256)   (reduced from 512; UNet 512×512 + batch 8 = OOM)
    loss          = CrossEntropyLoss
    scheduler     = ReduceLROnPlateau (patience=5, factor=0.5)
    save_every    = 5 epochs + best model (by val mIoU)

Prerequisites:
    Run precompute_masks.py first to generate output/train/semantic_mask/

Run from project root:
    python src/train/train_unet.py
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

# Ensure project root & src are importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from model.unet import UNet
from dataloader.segmentation_dataset import TrafficSegDataset, NUM_SEG_CLASSES

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Hyperparameters ───────────────────────────────────────────────────────────
DATA_ROOT = os.path.join(ROOT, "output", "train")
CKPT_DIR = os.path.join(ROOT, "src", "checkpoint", "unet")
LOG_DIR = os.path.join(ROOT, "src", "logs", "unet")

EPOCHS = 50
BATCH_SIZE = 2
LR = 1e-4
IMG_SIZE = (256, 256)
SAVE_EVERY = 5

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_iou_acc(preds: torch.Tensor, targets: torch.Tensor, num_classes: int):
    """
    Args:
        preds   : (B, H, W) long — predicted class per pixel
        targets : (B, H, W) long — ground truth
    Returns:
        iou_per_class : list[float] length num_classes
        pixel_acc     : float
    """
    iou_list = []
    for c in range(num_classes):
        pred_c = preds == c
        target_c = targets == c
        inter = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        iou_list.append(inter / (union + 1e-8))

    correct = (preds == targets).sum().item()
    total = targets.numel()
    pixel_acc = correct / (total + 1e-8)

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

        logits = model(images)  # (B, C, H, W)
        loss = criterion(logits, masks)
        total_loss += loss.item()

        preds = logits.argmax(dim=1)  # (B, H, W)
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

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = TrafficSegDataset(
        DATA_ROOT,
        split="train",
        img_size=IMG_SIZE,
        normalize_for_deeplab=False,
        augment=True,
    )
    val_ds = TrafficSegDataset(
        DATA_ROOT,
        split="val",
        img_size=IMG_SIZE,
        normalize_for_deeplab=False,
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

    # ── Model ──────────────────────────────────────────────────────────────────
    model = UNet(n_channels=3, n_classes=NUM_SEG_CLASSES, bilinear=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5  # verbose removed (deprecated)
    )
    writer = SummaryWriter(log_dir=LOG_DIR)
    best_miou = -1.0

    CLASS_NAMES = ["background", "car", "tree", "lamppost"]

    # ── Epoch loop ─────────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for images, masks in tqdm(
            train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS} [train]", leave=False
        ):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # ── Validation ────────────────────────────────────────────────────────
        val_loss, iou_list, mean_iou, pixel_acc = evaluate(
            model, val_loader, criterion, device, NUM_SEG_CLASSES
        )
        scheduler.step(mean_iou)
        current_lr = optimizer.param_groups[0]["lr"]

        iou_str = "  ".join(
            f"{CLASS_NAMES[c]}={iou_list[c]:.3f}" for c in range(NUM_SEG_CLASSES)
        )
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"TrainLoss={avg_loss:.4f} | "
            f"ValLoss={val_loss:.4f} | "
            f"mIoU={mean_iou:.4f} | "
            f"Acc={pixel_acc:.4f} | "
            f"LR={current_lr:.2e} | "
            f"Time={elapsed:.1f}s"
        )
        print(f"  IoU: {iou_str}")

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Metric/mIoU", mean_iou, epoch)
        writer.add_scalar("Metric/PixAcc", pixel_acc, epoch)
        writer.add_scalar("LR", current_lr, epoch)
        for c, name in enumerate(CLASS_NAMES):
            writer.add_scalar(f"IoU/{name}", iou_list[c], epoch)

        # ── Periodic checkpoint ────────────────────────────────────────────────
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

        # ── Best model ─────────────────────────────────────────────────────────
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
