"""
Training and evaluation engine.

train_one_epoch  — single training epoch with AMP
validate         — validation loop, returns loss + AUC metrics
test_tencrop     — test evaluation with 10-crop TTA
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Tuple

from src.metrics import compute_auc, compute_roc


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
    print_freq: int = 100,
) -> float:
    """Run one training epoch.

    Returns:
        avg_loss: mean BCEWithLogitsLoss over the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False, dynamic_ncols=True)
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches  += 1

        if (batch_idx + 1) % print_freq == 0:
            pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str],
) -> Tuple[float, float, List[float], np.ndarray, np.ndarray]:
    """Validate model on the validation set (CenterCrop, no TTA).

    Returns:
        val_loss:      mean BCE loss
        mean_auc:      mean AUC over valid classes
        per_class_auc: list of per-class AUC values
        gt_array:      (N, C) ground truth (binarised)
        pred_array:    (N, C) sigmoid predictions
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    all_preds  = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Validate", leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            logits = model(images)
            loss   = criterion(logits, labels)

        preds = torch.sigmoid(logits)

        all_preds.append(preds.cpu().float().numpy())
        all_labels.append(labels.cpu().float().numpy())

        total_loss += loss.item()
        n_batches  += 1

    val_loss   = total_loss / max(n_batches, 1)
    gt_array   = np.concatenate(all_labels, axis=0)  # (N, C)
    pred_array = np.concatenate(all_preds,  axis=0)  # (N, C)

    mean_auc, per_class_auc = compute_auc(gt_array, pred_array, class_names)

    return val_loss, mean_auc, per_class_auc, gt_array, pred_array


@torch.no_grad()
def test_tencrop(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Tuple[float, List[float], np.ndarray, np.ndarray]:
    """Evaluate on test set with 10-crop TTA.

    The test DataLoader must use the 'test' transform (returns [10, C, H, W]
    per image).

    Returns:
        mean_auc:      mean AUC over valid classes
        per_class_auc: per-class AUC list
        gt_array:      (N, C) ground truth
        pred_array:    (N, C) averaged sigmoid predictions
    """
    model.eval()

    all_preds  = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Test (10-crop)", leave=False, dynamic_ncols=True):
        # images: [B, 10, C, H, W]
        B, n_crops, C, H, W = images.shape
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass on all 10 crops simultaneously
        with torch.cuda.amp.autocast():
            logits_flat = model(images.view(B * n_crops, C, H, W))  # [B*10, num_classes]

        preds_flat = torch.sigmoid(logits_flat)                      # [B*10, num_classes]
        preds = preds_flat.view(B, n_crops, -1).mean(dim=1)          # [B, num_classes]

        all_preds.append(preds.cpu().float().numpy())
        all_labels.append(labels.cpu().float().numpy())

    gt_array   = np.concatenate(all_labels, axis=0)
    pred_array = np.concatenate(all_preds,  axis=0)

    mean_auc, per_class_auc = compute_auc(gt_array, pred_array, class_names)

    return mean_auc, per_class_auc, gt_array, pred_array
