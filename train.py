#!/usr/bin/env python3
"""
MIMIC-CXR Classification — Main Training Script
================================================
Fixes vs. original:
  1. Frontal-only filter  — PA+AP views only (dataset.py handles this)
  2. Weighted BCELoss     — pos_weight per class from training set statistics
  3. CXR normalization    — chest-X-ray mean/std instead of ImageNet
  4. Lower LR + warmup   — 5e-5 peak, 10-epoch warmup
  5. Stable DataLoader   — persistent_workers only for train; val/test use 4 workers
  6. Interim checkpoint  — saved BEFORE validation so a val crash doesn't lose progress
  7. Resume              — auto-resumes from checkpoints/latest.pth

Usage:
    python train.py --config configs/config.yaml [--smoke_test]
"""

import argparse
import heapq
import logging
import math
import os
import random
import sys
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from src.dataset       import MIMICCXRDataset, MIMIC_CLASSES, build_transform
from src.model         import build_model
from src.engine        import train_one_epoch, validate, test_tencrop
from src.metrics       import compute_roc, format_auc_table
from src.visualization import update_plots


# ============================================================================
# Config
# ============================================================================

def load_config(path: str) -> SimpleNamespace:
    with open(path) as f:
        raw = yaml.safe_load(f)
    def _ns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: _ns(v) for k, v in d.items()})
        return d
    return _ns(raw)


# ============================================================================
# LR schedule: linear warmup → cosine decay
# ============================================================================

def get_lr(epoch, base_lr, min_lr, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg['lr'] = lr


# ============================================================================
# Checkpoint helpers
# ============================================================================

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def save_latest(state, ckpt_dir):
    save_checkpoint(state, os.path.join(ckpt_dir, 'latest.pth'))


class TopKCheckpoints:
    """Keep only the best K checkpoints ranked by val mean AUC."""

    def __init__(self, k, ckpt_dir):
        self.k        = k
        self.ckpt_dir = ckpt_dir
        self._heap: List = []   # min-heap of (auc, path)

    def update(self, auc, epoch, state):
        path = os.path.join(
            self.ckpt_dir, f'best_epoch{epoch:03d}_auc{auc:.4f}.pth'
        )
        if len(self._heap) < self.k:
            save_checkpoint(state, path)
            heapq.heappush(self._heap, (auc, path))
            return True
        elif auc > self._heap[0][0]:
            _, old = heapq.heappop(self._heap)
            if os.path.exists(old):
                os.remove(old)
            save_checkpoint(state, path)
            heapq.heappush(self._heap, (auc, path))
            return True
        return False

    def best_auc(self):
        return max((a for a, _ in self._heap), default=0.0)


# ============================================================================
# Logging
# ============================================================================

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    logfile = os.path.join(log_dir, f'training_{ts}.log')

    logger = logging.getLogger('mimic_cxr')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(logfile)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log file: {logfile}")
    return logger


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',     default='configs/config.yaml')
    p.add_argument('--smoke_test', action='store_true',
                   help='2 batches per split to verify pipeline end-to-end')
    p.add_argument('--test_only', action='store_true',
                   help='Skip training; run test evaluation only')
    p.add_argument('--resume', default=None,
                   help='Checkpoint path to load for --test_only')
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # Seed
    seed = cfg.training.seed
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    # Directories
    ckpt_dir = cfg.output.checkpoint_dir
    log_dir  = cfg.output.log_dir
    plot_dir = cfg.output.plot_dir
    for d in (ckpt_dir, log_dir, plot_dir):
        os.makedirs(d, exist_ok=True)

    logger = setup_logging(log_dir)
    logger.info(f"Config: {args.config} | smoke_test={args.smoke_test}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}" +
                (f" — {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))

    # ------------------------------------------------------------------ #
    # Datasets
    # ------------------------------------------------------------------ #
    d = cfg.data
    logger.info("Building datasets …")

    # csv_dir = original MIMIC root (CSVs never move)
    # data_dir = resized images directory (or original if not pre-processed)
    csv_dir = getattr(d, 'mimic_root', d.data_dir)

    train_dataset = MIMICCXRDataset(
        data_dir=d.data_dir, split_csv=d.split_csv, label_csv=d.label_csv,
        metadata_csv=d.metadata_csv, split='train',
        transform=build_transform('train', d.img_size, d.crop_size, d.normalization),
        uncertain_label=d.uncertain_label, unknown_label=d.unknown_label,
        frontal_only=d.frontal_only, csv_dir=csv_dir,
    )
    val_dataset = MIMICCXRDataset(
        data_dir=d.data_dir, split_csv=d.split_csv, label_csv=d.label_csv,
        metadata_csv=d.metadata_csv, split='validate',
        transform=build_transform('val', d.img_size, d.crop_size, d.normalization),
        uncertain_label='Ones', unknown_label=d.unknown_label,
        frontal_only=d.frontal_only, csv_dir=csv_dir,
    )
    test_dataset = MIMICCXRDataset(
        data_dir=d.data_dir, split_csv=d.split_csv, label_csv=d.label_csv,
        metadata_csv=d.metadata_csv, split='test',
        transform=build_transform('test', d.img_size, d.crop_size, d.normalization),
        uncertain_label='Ones', unknown_label=d.unknown_label,
        frontal_only=d.frontal_only, csv_dir=csv_dir,
    )

    # ------------------------------------------------------------------ #
    # FIX: Weighted BCE — compute pos_weight from training set statistics
    # ------------------------------------------------------------------ #
    pos_weight = train_dataset.get_pos_weight().to(device)
    logger.info("Per-class pos_weight (neg/pos ratio, clamped [0.1, 50]):")
    for name, w in zip(MIMIC_CLASSES, pos_weight.tolist()):
        logger.info(f"  {name:<35s} {w:.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ------------------------------------------------------------------ #
    # DataLoaders
    # ------------------------------------------------------------------ #
    t = cfg.training
    n_train_workers = t.workers
    n_val_workers   = min(t.workers, 4)   # cap val/test to avoid NFS overload

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=t.batch_size, shuffle=True,
        num_workers=n_train_workers, pin_memory=True, drop_last=True,
        persistent_workers=(n_train_workers > 0),
        prefetch_factor=2 if n_train_workers > 0 else None,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=t.batch_size, shuffle=False,
        num_workers=n_val_workers, pin_memory=True,
        prefetch_factor=2 if n_val_workers > 0 else None,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=max(1, t.batch_size // 2), shuffle=False,
        num_workers=n_val_workers, pin_memory=True,
        prefetch_factor=2 if n_val_workers > 0 else None,
    )

    if args.smoke_test:
        logger.info("=== SMOKE TEST — 2 batches per split ===")
        train_loader = _SmokeLoader(train_loader, 2)
        val_loader   = _SmokeLoader(val_loader,   2)
        test_loader  = _SmokeLoader(test_loader,  2)

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    logger.info("Building model …")
    model = build_model(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t.lr, weight_decay=t.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler()

    # ------------------------------------------------------------------ #
    # Resume
    # ------------------------------------------------------------------ #
    start_epoch = 0
    history: Dict = dict(train_loss=[], val_loss=[], mean_auc=[],
                         per_class_auc=[], lr=[])
    top_k = TopKCheckpoints(cfg.output.top_k_models, ckpt_dir)

    latest_ckpt = os.path.join(ckpt_dir, 'latest.pth')
    if os.path.exists(latest_ckpt):
        logger.info(f"Resuming from {latest_ckpt} …")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch']
        history     = ckpt.get('history', history)
        if 'top_k_heap' in ckpt:
            top_k._heap = ckpt['top_k_heap']
        logger.info(f"Resumed at epoch {start_epoch} | best AUC: {top_k.best_auc():.4f}")
    else:
        logger.info("No checkpoint found — starting from scratch.")

    # ------------------------------------------------------------------ #
    # Test-only mode: skip training, load specified checkpoint, run test
    # ------------------------------------------------------------------ #
    if args.test_only:
        ckpt_path = args.resume or os.path.join(ckpt_dir, 'latest.pth')
        logger.info(f"[test_only] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info("Running test evaluation with 10-crop TTA …")
        test_mean_auc, test_per_class, test_gt, test_pred = test_tencrop(
            model, test_loader, device, MIMIC_CLASSES,
        )
        logger.info(f"Test Mean AUC (10-crop TTA): {test_mean_auc:.4f}")
        logger.info("\n" + format_auc_table(MIMIC_CLASSES, test_per_class))
        test_roc = compute_roc(test_gt, test_pred, MIMIC_CLASSES)
        try:
            _save_final_roc(
                test_roc, MIMIC_CLASSES, test_mean_auc,
                os.path.join(plot_dir, 'final_test_roc.png'),
            )
            logger.info(f"Final ROC curves saved to {plot_dir}/final_test_roc.png")
        except Exception as e:
            logger.warning(f"Final ROC plot failed: {e}")
        return

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    plot_path = os.path.join(plot_dir, cfg.output.plot_file)
    logger.info(f"Training epochs {start_epoch+1}–{t.epochs} | "
                f"classes={len(MIMIC_CLASSES)}")
    logger.info("=" * 80)

    for epoch in range(start_epoch, t.epochs):
        t0 = time.time()

        lr = get_lr(epoch, t.lr, t.min_lr, t.warmup_epochs, t.epochs)
        set_lr(optimizer, lr)

        # ---- Train -------------------------------------------------------
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch + 1, grad_clip=t.grad_clip, print_freq=t.print_freq,
        )

        # ---- FIX: Save interim checkpoint BEFORE validation --------------
        # If validation crashes, the trained weights are still preserved.
        interim = {
            'epoch':                epoch + 1,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict':    scaler.state_dict(),
            'val_auc':              None,
            'val_loss':             None,
            'train_loss':           train_loss,
            'history':              history,
            'top_k_heap':           top_k._heap,
        }
        save_latest(interim, ckpt_dir)
        logger.info(f"Epoch {epoch+1} | train_loss={train_loss:.4f} | "
                    f"interim checkpoint saved | lr={lr:.2e}")

        # ---- Validate ----------------------------------------------------
        val_loss, mean_auc, per_class_auc, gt_arr, pred_arr = validate(
            model, val_loader, criterion, device, MIMIC_CLASSES,
        )
        roc_data = compute_roc(gt_arr, pred_arr, MIMIC_CLASSES)
        elapsed  = time.time() - t0

        # ---- Log ---------------------------------------------------------
        logger.info(
            f"Epoch {epoch+1:3d}/{t.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"mean_AUC={mean_auc:.4f} | lr={lr:.2e} | time={elapsed:.0f}s"
        )
        logger.info("\n" + format_auc_table(MIMIC_CLASSES, per_class_auc))

        # ---- History & plots ---------------------------------------------
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['mean_auc'].append(mean_auc)
        history['per_class_auc'].append(per_class_auc)
        history['lr'].append(lr)

        try:
            update_plots(history, roc_data, MIMIC_CLASSES, plot_path)
        except Exception as e:
            logger.warning(f"Plot update failed: {e}")

        # ---- Final checkpoint with val metrics ---------------------------
        state = {
            'epoch':                epoch + 1,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict':    scaler.state_dict(),
            'val_auc':              mean_auc,
            'val_loss':             val_loss,
            'train_loss':           train_loss,
            'history':              history,
            'top_k_heap':           top_k._heap,
        }
        save_latest(state, ckpt_dir)

        saved = top_k.update(mean_auc, epoch + 1, state)
        if saved:
            logger.info(f"  ★ Saved to top-{cfg.output.top_k_models} "
                        f"(AUC={mean_auc:.4f})")
        logger.info("=" * 80)

    # ------------------------------------------------------------------ #
    # Final test with 10-crop TTA
    # ------------------------------------------------------------------ #
    logger.info("Running final test evaluation with 10-crop TTA …")

    # Load best available checkpoint
    best_ckpt_path = latest_ckpt
    if top_k._heap:
        _, best_path = max(top_k._heap, key=lambda x: x[0])
        if os.path.exists(best_path):
            best_ckpt_path = best_path
            logger.info(f"Best checkpoint: {best_path}")

    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])

    test_mean_auc, test_per_class, test_gt, test_pred = test_tencrop(
        model, test_loader, device, MIMIC_CLASSES,
    )
    logger.info(f"Test Mean AUC (10-crop TTA): {test_mean_auc:.4f}")
    logger.info("\n" + format_auc_table(MIMIC_CLASSES, test_per_class))

    # Save final ROC plot
    test_roc = compute_roc(test_gt, test_pred, MIMIC_CLASSES)
    try:
        _save_final_roc(
            test_roc, MIMIC_CLASSES, test_mean_auc,
            os.path.join(plot_dir, 'final_test_roc.png'),
        )
        logger.info(f"Final ROC curves saved to {plot_dir}/final_test_roc.png")
    except Exception as e:
        logger.warning(f"Final ROC plot failed: {e}")

    logger.info("Training complete.")


# ============================================================================
# Utilities
# ============================================================================

class _SmokeLoader:
    """Wraps a DataLoader and yields only the first n_batches each iteration."""
    def __init__(self, loader, n):
        self._loader = loader
        self._n      = n
    def __iter__(self):
        for i, batch in enumerate(self._loader):
            if i >= self._n:
                break
            yield batch
    def __len__(self):
        return self._n


def _save_final_roc(roc_data, class_names, mean_auc, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, ax = plt.subplots(figsize=(10, 9))
    colors  = cm.tab20(np.linspace(0, 1, len(class_names)))
    for i, name in enumerate(class_names):
        if name not in roc_data:
            continue
        fpr, tpr, auc = roc_data[name]
        lbl = f"{name} ({auc:.3f})" if not np.isnan(auc) else name
        ax.plot(fpr, tpr, color=colors[i], lw=1.5, label=lbl)

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves — Test Set (10-crop TTA)\nMean AUC = {mean_auc:.4f}')
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
