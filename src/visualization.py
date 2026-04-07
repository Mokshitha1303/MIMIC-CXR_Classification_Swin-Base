"""
Live training visualisation — 4-panel figure updated after every epoch.

Panel layout:
  [0,0] Train vs Val Loss (per epoch)
  [0,1] Mean Val AUC + per-class AUC (per epoch)
  [1,0] Learning Rate schedule (log scale)
  [1,1] ROC curves for all 14 classes on the validation set (latest epoch)

The PNG is overwritten in place after each epoch so a live file viewer
always shows the current state.
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for server-side rendering
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore', category=UserWarning)


def update_plots(
    history: Dict,
    roc_data: Optional[Dict[str, Tuple[np.ndarray, np.ndarray, float]]],
    class_names: List[str],
    output_path: str,
) -> None:
    """Redraw and save the 4-panel training progress figure.

    Args:
        history:     dict with keys:
                       'train_loss'  : list[float] (one per epoch)
                       'val_loss'    : list[float]
                       'mean_auc'    : list[float]
                       'per_class_auc': list[list[float]]  (epoch × class)
                       'lr'          : list[float]
        roc_data:    output of compute_roc() for the latest validation epoch,
                     or None on the first epoch before validation.
        class_names: list of class label strings (length = num_classes)
        output_path: file path to save PNG (will be overwritten each call)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MIMIC-CXR Classification — Training Progress', fontsize=15, y=1.01)

    epochs = list(range(1, len(history['train_loss']) + 1))

    # ------------------------------------------------------------------ #
    # Panel 1: Train vs Val Loss
    # ------------------------------------------------------------------ #
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train Loss', color='royalblue', lw=2)
    ax.plot(epochs, history['val_loss'],   label='Val Loss',   color='tomato',    lw=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('BCE Loss')
    ax.set_title('Train vs Validation Loss')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    if epochs:
        ax.set_xlim(1, max(epochs) + 0.5)

    # ------------------------------------------------------------------ #
    # Panel 2: Mean Val AUC + per-class (faded)
    # ------------------------------------------------------------------ #
    ax = axes[0, 1]
    n_classes = len(class_names)

    if history['per_class_auc']:
        per_class = np.array(history['per_class_auc'])  # (epochs, classes)
        colors = cm.tab20(np.linspace(0, 1, n_classes))
        for i, (name, c) in enumerate(zip(class_names, colors)):
            vals = per_class[:, i]
            ax.plot(epochs, vals, color=c, lw=1, alpha=0.45, label=name)

    if history['mean_auc']:
        ax.plot(epochs, history['mean_auc'], color='black', lw=2.5,
                label='Mean AUC', zorder=5)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.set_title('Validation AUC')
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc='lower right', fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    if epochs:
        ax.set_xlim(1, max(epochs) + 0.5)

    # ------------------------------------------------------------------ #
    # Panel 3: Learning Rate (log scale)
    # ------------------------------------------------------------------ #
    ax = axes[1, 0]
    if history['lr']:
        ax.semilogy(epochs, history['lr'], color='seagreen', lw=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    if epochs:
        ax.set_xlim(1, max(epochs) + 0.5)

    # ------------------------------------------------------------------ #
    # Panel 4: ROC curves (latest val epoch)
    # ------------------------------------------------------------------ #
    ax = axes[1, 1]
    if roc_data:
        colors = cm.tab20(np.linspace(0, 1, n_classes))
        for i, name in enumerate(class_names):
            if name not in roc_data:
                continue
            fpr, tpr, auc_val = roc_data[name]
            lbl = f"{name} ({auc_val:.3f})" if not np.isnan(auc_val) else name
            ax.plot(fpr, tpr, color=colors[i], lw=1.2, alpha=0.85, label=lbl)

        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax.legend(loc='lower right', fontsize=5.5, ncol=2)
    else:
        ax.text(0.5, 0.5, 'Waiting for first\nvalidation epoch…',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (latest val epoch)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
