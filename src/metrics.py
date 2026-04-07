"""
Evaluation metrics for multi-label chest X-ray classification.

compute_auc  — per-class ROC-AUC + mean AUC
compute_roc  — per-class FPR/TPR curves (for plotting)
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict, List, Tuple


def compute_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Tuple[float, List[float]]:
    """Compute per-class AUC and mean AUC.

    Args:
        y_true:      (N, C) ground-truth binary labels (may contain soft labels)
        y_pred:      (N, C) predicted probabilities in [0, 1]
        class_names: list of length C

    Returns:
        mean_auc:    scalar mean AUC over valid classes
        auc_list:    per-class AUC (nan for classes with < 2 unique labels)
    """
    n_classes = y_true.shape[1]
    auc_list  = []

    for i in range(n_classes):
        gt_col   = y_true[:, i]
        pred_col = y_pred[:, i]

        # Binarise ground truth for AUC (threshold at 0.5 for soft labels)
        gt_bin = (gt_col >= 0.5).astype(int)

        if len(np.unique(gt_bin)) < 2:
            # Skip classes that only have one unique label in this split
            auc_list.append(float('nan'))
            continue

        try:
            auc = roc_auc_score(gt_bin, pred_col)
        except Exception:
            auc = float('nan')

        auc_list.append(auc)

    valid_aucs = [a for a in auc_list if not np.isnan(a)]
    mean_auc   = float(np.mean(valid_aucs)) if valid_aucs else float('nan')

    return mean_auc, auc_list


def compute_roc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, Tuple[np.ndarray, np.ndarray, float]]:
    """Compute ROC curves for all classes.

    Returns:
        dict mapping class_name → (fpr, tpr, auc_score)
    """
    n_classes = y_true.shape[1]
    roc_data  = {}

    for i, name in enumerate(class_names):
        gt_bin = (y_true[:, i] >= 0.5).astype(int)

        if len(np.unique(gt_bin)) < 2:
            roc_data[name] = (np.array([0.0, 1.0]), np.array([0.0, 1.0]), float('nan'))
            continue

        try:
            fpr, tpr, _ = roc_curve(gt_bin, y_pred[:, i])
            auc          = roc_auc_score(gt_bin, y_pred[:, i])
        except Exception:
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.0, 1.0])
            auc = float('nan')

        roc_data[name] = (fpr, tpr, auc)

    return roc_data


def format_auc_table(class_names: List[str], auc_list: List[float]) -> str:
    """Return a pretty-printed AUC table string."""
    lines = ["  Class                         AUC"]
    lines.append("  " + "-" * 36)
    for name, auc in zip(class_names, auc_list):
        val = f"{auc:.4f}" if not np.isnan(auc) else "  N/A "
        lines.append(f"  {name:<30s} {val}")
    valid = [a for a in auc_list if not np.isnan(a)]
    lines.append("  " + "-" * 36)
    lines.append(f"  {'Mean AUC':<30s} {np.mean(valid):.4f}" if valid else "  Mean AUC  N/A")
    return "\n".join(lines)
