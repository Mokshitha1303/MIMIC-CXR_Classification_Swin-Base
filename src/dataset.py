"""
MIMIC-CXR Dataset for multi-label chest X-ray classification.

Fixes applied vs. original:
  1. Frontal-only filter  — uses metadata CSV to keep only PA and AP views.
                            Lateral views share the same study-level label but
                            look completely different → systematic label noise.
  2. CXR normalization   — mean/std computed from chest X-ray images rather
                            than ImageNet natural photos.
  3. Weighted BCE support — returns per-class positive counts so train.py can
                            compute pos_weight for BCEWithLogitsLoss.
  4. LSR-Ones uncertain   — -1 labels → uniform(0.55, 0.85) during training,
                            hard 1.0 during validation/test.
"""

import os
import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# 14-class disease labels in CheXpert / ARK+ order
# ---------------------------------------------------------------------------
MIMIC_CLASSES: List[str] = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices',
]

# Column names as they appear in mimic-cxr-2.0.0-chexpert.csv
_CHEXPERT_COLS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
    'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
    'Pneumothorax', 'Support Devices',
]

# Frontal view codes accepted by the frontal-only filter
_FRONTAL_VIEWS = {'PA', 'AP'}


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transform(
    mode: str,
    img_size: int = 256,
    crop_size: int = 224,
    normalize: str = 'chestx-ray',
):
    """Return the torchvision transform for train / val / test.

    Args:
        mode:       'train', 'val' / 'validate', or 'test'
        img_size:   resize target before cropping (val/test)
        crop_size:  final crop size fed to the model
        normalize:  'chestx-ray' (recommended) or 'imagenet'

    Test mode returns a [10, C, H, W] tensor per image (TenCrop TTA).
    """
    if normalize == 'chestx-ray':
        norm = transforms.Normalize(
            [0.5056, 0.5056, 0.5056],
            [0.252,  0.252,  0.252 ],
        )
    else:  # imagenet
        norm = transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )

    if mode == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.ToTensor(),
            norm,
        ])

    elif mode in ('val', 'validate'):
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            norm,
        ])

    elif mode == 'test':
        # TenCrop TTA → [10, C, H, W]
        to_tensor_norm = transforms.Compose([transforms.ToTensor(), norm])
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.TenCrop(crop_size),
            transforms.Lambda(
                lambda crops: torch.stack([to_tensor_norm(c) for c in crops])
            ),
        ])

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'train', 'val', or 'test'.")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MIMICCXRDataset(Dataset):
    """MIMIC-CXR multi-label chest X-ray classification dataset.

    Args:
        data_dir:        Root path of MIMIC dataset (contains /files and CSVs).
        split_csv:       Filename of the official split CSV (relative to data_dir).
        label_csv:       Filename of the CheXpert label CSV (relative to data_dir).
        metadata_csv:    Filename of the metadata CSV with ViewPosition column.
        split:           'train', 'validate', or 'test'.
        transform:       torchvision transform applied to each image.
        uncertain_label: 'LSR-Ones' | 'Ones' | 'Zeros' | 'LSR-Zeros'
        unknown_label:   Value assigned to blank/NaN labels (default 0.0).
        frontal_only:    If True, keep only PA and AP views (recommended).
    """

    def __init__(
        self,
        data_dir: str,
        split_csv: str,
        label_csv: str,
        metadata_csv: str,
        split: str,
        transform=None,
        uncertain_label: str = 'LSR-Ones',
        unknown_label: float = 0.0,
        frontal_only: bool = True,
        csv_dir: Optional[str] = None,
    ):
        """
        Args:
            data_dir:     Root for image files (may point to resized images).
            csv_dir:      Root for CSV files (defaults to data_dir).
                          Set to original MIMIC root when using resized images.
        """
        assert uncertain_label in ('Ones', 'Zeros', 'LSR-Ones', 'LSR-Zeros')

        self.data_dir        = data_dir
        self.split           = split
        self.transform       = transform
        self.uncertain_label = uncertain_label
        self.unknown_label   = unknown_label

        csv_root = csv_dir if csv_dir is not None else data_dir

        # ------------------------------------------------------------------ #
        # Load CSVs
        # ------------------------------------------------------------------ #
        split_df    = pd.read_csv(os.path.join(csv_root, split_csv))
        label_df    = pd.read_csv(os.path.join(csv_root, label_csv))
        metadata_df = pd.read_csv(
            os.path.join(csv_root, metadata_csv),
            usecols=['dicom_id', 'ViewPosition'],
        )

        # Keep only the requested split
        split_df = split_df[split_df['split'] == split].reset_index(drop=True)

        # Frontal-only filter  ← FIX #1
        if frontal_only:
            metadata_df = metadata_df[
                metadata_df['ViewPosition'].isin(_FRONTAL_VIEWS)
            ]
            split_df = split_df[
                split_df['dicom_id'].isin(metadata_df['dicom_id'])
            ].reset_index(drop=True)

        # Merge labels (study-level) onto the image-level split rows
        df = split_df.merge(label_df, on=['subject_id', 'study_id'], how='left')

        # ------------------------------------------------------------------ #
        # Build image paths and label arrays
        # ------------------------------------------------------------------ #
        self.img_paths: List[str] = []
        self._raw_labels: List[List[float]] = []   # store raw for pos_weight

        for _, row in df.iterrows():
            dicom_id   = str(row['dicom_id'])
            subject_id = str(int(row['subject_id']))
            study_id   = str(int(row['study_id']))

            img_path = os.path.join(
                data_dir, 'files',
                f'p{subject_id[:2]}',
                f'p{subject_id}',
                f's{study_id}',
                f'{dicom_id}.jpg',
            )
            self.img_paths.append(img_path)

            label = []
            for cls in MIMIC_CLASSES:
                val = row.get(cls, np.nan)
                if pd.isna(val):
                    label.append(float(unknown_label))
                else:
                    label.append(float(val))
            self._raw_labels.append(label)

        view_note = 'frontal-only (PA+AP)' if frontal_only else 'all views'
        print(
            f"[MIMICCXRDataset] split={split} | {view_note} | "
            f"images={len(self.img_paths):,} | classes={len(MIMIC_CLASSES)}"
        )

    # ---------------------------------------------------------------------- #
    # Positive-weight helper (used by train.py to build weighted BCE)
    # ---------------------------------------------------------------------- #
    def get_pos_weight(self) -> torch.Tensor:
        """Return per-class pos_weight = (N - pos) / pos for BCEWithLogitsLoss.

        Clamps to [0.1, 50] to avoid extreme weights.
        """
        labels = np.array(self._raw_labels)   # (N, 14)
        N      = labels.shape[0]
        pos_w  = []
        for i in range(labels.shape[1]):
            col    = labels[:, i]
            n_pos  = float(np.sum(col >= 0.5))
            n_pos  = max(n_pos, 1.0)           # avoid /0
            weight = (N - n_pos) / n_pos
            weight = float(np.clip(weight, 0.1, 50.0))
            pos_w.append(weight)
        return torch.FloatTensor(pos_w)

    # ---------------------------------------------------------------------- #
    # Internal: resolve uncertain label to training target
    # ---------------------------------------------------------------------- #
    def _resolve_label(self, v: float) -> float:
        if v == -1.0:
            if self.uncertain_label == 'Ones':
                return 1.0
            elif self.uncertain_label == 'Zeros':
                return 0.0
            elif self.uncertain_label == 'LSR-Ones':
                return random.uniform(0.55, 0.85)
            elif self.uncertain_label == 'LSR-Zeros':
                return random.uniform(0.0, 0.3)
        return v

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.img_paths[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        label = [self._resolve_label(v) for v in self._raw_labels[index]]
        return image, torch.FloatTensor(label)
