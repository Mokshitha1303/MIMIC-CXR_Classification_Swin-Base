# MIMIC-CXR Multi-Label Classification with Swin Transformer Base

Fine-tuning a **Swin Transformer Base** (ImageNet-21k pretrained) on the [MIMIC-CXR v2.0.0](https://physionet.org/content/mimic-cxr/2.0.0/) dataset for 14-class chest X-ray pathology classification, benchmarked against [Foundation X (WACV 2025)](https://openaccess.thecvf.com/content/WACV2025/papers/Islam_Foundation_X_Integrating_Classification_Localization_and_Segmentation_through_Lock-Release_Pretraining_WACV_2025_paper.pdf).

---

## Model Architecture

| Component | Details |
|-----------|---------|
| Backbone | `swin_base_patch4_window7_224.ms_in22k` (timm) |
| Pretraining | ImageNet-21k |
| Parameters | 86.8M |
| Input size | 224 × 224 (resized from 256) |
| Output | 14-class sigmoid (multi-label) |
| Loss | Weighted `BCEWithLogitsLoss` (per-class pos_weight) |

---

## Dataset — MIMIC-CXR v2.0.0

| Split | Images |
|-------|--------|
| Train | 237,972 |
| Validation | 1,959 |
| Test | 3,403 |

**Settings:**
- Frontal views only (PA + AP) — lateral views excluded to avoid label noise
- Uncertain labels (`-1`) treated via Label Smoothing Regularization (LSR-Ones): uniform(0.55, 0.85)
- Blank/NaN labels mapped to `0`
- CXR-specific normalization (chest X-ray mean/std)

**14 Pathology Classes:**
No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 5e-5 (peak) |
| Weight decay | 0.05 |
| Warmup epochs | 10 |
| Total epochs | 50 |
| Batch size | 128 |
| LR schedule | Linear warmup → cosine decay |
| Gradient clipping | 1.0 |
| Min LR | 1e-6 |
| Seed | 42 |
| Hardware | NVIDIA A100-SXM4-80GB |

**Test-time augmentation:** 10-crop TTA

---

## Results

### Per-Class AUC — Test Set (10-crop TTA, best checkpoint at epoch 15)

| Pathology | Ours (Swin-B) | Foundation X (MIMIC-II†) |
|-----------|:-------------:|:------------------------:|
| **Mean AUC** | **0.7810** | **0.7894** |

> †Foundation X reports a single mean AUC of **0.7894** on their MIMIC-II split (baseline Swin-B: 0.7912). Per-class breakdown is not reported in the paper. Foundation X trains jointly on 11 datasets including classification, localization, and segmentation tasks; direct dataset and split alignment with MIMIC-CXR v2.0.0 may differ.

### Training Curves

![Training Progress](plots/training_progress.png)

### ROC Curves — Test Set

![Final ROC](plots/final_test_roc.png)

---

## Repository Structure

```
MIMIC_CXR_Classification/
├── train.py                    # Main training + test-only entry point
├── configs/
│   └── config.yaml             # All hyperparameters and paths
├── src/
│   ├── dataset.py              # MIMICCXRDataset with frontal-only filter
│   ├── model.py                # Swin-B builder (timm / ARK+ weights)
│   ├── engine.py               # train_one_epoch, validate, test_tencrop
│   ├── metrics.py              # AUC computation, ROC curves
│   └── visualization.py        # Training progress plots
├── scripts/
│   ├── run_train.sh            # Training launch script
│   └── preprocess_resize.py    # Resize raw DICOM images to 256px
├── logs/                       # Training and test logs
├── plots/                      # training_progress.png, final_test_roc.png
├── Foundation_X/               # Foundation X codebase (reference)
└── Ark/                        # Ark / Ark+ codebase (reference)
```

---

## How to Run

### 1. Preprocess images (one-time)

```bash
python scripts/preprocess_resize.py
```

### 2. Train

```bash
bash scripts/run_train.sh
# or with smoke test (2 batches, verifies pipeline)
bash scripts/run_train.sh --smoke_test
```

Training auto-resumes from `checkpoints/latest.pth` if interrupted.

### 3. Evaluate best checkpoint on test set

```bash
python train.py --config configs/config.yaml \
    --test_only \
    --resume checkpoints/best_epoch015_auc0.8093.pth
```

---

## Comparison with Foundation X

[Foundation X (Islam et al., WACV 2025)](https://github.com/jlianglab/Foundation_X) is a multi-task chest X-ray foundation model trained on 11 public datasets with a Cyclic & Lock-Release pretraining strategy integrating classification, localization, and segmentation. It uses a Swin-B backbone — the same architecture used here.

Key differences:

| Aspect | Ours | Foundation X |
|--------|------|--------------|
| Pretraining | ImageNet-21k (timm) | 11 CXR datasets (cls + loc + seg) |
| Training data | MIMIC-CXR only | 11 diverse CXR datasets |
| Tasks | Classification only | Classification + Localization + Segmentation |
| Test AUC (MIMIC) | **0.7810** | **0.7894** |

---

## References

```bibtex
@InProceedings{Islam_2025_WACV,
    author    = {Islam, Nahid Ul and Ma, DongAo and Pang, Jiaxuan and Velan, Shivasakthi Senthil and Gotway, Michael and Liang, Jianming},
    title     = {Foundation X: Integrating Classification Localization and Segmentation through Lock-Release Pretraining Strategy for Chest X-ray Analysis},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {3647-3656}
}

@article{johnson2019mimic,
    title     = {MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports},
    author    = {Johnson, Alistair EW and others},
    journal   = {Scientific data},
    volume    = {6},
    number    = {1},
    pages     = {317},
    year      = {2019},
    publisher = {Nature Publishing Group}
}
```
