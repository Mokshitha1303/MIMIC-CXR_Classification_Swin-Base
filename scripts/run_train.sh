#!/bin/bash
# ============================================================
# MIMIC-CXR Classification — Training Launch Script
# ============================================================
# Usage:
#   bash scripts/run_train.sh              (full training)
#   bash scripts/run_train.sh --smoke_test (2-batch smoke test)
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# Activate conda base environment
source /scratch/mmohan21/miniconda3/etc/profile.d/conda.sh
conda activate unimiss

echo "============================================"
echo "Project dir : $PROJECT_DIR"
echo "Python      : $(which python)"
echo "PyTorch     : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA        : $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "============================================"

python train.py --config configs/config.yaml "$@"
