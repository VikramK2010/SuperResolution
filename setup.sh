#!/bin/bash
# =============================================================================
# RunPod Setup Script for Super-GAN
#
# Usage: bash setup.sh
#
# This script:
#   1. Installs Python dependencies
#   2. Clones the PlantVillage dataset from GitHub
#   3. Splits the data into train/val/test sets
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints"

echo "========================================="
echo " Super-GAN RunPod Setup"
echo "========================================="

# --- 1. Install Python dependencies ---
echo ""
echo "[1/3] Installing Python dependencies..."
pip install -q -r "$SCRIPT_DIR/requirements.txt"
echo "  Done."

# --- 2. Download PlantVillage dataset ---
echo ""
echo "[2/3] Downloading PlantVillage dataset..."

mkdir -p "$DATA_DIR"
mkdir -p "$CHECKPOINT_DIR"

if [ -d "$DATA_DIR/plantvillage/color_images/train" ]; then
    echo "  Dataset already prepared. Skipping download."
else
    REPO_DIR="$DATA_DIR/PlantVillage-Dataset"

    if [ ! -d "$REPO_DIR" ]; then
        echo "  Cloning PlantVillage dataset from GitHub (this may take a few minutes)..."
        git clone --depth 1 https://github.com/spMohanty/PlantVillage-Dataset.git "$REPO_DIR"
    else
        echo "  PlantVillage repo already cloned."
    fi

    COLOR_DIR="$REPO_DIR/raw/color"

    if [ ! -d "$COLOR_DIR" ]; then
        echo "  ERROR: Expected directory not found at: $COLOR_DIR"
        echo "  Check the repo structure and try again."
        exit 1
    fi

    # --- 3. Split data ---
    echo ""
    echo "[3/3] Splitting data into train/val/test..."
    python3 -c "
import sys, shutil
sys.path.insert(0, '$SCRIPT_DIR')
from split_data import split_data

# Copy raw color images to a working directory first (split_data uses os.rename)
import os
src = '$COLOR_DIR'
work_dir = '$DATA_DIR/plantvillage/raw/color'
if not os.path.exists(work_dir):
    print('  Copying images to working directory...')
    shutil.copytree(src, work_dir)

split_data(
    work_dir,
    '$DATA_DIR/plantvillage/color_images',
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2
)
print('  Done. Data split into train/val/test.')
"
fi

echo ""
echo "========================================="
echo " Setup complete!"
echo ""
echo " Data location:       $DATA_DIR/plantvillage/color_images/"
echo " Checkpoint location: $CHECKPOINT_DIR/"
echo ""
echo " Next steps:"
echo "   1. Open JupyterLab from the RunPod dashboard"
echo "   2. Open 'AgriculturalGenerator_RunPod.ipynb'"
echo "   3. Run the cells to start training"
echo "========================================="
