#!/bin/bash
# =============================================================================
# RunPod Setup Script for Super-GAN
#
# Usage: bash setup.sh
#
# This script:
#   1. Installs Python dependencies
#   2. Clones PlantVillage color images only (sparse checkout)
#   3. Creates train/val/test split via symlinks (no file copying)
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

# --- 2. Download PlantVillage dataset (color images only) ---
echo ""
echo "[2/3] Downloading PlantVillage dataset..."

mkdir -p "$DATA_DIR"
mkdir -p "$CHECKPOINT_DIR"

TRAIN_DIR="$DATA_DIR/plantvillage/color_images/train"

if [ -d "$TRAIN_DIR" ] && [ "$(ls -A "$TRAIN_DIR" 2>/dev/null)" ]; then
    echo "  Dataset already prepared. Skipping download."
else
    REPO_DIR="$DATA_DIR/PlantVillage-Dataset"

    if [ ! -d "$REPO_DIR" ]; then
        echo "  Cloning PlantVillage dataset (color images only via sparse checkout)..."
        git clone --depth 1 --filter=blob:none --sparse \
            https://github.com/spMohanty/PlantVillage-Dataset.git "$REPO_DIR"
        cd "$REPO_DIR"
        git sparse-checkout set raw/color
        cd "$SCRIPT_DIR"
    else
        echo "  PlantVillage repo already cloned."
    fi

    COLOR_DIR="$REPO_DIR/raw/color"

    if [ ! -d "$COLOR_DIR" ]; then
        echo "  ERROR: Expected directory not found at: $COLOR_DIR"
        echo "  Check the repo structure and try again."
        exit 1
    fi

    # --- 3. Split into train/val/test via symlinks ---
    echo ""
    echo "[3/3] Splitting data into train/val/test (symlinks, no copying)..."
    python3 "$SCRIPT_DIR/split_dataset.py" --color-dir "$COLOR_DIR"
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
