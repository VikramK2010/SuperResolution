# Evaluation Notebooks

Two Jupyter notebooks for evaluating trained models. Both auto-discover checkpoints and save figures as PNGs.

## Prerequisites

1. Dependencies installed (`pip install -r requirements.txt`)
2. PlantVillage dataset available at `data/plantvillage/color_images/train/` (run `bash setup.sh`)
3. At least one trained model under `checkpoints/`

## EvaluateGANModel.ipynb

Evaluates a single SR generator on image quality metrics.

**Requires:** A trained GAN checkpoint (from `crop_sr_experiments.py` or the training notebooks).

**What it does:**
- Generates side-by-side LR / SR / HR sample images
- Computes PSNR, SSIM vs bicubic baseline
- LPIPS perceptual similarity (requires `lpips` package)
- FID score using Inception v3 features
- Plots metric distributions

**Outputs:**
- `eval_sr_samples.png` — visual comparison grid
- `eval_metric_distributions.png` — PSNR/SSIM histograms

**Selecting a model:**

```python
# Option 1: Set the directory name directly
MODEL_DIR = "agricultural_4blocks_total50epochs"

# Option 2: Leave as None and pick by index from the printed list
MODEL_DIR = None
CHECKPOINT_IDX = 0
```

## EvaluateClassifiers.ipynb

Evaluates the downstream classification experiment — compares LR, SR, and HR classifiers to measure whether super-resolution helps classification.

**Requires:** A completed classification experiment (from `crop_classification_experiments.py`), which produces three classifier checkpoints (`classifier_lr_*`, `classifier_sr_*`, `classifier_hr_*`) plus the original SR generator checkpoint.

**What it does:**
- Loads all 3 classifiers and the matching SR generator
- Runs inference on a balanced evaluation subset
- Prints sklearn classification reports
- Plots accuracy/F1 comparison, per-class F1 breakdown, confusion matrices, and training loss curves
- Prints a summary table with SR vs LR deltas

**Outputs:**
- `eval_classifier_comparison.png` — accuracy & macro F1 bar chart
- `eval_classifier_per_class_f1.png` — per-disease-class F1 bars
- `eval_classifier_confusion_matrices.png` — side-by-side confusion matrices
- `eval_classifier_training_curves.png` — training loss over epochs

**Selecting an experiment:**

```python
# Option 1: Set the experiment name directly
EXPERIMENT_NAME = "agricultural_4blocks_total10epochs"

# Option 2: Leave as None and pick by index from the printed list
EXPERIMENT_NAME = None
EXPERIMENT_IDX = 0
```

## Quick Start

```bash
# 1. Train a GAN
python crop_sr_experiments.py --experiment agricultural --epochs 50

# 2. Evaluate image quality
jupyter notebook EvaluateGANModel.ipynb

# 3. Run classification experiment
python crop_classification_experiments.py --discover --classifier-epochs 10

# 4. Evaluate classification impact
jupyter notebook EvaluateClassifiers.ipynb
```

## Swapping Models

Both notebooks list all available checkpoints when you run the discovery cell. To compare different models, change `MODEL_DIR` or `EXPERIMENT_NAME` in the selection cell and re-run from that cell onward.
