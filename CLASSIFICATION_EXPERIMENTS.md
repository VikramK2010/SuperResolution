# Downstream Classification Experiments

`crop_classification_experiments.py` measures whether super-resolution improves crop disease classification accuracy. For each SR generator checkpoint, it trains and evaluates three classifiers side-by-side:

| Classifier | Input | Purpose |
|---|---|---|
| **LR baseline** | 64x64 low-res images | Lower bound — what accuracy do we get without SR? |
| **SR** | Generator-upscaled images | Does our SR model help? |
| **HR upper bound** | 256x256 high-res images | Upper bound — best possible accuracy |

Each classifier is a `SimpleCNNClassifier` (3-layer CNN, 38 disease classes) trained with Adam and CrossEntropyLoss. After training, each is evaluated with a full F1 report.

---

## Prerequisites

1. **PlantVillage dataset** in `data/plantvillage/color_images/train/` (run `setup.sh` if needed).
2. **At least one trained SR generator checkpoint.** Train one first with:
   ```bash
   python crop_sr_experiments.py --experiment agricultural --epochs 10
   ```
   This saves checkpoints to `checkpoints/agricultural_4blocks_total10epochs/`.

---

## Quick Start

### Run all discovered checkpoints

```bash
python crop_classification_experiments.py --discover
```

The script scans `checkpoints/` for every `checkpoint_latest.pt`, infers the generator architecture and block count from the directory name, and runs the 3-way comparison for each.

### Run a single checkpoint

```bash
python crop_classification_experiments.py \
  --checkpoint checkpoints/agricultural_4blocks_total10epochs/checkpoint_latest.pt \
  --generator agricultural \
  --num-blocks 4
```

---

## CLI Reference

### Checkpoint selection (one is required)

| Flag | Description |
|---|---|
| `--discover` | Auto-discover all `checkpoint_latest.pt` files under `--checkpoint-dir` |
| `--checkpoint PATH` | Path to a specific checkpoint file |

### Generator config (used with `--checkpoint`)

| Flag | Default | Description |
|---|---|---|
| `--generator` | `agricultural` | Generator architecture: `agricultural` or `baseline` |
| `--num-blocks` | `4` | Number of residual blocks in the generator |

With `--discover`, these are inferred from the checkpoint directory name (e.g. `agricultural_8blocks_total20epochs` -> `agricultural`, 8 blocks).

### Data settings

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | `data/plantvillage/color_images/train` | Path to PlantVillage train directory |
| `--samples-per-class` | `500` | Images per class for training |
| `--eval-samples-per-class` | `0` | Images per class for evaluation (0 = reuse training data) |

### Classifier training

| Flag | Default | Description |
|---|---|---|
| `--classifier-epochs` | `10` | Epochs to train each classifier |
| `--batch-size` | `16` | Batch size |

### Output

| Flag | Default | Description |
|---|---|---|
| `--checkpoint-dir` | `checkpoints` | Root directory for SR checkpoints and classifier checkpoints |
| `--results-dir` | `results/classification` | Directory for JSON results |

---

## Example Workflows

### 1. Train SR models, then evaluate classification impact

```bash
# Step 1: Train two different SR generators
python crop_sr_experiments.py --experiment agricultural --epochs 10
python crop_sr_experiments.py --experiment baseline --epochs 10

# Step 2: Run classification experiments on both
python crop_classification_experiments.py --discover
```

### 2. Compare block counts

```bash
# Step 1: Train generators with different block counts
python crop_sr_experiments.py --experiment multiblock --epochs-per-block 3

# Step 2: Evaluate all
python crop_classification_experiments.py --discover --classifier-epochs 15
```

### 3. Quick test with fewer samples

```bash
python crop_classification_experiments.py \
  --discover \
  --samples-per-class 50 \
  --classifier-epochs 3 \
  --batch-size 32
```

### 4. Separate train/eval sets

```bash
python crop_classification_experiments.py \
  --discover \
  --samples-per-class 400 \
  --eval-samples-per-class 100
```

---

## Outputs

- **Console** — Per-epoch loss/accuracy/F1 during training, plus a full classification report and confusion matrix for each classifier variant.
- **`results/classification/experiment_summary.json`** — Maps each checkpoint to its three classifier model names for later loading.
- **`checkpoints/classifier_*`** — Saved classifier checkpoints (one per variant per SR model).

---

## Checkpoint directory naming convention

Auto-discovery infers the generator config from the directory name. The expected format is:

```
checkpoints/
  {name}_{N}blocks_total{E}epochs/
    checkpoint_latest.pt
```

- Directory names containing `baseline` or `residual` -> `ResidualGenerator`
- All others -> `AgriculturalAttentionGenerator`
- The `{N}blocks` part sets `num_residual_blocks`

If your directory names don't follow this convention, use `--checkpoint` with explicit `--generator` and `--num-blocks` flags instead.

---

## Relationship to other scripts

| Script | Purpose |
|---|---|
| `crop_sr_experiments.py` | Train SR generators (run first) |
| **`crop_classification_experiments.py`** | Evaluate SR impact on classification (this script) |
| `cnn_training.py` | Classifier training/evaluation functions (used internally) |
| `downstream_classifiers.py` | `SimpleCNNClassifier` architecture (used internally) |
| `workflow/experiment_03_classification_impact.py` | Older standalone evaluation script (Google Colab oriented) |
