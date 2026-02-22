# Crop Super-Resolution Experiments

Command-line script for running super-resolution experiments on crop disease images.

## Quick start

```bash
# See all options
python crop_sr_experiments.py --help

# Run the main agricultural experiment
python crop_sr_experiments.py --experiment agricultural

# Run with custom settings
python crop_sr_experiments.py --experiment baseline --epochs 20 --batch-size 32
```

## Requirements

- PlantVillage dataset in `data/plantvillage/color_images/train/` (or pass `--data-dir`)
- Python dependencies: torch, torchvision, PIL, numpy, matplotlib
- Optional: `wandb` for experiment tracking

## Available experiments

### Architecture experiments

These mirror the configs from the RunPod notebook (`AgriculturalGenerator_RunPod.ipynb`).

| Name | Trainer | Generator | Discriminator | Notes |
|------|---------|-----------|---------------|-------|
| `agricultural` | AgriculturalTrainer | AgriculturalAttentionGenerator | AgriculturalDiscriminator | Full agricultural pipeline with attention + crop-specific losses |
| `baseline` | EnhancedTrainer | ResidualGenerator | SpectralNormDiscriminator | Simpler model, good comparison point |
| `multiblock` | AgriculturalTrainer | AgriculturalAttentionGenerator | AgriculturalDiscriminator | Trains 3 models with 4, 8, and 16 residual blocks |
| `mix` | AgriculturalTrainer | ResidualGenerator | AgriculturalDiscriminator | Simpler generator + agricultural losses |

### Loss function experiments

Useful for debugging model collapse. All use ResidualGenerator + SpectralNormDiscriminator so the only variable is the loss function.

| Name | Loss type | Description |
|------|-----------|-------------|
| `loss-pixel` | MSE pixel loss | Optimizes pixel-level accuracy only |
| `loss-perceptual` | VGG perceptual loss | Optimizes feature-level similarity only |
| `loss-combined` | Pixel + perceptual | Both losses with scheduled weighting |

## CLI options

```
--experiment NAME       Which experiment to run (required)
--data-dir PATH         Path to PlantVillage train dir (default: data/plantvillage/color_images/train)
--samples-per-class N   Images to sample per class (default: 500)
--epochs N              Total epochs (default: 10)
--epochs-per-block N    Epochs per residual-block config (default: 2)
--batch-size N          Batch size (default: 16)
--wandb                 Enable Weights & Biases logging
--wandb-project NAME    W&B project name (default: Agricultural-Super-Resolution)
```

## Examples

```bash
# Compare loss functions to debug model collapse
python crop_sr_experiments.py --experiment loss-pixel --epochs 10
python crop_sr_experiments.py --experiment loss-perceptual --epochs 10
python crop_sr_experiments.py --experiment loss-combined --epochs 10

# Train agricultural model with more data
python crop_sr_experiments.py --experiment agricultural --samples-per-class 1000 --epochs 20

# Train multiple block sizes
python crop_sr_experiments.py --experiment multiblock --epochs-per-block 5

# Enable W&B tracking
python crop_sr_experiments.py --experiment agricultural --wandb --wandb-project my-project
```

## Output

Checkpoints are saved to `checkpoints/<model_name>/` with:
- `checkpoint_latest.pt` — most recent checkpoint (auto-resumed on restart)
- `checkpoint_epoch_NNN_step_NNN.pt` — periodic snapshots
- Final models saved as `generator_Nblocks.pth` and `discriminator_Nblocks.pth`
