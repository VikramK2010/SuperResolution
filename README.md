# Super-GAN: Crop Disease Super-Resolution

Super-resolution GAN models adapted for agricultural/crop disease images, using the [PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset).

## Running on RunPod

### 1. Create a Pod

- Go to [runpod.io](https://www.runpod.io/) and create a GPU pod
- Select the **RunPod PyTorch** template (comes with CUDA and PyTorch pre-installed)
- Pick a GPU with enough VRAM (16GB+ recommended, e.g. RTX A4000 or better)

### 2. Setup

Open a **terminal** in JupyterLab and run:

```bash
cd /workspace
git clone <your-repo-url> super-gan
cd super-gan
bash setup.sh
```

This will:
- Install all Python dependencies
- Clone the PlantVillage dataset from GitHub
- Split the images into train/val/test sets (60/20/20)

Setup only needs to be run once. Everything is saved under `/workspace`, which persists across pod restarts.

### 3. Train

Open `AgriculturalGenerator_RunPod.ipynb` in JupyterLab and run the cells from top to bottom.

Checkpoints are saved to `checkpoints/` automatically during training. If training is interrupted, it will resume from the last checkpoint when you re-run the training cell.

## Project Structure

```
super-gan/
├── AgriculturalGenerator_RunPod.ipynb  # Main notebook for RunPod
├── setup.sh                            # One-command setup script
├── requirements.txt                    # Python dependencies
│
├── agricultural_sr.py      # Agricultural generator, discriminator, trainer, losses
├── generators.py           # Generator architectures (Simple, Residual, Attention)
├── discriminators.py       # Discriminator architectures
├── training_loop.py        # Training loops with checkpointing
├── training_classes.py     # Loss functions and trainer base classes
├── training_utils.py       # Data loading and visualization utilities
├── dataset_utils.py        # PyTorch Dataset classes for super-resolution
├── split_data.py           # Train/val/test splitting utility
│
├── data/                   # Created by setup.sh (gitignored)
│   └── plantvillage/
│       └── color_images/
│           ├── train/
│           ├── val/
│           └── test/
│
└── checkpoints/            # Created during training (gitignored)
```

## Troubleshooting

**"Dataset not found" error in the notebook**
You need to run `bash setup.sh` in the terminal first.

**"No such file or directory: setup.sh"**
Make sure you `cd /workspace/super-gan` before running the script.

**Training is slow / out of memory**
- Reduce `batch_size` in the DataLoader cell (try 8 or 4)
- Reduce the sample size passed to `load_plant_balanced_image_file_sample` (try 200 instead of 500)

**Pod restarted and data is gone**
This shouldn't happen if you're using `/workspace`. Re-run `bash setup.sh` — it will skip steps that are already done.
