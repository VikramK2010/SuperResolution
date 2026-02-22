"""
crop_sr_experiments.py

Run super-resolution experiments from the command line.
Mirrors the working configs from AgriculturalGenerator_RunPod.ipynb.

Usage:
    python crop_sr_experiments.py --experiment agricultural --epochs 10
    python crop_sr_experiments.py --experiment baseline --data-dir data/plantvillage/color_images/train
    python crop_sr_experiments.py --experiment multiblock --epochs-per-block 3
    python crop_sr_experiments.py --help
"""

import argparse

from torch.utils.data import DataLoader

import dataset_utils
import training_loop
from agricultural_sr import (
    AgriculturalAttentionGenerator,
    AgriculturalDiscriminator,
    AgriculturalTrainer,
)
from discriminators import SpectralNormDiscriminator
from generators import ResidualGenerator
from training_classes import EnhancedTrainer
from training_utils import (
    build_plant_dict,
    load_class_names,
    load_plant_balanced_image_file_sample,
)

# Each experiment maps to a training_loop.enhanced_training_loop() call.
# "trainer_kwargs" are forwarded to the Trainer constructor (e.g. loss_type for EnhancedTrainer).
EXPERIMENTS = {
    # --- Architecture experiments (from the RunPod notebook) ---
    "agricultural": {
        "trainer": AgriculturalTrainer,
        "generator": AgriculturalAttentionGenerator,
        "discriminator": AgriculturalDiscriminator,
        "model_name": "agricultural",
    },
    "baseline": {
        "trainer": EnhancedTrainer,
        "generator": ResidualGenerator,
        "discriminator": SpectralNormDiscriminator,
        "model_name": "baseline_residual",
    },
    "multiblock": {
        "trainer": AgriculturalTrainer,
        "generator": AgriculturalAttentionGenerator,
        "discriminator": AgriculturalDiscriminator,
        "model_name": "ag_attention_multiblock",
        "residual_blocks_list": [4, 8, 16],
    },
    "mix": {
        "trainer": AgriculturalTrainer,
        "generator": ResidualGenerator,
        "discriminator": AgriculturalDiscriminator,
        "model_name": "residual_ag_losses",
    },
    # --- Loss function experiments (useful for debugging model collapse) ---
    "loss-pixel": {
        "trainer": EnhancedTrainer,
        "generator": ResidualGenerator,
        "discriminator": SpectralNormDiscriminator,
        "model_name": "loss_pixel_only",
        "trainer_kwargs": {"loss_type": "pixel"},
    },
    "loss-perceptual": {
        "trainer": EnhancedTrainer,
        "generator": ResidualGenerator,
        "discriminator": SpectralNormDiscriminator,
        "model_name": "loss_perceptual_only",
        "trainer_kwargs": {"loss_type": "perceptual"},
    },
    "loss-combined": {
        "trainer": EnhancedTrainer,
        "generator": ResidualGenerator,
        "discriminator": SpectralNormDiscriminator,
        "model_name": "loss_combined",
        "trainer_kwargs": {"loss_type": "combined"},
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Run crop super-resolution experiments."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        choices=EXPERIMENTS.keys(),
        help="Which experiment config to run.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/plantvillage/color_images/train",
        help="Path to PlantVillage train directory (default: data/plantvillage/color_images/train).",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=500,
        help="Images to sample per class (default: 500).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Total epochs (default: 10).",
    )
    parser.add_argument(
        "--epochs-per-block",
        type=int,
        default=2,
        help="Epochs per residual-block config (default: 2).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16).",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=5,
        help="Content-only pre-training epochs before adversarial training (default: 5).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        default="Agricultural-Super-Resolution",
        help="W&B project name (default: Agricultural-Super-Resolution).",
    )
    args = parser.parse_args()

    # --- Data loading ---
    class_names = load_class_names(args.data_dir)
    plant_dict = build_plant_dict(class_names)
    image_files, _, _ = load_plant_balanced_image_file_sample(
        args.data_dir, plant_dict, args.samples_per_class
    )

    dataset = dataset_utils.SuperResolutionDataset(image_files=image_files)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    print(f"Dataset: {len(dataset)} images, {len(dataloader)} batches per epoch")

    # --- Experiment config ---
    cfg = EXPERIMENTS[args.experiment]

    training_loop.enhanced_training_loop(
        dataset,
        dataloader,
        cfg["trainer"],
        cfg["generator"],
        cfg["discriminator"],
        residual_blocks_list=cfg.get("residual_blocks_list", [4]),
        model_name=cfg["model_name"],
        num_epochs_per_block=args.epochs_per_block,
        total_epochs=args.epochs,
        pretrain_epochs=args.pretrain_epochs,
        checkpoint_dir="checkpoints",
        use_wandb=args.wandb,
        wandb_project_name=args.wandb_project if args.wandb else None,
        **cfg.get("trainer_kwargs", {}),
    )


if __name__ == "__main__":
    main()
