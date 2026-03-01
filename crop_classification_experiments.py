"""
crop_classification_experiments.py

Run downstream classification experiments to measure the impact of
super-resolution on crop disease classification accuracy.

For each SR generator checkpoint, trains and evaluates classifiers on:
  - Low-resolution images (baseline)
  - Super-resolved images (generator output)
  - High-resolution images (upper bound)

Auto-discovers available generator checkpoints, or accepts explicit paths.

Usage:
    # Auto-discover checkpoints and run all experiments
    python crop_classification_experiments.py --discover

    # Run with a specific checkpoint
    python crop_classification_experiments.py --checkpoint checkpoints/agricultural_4blocks_total10epochs/checkpoint_latest.pt --generator agricultural

    # Specify generator type and block count
    python crop_classification_experiments.py --checkpoint path/to/checkpoint.pt --generator baseline --num-blocks 8

    # Change classifier epochs or data settings
    python crop_classification_experiments.py --discover --classifier-epochs 20 --samples-per-class 300

    python crop_classification_experiments.py --help
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import cnn_training
import dataset_utils
from agricultural_sr import AgriculturalAttentionGenerator
from downstream_classifiers import SimpleCNNClassifier
from generators import ResidualGenerator
from training_utils import (
    build_plant_dict,
    load_class_names,
    load_image_file_sample,
    load_plant_balanced_image_file_sample,
)

# Maps generator type name to class
GENERATOR_CLASSES = {
    "agricultural": AgriculturalAttentionGenerator,
    "baseline": ResidualGenerator,
}

NUM_CLASSES = 38


def discover_checkpoints(checkpoint_dir="checkpoints"):
    """Find all checkpoint_latest.pt files under checkpoint_dir."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return []

    found = sorted(checkpoint_dir.rglob("checkpoint_latest.pt"))
    return found


def infer_generator_config(checkpoint_path):
    """Try to infer generator type and block count from the checkpoint directory name.

    Expected naming: {model_name}_{N}blocks_total{E}epochs
    e.g. agricultural_4blocks_total10epochs
    """
    parent_name = Path(checkpoint_path).parent.name

    # Infer block count
    num_blocks = 4  # default
    for part in parent_name.split("_"):
        if "blocks" in part:
            try:
                num_blocks = int(part.replace("blocks", ""))
            except ValueError:
                pass

    # Infer generator type
    gen_type = "agricultural"  # default
    name_lower = parent_name.lower()
    if "baseline" in name_lower or "residual" in name_lower:
        gen_type = "baseline"

    return gen_type, num_blocks


def load_generator(checkpoint_path, generator_type, num_blocks, device):
    """Load a trained generator from a checkpoint file."""
    gen_cls = GENERATOR_CLASSES[generator_type]
    generator = gen_cls(num_residual_blocks=num_blocks).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "generator_state_dict" in checkpoint:
        generator.load_state_dict(checkpoint["generator_state_dict"])
    else:
        # Assume raw state dict
        generator.load_state_dict(checkpoint)

    generator.eval()
    return generator


def run_classification_experiment(
    generator,
    generator_name,
    dataloader,
    num_classes,
    classifier_epochs,
    device,
    checkpoint_dir,
    eval_dataloader=None,
):
    """Run the 3-way classification comparison for one generator.

    Trains and evaluates classifiers on:
      1. Low-resolution images
      2. Super-resolved images (via generator)
      3. High-resolution images
    """
    if eval_dataloader is None:
        eval_dataloader = dataloader

    results = {}
    lr = 1e-3
    criterion = nn.CrossEntropyLoss()

    # --- 1. Low-resolution baseline ---
    print(f"\n{'='*60}")
    print(f"  [1/3] Training classifier on LOW-RES images")
    print(f"{'='*60}")
    model_name = f"classifier_lr_{generator_name}"
    classifier_lr = SimpleCNNClassifier(num_classes=num_classes).to(device)
    optimizer_lr = optim.Adam(classifier_lr.parameters(), lr=lr)

    cnn_training.train_classifier(
        classifier_lr,
        optimizer_lr,
        criterion,
        data_loader=dataloader,
        num_epochs=classifier_epochs,
        device=device,
        checkpoint_path=checkpoint_dir,
        model_name=model_name,
        low_res_input=True,
    )

    print("\n  Evaluating LR classifier...")
    cnn_training.generate_f1_report(classifier_lr, eval_dataloader, device, low_res_input=True)
    results["lr_baseline"] = model_name

    # --- 2. Super-resolved images ---
    print(f"\n{'='*60}")
    print(f"  [2/3] Training classifier on SUPER-RESOLVED images ({generator_name})")
    print(f"{'='*60}")
    model_name = f"classifier_sr_{generator_name}"
    classifier_sr = SimpleCNNClassifier(num_classes=num_classes).to(device)
    optimizer_sr = optim.Adam(classifier_sr.parameters(), lr=lr)

    cnn_training.train_classifier_w_generator(
        classifier_sr,
        generator,
        optimizer_sr,
        criterion,
        data_loader=dataloader,
        num_epochs=classifier_epochs,
        device=device,
        checkpoint_path=checkpoint_dir,
        model_name=model_name,
    )

    print("\n  Evaluating SR classifier...")
    cnn_training.generate_f1_report_w_generator(classifier_sr, generator, eval_dataloader, device)
    results["sr"] = model_name

    # --- 3. High-resolution upper bound ---
    print(f"\n{'='*60}")
    print(f"  [3/3] Training classifier on HIGH-RES images (upper bound)")
    print(f"{'='*60}")
    model_name = f"classifier_hr_{generator_name}"
    classifier_hr = SimpleCNNClassifier(num_classes=num_classes).to(device)
    optimizer_hr = optim.Adam(classifier_hr.parameters(), lr=lr)

    cnn_training.train_classifier(
        classifier_hr,
        optimizer_hr,
        criterion,
        data_loader=dataloader,
        num_epochs=classifier_epochs,
        device=device,
        checkpoint_path=checkpoint_dir,
        model_name=model_name,
        low_res_input=False,
    )

    print("\n  Evaluating HR classifier...")
    cnn_training.generate_f1_report(classifier_hr, eval_dataloader, device, low_res_input=False)
    results["hr_upper_bound"] = model_name

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run downstream classification experiments to evaluate SR impact."
    )

    # Checkpoint selection (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--discover",
        action="store_true",
        help="Auto-discover all checkpoints in --checkpoint-dir and run experiments for each.",
    )
    group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a specific generator checkpoint file.",
    )

    # Generator config (used with --checkpoint)
    parser.add_argument(
        "--generator",
        choices=GENERATOR_CLASSES.keys(),
        default="agricultural",
        help="Generator architecture type (default: agricultural).",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=4,
        help="Number of residual blocks in the generator (default: 4).",
    )

    # Data settings
    parser.add_argument(
        "--data-dir",
        default="data/plantvillage/color_images/train",
        help="Path to PlantVillage train directory (default: data/plantvillage/color_images/train).",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=500,
        help="Images to sample per class for training (default: 500).",
    )
    parser.add_argument(
        "--eval-samples-per-class",
        type=int,
        default=0,
        help="Images per class for evaluation. 0 = reuse training data (default: 0).",
    )

    # Classifier training settings
    parser.add_argument(
        "--classifier-epochs",
        type=int,
        default=10,
        help="Epochs to train each classifier (default: 10).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16).",
    )

    # Output settings
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Directory containing SR model checkpoints (default: checkpoints).",
    )
    parser.add_argument(
        "--results-dir",
        default="results/classification",
        help="Directory to save classification results (default: results/classification).",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data loading ---
    print("\nLoading dataset...")
    class_names = load_class_names(args.data_dir)
    plant_dict = build_plant_dict(class_names)
    image_files, labels, _ = load_plant_balanced_image_file_sample(
        args.data_dir, plant_dict, args.samples_per_class
    )

    dataset = dataset_utils.SuperResolutionDatasetWithLabels(
        image_files=image_files, image_labels=labels
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    print(f"Dataset: {len(dataset)} images, {len(dataloader)} batches")

    # Optional separate eval set
    eval_dataloader = None
    if args.eval_samples_per_class > 0:
        eval_files, eval_labels = load_image_file_sample(
            args.data_dir, class_names, args.eval_samples_per_class
        )
        eval_dataset = dataset_utils.SuperResolutionDatasetWithLabels(
            image_files=eval_files, image_labels=eval_labels
        )
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        print(f"Eval set: {len(eval_dataset)} images")

    # --- Build list of experiments to run ---
    experiments = []

    if args.discover:
        checkpoints = discover_checkpoints(args.checkpoint_dir)
        if not checkpoints:
            print(f"No checkpoints found under {args.checkpoint_dir}/")
            return
        print(f"\nDiscovered {len(checkpoints)} checkpoint(s):")
        for cp in checkpoints:
            gen_type, num_blocks = infer_generator_config(cp)
            name = cp.parent.name
            print(f"  {name}  (generator={gen_type}, blocks={num_blocks})")
            experiments.append((str(cp), gen_type, num_blocks, name))
    else:
        name = Path(args.checkpoint).parent.name or "custom"
        experiments.append((args.checkpoint, args.generator, args.num_blocks, name))

    # --- Run experiments ---
    os.makedirs(args.results_dir, exist_ok=True)
    all_results = {}

    for checkpoint_path, gen_type, num_blocks, exp_name in experiments:
        print(f"\n{'#'*60}")
        print(f"# EXPERIMENT: {exp_name}")
        print(f"# Generator: {gen_type}, blocks: {num_blocks}")
        print(f"# Checkpoint: {checkpoint_path}")
        print(f"{'#'*60}")

        generator = load_generator(checkpoint_path, gen_type, num_blocks, device)

        exp_results = run_classification_experiment(
            generator=generator,
            generator_name=exp_name,
            dataloader=dataloader,
            num_classes=NUM_CLASSES,
            classifier_epochs=args.classifier_epochs,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            eval_dataloader=eval_dataloader,
        )

        all_results[exp_name] = exp_results

    # --- Summary ---
    results_path = Path(args.results_dir) / "experiment_summary.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("ALL CLASSIFICATION EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {results_path}")
    print(f"Classifier checkpoints saved under: {args.checkpoint_dir}/")
    print(f"\nExperiments run:")
    for exp_name, exp_results in all_results.items():
        print(f"  {exp_name}:")
        for variant, model_name in exp_results.items():
            print(f"    {variant}: {model_name}")


if __name__ == "__main__":
    main()
