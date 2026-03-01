"""
split_dataset.py

Create a train/val/test split using symlinks (no file copying).
Reads from the cloned PlantVillage repo and symlinks each image file
into the appropriate split directory, preserving class subdirectories.

Usage:
    python split_dataset.py
    python split_dataset.py --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
"""

import argparse
import os
import random
from pathlib import Path

from tqdm import tqdm


def split_dataset(color_dir, output_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    color_dir = Path(color_dir)
    output_dir = Path(output_dir)

    total_linked = {"train": 0, "val": 0, "test": 0}

    class_dirs = sorted([d for d in color_dir.iterdir() if d.is_dir()])
    print(f"Found {len(class_dirs)} classes in {color_dir}")

    # Collect all files first so tqdm knows the total
    all_work = []  # list of (file_path, split_name, class_name)
    for class_dir in class_dirs:
        files = sorted([f for f in class_dir.iterdir() if f.is_file()])
        if not files:
            continue

        random.shuffle(files)

        n = len(files)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        for f in files[:train_end]:
            all_work.append((f, "train", class_dir.name))
        for f in files[train_end:val_end]:
            all_work.append((f, "val", class_dir.name))
        for f in files[val_end:]:
            all_work.append((f, "test", class_dir.name))

    for f, split_name, class_name in tqdm(all_work, desc="Linking images", unit="img"):
        split_class_dir = output_dir / split_name / class_name
        split_class_dir.mkdir(parents=True, exist_ok=True)

        link = split_class_dir / f.name
        if link.exists() or link.is_symlink():
            link.unlink()
        # Use relative symlink so it survives directory moves
        rel_target = os.path.relpath(f.resolve(), split_class_dir.resolve())
        link.symlink_to(rel_target)
        total_linked[split_name] += 1

    for split_name, count in total_linked.items():
        print(f"  {split_name}: {count} images")
    print(f"  Total: {sum(total_linked.values())} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split PlantVillage color images via symlinks.")
    parser.add_argument("--color-dir", default=None, help="Path to raw/color directory.")
    parser.add_argument("--output-dir", default=None, help="Output directory for splits.")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    color_dir = args.color_dir or script_dir / "data" / "PlantVillage-Dataset" / "raw" / "color"
    output_dir = args.output_dir or script_dir / "data" / "plantvillage" / "color_images"

    if not Path(color_dir).is_dir():
        print(f"ERROR: Color directory not found: {color_dir}")
        print("Run setup.sh first to clone the dataset.")
        raise SystemExit(1)

    ratios = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratios - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {ratios}")

    split_dataset(color_dir, output_dir, args.train_ratio, args.val_ratio, args.test_ratio)
