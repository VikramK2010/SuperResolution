import os 
from pathlib import Path
import random
import math

# Data directory structure:
# input_dir/class1/..., input_dir/class2/..., etc.
# Output directory structure:
# output_dir/train/class1/..., output_dir/val/class1/..., output_dir/test/class1/..., etc.

def split_single_class_data(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits data for a single class into train, val, and test sets.
    
    Args:
        input_dir (Path): Directory containing data files for a single class.
        output_dir (Path): Base directory where the split data will be saved.
        train_ratio (float): Proportion of data to use for training.
        val_ratio (float): Proportion of data to use for validation.
        test_ratio (float): Proportion of data to use for testing.
    """
    files = list(input_dir.glob('*'))
    random.shuffle(files)
    
    total_files = len(files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    splits = {
        'train': files[:train_end],
        'val': files[train_end:val_end],
        'test': files[val_end:]
    }
    
    class_name = input_dir.name
    
    for split_name, split_files in splits.items():
        split_dir = output_dir / split_name / class_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for file_path in split_files:
            destination = split_dir / file_path.name
            os.rename(file_path, destination)


def split_data(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits data from input_dir into train, val, and test sets in output_dir.
    
    Args:
        input_dir (str or Path): Directory containing class subdirectories with data files.
        output_dir (str or Path): Directory where the split data will be saved.
        train_ratio (float): Proportion of data to use for training.
        val_ratio (float): Proportion of data to use for validation.
        test_ratio (float): Proportion of data to use for testing.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.")
    
    for class_dir in input_dir.iterdir():
        if class_dir.is_dir():
            split_single_class_data(class_dir, output_dir, train_ratio, val_ratio, test_ratio)