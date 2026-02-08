"""
Experiment 01: Data Preparation & Strategic Subsampling

GOALS:
1. Download and explore the PlantVillage dataset
2. Create multiple strategic subsets for experimentation
3. Turn data constraints into a research question

RESEARCH QUESTION: 
Can we achieve comparable performance with 10-25% of the full dataset 
if we sample strategically?
"""

import os
import random
from pathlib import Path
from collections import defaultdict
import json

# Configuration
class Config:
    GDRIVE_ROOT = "/content/drive/MyDrive/AgriSR_Project"
    RAW_DATA_DIR = f"{GDRIVE_ROOT}/data/raw"
    PROCESSED_DATA_DIR = f"{GDRIVE_ROOT}/data/processed"
    SUBSET_SIZES = [0.1, 0.25, 0.5, 1.0]
    RANDOM_SEED = 42

def setup_directories():
    """Create necessary directory structure"""
    dirs = [
        Config.GDRIVE_ROOT,
        Config.RAW_DATA_DIR,
        Config.PROCESSED_DATA_DIR,
        f"{Config.PROCESSED_DATA_DIR}/subsets",
        f"{Config.PROCESSED_DATA_DIR}/stats",
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✓ Created/verified: {d}")

def explore_dataset(data_dir):
    """Analyze dataset structure and statistics"""
    print("\n" + "="*60)
    print("DATASET EXPLORATION")
    print("="*60)
    
    stats = {
        'total_images': 0,
        'classes': {},
    }
    
    class_dirs = [d for d in Path(data_dir).iterdir() if d.is_dir()]
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.JPG'))
        
        stats['classes'][class_name] = len(image_files)
        stats['total_images'] += len(image_files)
    
    print(f"\nTotal Images: {stats['total_images']:,}")
    print(f"Number of Classes: {len(stats['classes'])}")
    print("\nClass Distribution:")
    for class_name, count in sorted(stats['classes'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count:,} images ({count/stats['total_images']*100:.1f}%)")
    
    return stats

def stratified_sample(data_dir, subset_fraction, output_dir, seed=42):
    """Create a stratified sample maintaining class proportions"""
    random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"STRATIFIED SAMPLING: {subset_fraction*100:.0f}% of data")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    subset_stats = {'total': 0, 'classes': {}}
    
    class_dirs = [d for d in Path(data_dir).iterdir() if d.is_dir()]
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.JPG'))
        
        n_samples = max(1, int(len(image_files) * subset_fraction))
        sampled_files = random.sample(image_files, n_samples)
        
        output_class_dir = Path(output_dir) / class_name
        output_class_dir.mkdir(exist_ok=True)
        
        for img_path in sampled_files:
            dest_path = output_class_dir / img_path.name
            if not dest_path.exists():
                os.symlink(img_path, dest_path)
        
        subset_stats['classes'][class_name] = n_samples
        subset_stats['total'] += n_samples
        
        print(f"  {class_name}: {n_samples}/{len(image_files)} images")
    
    with open(Path(output_dir) / 'subset_info.json', 'w') as f:
        json.dump(subset_stats, f, indent=2)
    
    print(f"\n✓ Created subset with {subset_stats['total']:,} images")
    
    return subset_stats

if __name__ == "__main__":
    setup_directories()
    print("\n⚠️  Download dataset to:", Config.RAW_DATA_DIR)
    print("Then run this script again to create subsets.")
