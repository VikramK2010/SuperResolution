"""
Experiment 03: Evaluate SR Impact on Disease Classification

GOALS:
1. Generate SR images using trained models
2. Train disease classifiers on LR vs SR images
3. Measure diagnostic improvement

RESEARCH QUESTIONS:
- Does SR improve classification accuracy?
- Which diseases benefit most from SR?
- Is the computational cost worth it?
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

class EvalConfig:
    GDRIVE_ROOT = "/content/drive/MyDrive/AgriSR_Project"
    DATA_DIR = f"{GDRIVE_ROOT}/data/processed/subsets/stratified_25pct"
    MODEL_DIR = f"{GDRIVE_ROOT}/results"
    SR_OUTPUT_DIR = f"{GDRIVE_ROOT}/data/sr_generated"
    RESULTS_DIR = f"{GDRIVE_ROOT}/results/classification"
    
    # Models to evaluate
    SR_MODELS = ['simple', 'residual_4blocks', 'residual_8blocks']
    
    # Classification settings
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# STEP 1: GENERATE SR IMAGES
# ============================================================================

class ImageDataset(Dataset):
    """Simple dataset for SR generation"""
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, str(img_path)


def generate_sr_images(model_name, model_path, data_dir, output_dir):
    """Generate SR images using trained model"""
    
    print(f"\n{'='*60}")
    print(f"Generating SR images with: {model_name}")
    print(f"{'='*60}")
    
    device = torch.device(EvalConfig.DEVICE)
    
    # Load generator
    from generators import SimpleGenerator, ResidualGenerator
    
    if 'simple' in model_name:
        generator = SimpleGenerator().to(device)
    else:
        num_blocks = int(model_name.split('_')[1].replace('blocks', ''))
        generator = ResidualGenerator(num_residual_blocks=num_blocks).to(device)
    
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    
    # Prepare output directory
    output_dir = Path(output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = []
    for class_dir in Path(data_dir).iterdir():
        if class_dir.is_dir():
            image_files.extend(list(class_dir.glob('*.jpg')))
    
    print(f"Processing {len(image_files)} images...")
    
    # Transform for SR input (64x64)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    dataset = ImageDataset(image_files, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Generate SR images
    generated_count = 0
    with torch.no_grad():
        for batch_imgs, batch_paths in dataloader:
            batch_imgs = batch_imgs.to(device)
            sr_imgs = generator(batch_imgs)
            
            # Save SR images
            for sr_img, orig_path in zip(sr_imgs, batch_paths):
                # Denormalize
                sr_img = (sr_img + 1) / 2
                sr_img = torch.clamp(sr_img, 0, 1)
                
                # Get relative path structure
                orig_path = Path(orig_path)
                class_name = orig_path.parent.name
                
                # Create class directory
                class_output_dir = output_dir / class_name
                class_output_dir.mkdir(exist_ok=True)
                
                # Save
                sr_img_pil = transforms.ToPILImage()(sr_img.cpu())
                sr_img_pil.save(class_output_dir / orig_path.name)
                generated_count += 1
            
            if generated_count % 100 == 0:
                print(f"  Generated {generated_count}/{len(image_files)} images")
    
    print(f"✓ Generated {generated_count} SR images")
    
    return str(output_dir)


# ============================================================================
# STEP 2: TRAIN DISEASE CLASSIFIERS
# ============================================================================

class DiseaseDataset(Dataset):
    """Dataset for disease classification"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_files = []
        self.labels = []
        self.class_names = []
        
        # Get all classes
        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls.name: idx for idx, cls in enumerate(class_dirs)}
        self.class_names = [cls.name for cls in class_dirs]
        
        # Get all images
        for class_dir in class_dirs:
            class_images = list(class_dir.glob('*.jpg'))
            self.image_files.extend(class_images)
            self.labels.extend([self.class_to_idx[class_dir.name]] * len(class_images))
        
        print(f"Loaded {len(self.image_files)} images from {len(self.class_names)} classes")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def train_classifier(data_dir, model_name, num_classes, num_epochs=20):
    """Train a disease classifier"""
    
    print(f"\n{'='*60}")
    print(f"Training classifier: {model_name}")
    print(f"{'='*60}")
    
    device = torch.device(EvalConfig.DEVICE)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = DiseaseDataset(data_dir, transform=transform)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=EvalConfig.BATCH_SIZE,
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=EvalConfig.BATCH_SIZE,
                           shuffle=False, num_workers=2)
    
    # Create model (ResNet18 for speed)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=EvalConfig.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    
    # Training loop
    best_val_acc = 0
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    print(f"✓ Best validation accuracy: {best_val_acc:.2f}%")
    
    return model, train_accs, val_accs, best_val_acc, dataset.class_names


# ============================================================================
# STEP 3: EVALUATE AND COMPARE
# ============================================================================

def evaluate_classifier(model, data_dir, class_names):
    """Detailed evaluation of classifier"""
    
    device = torch.device(EvalConfig.DEVICE)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = DiseaseDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=EvalConfig.BATCH_SIZE,
                           shuffle=False, num_workers=2)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Classification report
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names,
                                   output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return report, cm


def plot_comparison(results, output_path):
    """Create comparison plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SR Impact on Disease Classification', fontsize=16, fontweight='bold')
    
    # Plot 1: Overall accuracy comparison
    ax = axes[0, 0]
    models = list(results.keys())
    accs = [results[m]['best_val_acc'] for m in models]
    colors = ['#d62728' if m == 'lr_baseline' else '#2ca02c' for m in models]
    
    bars = ax.bar(models, accs, color=colors, alpha=0.7)
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Overall Classification Accuracy')
    ax.set_ylim([0, 100])
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    # Plot 2: Per-class accuracy (F1 scores)
    ax = axes[0, 1]
    class_names = results['lr_baseline']['class_names']
    x = np.arange(len(class_names))
    width = 0.2
    
    for idx, model_name in enumerate(models):
        f1_scores = [results[model_name]['report'][c]['f1-score'] * 100 
                    for c in class_names]
        ax.bar(x + idx*width, f1_scores, width, label=model_name, alpha=0.7)
    
    ax.set_ylabel('F1 Score (%)')
    ax.set_title('Per-Class Performance')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 100])
    
    # Plot 3: Training curves (example from best SR model)
    ax = axes[1, 0]
    best_sr_model = max([m for m in models if m != 'lr_baseline'],
                       key=lambda m: results[m]['best_val_acc'])
    
    epochs = range(1, len(results[best_sr_model]['train_accs']) + 1)
    ax.plot(epochs, results[best_sr_model]['train_accs'], label='Train')
    ax.plot(epochs, results[best_sr_model]['val_accs'], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Training Curves: {best_sr_model}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Improvement summary
    ax = axes[1, 1]
    ax.axis('off')
    
    lr_acc = results['lr_baseline']['best_val_acc']
    improvements = {}
    for model in [m for m in models if m != 'lr_baseline']:
        improvements[model] = results[model]['best_val_acc'] - lr_acc
    
    summary_text = "IMPROVEMENT OVER LR BASELINE\n\n"
    summary_text += f"LR Baseline: {lr_acc:.2f}%\n\n"
    
    for model, improvement in sorted(improvements.items(), 
                                    key=lambda x: x[1], reverse=True):
        summary_text += f"{model}:\n"
        summary_text += f"  Accuracy: {results[model]['best_val_acc']:.2f}%\n"
        summary_text += f"  Improvement: {improvement:+.2f}%\n\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison plot saved: {output_path}")
    plt.close()


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Complete evaluation workflow"""
    
    print("\n" + "="*60)
    print("EXPERIMENT 03: SR IMPACT ON DISEASE CLASSIFICATION")
    print("="*60)
    
    os.makedirs(EvalConfig.SR_OUTPUT_DIR, exist_ok=True)
    os.makedirs(EvalConfig.RESULTS_DIR, exist_ok=True)
    
    results = {}
    
    # Step 1: Train classifier on LR baseline
    print("\n### BASELINE: Low Resolution Images ###")
    model_lr, train_accs, val_accs, best_acc, class_names = train_classifier(
        EvalConfig.DATA_DIR,
        'lr_baseline',
        num_classes=len(list(Path(EvalConfig.DATA_DIR).iterdir()))
    )
    report_lr, cm_lr = evaluate_classifier(model_lr, EvalConfig.DATA_DIR, class_names)
    
    results['lr_baseline'] = {
        'best_val_acc': best_acc,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'report': report_lr,
        'class_names': class_names
    }
    
    # Step 2: Generate SR images and train classifiers
    for sr_model in EvalConfig.SR_MODELS:
        print(f"\n### SR MODEL: {sr_model} ###")
        
        # Generate SR images
        model_path = Path(EvalConfig.MODEL_DIR) / sr_model / 'generator_final.pth'
        
        if not model_path.exists():
            print(f"⚠️  Model not found: {model_path}")
            continue
        
        sr_data_dir = generate_sr_images(
            sr_model,
            str(model_path),
            EvalConfig.DATA_DIR,
            EvalConfig.SR_OUTPUT_DIR
        )
        
        # Train classifier on SR images
        model_sr, train_accs, val_accs, best_acc, _ = train_classifier(
            sr_data_dir,
            sr_model,
            num_classes=len(class_names)
        )
        
        report_sr, cm_sr = evaluate_classifier(model_sr, sr_data_dir, class_names)
        
        results[sr_model] = {
            'best_val_acc': best_acc,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'report': report_sr,
            'class_names': class_names
        }
    
    # Step 3: Create comparison plots
    plot_comparison(results, 
                   Path(EvalConfig.RESULTS_DIR) / 'sr_comparison.png')
    
    # Save results
    # Convert numpy types for JSON serialization
    results_json = {}
    for model_name, model_results in results.items():
        results_json[model_name] = {
            'best_val_acc': float(model_results['best_val_acc']),
            'class_names': model_results['class_names']
        }
    
    with open(Path(EvalConfig.RESULTS_DIR) / 'results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT 03 COMPLETE!")
    print("="*60)
    print("\nKey Findings:")
    lr_acc = results['lr_baseline']['best_val_acc']
    print(f"  LR Baseline: {lr_acc:.2f}%")
    
    for model in EvalConfig.SR_MODELS:
        if model in results:
            improvement = results[model]['best_val_acc'] - lr_acc
            print(f"  {model}: {results[model]['best_val_acc']:.2f}% "
                  f"({improvement:+.2f}%)")
    
    print("\nNext steps:")
    print("1. Review sr_comparison.png for detailed analysis")
    print("2. Consider which SR model provides best trade-off")
    print("3. Proceed to domain-specific experiments (LASSR if time permits)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
