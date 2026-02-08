"""
Experiment 02: Train Baseline SR Models with Checkpointing

GOALS:
1. Train SimpleGenerator and ResidualGenerator
2. Handle Colab disconnections gracefully
3. Compare training efficiency vs. performance

RESEARCH QUESTIONS:
- What's the minimum model complexity needed?
- How much can we train in a 12-hour Colab session?
- Does early stopping help with limited compute?

KEY FEATURES:
- Automatic checkpoint saving/loading
- Resume from interruption
- Memory-efficient data loading
- Progress tracking
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt

# Import your modules
import sys
sys.path.append('/content/drive/MyDrive/AgriSR_Project/code')
from generators import SimpleGenerator, ResidualGenerator
from discriminators import SimpleDiscriminator
from training_classes import AdaptiveTrainer
from dataset_utils import SuperResolutionDataset

# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Centralized training configuration"""
    
    # Paths
    GDRIVE_ROOT = "/content/drive/MyDrive/AgriSR_Project"
    DATA_DIR = f"{GDRIVE_ROOT}/data/processed/subsets/stratified_25pct"
    CHECKPOINT_DIR = f"{GDRIVE_ROOT}/checkpoints"
    RESULTS_DIR = f"{GDRIVE_ROOT}/results"
    
    # Training hyperparameters
    BATCH_SIZE = 16  # Reduced for Colab
    NUM_EPOCHS = 50  # Will likely not complete in one session
    LEARNING_RATE = 0.0002
    
    # Image sizes
    LOW_RES_SIZE = 64
    HIGH_RES_SIZE = 256
    
    # Checkpoint frequency
    CHECKPOINT_EVERY_N_EPOCHS = 1  # Save frequently!
    VALIDATE_EVERY_N_EPOCHS = 2
    
    # Early stopping
    PATIENCE = 5
    MIN_DELTA = 0.001
    
    # Model variants to train
    MODELS_TO_TRAIN = [
        {'name': 'simple', 'generator': 'SimpleGenerator', 'residual_blocks': None},
        {'name': 'residual_4blocks', 'generator': 'ResidualGenerator', 'residual_blocks': 4},
        {'name': 'residual_8blocks', 'generator': 'ResidualGenerator', 'residual_blocks': 8},
    ]
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

class CheckpointManager:
    """Handles saving and loading checkpoints"""
    
    def __init__(self, checkpoint_dir, model_name):
        self.checkpoint_dir = Path(checkpoint_dir) / model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        
    def save_checkpoint(self, epoch, generator, discriminator, 
                       g_optimizer, d_optimizer, trainer, metrics):
        """Save a complete checkpoint"""
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            'trainer_state': {
                'g_loss_history': trainer.g_loss_history,
                'd_loss_history': trainer.d_loss_history,
                'g_lr': trainer.g_lr,
                'd_lr': trainer.d_lr,
            },
            'metrics': metrics,
            'timestamp': time.time(),
        }
        
        # Save both specific epoch and latest
        torch.save(checkpoint, checkpoint_path)
        torch.save(checkpoint, latest_path)
        
        print(f"✓ Checkpoint saved: {checkpoint_path.name}")
        
        # Keep only last 3 checkpoints to save space
        self._cleanup_old_checkpoints(keep_n=3)
        
    def load_checkpoint(self, generator, discriminator, g_optimizer, d_optimizer, trainer):
        """Load the latest checkpoint if it exists"""
        
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        
        if not latest_path.exists():
            print("No checkpoint found. Starting from scratch.")
            return 0, {}
        
        print(f"Loading checkpoint from: {latest_path}")
        checkpoint = torch.load(latest_path, map_location=TrainingConfig.DEVICE)
        
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        # Restore trainer state
        trainer.g_loss_history = checkpoint['trainer_state']['g_loss_history']
        trainer.d_loss_history = checkpoint['trainer_state']['d_loss_history']
        trainer.g_lr = checkpoint['trainer_state']['g_lr']
        trainer.d_lr = checkpoint['trainer_state']['d_lr']
        
        epoch = checkpoint['epoch']
        metrics = checkpoint['metrics']
        
        print(f"✓ Resumed from epoch {epoch}")
        
        return epoch, metrics
    
    def _cleanup_old_checkpoints(self, keep_n=3):
        """Keep only the N most recent checkpoints"""
        checkpoints = sorted(
            [f for f in self.checkpoint_dir.glob("checkpoint_epoch_*.pt")],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        for old_checkpoint in checkpoints[keep_n:]:
            old_checkpoint.unlink()


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class MetricsTracker:
    """Track and save training metrics"""
    
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {
            'epochs': [],
            'g_losses': [],
            'd_losses': [],
            'train_time': [],
        }
    
    def add(self, epoch, g_loss, d_loss, epoch_time):
        self.metrics['epochs'].append(epoch)
        self.metrics['g_losses'].append(g_loss)
        self.metrics['d_losses'].append(d_loss)
        self.metrics['train_time'].append(epoch_time)
    
    def save(self):
        """Save metrics to JSON"""
        metrics_file = self.save_dir / 'training_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def plot(self):
        """Create training plots"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax = axes[0]
        ax.plot(self.metrics['epochs'], self.metrics['g_losses'], label='Generator')
        ax.plot(self.metrics['epochs'], self.metrics['d_losses'], label='Discriminator')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Time plot
        ax = axes[1]
        ax.plot(self.metrics['epochs'], self.metrics['train_time'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Training Time per Epoch')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=150)
        plt.close()


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# ============================================================================
# DATA LOADING
# ============================================================================

def create_dataloaders(data_dir, batch_size, num_workers=2):
    """Create training and validation dataloaders"""
    
    # Get all image files
    image_files = []
    for class_dir in Path(data_dir).iterdir():
        if class_dir.is_dir():
            image_files.extend(list(class_dir.glob('*.jpg')))
            image_files.extend(list(class_dir.glob('*.JPG')))
    
    print(f"Found {len(image_files)} images")
    
    # Split into train/val (90/10)
    import random
    random.seed(42)
    random.shuffle(image_files)
    
    split_idx = int(0.9 * len(image_files))
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Create datasets
    train_dataset = SuperResolutionDataset(
        train_files,
        low_res_size=TrainingConfig.LOW_RES_SIZE,
        high_res_size=TrainingConfig.HIGH_RES_SIZE
    )
    
    val_dataset = SuperResolutionDataset(
        val_files,
        low_res_size=TrainingConfig.LOW_RES_SIZE,
        high_res_size=TrainingConfig.HIGH_RES_SIZE
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if TrainingConfig.DEVICE == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if TrainingConfig.DEVICE == 'cuda' else False
    )
    
    return train_loader, val_loader


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_model(model_config):
    """Train a single model with checkpointing"""
    
    model_name = model_config['name']
    print("\n" + "="*60)
    print(f"TRAINING MODEL: {model_name}")
    print("="*60)
    
    device = torch.device(TrainingConfig.DEVICE)
    print(f"Using device: {device}")
    
    # Initialize models
    if model_config['generator'] == 'SimpleGenerator':
        generator = SimpleGenerator().to(device)
    else:
        generator = ResidualGenerator(
            num_residual_blocks=model_config['residual_blocks']
        ).to(device)
    
    discriminator = SimpleDiscriminator().to(device)
    
    # Initialize trainer
    trainer = AdaptiveTrainer(generator, discriminator, device)
    
    # Setup checkpoint manager
    checkpoint_mgr = CheckpointManager(TrainingConfig.CHECKPOINT_DIR, model_name)
    
    # Try to load checkpoint
    start_epoch, prev_metrics = checkpoint_mgr.load_checkpoint(
        generator, discriminator,
        trainer.g_optimizer, trainer.d_optimizer,
        trainer
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        TrainingConfig.DATA_DIR,
        TrainingConfig.BATCH_SIZE
    )
    
    # Metrics tracking
    metrics_tracker = MetricsTracker(
        Path(TrainingConfig.RESULTS_DIR) / model_name
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=TrainingConfig.PATIENCE,
        min_delta=TrainingConfig.MIN_DELTA
    )
    
    # Training loop
    step = 0
    print(f"\nStarting training from epoch {start_epoch}")
    
    for epoch in range(start_epoch, TrainingConfig.NUM_EPOCHS):
        epoch_start_time = time.time()
        
        # Training phase
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        num_batches = 0
        
        for batch_idx, (low_res, high_res) in enumerate(train_loader):
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            
            d_loss, g_loss = trainer.train_step(low_res, high_res, step)
            
            epoch_g_loss += g_loss
            epoch_d_loss += d_loss
            num_batches += 1
            step += 1
            
            if d_loss > 0 and g_loss > 0:
                trainer.adapt_training_schedule(d_loss, g_loss)
            
            if batch_idx % 50 == 0:
                print(f"  Batch [{batch_idx}/{len(train_loader)}] "
                      f"D_loss: {d_loss:.4f} G_loss: {g_loss:.4f}")
        
        # Calculate average losses
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        epoch_time = time.time() - epoch_start_time
        
        print(f"\nEpoch [{epoch+1}/{TrainingConfig.NUM_EPOCHS}] "
              f"Avg D_loss: {avg_d_loss:.4f} Avg G_loss: {avg_g_loss:.4f} "
              f"Time: {epoch_time:.2f}s")
        
        # Track metrics
        metrics_tracker.add(epoch, avg_g_loss, avg_d_loss, epoch_time)
        
        # Save checkpoint
        if (epoch + 1) % TrainingConfig.CHECKPOINT_EVERY_N_EPOCHS == 0:
            checkpoint_mgr.save_checkpoint(
                epoch + 1, generator, discriminator,
                trainer.g_optimizer, trainer.d_optimizer,
                trainer, metrics_tracker.metrics
            )
        
        # Validation (optional, less frequent)
        if (epoch + 1) % TrainingConfig.VALIDATE_EVERY_N_EPOCHS == 0:
            val_g_loss = validate(generator, discriminator, val_loader, device)
            print(f"Validation G_loss: {val_g_loss:.4f}")
            
            # Early stopping check
            early_stopping(val_g_loss)
            if early_stopping.should_stop:
                print("Early stopping triggered!")
                break
        
        # Save metrics and plots
        metrics_tracker.save()
        metrics_tracker.plot()
    
    print(f"\n✓ Training complete for {model_name}")
    
    return generator, discriminator, metrics_tracker.metrics


def validate(generator, discriminator, val_loader, device):
    """Run validation"""
    generator.eval()
    criterion = nn.MSELoss()
    
    total_loss = 0
    with torch.no_grad():
        for low_res, high_res in val_loader:
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            
            fake_high_res = generator(low_res)
            loss = criterion(fake_high_res, high_res)
            total_loss += loss.item()
    
    generator.train()
    return total_loss / len(val_loader)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training function"""
    
    print("\n" + "="*60)
    print("EXPERIMENT 02: BASELINE SR MODEL TRAINING")
    print("="*60)
    
    # Create directories
    os.makedirs(TrainingConfig.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TrainingConfig.RESULTS_DIR, exist_ok=True)
    
    # Check data exists
    if not os.path.exists(TrainingConfig.DATA_DIR):
        print(f"\n⚠️  Data not found at: {TrainingConfig.DATA_DIR}")
        print("Run experiment_01 first to prepare data!")
        return
    
    # Train each model
    results = {}
    for model_config in TrainingConfig.MODELS_TO_TRAIN:
        try:
            generator, discriminator, metrics = train_model(model_config)
            results[model_config['name']] = metrics
            
            # Save final model separately
            final_model_dir = Path(TrainingConfig.RESULTS_DIR) / model_config['name']
            torch.save(generator.state_dict(), 
                      final_model_dir / 'generator_final.pth')
            torch.save(discriminator.state_dict(), 
                      final_model_dir / 'discriminator_final.pth')
            
        except Exception as e:
            print(f"Error training {model_config['name']}: {e}")
            continue
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nResults saved to:", TrainingConfig.RESULTS_DIR)
    print("\nNext steps:")
    print("1. Review training curves in results folder")
    print("2. Proceed to Experiment 03 to evaluate SR quality")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
