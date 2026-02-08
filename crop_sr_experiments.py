"""
experimental_framework.py

Comprehensive experimental setup for comparing architectures and loss functions
on crop disease super-resolution task.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
import os


# ============================================================================
# EXPERIMENT 1: ARCHITECTURE COMPARISON
# ============================================================================

class ArchitectureExperiment:
    """
    Compare SimpleGenerator, ResidualGenerator, and AgriculturalAttentionGenerator
    on crop disease images.
    """
    def __init__(self, dataset, device, batch_size=16):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.results = {}
        
    def run_experiment(self, num_epochs=50, save_dir='experiment1_results'):
        """Run all architecture comparisons"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Define architectures to test
        architectures = {
            'SimpleGenerator': {
                'generator': self._get_simple_generator,
                'discriminator': self._get_simple_discriminator,
                'description': 'Basic CNN with upsampling'
            },
            'ResidualGenerator': {
                'generator': lambda: self._get_residual_generator(num_blocks=4),
                'discriminator': self._get_simple_discriminator,
                'description': '4 residual blocks with PixelShuffle'
            },
            'ResidualGenerator_8blocks': {
                'generator': lambda: self._get_residual_generator(num_blocks=8),
                'discriminator': self._get_simple_discriminator,
                'description': '8 residual blocks (deeper)'
            },
            'AgriculturalAttention': {
                'generator': lambda: self._get_attention_generator(num_blocks=8),
                'discriminator': self._get_spectral_discriminator,
                'description': 'Attention-enhanced for crops (RGB)'
            }
        }
        
        # Train each architecture
        for arch_name, arch_config in architectures.items():
            print(f"\n{'='*60}")
            print(f"Training: {arch_name}")
            print(f"Description: {arch_config['description']}")
            print(f"{'='*60}\n")
            
            # Initialize models
            generator = arch_config['generator']().to(self.device)
            discriminator = arch_config['discriminator']().to(self.device)
            
            # Train with standard combined loss
            from training_classes import EnhancedTrainer
            trainer = EnhancedTrainer(generator, discriminator, self.device, loss_type='combined')
            
            # Track metrics
            metrics = self._train_architecture(
                trainer, generator, discriminator, 
                arch_name, num_epochs, save_dir
            )
            
            self.results[arch_name] = {
                'metrics': metrics,
                'description': arch_config['description'],
                'params': self._count_parameters(generator)
            }
        
        # Generate comparison report
        self._generate_architecture_report(save_dir)
        
    def _get_simple_generator(self):
        """Create RGB-compatible SimpleGenerator"""
        from generators import SimpleGenerator
        
        class RGBSimpleGenerator(nn.Module):
            def __init__(self):
                super(RGBSimpleGenerator, self).__init__()
                self.network = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(64, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(32, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 3, kernel_size=3, padding=1),
                    nn.Tanh(),
                )
            
            def forward(self, x):
                return self.network(x)
        
        return RGBSimpleGenerator()
    
    def _get_residual_generator(self, num_blocks=4):
        """Create RGB ResidualGenerator"""
        from generators import ResidualGenerator
        return ResidualGenerator(num_residual_blocks=num_blocks)
    
    def _get_attention_generator(self, num_blocks=8):
        """Create agricultural attention generator"""
        from agricultural_sr import AgriculturalAttentionGenerator
        return AgriculturalAttentionGenerator(num_residual_blocks=num_blocks)
    
    def _get_simple_discriminator(self):
        """Create RGB SimpleDiscriminator"""
        from discriminators import SimpleDiscriminator
        return SimpleDiscriminator()
    
    def _get_spectral_discriminator(self):
        """Create RGB SpectralNormDiscriminator"""
        from agricultural_sr import AgriculturalDiscriminator
        return AgriculturalDiscriminator()
    
    def _train_architecture(self, trainer, generator, discriminator, 
                           arch_name, num_epochs, save_dir):
        """Train a single architecture and track metrics"""
        metrics = {
            'd_loss': [],
            'g_total_loss': [],
            'g_adv_loss': [],
            'g_content_loss': [],
            'epoch_times': []
        }
        
        for epoch in range(num_epochs):
            import time
            epoch_start = time.time()
            
            trainer.update_loss_weights(epoch, num_epochs)
            
            epoch_d = 0
            epoch_g_total = 0
            epoch_g_adv = 0
            epoch_g_content = 0
            num_batches = 0
            
            for step, (low_res, high_res) in enumerate(self.dataloader):
                low_res, high_res = low_res.to(self.device), high_res.to(self.device)
                
                d_loss, g_total, g_adv, g_content = trainer.train_step(
                    low_res, high_res, step
                )
                
                epoch_d += d_loss
                epoch_g_total += g_total
                epoch_g_adv += g_adv
                epoch_g_content += g_content
                num_batches += 1
                
                if d_loss > 0 and g_adv > 0:
                    trainer.adapt_training_schedule(d_loss, g_adv)
            
            # Record metrics
            metrics['d_loss'].append(epoch_d / num_batches)
            metrics['g_total_loss'].append(epoch_g_total / num_batches)
            metrics['g_adv_loss'].append(epoch_g_adv / num_batches)
            metrics['g_content_loss'].append(epoch_g_content / num_batches)
            metrics['epoch_times'].append(time.time() - epoch_start)
            
            if epoch % 10 == 0:
                print(f"[{arch_name}] Epoch {epoch}/{num_epochs} - "
                      f"D: {metrics['d_loss'][-1]:.4f}, "
                      f"G: {metrics['g_total_loss'][-1]:.4f}")
                
                # Save sample outputs
                self._save_sample_outputs(generator, arch_name, epoch, save_dir)
        
        # Save final model
        torch.save(generator.state_dict(), 
                  f"{save_dir}/{arch_name}_generator.pth")
        
        return metrics
    
    def _save_sample_outputs(self, generator, arch_name, epoch, save_dir):
        """Save sample super-resolution outputs"""
        generator.eval()
        with torch.no_grad():
            test_loader = DataLoader(self.dataset, batch_size=4, shuffle=True)
            low_res, high_res = next(iter(test_loader))
            low_res, high_res = low_res.to(self.device), high_res.to(self.device)
            
            fake_high_res = generator(low_res)
            
            # Denormalize images
            def denorm(x):
                return (x + 1) / 2
            
            fig, axes = plt.subplots(3, 4, figsize=(12, 9))
            for i in range(4):
                axes[0, i].imshow(denorm(low_res[i]).cpu().permute(1, 2, 0))
                axes[0, i].set_title("Low Res Input")
                axes[0, i].axis("off")
                
                axes[1, i].imshow(denorm(fake_high_res[i]).cpu().permute(1, 2, 0))
                axes[1, i].set_title("Generated SR")
                axes[1, i].axis("off")
                
                axes[2, i].imshow(denorm(high_res[i]).cpu().permute(1, 2, 0))
                axes[2, i].set_title("Ground Truth")
                axes[2, i].axis("off")
            
            plt.suptitle(f"{arch_name} - Epoch {epoch}")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{arch_name}_epoch{epoch}.png", dpi=150)
            plt.close()
        
        generator.train()
    
    def _count_parameters(self, model):
        """Count trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def _generate_architecture_report(self, save_dir):
        """Generate comparison plots and summary"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Generator Total Loss
        ax = axes[0, 0]
        for arch_name, data in self.results.items():
            ax.plot(data['metrics']['g_total_loss'], label=arch_name, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Generator Total Loss')
        ax.set_title('Generator Loss Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Discriminator Loss
        ax = axes[0, 1]
        for arch_name, data in self.results.items():
            ax.plot(data['metrics']['d_loss'], label=arch_name, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Discriminator Loss')
        ax.set_title('Discriminator Loss Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Content Loss
        ax = axes[1, 0]
        for arch_name, data in self.results.items():
            ax.plot(data['metrics']['g_content_loss'], label=arch_name, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Content Loss')
        ax.set_title('Content Loss Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Training Time
        ax = axes[1, 1]
        arch_names = list(self.results.keys())
        avg_times = [np.mean(self.results[name]['metrics']['epoch_times']) 
                    for name in arch_names]
        ax.bar(arch_names, avg_times)
        ax.set_ylabel('Avg Time per Epoch (s)')
        ax.set_title('Training Speed Comparison')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/architecture_comparison.png", dpi=150)
        plt.show()
        
        # Save summary JSON
        summary = {}
        for arch_name, data in self.results.items():
            summary[arch_name] = {
                'description': data['description'],
                'parameters': data['params'],
                'final_g_loss': data['metrics']['g_total_loss'][-1],
                'final_d_loss': data['metrics']['d_loss'][-1],
                'final_content_loss': data['metrics']['g_content_loss'][-1],
                'avg_epoch_time': np.mean(data['metrics']['epoch_times'])
            }
        
        with open(f"{save_dir}/architecture_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*60)
        print("ARCHITECTURE COMPARISON SUMMARY")
        print("="*60)
        for arch_name, stats in summary.items():
            print(f"\n{arch_name}:")
            print(f"  Parameters: {stats['parameters']:,}")
            print(f"  Final G Loss: {stats['final_g_loss']:.4f}")
            print(f"  Final Content Loss: {stats['final_content_loss']:.4f}")
            print(f"  Avg Epoch Time: {stats['avg_epoch_time']:.2f}s")


# ============================================================================
# EXPERIMENT 2: LOSS FUNCTION COMPARISON
# ============================================================================

class LossFunctionExperiment:
    """
    Compare different loss functions on the same architecture.
    Tests: pixel-only, perceptual-only, combined, and agricultural-specific.
    """
    def __init__(self, dataset, device, batch_size=16):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.results = {}
    
    def run_experiment(self, num_epochs=50, save_dir='experiment2_results'):
        """Run all loss function comparisons"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Use ResidualGenerator as baseline architecture
        from generators import ResidualGenerator
        from discriminators import SimpleDiscriminator
        
        loss_configs = {
            'pixel_only': {
                'loss_type': 'pixel',
                'description': 'MSE pixel loss only'
            },
            'perceptual_only': {
                'loss_type': 'perceptual',
                'description': 'VGG perceptual loss only'
            },
            'combined_balanced': {
                'loss_type': 'combined',
                'pixel_weight': 0.5,
                'perceptual_weight': 0.5,
                'description': 'Equal pixel + perceptual'
            },
            'combined_pixel_heavy': {
                'loss_type': 'combined',
                'pixel_weight': 0.8,
                'perceptual_weight': 0.2,
                'description': 'Pixel-focused (80/20)'
            },
            'combined_perceptual_heavy': {
                'loss_type': 'combined',
                'pixel_weight': 0.2,
                'perceptual_weight': 0.8,
                'description': 'Perceptual-focused (20/80)'
            },
            'agricultural_specialized': {
                'loss_type': 'agricultural',
                'description': 'Agricultural loss (color+texture+perceptual)'
            }
        }
        
        for loss_name, loss_config in loss_configs.items():
            print(f"\n{'='*60}")
            print(f"Training with: {loss_name}")
            print(f"Description: {loss_config['description']}")
            print(f"{'='*60}\n")
            
            # Initialize fresh models
            generator = ResidualGenerator(num_residual_blocks=8).to(self.device)
            discriminator = SimpleDiscriminator().to(self.device)
            
            # Create trainer based on loss type
            if loss_config['loss_type'] == 'agricultural':
                from agricultural_sr import (AgriculturalTrainer, 
                                            AgriculturalAttentionGenerator,
                                            AgriculturalDiscriminator)
                # Use agricultural-specific models
                generator = AgriculturalAttentionGenerator(num_residual_blocks=8).to(self.device)
                discriminator = AgriculturalDiscriminator().to(self.device)
                trainer = AgriculturalTrainer(generator, discriminator, self.device)
            else:
                from training_classes import EnhancedTrainer
                trainer = EnhancedTrainer(generator, discriminator, self.device, 
                                        loss_type=loss_config['loss_type'])
                
                # Set custom weights if specified
                if 'pixel_weight' in loss_config:
                    trainer.content_loss.update_weights(
                        loss_config['pixel_weight'],
                        loss_config['perceptual_weight']
                    )
            
            # Train and track metrics
            metrics = self._train_with_loss(
                trainer, generator, discriminator,
                loss_name, num_epochs, save_dir
            )
            
            self.results[loss_name] = {
                'metrics': metrics,
                'description': loss_config['description']
            }
        
        # Generate comparison report
        self._generate_loss_report(save_dir)
    
    def _train_with_loss(self, trainer, generator, discriminator,
                        loss_name, num_epochs, save_dir):
        """Train with specific loss function"""
        metrics = {
            'd_loss': [],
            'g_total_loss': [],
            'g_adv_loss': [],
            'g_content_loss': []
        }
        
        for epoch in range(num_epochs):
            if hasattr(trainer, 'update_loss_weights'):
                trainer.update_loss_weights(epoch, num_epochs)
            
            epoch_d = 0
            epoch_g_total = 0
            epoch_g_adv = 0
            epoch_g_content = 0
            num_batches = 0
            
            for step, (low_res, high_res) in enumerate(self.dataloader):
                low_res, high_res = low_res.to(self.device), high_res.to(self.device)
                
                d_loss, g_total, g_adv, g_content = trainer.train_step(
                    low_res, high_res, step
                )
                
                epoch_d += d_loss
                epoch_g_total += g_total
                epoch_g_adv += g_adv
                epoch_g_content += g_content
                num_batches += 1
                
                if d_loss > 0 and g_adv > 0:
                    trainer.adapt_training_schedule(d_loss, g_adv)
            
            metrics['d_loss'].append(epoch_d / num_batches)
            metrics['g_total_loss'].append(epoch_g_total / num_batches)
            metrics['g_adv_loss'].append(epoch_g_adv / num_batches)
            metrics['g_content_loss'].append(epoch_g_content / num_batches)
            
            if epoch % 10 == 0:
                print(f"[{loss_name}] Epoch {epoch}/{num_epochs} - "
                      f"Content: {metrics['g_content_loss'][-1]:.4f}")
                
                self._save_loss_comparison_outputs(
                    generator, loss_name, epoch, save_dir
                )
        
        torch.save(generator.state_dict(),
                  f"{save_dir}/{loss_name}_generator.pth")
        
        return metrics
    
    def _save_loss_comparison_outputs(self, generator, loss_name, epoch, save_dir):
        """Save outputs for visual comparison"""
        generator.eval()
        with torch.no_grad():
            test_loader = DataLoader(self.dataset, batch_size=4, shuffle=True)
            low_res, high_res = next(iter(test_loader))
            low_res, high_res = low_res.to(self.device), high_res.to(self.device)
            
            fake_high_res = generator(low_res)
            
            def denorm(x):
                return torch.clamp((x + 1) / 2, 0, 1)
            
            fig, axes = plt.subplots(3, 4, figsize=(12, 9))
            for i in range(4):
                # Focus on disease-relevant details
                axes[0, i].imshow(denorm(low_res[i]).cpu().permute(1, 2, 0))
                axes[0, i].set_title("Input (64x64)")
                axes[0, i].axis("off")
                
                axes[1, i].imshow(denorm(fake_high_res[i]).cpu().permute(1, 2, 0))
                axes[1, i].set_title(f"Generated ({loss_name})")
                axes[1, i].axis("off")
                
                axes[2, i].imshow(denorm(high_res[i]).cpu().permute(1, 2, 0))
                axes[2, i].set_title("Ground Truth (256x256)")
                axes[2, i].axis("off")
            
            plt.suptitle(f"{loss_name} - Epoch {epoch}")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{loss_name}_epoch{epoch}.png", dpi=150)
            plt.close()
        
        generator.train()
    
    def _generate_loss_report(self, save_dir):
        """Generate loss function comparison report"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Content Loss Evolution
        ax = axes[0, 0]
        for loss_name, data in self.results.items():
            ax.plot(data['metrics']['g_content_loss'], 
                   label=loss_name, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Content Loss')
        ax.set_title('Content Loss: Preserving Disease Features')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Total Generator Loss
        ax = axes[0, 1]
        for loss_name, data in self.results.items():
            ax.plot(data['metrics']['g_total_loss'], 
                   label=loss_name, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Generator Loss')
        ax.set_title('Total Generator Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Adversarial Loss
        ax = axes[1, 0]
        for loss_name, data in self.results.items():
            ax.plot(data['metrics']['g_adv_loss'], 
                   label=loss_name, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Adversarial Loss')
        ax.set_title('Adversarial Loss: Realism')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Final Convergence Comparison
        ax = axes[1, 1]
        loss_names = list(self.results.keys())
        final_content = [self.results[name]['metrics']['g_content_loss'][-1] 
                        for name in loss_names]
        bars = ax.bar(range(len(loss_names)), final_content)
        ax.set_xticks(range(len(loss_names)))
        ax.set_xticklabels(loss_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Final Content Loss')
        ax.set_title('Final Content Loss Comparison')
        
        # Color bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/loss_function_comparison.png", dpi=150)
        plt.show()
        
        # Summary
        summary = {}
        for loss_name, data in self.results.items():
            summary[loss_name] = {
                'description': data['description'],
                'final_content_loss': data['metrics']['g_content_loss'][-1],
                'final_total_loss': data['metrics']['g_total_loss'][-1],
                'content_improvement': (
                    data['metrics']['g_content_loss'][0] - 
                    data['metrics']['g_content_loss'][-1]
                )
            }
        
        with open(f"{save_dir}/loss_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*60)
        print("LOSS FUNCTION COMPARISON SUMMARY")
        print("="*60)
        print("\nKey Question: Does perceptual loss preserve disease features?")
        for loss_name, stats in summary.items():
            print(f"\n{loss_name}:")
            print(f"  {stats['description']}")
            print(f"  Final Content Loss: {stats['final_content_loss']:.4f}")
            print(f"  Improvement: {stats['content_improvement']:.4f}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the experimental framework.
    
    To run:
    1. Load your crop disease dataset
    2. Run Experiment 1 (architectures)
    3. Run Experiment 2 (loss functions)
    4. Analyze results in the generated reports
    """
    
    # Setup (replace with your actual data loading)
    from dataset_utils import SuperResolutionDataset
    from torch.utils.data import DataLoader
    
    # Example setup
    # image_files = load_your_crop_images()  # Your image loading code
    # dataset = SuperResolutionDataset(image_files, low_res_size=64, high_res_size=256)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # EXPERIMENT 1: Architecture Comparison
    # print("\n" + "="*60)
    # print("EXPERIMENT 1: ARCHITECTURE COMPARISON")
    # print("="*60)
    # exp1 = ArchitectureExperiment(dataset, device, batch_size=16)
    # exp1.run_experiment(num_epochs=50, save_dir='exp1_architectures')
    
    # EXPERIMENT 2: Loss Function Comparison
    # print("\n" + "="*60)
    # print("EXPERIMENT 2: LOSS FUNCTION COMPARISON")
    # print("="*60)
    # exp2 = LossFunctionExperiment(dataset, device, batch_size=16)
    # exp2.run_experiment(num_epochs=50, save_dir='exp2_loss_functions')
    
    print("\n" + "="*60)
    print("EXPERIMENTAL FRAMEWORK READY")
    print("="*60)
    print("\nTo run experiments:")
    print("1. Load your crop disease dataset")
    print("2. Uncomment the experiment code above")
    print("3. Results will be saved to exp1_architectures/ and exp2_loss_functions/")
    print("\nExpected outputs:")
    print("- Training curves and comparisons")
    print("- Sample super-resolution outputs")
    print("- JSON summaries with metrics")
    print("- Visual comparison plots")
