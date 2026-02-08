import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from training_utils import visualize_results, visualize_results_rgb
import time
import wandb

#
# TRAINING LOOP
#

# example usage: 
# generator, discriminator = training_loop.enhanced_training_loop(dataset,
#                                         dataloader,
#                                         AgriculturalTrainer,
#                                         AgriculturalAttentionGenerator,
#                                         AgriculturalDiscriminator,
#                                         model_name="agricultural_gen_and_dis_model")

# for EnhancedTrainer based trainer classes
def enhanced_training_loop(dataset, 
                           dataloader, 
                           Trainer, 
                           Generator, 
                           Discriminator, 
                           residual_blocks_list=[4], 
                           visualize=True, 
                           rbg=True, 
                           checkpoint_dir="checkpoints",
                           model_name="enhanced_model",
                           num_epochs_per_block=5,
                           total_epochs=10,
                           wandb_project_name=None,
                           wandb_entity=None,
                           use_wandb=False):

    if os.path.exists(checkpoint_dir) == False:
        os.makedirs(checkpoint_dir)

    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # residual_blocks_list = [4, 8, 16, 32] <--- original list
    all_losses = {}

    for num_blocks in residual_blocks_list:
        # rename model directory for each num_blocks
        model_name = f"{model_name}_{num_blocks}blocks_total{total_epochs}epochs"
        if os.path.exists(os.path.join(checkpoint_dir, model_name)) == False:
            os.makedirs(os.path.join(checkpoint_dir, model_name))

        # Initialize wandb with config
        if use_wandb:
            run_name = f"{model_name}_{num_blocks}blocks"
            wandb_run = wandb.init(
                project = wandb_project_name,
                entity = wandb_entity,  
                name = run_name,
                config = {
                    "num_residual_blocks": num_blocks,
                    "num_epochs_per_block": num_epochs_per_block,
                    "total_epochs": total_epochs,
                    "model_name": model_name,
                    "generator": Generator.__name__,
                    "discriminator": Discriminator.__name__,
                    "trainer": Trainer.__name__,   
                },
                # reinit=True,
                resume="allow"
            )

        generator = Generator(num_residual_blocks=num_blocks).to(device)
        discriminator = Discriminator().to(device)

        trainer = Trainer(generator, discriminator, device, total_epochs=total_epochs)

        if use_wandb:
            wandb.watch(generator, log="all", log_freq=100, log_graph=True)
            wandb.watch(discriminator, log="all", log_freq=100, log_graph=True)

        # reload from latest checkpoint if available
        start_epoch = load_latest_checkpoint(generator, discriminator, 
                                            trainer.g_optimizer, trainer.d_optimizer, trainer, 
                                            checkpoint_dir=checkpoint_dir,
                                            model_name=model_name)

        num_epochs = num_epochs_per_block
        step = 0
        epoch_losses = []

        print("Starting training...")
        for epoch in range(0 + start_epoch, start_epoch + num_epochs):
            trainer.update_loss_weights(epoch)

            epoch_d_loss = 0
            epoch_g_total = 0
            epoch_g_adv = 0
            epoch_g_content = 0
            num_batches = 0

            for batch_idx, (low_res, high_res) in enumerate(dataloader):
                low_res, high_res = low_res.to(device), high_res.to(device)
                d_loss, g_total_loss, g_adv_loss, g_content_loss = trainer.train_step(low_res, high_res, step)

                epoch_d_loss += d_loss
                epoch_g_total += g_total_loss
                epoch_g_adv += g_adv_loss
                epoch_g_content += g_content_loss
                num_batches += 1
                step += 1

                if d_loss > 0 and g_adv_loss > 0 and g_content_loss > 0:
                    trainer.adapt_training_schedule(d_loss, g_adv_loss)

                if use_wandb and batch_idx % 10 == 0:
                    wandb.log({
                        "D_loss": d_loss,
                        "G_total_loss": g_total_loss,
                        "G_adv_loss": g_adv_loss,
                        "G_content_loss": g_content_loss,
                        "epoch": epoch,
                        "step": step,
                        "G_lr": trainer.g_lr,
                        "D_lr": trainer.d_lr
                    }, step=step)
                elif not use_wandb and batch_idx % 100 == 0:
                    print(
                        f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                        f"D_loss: {d_loss:.4f} G_loss: {g_total_loss:.4f}"
                    )
                
                if batch_idx % 400 == 0 and batch_idx > 0:
                    save_checkpoint(epoch, generator, discriminator,
                                    trainer.g_optimizer, trainer.d_optimizer, trainer,
                                    metrics={'d_loss': d_loss, 
                                             'g_total_loss': g_total_loss, 
                                             'g_adv_loss': g_adv_loss, 
                                             'g_content_loss': g_content_loss},
                                    checkpoint_dir=checkpoint_dir,
                                    model_name=model_name,
                                    step=batch_idx + len(dataloader)*epoch)

            # avg_d_loss = epoch_d_loss / num_batches
            avg_g_total = epoch_g_total / num_batches
            # avg_g_adv = epoch_g_adv / num_batches
            # avg_g_content = epoch_g_content / num_batches
            trainer.print_loss_summary(epoch)

            epoch_losses.append(avg_g_total)

            # Save checkpoint after EVERY epoch
            save_checkpoint(epoch, generator, discriminator, 
                            trainer.g_optimizer, trainer.d_optimizer, trainer,
                            metrics={'d_loss': d_loss, 
                                    'g_total_loss': g_total_loss, 
                                    'g_adv_loss': g_adv_loss, 
                                    'g_content_loss': g_content_loss},
                            checkpoint_dir=checkpoint_dir,
                            model_name=model_name)


            if visualize and epoch % 2 == 0:
                test_loader = DataLoader(dataset, batch_size=4, shuffle=True)
                for low_res, high_res in test_loader:
                    high_res = high_res.to(device)
                    low_res = low_res.to(device)
                    try:
                        if rbg:
                            visualize_results_rgb(generator, [(low_res, high_res)], device, epoch)
                        else:
                            visualize_results(generator, [(low_res, high_res)], device, epoch)
                    except Exception as e:
                        print(f"Visualization error at epoch {epoch}: {e}")
                    break
            
            
        # Save models after training with current num_blocks
        torch.save(generator.state_dict(), f"generator_{num_blocks}blocks.pth")
        torch.save(discriminator.state_dict(), f"discriminator_{num_blocks}blocks.pth")
        print(f"Models saved for {num_blocks} residual blocks!")

        all_losses[num_blocks] = epoch_losses

    if use_wandb:
        wandb.finish()

    plt.figure(figsize=(10,6))
    for blocks, losses in all_losses.items():
        plt.plot(losses, label=f'{blocks} residual blocks')
    plt.xlabel('Epoch')
    plt.ylabel('Generator Total Loss')
    plt.title('ResidualGenerator Loss per Epoch')
    plt.legend()
    plt.show()

    return generator, discriminator

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# for AdaptiveTrainer based trainer classes
def training_loop(dataset, 
                  dataloader, 
                  Trainer, 
                  Generator, 
                  Discriminator, 
                  num_residual_blocks=4,
                  visualize=True, 
                  rbg=True, 
                  checkpoint_dir="checkpoints",
                  model_name="enhanced_model",
                  num_epochs_per_block=5,
                  total_epochs=10):

    # Initialize models
    generator = Generator(num_residual_blocks=num_residual_blocks).to(device)
    discriminator = Discriminator().to(device)

    # Initialize trainer
    trainer = Trainer(generator, discriminator, device, total_epochs=total_epochs)

    model_name = f"{model_name}_{num_residual_blocks}blocks_total{total_epochs}epochs"
    if os.path.exists(os.path.join(checkpoint_dir, model_name)) == False:
        os.makedirs(os.path.join(checkpoint_dir, model_name))
    # reload from latest checkpoint if available
    start_epoch = load_latest_checkpoint(generator, discriminator, 
                                        trainer.g_optimizer, trainer.d_optimizer, trainer, 
                                        checkpoint_dir=checkpoint_dir,
                                        model_name=model_name)

    # Training loop
    num_epochs = num_epochs_per_block
    step = 0

    print("Starting training...")
    for epoch in range(0 + start_epoch, start_epoch + num_epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0
        num_batches = 0

        for batch_idx, (low_res, high_res) in enumerate(dataloader):
            low_res, high_res = low_res.to(device), high_res.to(device)
            # Train step
            d_loss, g_loss = trainer.train_step(low_res, high_res, step)

            epoch_d_loss += d_loss
            epoch_g_loss += g_loss
            num_batches += 1
            step += 1

            # Adapt training schedule
            if d_loss > 0 and g_loss > 0:
                trainer.adapt_training_schedule(d_loss, g_loss)

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                    f"D_loss: {d_loss:.4f} G_loss: {g_loss:.4f}"
                )
            if batch_idx % 500 == 0 and batch_idx > 0:
                save_checkpoint(epoch, generator, discriminator,
                                trainer.g_optimizer, trainer.d_optimizer, trainer,
                                metrics={'avg_d_loss': avg_d_loss, 
                                        'avg_g_loss': avg_g_loss},
                                checkpoint_dir=checkpoint_dir,
                                model_name=model_name,
                                step=batch_idx + len(dataloader)*epoch)

        # Epoch summary
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        print(
            f"Epoch [{epoch}/{num_epochs}] Avg D_loss: {avg_d_loss:.4f} Avg G_loss: {avg_g_loss:.4f}"
        )

        # Save checkpoint after EVERY epoch
        save_checkpoint(epoch, generator, discriminator,
                        trainer.g_optimizer, trainer.d_optimizer, trainer,
                        metrics={'avg_d_loss': avg_d_loss, 
                                 'avg_g_loss': avg_g_loss},
                        checkpoint_dir=checkpoint_dir,
                        model_name=model_name,
                        step=len(dataloader)*epoch)

        # Visualize results every 2 epochs
        if visualize and epoch % 2 == 0:
            test_loader = DataLoader(dataset, batch_size=8, shuffle=True)
            for low_res, high_res in test_loader:
                high_res = high_res.to(device)
                low_res = low_res.to(device)
                try:
                    if rbg:
                        visualize_results_rgb(generator, [(low_res, high_res)], device, epoch)
                    else:
                        visualize_results(generator, [(low_res, high_res)], device, epoch)
                except Exception as e:
                    print(f"Visualization error at epoch {epoch}: {e}")
                break


    print("Training completed!")


def save_checkpoint(epoch, generator, discriminator, 
                    g_optimizer, d_optimizer, trainer, metrics, 
                    checkpoint_dir="checkpoints", 
                    model_name="enhanced_model",
                    step=0):
    """Save a complete checkpoint"""
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir = checkpoint_dir / model_name
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}_step_{step}.pt"
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    
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
    
def load_latest_checkpoint(generator, discriminator, 
                         g_optimizer, d_optimizer, trainer, 
                         checkpoint_dir="checkpoints", 
                         model_name="enhanced_model"):
    """Load the latest checkpoint if available"""
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir = checkpoint_dir / model_name
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    
    if not latest_path.exists():
        print("No checkpoint found at latest path.")
        return 0  # Start from epoch 0
    
    checkpoint = torch.load(latest_path)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    
    trainer.g_loss_history = checkpoint['trainer_state']['g_loss_history']
    trainer.d_loss_history = checkpoint['trainer_state']['d_loss_history']
    trainer.g_lr = checkpoint['trainer_state']['g_lr']
    trainer.d_lr = checkpoint['trainer_state']['d_lr']
    
    start_epoch = checkpoint['epoch'] + 1
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    return start_epoch

def save_classifier_checkpoint(epoch, classifier, 
                    optimizer, running_loss, 
                    checkpoint_dir="checkpoints", 
                    model_name="basic_classifier_model"):
    """Save a complete checkpoint"""
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir = checkpoint_dir / model_name
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    
    checkpoint = {
        'epoch': epoch,
        'classifier_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'running_loss': running_loss,
        'timestamp': time.time(),
    }
    
    # Save both specific epoch and latest
    torch.save(checkpoint, checkpoint_path)
    torch.save(checkpoint, latest_path)
    
    print(f"✓ Checkpoint saved: {checkpoint_path.name}")

def load_latest_classifier_checkpoint(classifier, 
                         optimizer, 
                         checkpoint_dir="checkpoints", 
                         model_name="basic_classifier_model"):
    """Load the latest checkpoint if available"""
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir = checkpoint_dir / model_name
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    
    if not latest_path.exists():
        print("No checkpoint found at latest path.")
        return 0  # Start from epoch 0
    
    checkpoint = torch.load(latest_path)
    
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    return start_epoch

def load_classifier_from_path(classifier, 
                         optimizer, 
                         checkpoint_path):
    """Load a specific checkpoint from path"""
    
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at path: {checkpoint_path}")
        return 0  # Start from epoch 0
    
    checkpoint = torch.load(checkpoint_path)
    
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    print(f"✓ Loaded classifier checkpoint from epoch {checkpoint['epoch']}")
    
    return start_epoch