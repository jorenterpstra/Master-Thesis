import os
# Verify CUDA setup before importing torch
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")

import torch
from pathlib import Path
from models import PatchEmbeddingScorer
from training_loop import TrainingConfig, train_model
from dataloader import get_patch_rank_loader
from torch.utils.data import random_split
import json
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Train patch scoring model')
    parser.add_argument('--data-root', type=str, default='/scratch/6403840/data/imagenet',
                        help='Path to ImageNet dataset')
    parser.add_argument('--save-root', type=str, default='runs/imagenet_training',
                        help='Path to save results')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--plot-every', type=int, default=5,
                        help='Plot predictions every N epochs')
    return parser.parse_args()

def print_gpu_info():
    """Print GPU information and memory usage"""
    if torch.cuda.is_available():
        # we want to print the available GPUs and which one we are using
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"Using device: {torch.cuda.current_device()}")
        print(f"Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f}MB")

def main():
    args = parse_args()
    
    # Setup device - will automatically use the GPU specified by CUDA_VISIBLE_DEVICES
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print_gpu_info()
    
    # Data loading
    print("\nSetting up data loaders...")
    train_loader = get_patch_rank_loader(
        args.data_root, 
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    
    val_loader = get_patch_rank_loader(
        args.data_root, 
        split='val',
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Dataset sizes - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    # Initialize model
    model = PatchEmbeddingScorer(
        patch_size=16,
        hidden_dim=512,
        num_patches=14
    ).to(device)
    
    # Training configuration
    config = TrainingConfig(
        verbose=2,              # Show detailed progress
        save_dir=args.save_root,
        plot_every=args.plot_every,          # Visualize every 5 epochs
        save_best=True,
        num_epochs=args.epochs,         # Train for 50 epochs
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        optimizer={
            'name': 'sgd',
            'lr': args.lr,       # Initial learning rate
            'momentum': 0.9,
            'weight_decay': 1e-4
        },
        scheduler={
            'name': 'cosine',
            'T_max': args.epochs,       # Match with num_epochs
            'eta_min': 1e-6    # Minimum learning rate
        },
        loss={
            'alpha': 1.0,
            'beta': 0.0
        }
    )
    
    # Train model
    print("\nStarting training...")
    tracker = train_model(model, train_loader, val_loader, config)
    
    # Get best epoch information from tracker
    val_losses = tracker.history['val']['loss']
    best_epoch = val_losses.index(min(val_losses))
    
    # Save best epoch info
    best_epoch_info = {
        'best_epoch': best_epoch,
        'best_val_loss': min(val_losses),
        'final_val_loss': val_losses[-1],
        'total_epochs': len(val_losses)
    }
    
    with open(config.save_dir / 'best_epoch_info.json', 'w') as f:
        json.dump(best_epoch_info, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best epoch: {best_epoch} with validation loss: {min(val_losses):.4f}")
    
    # Monitor GPU usage during training
    print("\nGPU state after training:")
    print_gpu_info()

if __name__ == "__main__":
    main()
