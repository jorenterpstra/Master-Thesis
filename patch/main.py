import torch
from pathlib import Path
from models import PatchEmbeddingScorer
from training_loop import TrainingConfig, train_model
from dataloader import get_patch_rank_loader
from torch.utils.data import random_split
import json

def main():
    # Setup paths
    data_root = Path("/scratch/6403840/data/imagenet")
    save_root = Path("runs/imagenet_training")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loading
    print("\nSetting up data loaders...")
    train_loader = get_patch_rank_loader(
        data_root, 
        split='train',
        batch_size=32,
        num_workers=8,
        shuffle=True
    )
    
    val_loader = get_patch_rank_loader(
        data_root, 
        split='val',
        batch_size=32,
        num_workers=8
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
        save_dir=save_root,
        plot_every=5,          # Visualize every 5 epochs
        save_best=True,
        num_epochs=50,         # Train for 50 epochs
        batch_size=32,
        num_workers=8,
        optimizer={
            'name': 'sgd',
            'lr': 0.001,       # Initial learning rate
            'momentum': 0.9,
            'weight_decay': 1e-4
        },
        scheduler={
            'name': 'cosine',
            'T_max': 50,       # Match with num_epochs
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

if __name__ == "__main__":
    main()
