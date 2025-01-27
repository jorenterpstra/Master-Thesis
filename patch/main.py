import os
# Verify CUDA setup before importing torch
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from models import PatchEmbeddingScorer, get_model
from training_loop import TrainingConfig, train_model
from dataloader import get_patch_rank_loader
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
    parser.add_argument('--distributed', action='store_true',
                       help='Enable distributed training')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training')
    return parser.parse_args()

def setup_distributed(args):
    """Initialize distributed training if enabled"""
    if args.distributed and args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        args.world_size = dist.get_world_size()
    else:
        args.world_size = 1
        args.local_rank = -1
    return args

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

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
        print(torch.cuda.memory_summary(device=None, abbreviated=False))

def main():
    args = parse_args()
    args = setup_distributed(args) if args.distributed else args
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}" + (f" (local_rank: {args.local_rank})" if args.distributed else ""))
    
    # Data loading with optional distributed sampler
    print("\nSetting up data loaders...")
    train_sampler = None
    if args.distributed:
        train_sampler = DistributedSampler(
            dataset=get_patch_rank_loader(args.data_root, split='train').dataset,
            num_replicas=args.world_size,
            rank=args.local_rank
        )
    
    train_loader = get_patch_rank_loader(
        args.data_root, 
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=(train_sampler is None),
        sampler=train_sampler
    )
    
    val_loader = get_patch_rank_loader(
        args.data_root, 
        split='val',
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    if not args.distributed or args.local_rank <= 0:
        print(f"Dataset sizes - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    # Initialize model
    model = get_model(
        'resnet',  # or use an argument to select model type
        patch_size=16,
        hidden_dim=512,
        num_patches=14
    ).to(device)
    
    # Wrap model in DDP
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank])
    
    # Use model's default optimizer settings
    optimizer_config = getattr(model.module if args.distributed else model, 'default_optimizer', {
        'name': 'sgd',
        'lr': args.lr,
        'momentum': 0.9,
        'weight_decay': 1e-4
    })
    
    # Override learning rate if specified in args
    if args.lr is not None:
        optimizer_config['lr'] = args.lr
    
    # Training configuration
    config = TrainingConfig(
        verbose=2 if not args.distributed or args.local_rank <= 0 else 0,  # Only print on main process
        save_dir=args.save_root,
        plot_every=args.plot_every,
        save_best=True,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed={'enabled': args.distributed,
                    'local_rank': args.local_rank,
                    'world_size': args.world_size,
                    'sampler': train_sampler} if args.distributed else None,
        optimizer=optimizer_config,
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
    
    try:
        # Train model
        if not args.distributed or args.local_rank <= 0:
            print_gpu_info()
            print("\nStarting training...")
        tracker = train_model(model, train_loader, val_loader, config)
        
        # Save results only on main process
        if not args.distributed or args.local_rank <= 0:
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
            print("\nGPU state after training:")
            print_gpu_info()
    
    finally:
        if args.distributed:
            cleanup_distributed()

if __name__ == "__main__":
    main()
