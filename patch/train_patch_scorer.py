import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from patch_rank import get_patch_rank_loader
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from torch.nn import functional as F
import json
from datetime import datetime
from utils import AverageMeter, LossTracker, visualize_predictions

class PatchScoreLoss(nn.Module):
    """Custom loss for patch score prediction"""
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        assert abs(alpha + beta - 1.0) < 1e-6, "Alpha and beta should sum to 1"
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, pred, target):
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
            
        # MSE for direct value prediction
        mse_loss = F.mse_loss(pred, target)
        
        # Ranking loss using KL divergence
        rank_loss = F.kl_div(
            F.log_softmax(pred, dim=1),
            F.softmax(target, dim=1),
            reduction='batchmean',
            log_target=False
        )
        
        return self.alpha * mse_loss + self.beta * rank_loss

class TrainingConfig:
    """Configuration for training verbosity and logging"""
    def __init__(self, 
                 verbose=1,           # 0: silent, 1: progress bars, 2: + epoch stats, 3: + batch stats
                 use_wandb=False,     # Whether to use Weights & Biases logging
                 wandb_project=None,  # Project name for wandb
                 wandb_entity=None,   # Username or team name for wandb
                 save_dir='runs/patch_scorer',
                 plot_every=5,        # Plot predictions every N epochs
                 save_best=True):     # Whether to save best model
        self.verbose = verbose
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.save_dir = Path(save_dir)
        self.plot_every = plot_every
        self.save_best = save_best

        if self.use_wandb:
            import wandb
            if not wandb.api.api_key:
                print("Warning: wandb not authenticated. Run 'wandb login' first.")
                self.use_wandb = False
            else:
                wandb.init(
                    project=self.wandb_project,
                    entity=self.wandb_entity,
                    config={
                        "verbose": verbose,
                        "plot_every": plot_every,
                        "save_best": save_best
                    }
                )

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config):
    model.train()
    losses = AverageMeter()
    rmse = AverageMeter()
    
    # Create progress bar if verbose enough
    iterator = train_loader
    if config.verbose >= 1:
        iterator = tqdm(train_loader, desc=f'Epoch {epoch} [TRAIN]')
    
    metrics = {
        'loss': losses.avg,
        'predictions': [],
        'targets': []
    }
    
    for i, (images, targets, _) in enumerate(iterator):
        images = images.to(device)
        targets = targets.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_rmse = torch.sqrt(torch.mean((outputs - targets) ** 2))
        
        losses.update(loss.item(), images.size(0))
        rmse.update(batch_rmse.item(), images.size(0))
        
        metrics['predictions'].extend(outputs.cpu().detach().numpy().flatten())
        metrics['targets'].extend(targets.cpu().numpy().flatten())
        
        # Update progress bar or print batch stats
        if config.verbose >= 1:
            iterator.set_postfix({'Loss': f'{losses.avg:.4f}', 'RMSE': f'{rmse.avg:.4f}'})
        elif config.verbose >= 3:
            print(f"Batch {i}: Loss = {loss.item():.4f}, RMSE = {batch_rmse.item():.4f}")
    
    return metrics

@torch.no_grad()
def validate(model, val_loader, criterion, device, config):
    model.eval()
    losses = AverageMeter()
    rmse = AverageMeter()
    
    pbar = tqdm(val_loader, desc='[VAL]')
    metrics = {
        'loss': losses.avg,
        'predictions': [],
        'targets': []
    }
    
    for images, targets, _ in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Calculate RMSE
        batch_rmse = torch.sqrt(torch.mean((outputs - targets) ** 2))
        
        # Update metrics
        losses.update(loss.item(), images.size(0))
        rmse.update(batch_rmse.item(), images.size(0))
        
        # Store predictions and targets for distribution visualization
        metrics['predictions'].extend(outputs.cpu().detach().numpy().flatten())
        metrics['targets'].extend(targets.cpu().numpy().flatten())
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'RMSE': f'{rmse.avg:.4f}'
        })
    
    return metrics

def train_model(model, train_loader, val_loader, num_epochs=100, 
                device='cuda', learning_rate=1e-4, config=None):
    """
    Main training loop with configurable verbosity
    Args:
        model: model to train
        train_loader: training data loader
        val_loader: validation data loader
        num_epochs: number of epochs to train
        device: device to train on
        learning_rate: initial learning rate
        config: TrainingConfig object for controlling verbosity and logging
    """
    if config is None:
        config = TrainingConfig()
    
    save_dir = config.save_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    tracker = LossTracker(save_dir)
    criterion = PatchScoreLoss(alpha=0.7, beta=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, 
        verbose=(config.verbose >= 2)
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train and validate
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, config)
        val_metrics = validate(model, val_loader, criterion, device, config)
        
        # Update tracking
        tracker.update(epoch, train_metrics, val_metrics)
        scheduler.step(val_metrics['loss'])
        
        # Save best model
        if config.save_best and val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, save_dir / 'best_model.pth')
        
        # Visualize predictions
        if epoch % config.plot_every == 0:
            visualize_predictions(model, val_loader, device, epoch, save_dir)
        
        # Log metrics
        if config.use_wandb:
            wandb.log({
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch
            })
        
        # Print epoch stats
        if config.verbose >= 2:
            print(f"\nEpoch {epoch}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}")
    
    if config.use_wandb:
        wandb.finish()
    
    return tracker

if __name__ == "__main__":
    config = TrainingConfig(
        verbose=2,              # Show progress bars and epoch stats
        use_wandb=True,        # Enable wandb logging
        wandb_project='patch-scorer',
        wandb_entity='your_username',
        plot_every=5,          # Plot every 5 epochs
        save_best=True         # Save best model
    )
    
    # model = PatchScorer()
    # train_loader = get_patch_rank_loader("path/to/data", split="train")
    # val_loader = get_patch_rank_loader("path/to/data", split="val")
    # train_model(model, train_loader, val_loader, config=config)
    pass
