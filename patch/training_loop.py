import torch
import torch.nn as nn
from main import print_gpu_info
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn import functional as F
from datetime import datetime
from utils import AverageMeter, LossTracker, visualize_predictions
import json
import torch.autograd

class PatchScoreLoss(nn.Module):
    """Custom loss for patch score prediction"""
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        assert abs(alpha + beta - 1.0) < 1e-6, "Alpha and beta should sum to 1"
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, pred, target):
        if self.alpha == 1.0:
            return F.mse_loss(pred, target)
        elif self.beta == 1.0:
            return F.kl_div(F.log_softmax(pred, dim=1), 
                            F.softmax(target, dim=1), 
                            reduction='batchmean')
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
            
        # MSE for direct value prediction
        mse_loss = F.mse_loss(pred, target)
        
        # Ranking loss using KL divergence on softmax distributions
        pred_dist = F.softmax(pred * 10, dim=1)  # Scale logits for sharper distribution
        target_dist = F.softmax(target * 10, dim=1)
        rank_loss = F.kl_div(
            pred_dist.log(),
            target_dist,
            reduction='batchmean',
            log_target=False
        )
        
        return self.alpha * mse_loss + self.beta * rank_loss

class TrainingConfig:
    """Enhanced configuration for training settings"""
    def __init__(self, 
                 # Logging settings
                 verbose=1,           # 0: silent, 1: progress bars, 2: + epoch stats
                 save_dir='runs/patch_scorer',
                 plot_every=5,        # Plot predictions every N epochs
                 save_best=True,      # Whether to save best model
                 
                 # Training settings
                 num_epochs=100,
                 batch_size=32,
                 num_workers=4,
                 device='cuda',
                 
                 # Optimizer settings
                 optimizer=None,  # Now accepts None
                 
                 # Scheduler settings
                 scheduler=None,  # Now accepts None
                 
                 # Loss settings
                 loss=None):     # Now accepts None
        
        # Basic settings
        self.verbose = verbose
        self.plot_every = plot_every
        self.save_best = save_best
        
        # Training settings
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Set default configurations if None
        self.optimizer = optimizer if optimizer is not None else {
            'name': 'sgd',
            'lr': 0.01,
            'momentum': 0.9,
            'weight_decay': 1e-4
        }
        
        self.scheduler = scheduler  # Can be None
        
        self.loss = loss if loss is not None else {
            'alpha': 0.7,
            'beta': 0.3
        }
        
        # Create save directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(save_dir) / timestamp
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        self.save_config()
    
    def save_config(self):
        """Save configuration to JSON"""
        config = {
            'verbose': self.verbose,
            'save_dir': str(self.save_dir),
            'plot_every': self.plot_every,
            'save_best': self.save_best,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'device': self.device,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'loss': self.loss
        }
        
        with open(self.save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config):
    model.train()
    metrics = {
        'loss': AverageMeter(),
        'rmse': AverageMeter()
    }
    
    iterator = train_loader
    if config.verbose >= 1:
        iterator = tqdm(train_loader, desc=f'Epoch {epoch} [TRAIN]')
    
    for images, targets in iterator:
        images = images.to(device)
        targets = targets.to(device)
        
        print_gpu_info()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Compute additional metrics
        with torch.no_grad():
            rmse = torch.sqrt(F.mse_loss(outputs, targets))
       # with torch.autograd.detect_anomaly():
            # Backward pass
        optimizer.zero_grad()
        loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics
        batch_size = images.size(0)
        metrics['loss'].update(loss.item(), batch_size)
        metrics['rmse'].update(rmse.item(), batch_size)
        
        # Update progress bar
        if config.verbose >= 1:
            iterator.set_postfix({
                name: f"{meter.avg:.4f}"
                for name, meter in metrics.items()
            })
    
    return {name: meter.avg for name, meter in metrics.items()}

@torch.no_grad()
def validate(model, val_loader, criterion, device, config):
    model.eval()
    metrics = {
        'loss': AverageMeter(),
        'rmse': AverageMeter()
    }
    
    pbar = tqdm(val_loader, desc='[VAL]') if config.verbose >= 1 else val_loader
    
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, targets)
        rmse = torch.sqrt(F.mse_loss(outputs, targets))
        
        # Update metrics
        batch_size = images.size(0)
        metrics['loss'].update(loss.item(), batch_size)
        metrics['rmse'].update(rmse.item(), batch_size)
        
        if config.verbose >= 1:
            pbar.set_postfix({
                name: f"{meter.avg:.4f}"
                for name, meter in metrics.items()
            })
    
    return {name: meter.avg for name, meter in metrics.items()}

def get_optimizer(name, parameters, **kwargs):
    """Get optimizer by name"""
    optimizers = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
        'rmsprop': torch.optim.RMSprop
    }
    
    if name not in optimizers:
        raise ValueError(f"Optimizer {name} not found. Available optimizers: {list(optimizers.keys())}")
    
    return optimizers[name](parameters, **kwargs)

def get_scheduler(name, optimizer, **kwargs):
    """Get learning rate scheduler by name"""
    schedulers = {
        'step': torch.optim.lr_scheduler.StepLR,
        'multistep': torch.optim.lr_scheduler.MultiStepLR,
        'exponential': torch.optim.lr_scheduler.ExponentialLR,
        'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
        'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'warmup_cosine': torch.optim.lr_scheduler.OneCycleLR
    }
    
    if name not in schedulers:
        raise ValueError(f"Scheduler {name} not found. Available schedulers: {list(schedulers.keys())}")
    
    return schedulers[name](optimizer, **kwargs)

def train_model(model, train_loader, val_loader, config=None):
    """Simplified training loop using configuration object"""
    if config is None:
        config = TrainingConfig()
    
    # Setup components using config (use same save_dir for all components)
    tracker = LossTracker(config.save_dir)
    criterion = PatchScoreLoss(**config.loss)
    
    # Get optimizer
    opt_config = config.optimizer.copy()  # Make copy to avoid modifying original
    opt_name = opt_config.pop('name')
    optimizer = get_optimizer(opt_name, model.parameters(), **opt_config)
    
    # Get scheduler (if configured)
    scheduler = None
    if config.scheduler is not None:
        sched_config = config.scheduler.copy()  # Make copy to avoid modifying original
        sched_name = sched_config.pop('name')
        scheduler = get_scheduler(sched_name, optimizer, **sched_config)
    
    # Move model to device
    model = model.to(config.device)
    
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        # Train and validate
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, 
                                  config.device, epoch, config)
        val_metrics = validate(model, val_loader, criterion, config.device, config)
        
        # Update tracking and scheduler if it exists
        tracker.update(epoch, train_metrics, val_metrics)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Print epoch stats
        if config.verbose >= 2:
            print(f"\nEpoch {epoch}")
            for metric in train_metrics.keys():
                print(f"{metric.upper():>8} - Train: {train_metrics[metric]:.4f}, Val: {val_metrics[metric]:.4f}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model (now using config.save_dir directly)
        if config.save_best and val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': {
                    'train': train_metrics,
                    'val': val_metrics
                }
            }
            if scheduler is not None:
                save_dict['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(save_dict, config.save_dir / 'best_model.pth')
            
        # Visualize predictions (now using config.save_dir directly)
        if epoch % config.plot_every == 0:
            visualize_predictions(model, val_loader, config.device, epoch, config.save_dir)
    
    return tracker

if __name__ == "__main__":
    pass