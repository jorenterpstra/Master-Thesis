import torch
import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class LossTracker:
    """Track and save training progress"""
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': {},
            'val_metrics': {},
            'epochs': []
        }
        
    def update(self, epoch, train_metrics, val_metrics):
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['val_loss'].append(val_metrics['loss'])
        
        self.save()
        self.plot_progress()
    
    def save(self):
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f)
    
    def plot_progress(self):
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history['epochs'], self.history['train_loss'], label='Train')
        plt.plot(self.history['epochs'], self.history['val_loss'], label='Validation')
        plt.title('Loss Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Distribution plot
        plt.subplot(1, 2, 2)
        if len(self.history['train_metrics']) > 0:
            latest = self.history['train_metrics'][-1]
            plt.hist(latest['predictions'], bins=50, alpha=0.5, label='Pred')
            plt.hist(latest['targets'], bins=50, alpha=0.5, label='True')
            plt.title('Score Distribution')
            plt.xlabel('Score Value')
            plt.ylabel('Count')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_progress.png')
        plt.close()

def add_grid(ax, size, patch_size, color='white', alpha=0.3):
    """Add grid lines to show patch boundaries"""
    num_patches = size // patch_size
    for i in range(num_patches + 1):
        ax.axhline(y=i * patch_size, color=color, alpha=alpha)
        ax.axvline(x=i * patch_size, color=color, alpha=alpha)

def create_patch_score_map(scores, image_size=224):
    """Create a full-resolution score map from patch scores"""
    num_patches = int(np.sqrt(len(scores)))
    score_grid = scores.reshape(num_patches, num_patches)
    patch_size = image_size // num_patches
    
    score_map = np.zeros((image_size, image_size))
    for i in range(num_patches):
        for j in range(num_patches):
            y_start = i * patch_size
            y_end = (i + 1) * patch_size
            x_start = j * patch_size
            x_end = (j + 1) * patch_size
            score_map[y_start:y_end, x_start:x_end] = score_grid[i, j]
    
    return score_map

def visualize_predictions(model, val_loader, device, epoch, save_dir):
    """Visualize model predictions vs ground truth"""
    model.eval()
    images, targets, _ = next(iter(val_loader))
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Ground truth
    ax1.imshow(targets[0].reshape(14, 14).cpu(), cmap='hot')
    ax1.set_title('Ground Truth')
    add_grid(ax1, 224, 16)
    
    # Prediction
    ax2.imshow(outputs[0].reshape(14, 14).cpu().detach(), cmap='hot')
    ax2.set_title('Prediction')
    add_grid(ax2, 224, 16)
    
    plt.savefig(save_dir / f'pred_vs_true_epoch_{epoch}.png')
    plt.close()
