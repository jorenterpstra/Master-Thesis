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
    """Enhanced training progress tracker with flexible metrics"""
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.history = {
            'epochs': [],
            'train': {},  # Flexible metric dictionary for training
            'val': {}     # Flexible metric dictionary for validation
        }
        
    def update(self, epoch, train_metrics, val_metrics):
        """Update history with new metrics"""
        self.history['epochs'].append(epoch)
        
        # Update all metrics
        for phase in ['train', 'val']:
            metrics = train_metrics if phase == 'train' else val_metrics
            
            # Initialize metric trackers if needed
            for metric_name, value in metrics.items():
                if metric_name not in self.history[phase]:
                    self.history[phase][metric_name] = []
                self.history[phase][metric_name].append(value)
        
        # Save and plot after each update
        self.save()
        self.plot_progress()
    
    def save(self):
        """Save training history to JSON"""
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_progress(self):
        """Plot training progress for all metrics"""
        epochs = self.history['epochs']
        metrics = set()
        for phase in ['train', 'val']:
            metrics.update(self.history[phase].keys())
        
        num_metrics = len(metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=(6*num_metrics, 4))
        if num_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, sorted(metrics)):
            ax.set_title(f'{metric.capitalize()} Progress')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            
            for phase in ['train', 'val']:
                if metric in self.history[phase]:
                    values = self.history[phase][metric]
                    ax.plot(epochs, values, label=phase.capitalize())
            
            ax.legend()
            ax.grid(True)
        
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
    images, targets = next(iter(val_loader))
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
    
    # Get value range from ground truth
    vmin = targets[0].min().item()
    vmax = targets[0].max().item()
    
    # Create figure with proper DPI and size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
    
    # Ground truth plot with proper extent and value range
    im1 = ax1.imshow(targets[0].reshape(14, 14).cpu(), 
                     cmap='hot',
                     interpolation='nearest',
                     extent=[0, 224, 224, 0],
                     vmin=vmin,
                     vmax=vmax)
    ax1.set_title('Ground Truth')
    add_grid(ax1, 224, 16)
    plt.colorbar(im1, ax=ax1)
    
    # Prediction plot with same value range as ground truth
    im2 = ax2.imshow(outputs[0].reshape(14, 14).cpu().detach(),
                     cmap='hot',
                     interpolation='nearest',
                     extent=[0, 224, 224, 0],
                     vmin=vmin,
                     vmax=vmax)
    ax2.set_title('Prediction')
    add_grid(ax2, 224, 16)
    plt.colorbar(im2, ax=ax2)
    
    # Set proper dimensions and layout
    ax1.set_xlim(0, 224)
    ax1.set_ylim(224, 0)
    ax2.set_xlim(0, 224)
    ax2.set_ylim(224, 0)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'pred_vs_true_epoch_{epoch}.png')
    plt.close()

def compare_model_states(state_dict1, state_dict2, name1="Model1", name2="Model2"):
    """Compare two model state dictionaries and show their differences."""
    # Get all unique keys
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    all_keys = keys1.union(keys2)
    
    # Compare shapes and parameters
    print(f"\n{'='*20} Model State Comparison {'='*20}")
    print(f"Comparing {name1} vs {name2}")
    print(f"Total parameters: {len(keys1)} vs {len(keys2)}")
    
    # Show unique keys
    if keys1 - keys2:
        print(f"\nKeys unique to {name1}:")
        for k in sorted(keys1 - keys2):
            print(f"  {k}: {state_dict1[k].shape}")
    
    if keys2 - keys1:
        print(f"\nKeys unique to {name2}:")
        for k in sorted(keys2 - keys1):
            print(f"  {k}: {state_dict2[k].shape}")
    
    # Compare common keys
    common_keys = keys1.intersection(keys2)
    shape_mismatches = []
    value_mismatches = []
    
    print(f"\nCommon keys with differences:")
    for k in sorted(common_keys):
        tensor1, tensor2 = state_dict1[k], state_dict2[k]
        
        # Check shapes
        if tensor1.shape != tensor2.shape:
            shape_mismatches.append((k, tensor1.shape, tensor2.shape))
            continue
            
        # Check values (if shapes match)
        if not torch.allclose(tensor1, tensor2, rtol=1e-4, atol=1e-4):
            max_diff = (tensor1 - tensor2).abs().max().item()
            value_mismatches.append((k, max_diff))
    
    if shape_mismatches:
        print("\nShape mismatches:")
        for k, s1, s2 in shape_mismatches:
            print(f"  {k}: {s1} vs {s2}")
    
    if value_mismatches:
        print("\nValue differences (max absolute difference):")
        for k, diff in value_mismatches:
            print(f"  {k}: {diff:.6f}")
            
    print("\nSummary:")
    print(f"Total keys: {len(all_keys)}")
    print(f"Common keys: {len(common_keys)}")
    print(f"Shape mismatches: {len(shape_mismatches)}")
    print(f"Value differences: {len(value_mismatches)}")
    print('='*60)
    
    return {
        'shape_mismatches': shape_mismatches,
        'value_mismatches': value_mismatches,
        'unique_keys1': keys1 - keys2,
        'unique_keys2': keys2 - keys1
    }
