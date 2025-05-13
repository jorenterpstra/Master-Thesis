#!/usr/bin/env python
# Test file for HeatmapImageFolder dataset functionality

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import albumentations as A
import cv2
import random

from datasets import build_dataset, HeatmapImageFolder
from transforms import build_transform, AlbumentationsRandAugment


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('Test HeatmapImageFolder dataset', add_help=True)
    # Required parameters
    parser.add_argument('--data-path', required=True,
                        default=r"C:\Users\joren\Documents\_Uni\Master\Thesis\imagenet_subset\train",
                        help='Path to ImageNet or subset')
    parser.add_argument('--heatmap-path', default=None, help='Path to heatmap folder')
    
    # Dataset parameters
    parser.add_argument('--data-set', default='IMNET_HEAT', 
                        choices=['CIFAR', 'IMNET', 'IMNET_HEAT'],
                        help='Dataset type')
    parser.add_argument('--input-size', default=224, type=int, help='Image input size')
    parser.add_argument('--global-heatmap-path', default=None, type=str, 
                        help='Path to global heatmap for all images')
    
    # Transform parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, 
                        help='Color jitter factor')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1',
                        help='Auto augment policy')
    parser.add_argument('--train-interpolation', default='bicubic',
                        help='Training interpolation')
    parser.add_argument('--reprob', type=float, default=0.25,
                        help='Random erase probability')
    parser.add_argument('--remode', default='pixel',
                        help='Random erase mode')
    parser.add_argument('--recount', default=1, type=int,
                        help='Random erase count')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float,
                        help='Evaluation crop ratio')
    
    # Test specific parameters
    parser.add_argument('--num-samples', default=5, type=int,
                        help='Number of samples to test')
    parser.add_argument('--output-dir', default='dataset_test_results',
                        help='Directory to save test results')
    parser.add_argument('--return-rankings', action='store_true',
                        help='Return rankings from dataset')
    parser.add_argument('--return-heatmap', action='store_true',
                        help='Return heatmap from dataset')
    parser.add_argument('--return-path', action='store_true',
                        help='Return image paths')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed')
    
    return parser.parse_args()


def test_different_configurations(args):
    """Test all possible dataset configurations."""
    print("\n==== Testing Different Dataset Configurations ====")
    
    # List of configuration options
    config_options = [
        {"return_rankings": True, "return_heatmap": False, "return_path": False},
        {"return_rankings": False, "return_heatmap": True, "return_path": False},
        {"return_rankings": True, "return_heatmap": True, "return_path": False},
        {"return_rankings": False, "return_heatmap": False, "return_path": True},
        {"return_rankings": False, "return_heatmap": False, "return_path": False},
    ]
    
    for i, config in enumerate(config_options):
        print(f"\nConfiguration {i+1}:")
        print(f"  - return_rankings: {config['return_rankings']}")
        print(f"  - return_heatmap: {config['return_heatmap']}")
        print(f"  - return_path: {config['return_path']}")
        
        # Set args for this configuration
        args.return_rankings = config['return_rankings']
        args.return_heatmap = config['return_heatmap']
        args.return_path = config['return_path']
        
        try:
            # Build dataset
            dataset, nb_classes = build_dataset(is_train=True, args=args)
            
            # Get one sample and check the return value
            sample = dataset[0]
            
            print(f"  - Return format: {type(sample)} with {len(sample)} elements")
            
            if isinstance(sample, tuple):
                for j, item in enumerate(sample):
                    if isinstance(item, torch.Tensor):
                        print(f"    - Item {j}: Tensor of shape {item.shape}")
                    elif isinstance(item, str):
                        print(f"    - Item {j}: Path string")
                    else:
                        print(f"    - Item {j}: {type(item)}")
            
            print("  ✓ Configuration works!")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")


def test_transform_consistency(args):
    """Test that transforms are applied consistently to images and heatmaps."""
    print("\n==== Testing Transform Consistency ====")
    
    # Set args for this test
    args.return_rankings = False
    args.return_heatmap = True
    
    # Create train transforms
    transform = build_transform(is_train=True, args=args)
    
    # Build dataset
    dataset, _ = build_dataset(is_train=True, args=args)
    
    # Directory for saving output
    os.makedirs(os.path.join(args.output_dir, "transform_consistency"), exist_ok=True)
    
    for i in range(min(args.num_samples, len(dataset))):
        try:
            # Get a random sample
            idx = random.randint(0, len(dataset)-1)
            image, _, heatmap = dataset[idx]
            
            # Convert tensors to numpy for visualization
            if isinstance(image, torch.Tensor):
                image_np = image.permute(1, 2, 0).numpy()
                # Denormalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image_np = image_np * std + mean
                image_np = np.clip(image_np, 0, 1)
            else:
                image_np = np.array(image)
            
            if isinstance(heatmap, torch.Tensor):
                # Convert tensor to numpy - heatmaps are already in correct spatial orientation
                if heatmap.dim() == 3 and heatmap.shape[0] == 1:
                    # If single-channel tensor in CHW format, just take the first channel
                    heatmap_np = heatmap[0].numpy()
                else:
                    # Otherwise just convert as-is - no permutation needed
                    heatmap_np = heatmap.numpy()
            else:
                # Already numpy array
                heatmap_np = np.array(heatmap)
                
            # Plot the image and heatmap side by side
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(image_np)
            axes[0].set_title("Transformed Image")
            axes[0].axis('off')
            
            axes[1].imshow(heatmap_np, cmap='viridis')
            axes[1].set_title("Transformed Heatmap")
            axes[1].axis('off')
            
            plt.suptitle(f"Sample {i+1} - Consistent Transform Application")
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "transform_consistency", f"sample_{i+1}.png"))
            plt.close()
            
            print(f"  ✓ Sample {i+1} transforms checked.")
            
        except Exception as e:
            print(f"  ✗ Error on sample {i+1}: {str(e)}")


def test_ranking_generation(args):
    """Test ranking generation from heatmaps."""
    print("\n==== Testing Ranking Generation ====")
    
    # Set args for this test
    args.return_rankings = True
    args.return_heatmap = True
    
    # Build dataset
    dataset, _ = build_dataset(is_train=True, args=args)
    
    # Directory for saving output
    os.makedirs(os.path.join(args.output_dir, "ranking_generation"), exist_ok=True)
    
    for i in range(min(args.num_samples, len(dataset))):
        try:
            # Get a random sample
            idx = random.randint(0, len(dataset)-1)
            image, _, ranking, heatmap = dataset[idx]
            
            # Convert tensors to numpy for visualization
            if isinstance(image, torch.Tensor):
                image_np = image.permute(1, 2, 0).numpy()
                # Denormalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image_np = image_np * std + mean
                image_np = np.clip(image_np, 0, 1)
            else:
                image_np = np.array(image)
            
            if isinstance(heatmap, torch.Tensor):
                # Convert tensor to numpy - heatmaps are already in correct spatial orientation
                if heatmap.dim() == 3 and heatmap.shape[0] == 1:
                    # If single-channel tensor in CHW format, just take the first channel
                    heatmap_np = heatmap[0].numpy()
                else:
                    # Otherwise just convert as-is - no permutation needed
                    heatmap_np = heatmap.numpy()
            else:
                # Already numpy array
                heatmap_np = np.array(heatmap)
            
            # Visualize the ranking as a 2D grid
            num_patches_per_dim = int(np.sqrt(ranking.shape[0]))
            rank_grid = np.zeros((num_patches_per_dim, num_patches_per_dim), dtype=int)
            
            # Fill the grid with ranks
            for rank, idx in enumerate(ranking.numpy()):
                y, x = divmod(idx, num_patches_per_dim)
                rank_grid[y, x] = rank
            
            # Plot the image, heatmap, and ranking side by side
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(image_np)
            axes[0].set_title("Image")
            axes[0].axis('off')
            
            axes[1].imshow(heatmap_np, cmap='viridis')
            axes[1].set_title("Heatmap")
            axes[1].axis('off')
            
            im = axes[2].imshow(rank_grid, cmap='hot')
            axes[2].set_title("Patch Rankings")
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            
            plt.suptitle(f"Sample {i+1} - Ranking Generation")
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "ranking_generation", f"sample_{i+1}.png"))
            plt.close()
            
            print(f"  ✓ Sample {i+1} ranking checked.")
            
        except Exception as e:
            print(f"  ✗ Error on sample {i+1}: {str(e)}")


def test_data_loading_speed(args):
    """Test the speed of data loading."""
    print("\n==== Testing Data Loading Speed ====")
    
    # Set args for data loading
    args.return_rankings = True
    args.return_heatmap = False
    
    # Create dataset and dataloader
    dataset, _ = build_dataset(is_train=True, args=args)
    
    num_workers = 4
    batch_size = 32
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)} images")
    print(f"Batch size: {batch_size}, Num workers: {num_workers}")
    
    # Test data loading speed
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    total_batches = min(10, len(dataloader))
    
    start_time.record()
    
    for i, batch in enumerate(dataloader):
        if i >= total_batches:
            break
        
        # Simulate device transfer
        if torch.cuda.is_available():
            if args.return_rankings:
                images, targets, rankings = batch
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
                rankings = rankings.cuda(non_blocking=True)
            else:
                images, targets = batch
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
        
        # Print batch shapes for the first batch
        if i == 0:
            print(f"\nFirst batch shapes:")
            print(f"  - Images: {images.shape}")
            print(f"  - Targets: {targets.shape}")
            if args.return_rankings:
                print(f"  - Rankings: {rankings.shape}")
    
    end_time.record()
    torch.cuda.synchronize()
    
    elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
    images_per_second = (total_batches * batch_size) / elapsed_time
    
    print(f"\nLoaded {total_batches * batch_size} images in {elapsed_time:.2f} seconds")
    print(f"Loading speed: {images_per_second:.2f} images/second")


def test_global_heatmap(args):
    """Test using a global heatmap for all images."""
    print("\n==== Testing Global Heatmap ====")
    
    if args.global_heatmap_path is None:
        print("No global heatmap provided, creating a synthetic one")
        # Create a synthetic heatmap (centered gaussian)
        size = args.input_size
        x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
        d = np.sqrt(x*x+y*y)
        sigma = 0.5
        heatmap = np.exp(-( d**2 / (2.0 * sigma**2) ))
        
        # Save the heatmap
        os.makedirs(args.output_dir, exist_ok=True)
        global_heatmap_path = os.path.join(args.output_dir, "global_heatmap.png")
        plt.imsave(global_heatmap_path, heatmap, cmap='viridis')
        args.global_heatmap_path = global_heatmap_path
        print(f"Created and saved global heatmap to {global_heatmap_path}")
    
    # Set args for this test
    args.return_rankings = True
    args.return_heatmap = True
    
    # Build dataset
    dataset, _ = build_dataset(is_train=True, args=args)
    
    # Directory for saving output
    os.makedirs(os.path.join(args.output_dir, "global_heatmap"), exist_ok=True)
    
    # Check that all samples return the same ranking
    all_rankings = []
    num_samples = min(args.num_samples, len(dataset))
    
    # Create a figure with subplots for all samples
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    # If only one sample, make sure axes is still indexed as a 2D array
    if num_samples == 1:
        axes = np.array([axes])
    
    for i in range(num_samples):
        try:
            # Get a random sample
            idx = random.randint(0, len(dataset)-1)
            image, _, ranking, heatmap = dataset[idx]
            
            all_rankings.append(ranking.clone())
            
            # Convert tensors to numpy for visualization
            if isinstance(image, torch.Tensor):
                image_np = image.permute(1, 2, 0).numpy()
                # Denormalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image_np = image_np * std + mean
                image_np = np.clip(image_np, 0, 1)
            else:
                image_np = np.array(image)
            
            if isinstance(heatmap, torch.Tensor):
                # Convert tensor to numpy - heatmaps are already in correct spatial orientation
                if heatmap.dim() == 3 and heatmap.shape[0] == 1:
                    # If single-channel tensor in CHW format, just take the first channel
                    heatmap_np = heatmap[0].numpy()
                else:
                    # Otherwise just convert as-is - no permutation needed
                    heatmap_np = heatmap.numpy()
            else:
                # Already numpy array
                heatmap_np = np.array(heatmap)
                
            # Visualize the ranking as a 2D grid
            num_patches_per_dim = int(np.sqrt(ranking.shape[0]))
            rank_grid = np.zeros((num_patches_per_dim, num_patches_per_dim), dtype=int)
            
            # Fill the grid with ranks
            for rank, idx in enumerate(ranking.numpy()):
                y, x = divmod(idx, num_patches_per_dim)
                rank_grid[y, x] = rank
            
            # Plot the results for this sample
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title(f"Image {i+1}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(heatmap_np, cmap='viridis')
            axes[i, 1].set_title(f"Global Heatmap {i+1}")
            axes[i, 1].axis('off')
            
            im = axes[i, 2].imshow(rank_grid, cmap='hot')
            axes[i, 2].set_title(f"Patch Rankings {i+1}")
            axes[i, 2].axis('off')
            plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
            
            print(f"  ✓ Sample {i+1} checked.")
            
        except Exception as e:
            print(f"  ✗ Error on sample {i+1}: {str(e)}")
    
    plt.suptitle("Global Heatmap Test - Multiple Samples")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "global_heatmap", "visualization_multi.png"))
    plt.close()
    
    # Check if all rankings are equal
    if len(all_rankings) > 1:
        all_equal = all(torch.all(all_rankings[0] == r) for r in all_rankings[1:])
        print(f"\nAll rankings equal: {all_equal}")
        if not all_equal:
            print("WARNING: Global heatmap should produce identical rankings for all images!")


def test_auto_augment(args):
    """Test the RandAugment implementation."""
    print("\n==== Testing Auto Augment ====")
    
    # Set args for this test
    args.return_rankings = False
    args.return_heatmap = True
    args.aa = 'rand-m9-mstd0.5-inc1'  # Set auto augment policy
    
    # Build dataset
    dataset, _ = build_dataset(is_train=True, args=args)
    
    # Directory for saving output
    os.makedirs(os.path.join(args.output_dir, "auto_augment"), exist_ok=True)
    
    # Get a fixed sample
    idx = 153
    image_before_aug, _, heatmap_before_aug = dataset[idx]
    
    # Convert to numpy for visualization
    if isinstance(image_before_aug, torch.Tensor):
        image_before_np = image_before_aug.permute(1, 2, 0).numpy()
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_before_np = image_before_np * std + mean
        image_before_np = np.clip(image_before_np, 0, 1)
    else:
        image_before_np = np.array(image_before_aug)
    
    if isinstance(heatmap_before_aug, torch.Tensor):
        if heatmap_before_aug.dim() == 3 and heatmap_before_aug.shape[0] == 1:
            heatmap_before_np = heatmap_before_aug[0].numpy()
        else:
            heatmap_before_np = heatmap_before_aug.numpy()
            if heatmap_before_np.shape[2] > 1:
                heatmap_before_np = np.mean(heatmap_before_np, axis=2)
    else:
        heatmap_before_np = np.array(heatmap_before_aug)
    
    # Apply RandAugment directly multiple times to see the effects
    num_samples = 5
    augmenter = AlbumentationsRandAugment(num_ops=2, magnitude=9, std=0.5)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    
    # Original image in the first row
    axes[0, 0].imshow(image_before_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(heatmap_before_np, cmap='viridis')
    axes[0, 1].set_title("Original Heatmap")
    axes[0, 1].axis('off')
    
    axes[0, 2].set_visible(False)
    
    # Apply RandAugment multiple times to demonstrate variety
    for i in range(1, num_samples):
        try:
            # Apply RandAugment
            augmented = augmenter(image=image_before_np, heatmap=heatmap_before_np)
            aug_image = augmented['image']
            aug_heatmap = augmented['heatmap']
            
            # Display
            axes[i, 0].imshow(aug_image)
            axes[i, 0].set_title(f"Augmented Image {i}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(aug_heatmap, cmap='viridis')
            axes[i, 1].set_title(f"Augmented Heatmap {i}")
            axes[i, 1].axis('off')
            
            # Optional: Show operations applied in the third column
            ops_text = "\n".join([op for op in augmenter._selected_ops])
            axes[i, 2].text(0.5, 0.5, f"Applied operations:\n{ops_text}", 
                          ha='center', va='center', fontsize=10)
            axes[i, 2].axis('off')
            
            print(f"  ✓ Generated augmentation sample {i}")
            
        except Exception as e:
            print(f"  ✗ Error on augmentation {i}: {str(e)}")
            axes[i, 0].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            axes[i, 0].axis('off')
            axes[i, 1].axis('off')
            axes[i, 2].axis('off')
    
    plt.suptitle("RandAugment Samples")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "auto_augment", "samples.png"))
    plt.close()


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Testing HeatmapImageFolder dataset")
    print(f"Data path: {args.data_path}")
    print(f"Heatmap path: {args.heatmap_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Run tests
    test_different_configurations(args)
    test_transform_consistency(args)
    test_ranking_generation(args)
    if torch.cuda.is_available():
        test_data_loading_speed(args)
    else:
        print("Skipping data loading speed test (CUDA not available)")
    test_auto_augment(args)
    test_global_heatmap(args)
    
    print("\nAll tests completed. Results saved to:", args.output_dir)


if __name__ == "__main__":
    main()