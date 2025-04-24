#!/usr/bin/env python3
"""
Simple test script to validate rankings from the RankedImageFolder dataset.
This script only works if you add the paths of the batch as output to the dataset.
It means that you will have to modify the dataset class to return the paths of the images in the batch.
"""

import os
import argparse
import numpy as np
import torch
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
import time

from datasets import RankedImageFolder
from timm.data import create_transform


def load_csv_rankings(csv_path):
    """Load rankings from a CSV file into a dictionary mapping filename to ranking array"""
    rankings_dict = {}
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header row
            
            for row in reader:
                if len(row) < 2:
                    continue
                    
                filename = row[0]
                rankings_str = row[1]
                
                # Convert the rankings string to a numpy array
                rankings = np.array(list(map(int, rankings_str.split(','))), dtype=np.int32)
                rankings_dict[filename] = rankings
                
    except Exception as e:
        print(f"Error loading CSV file {csv_path}: {e}")
        
    return rankings_dict


def visualize_ranking_match(image, expected_ranking, actual_ranking, filename, output_dir='.'):
    """Create a visualization comparing expected vs. actual rankings"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate match percentage
    match_percentage = np.mean(expected_ranking == actual_ranking) * 100
    
    # Create heatmaps for both rankings
    height, width = 224, 224  # Assuming 224x224 images
    patch_size = 16
    stride = 16
    num_patches_y = (height - patch_size) // stride + 1
    num_patches_x = (width - patch_size) // stride + 1
    
    # Helper function to create heatmap from ranking
    def create_heatmap(ranking):
        heatmap = np.zeros((height, width))
        for rank_idx, patch_idx in enumerate(ranking):
            importance = 1.0 - (rank_idx / len(ranking))
            y = (patch_idx // num_patches_x) * stride
            x = (patch_idx % num_patches_x) * stride
            if y + patch_size <= height and x + patch_size <= width:
                heatmap[y:y+patch_size, x:x+patch_size] = importance
        return heatmap
    
    # Convert image tensor to numpy array for visualization
    if isinstance(image, torch.Tensor):
        img_np = image.permute(1, 2, 0).numpy()
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
    else:
        img_np = image
    
    # Create heatmaps
    expected_heatmap = create_heatmap(expected_ranking)
    actual_heatmap = create_heatmap(actual_ranking)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title(f"Image: {filename}")
    axes[0].axis('off')
    
    # Expected ranking
    axes[1].imshow(expected_heatmap, cmap='viridis')
    axes[1].set_title("Expected Ranking from CSV")
    axes[1].axis('off')
    
    # Actual ranking
    axes[2].imshow(actual_heatmap, cmap='viridis')
    axes[2].set_title(f"Actual Ranking from DataLoader\nMatch: {match_percentage:.1f}%")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ranking_match_{filename}.png"))
    plt.close()
    
    return match_percentage


def test_rankings(args):
    """Test rankings from the dataloader against CSV files"""
    print(f"Testing rankings from dataloader against CSV files")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create transform
    transform = create_transform(
        input_size=args.input_size,
        is_training=True,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation=args.interpolation,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
    )
    
    # Create dataset
    print(f"Creating dataset with root: {args.data_path}, rankings_dir: {args.rankings_dir}")
    dataset = RankedImageFolder(
        root=args.data_path,
        rankings_dir=args.rankings_dir,
        transform=transform
    )
    
    # Create dataloader with deterministic behavior for reproducibility
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Dataset contains {len(dataset)} images")
    
    # Check ranking shapes
    print("Checking ranking shapes from dataloader...")
    batch = next(iter(dataloader))
    
    images, _, rankings, _ = batch
    batch_size = images.size(0)
    expected_ranking_shape = (batch_size, 196)  # 196 patches for 224x224 image with 16x16 patches
    
    print(f"Batch shape: {rankings.shape}, Expected: {expected_ranking_shape}")
    if rankings.shape != expected_ranking_shape:
        print(f"ERROR: Ranking shape mismatch!")
    else:
        print(f"OK: Ranking shape matches expected shape")
    
    # Load CSV rankings for comparison
    print("Loading CSV rankings for comparison...")
    
    # Load ALL classes instead of just a subset
    class_folders = list(os.scandir(args.data_path))
    print(f"Found {len(class_folders)} class folders in dataset")
    csv_rankings = {}
    
    for class_folder in tqdm(class_folders, desc="Loading CSV files"):
        class_name = class_folder.name
        class_csv = os.path.join(args.rankings_dir, f"{class_name}_rankings.csv")
        
        if not os.path.exists(class_csv):
            print(f"Warning: No CSV file for class {class_name}")
            continue
            
        csv_rankings[class_name] = load_csv_rankings(class_csv)
        print(f"  Loaded rankings for class {class_name}: {len(csv_rankings[class_name])} images")
    
    print(f"Loaded rankings for {len(csv_rankings)} classes out of {len(class_folders)} in dataset")
    
    # Check if the ranking CSV files match the expected format
    print("Checking ranking CSV format...")
    sample_csv_path = next((os.path.join(args.rankings_dir, f"{class_folder.name}_rankings.csv") 
                           for class_folder in class_folders 
                           if os.path.exists(os.path.join(args.rankings_dir, f"{class_folder.name}_rankings.csv"))), None)
    
    if sample_csv_path:
        print(f"Checking sample CSV file: {sample_csv_path}")
        with open(sample_csv_path, 'r') as f:
            sample_lines = [line for idx, line in enumerate(f) if idx < 5]
            print("CSV file format (first few lines):")
            for line in sample_lines:
                print(f"  {line.strip()}")
    
    # Compare rankings from dataloader with CSV rankings
    print("Comparing dataloader rankings with CSV rankings...")
    
    # Create a mapping from image paths to filenames for easier lookup
    path_to_filename = {}
    for path, _ in dataset.samples:
        path_to_filename[path] = os.path.basename(path)
    
    # Keep track of classes and matches
    class_matches = defaultdict(list)
    
    # Test multiple batches - process all available batches if num_batches is 0
    num_batches = len(dataloader) if args.num_batches == 0 else min(args.num_batches, len(dataloader))
    print(f"Testing {num_batches} batches out of {len(dataloader)} total")
    
    dataloader_iter = iter(dataloader)
    match_percentages = []
    
    for batch_idx in tqdm(range(num_batches), desc="Testing batches"):
        try:
            images, targets, rankings, paths = next(dataloader_iter)
        except StopIteration:
            break
            
        # For each image in the batch
        for i in range(images.size(0)):
            # Get the image path from dataset.samples
            path = paths[i]
            filename = path_to_filename[path]
            class_name = os.path.basename(os.path.dirname(path))
                
            # Only process if we have CSV rankings for this class
            if class_name not in csv_rankings:
                print(f"  Skipping {filename} - No rankings for class {class_name}")
                continue
                
            # Get the expected ranking from CSV
            expected_ranking = csv_rankings[class_name].get(filename)
            if expected_ranking is None:
                print(f"  Skipping {filename} - No ranking entry in CSV")
                continue
                
            # Get the actual ranking from dataloader
            actual_ranking = rankings[i].numpy()
            
            # Check if rankings match
            match = np.array_equal(expected_ranking, actual_ranking)
            match_status = "✓ MATCH" if match else "✗ MISMATCH"
            print(f"  Testing image: {filename} - {match_status}")
            if not match:
                print(f"    Expected: {expected_ranking}")
                print(f"    Actual:   {actual_ranking}")
            
            class_matches[class_name].append(match)
            
            # Visualize a subset of examples
            if len(match_percentages) < args.num_visualizations:
                print(f"  Creating visualization for {filename}")
                match_percentage = visualize_ranking_match(
                    images[i], expected_ranking, actual_ranking, 
                    filename, args.output_dir
                )
                match_percentages.append(match_percentage)
    
    # Print results
    print("\n===== Results =====")
    all_matches = []
    
    for class_name, matches in class_matches.items():
        match_rate = sum(matches) / len(matches) if matches else 0
        all_matches.extend(matches)
        print(f"Class {class_name}: {match_rate*100:.1f}% match ({sum(matches)}/{len(matches)})")
    
    overall_match_rate = sum(all_matches) / len(all_matches) if all_matches else 0
    print(f"\nOverall: {overall_match_rate*100:.1f}% match ({sum(all_matches)}/{len(all_matches)})")
    
    if match_percentages:
        print(f"Average visualization match percentage: {np.mean(match_percentages):.1f}%")
    
    return overall_match_rate == 1.0  # Return True if all rankings match perfectly


def parse_args():
    parser = argparse.ArgumentParser(description='Test rankings dataloader')
    
    parser.add_argument('--data-path', type=str, default=r"C:\Users\joren\Documents\_Uni\Master\Thesis\imagenet_subset\train",
                        help='Path to the dataset root directory')
    parser.add_argument('--rankings-dir', type=str, default=r"C:\Users\joren\Documents\_Uni\Master\Thesis\imagenet_subset\train_rankings",
                        help='Path to directory containing ranking CSV files')
    parser.add_argument('--output-dir', type=str, default='./ranking_tests',
                        help='Output directory for visualizations and results')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for dataloader')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for dataloader')
    parser.add_argument('--num-classes', type=int, default=0,
                        help='Number of classes to test (0 for all classes)')
    parser.add_argument('--num-batches', type=int, default=0,
                        help='Number of batches to test (0 for all batches)')
    parser.add_argument('--num-visualizations', type=int, default=5,
                        help='Number of ranking comparisons to visualize')
    
    # Transform parameters (to match what's used in training)
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--color-jitter', type=float, default=0.3,
                        help='Color jitter factor')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1',
                        help='Auto-augment policy')
    parser.add_argument('--interpolation', type=str, default='bicubic',
                        help='Interpolation method')
    parser.add_argument('--reprob', type=float, default=0.25,
                        help='Random erase probability')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set global random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Calculate and print time estimates
    start_time = time.time()
    test_result = test_rankings(args)
    total_time = time.time() - start_time
    
    print(f"\nFinal result: {'PASS' if test_result else 'FAIL'}")
    print(f"Total runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
