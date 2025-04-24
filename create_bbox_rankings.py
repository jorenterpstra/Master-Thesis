#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

def parse_bbox_xml(xml_path):
    """Parse a PASCAL VOC format XML file to extract bounding box coordinates."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extract image size
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        # Extract all bounding boxes
        bboxes = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
            
            # Ensure coordinates are within image bounds
            xmin = max(0, min(xmin, width-1))
            ymin = max(0, min(ymin, height-1))
            xmax = max(0, min(xmax, width-1))
            ymax = max(0, min(ymax, height-1))
            
            bboxes.append((xmin, ymin, xmax, ymax))
        
        return bboxes, width, height
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return [], 0, 0

def find_all_bbox_files(dataset_path):
    """Find all bounding box XML files in the dataset."""
    bbox_files = []
    
    # Check for train_bbox directory
    train_bbox_dir = os.path.join(dataset_path, 'train_bbox')
    if os.path.exists(train_bbox_dir):
        for root, _, files in os.walk(train_bbox_dir):
            for file in files:
                if file.lower().endswith('.xml'):
                    bbox_files.append(os.path.join(root, file))
    
    # Check for val_bbox directory
    val_bbox_dir = os.path.join(dataset_path, 'val_bbox')
    if os.path.exists(val_bbox_dir):
        for root, _, files in os.walk(val_bbox_dir):
            for file in files:
                if file.lower().endswith('.xml'):
                    bbox_files.append(os.path.join(root, file))
    
    return bbox_files

def generate_bbox_heatmap(dataset_path, resolution=(224, 224)):
    """Generate a heatmap of bounding box locations."""
    bbox_files = find_all_bbox_files(dataset_path)
    
    if not bbox_files:
        print(f"No bounding box files found in {dataset_path}")
        return None, []
    
    print(f"Found {len(bbox_files)} bounding box files")
    
    # Initialize heatmap
    heatmap = np.zeros(resolution, dtype=np.float32)
    
    # Store all normalized bounding boxes for visualization
    all_bboxes = []
    image_sizes = []
    
    # Process each bbox file
    for xml_file in tqdm(bbox_files[:10000], desc="Processing bounding boxes"):
        bboxes, width, height = parse_bbox_xml(xml_file)
        
        if width == 0 or height == 0:
            continue
            
        image_sizes.append((width, height))
        
        # Normalize bboxes to the heatmap resolution
        for xmin, ymin, xmax, ymax in bboxes:
            # Store normalized bbox for visualization
            norm_bbox = (
                xmin / width,
                ymin / height,
                xmax / width,
                ymax / height
            )
            all_bboxes.append(norm_bbox)
            
            # Map to heatmap resolution
            h_xmin = int(xmin * resolution[1] / width)
            h_ymin = int(ymin * resolution[0] / height)
            h_xmax = int(xmax * resolution[1] / width)
            h_ymax = int(ymax * resolution[0] / height)
            
            # Ensure valid coordinates
            h_xmin = max(0, min(h_xmin, resolution[1]-1))
            h_ymin = max(0, min(h_ymin, resolution[0]-1))
            h_xmax = max(0, min(h_xmax, resolution[1]-1))
            h_ymax = max(0, min(h_ymax, resolution[0]-1))
            
            # Update heatmap - increment the area covered by this bbox
            heatmap[h_ymin:h_ymax+1, h_xmin:h_xmax+1] += 1
    
    # Normalize heatmap for visualization
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    return heatmap, all_bboxes, image_sizes

def compute_patch_scores(heatmap, patch_size=16):
    """Compute scores for each patch in the grid based on heatmap values."""
    height, width = heatmap.shape
    grid_size_h = height // patch_size
    grid_size_w = width // patch_size
    
    patch_scores = []
    
    for i in range(grid_size_h):
        for j in range(grid_size_w):
            # Calculate patch boundaries
            y_start = i * patch_size
            y_end = (i + 1) * patch_size if i < grid_size_h - 1 else height
            x_start = j * patch_size
            x_end = (j + 1) * patch_size if j < grid_size_w - 1 else width
            
            # Extract patch region
            patch = heatmap[y_start:y_end, x_start:x_end]
            
            # Compute average score for this patch
            avg_score = np.mean(patch)
            
            # Store patch info: (row, col, score)
            patch_scores.append((i, j, avg_score))
    
    # Sort patches by score in descending order
    patch_scores.sort(key=lambda x: x[2], reverse=True)
    
    return patch_scores, grid_size_h, grid_size_w

def create_ranking_tensor(patch_scores, grid_size_h, grid_size_w, batch_size=1, strategy="highest_first"):
    """
    Create a ranking tensor based on patch scores that can be used with the custom_rank parameter.
    
    Args:
        patch_scores: List of (row, col, score) tuples sorted by score
        grid_size_h: Number of patches in height dimension
        grid_size_w: Number of patches in width dimension
        batch_size: Batch size for the tensor
        strategy: Ranking strategy to use (highest_first, lowest_first, center_outward, periphery_inward)
        
    Returns:
        torch.Tensor of shape [batch_size, num_patches] with indices for reordering patches
    """
    num_patches = grid_size_h * grid_size_w
    
    # Create a 2D grid of indices (i, j)
    indices_grid = np.zeros((grid_size_h, grid_size_w), dtype=np.int64)
    
    if strategy == "highest_first":
        # Order by patch scores (highest first)
        for rank, (i, j, _) in enumerate(patch_scores):
            # Convert 2D position to 1D index
            flat_idx = i * grid_size_w + j
            indices_grid[i, j] = rank
            
    elif strategy == "lowest_first":
        # Order by patch scores (lowest first)
        for rank, (i, j, _) in enumerate(reversed(patch_scores)):
            flat_idx = i * grid_size_w + j
            indices_grid[i, j] = rank
            
    elif strategy == "center_outward":
        # Calculate distance from center for each patch
        center_h, center_w = grid_size_h / 2, grid_size_w / 2
        distance_scores = []
        
        for i in range(grid_size_h):
            for j in range(grid_size_w):
                distance = np.sqrt((i + 0.5 - center_h)**2 + (j + 0.5 - center_w)**2)
                distance_scores.append((i, j, distance))
        
        # Sort by distance (center first)
        distance_scores.sort(key=lambda x: x[2])
        
        for rank, (i, j, _) in enumerate(distance_scores):
            indices_grid[i, j] = rank
            
    elif strategy == "periphery_inward":
        # Calculate distance from center for each patch
        center_h, center_w = grid_size_h / 2, grid_size_w / 2
        distance_scores = []
        
        for i in range(grid_size_h):
            for j in range(grid_size_w):
                distance = np.sqrt((i + 0.5 - center_h)**2 + (j + 0.5 - center_w)**2)
                distance_scores.append((i, j, distance))
        
        # Sort by distance (periphery first)
        distance_scores.sort(key=lambda x: x[2], reverse=True)
        
        for rank, (i, j, _) in enumerate(distance_scores):
            indices_grid[i, j] = rank
            
    elif strategy == "horizontal_scan":
        # Simple left-to-right, top-to-bottom scan
        rank = 0
        for i in range(grid_size_h):
            for j in range(grid_size_w):
                indices_grid[i, j] = rank
                rank += 1
                
    elif strategy == "vertical_scan":
        # Simple top-to-bottom, left-to-right scan
        rank = 0
        for j in range(grid_size_w):
            for i in range(grid_size_h):
                indices_grid[i, j] = rank
                rank += 1
    
    # Reshape to match expected format for custom_rank
    # First flatten the grid to get a list of positions in order
    flat_indices = indices_grid.flatten()
    
    # Create a tensor of shape [batch_size, num_patches]
    ranking_tensor = torch.tensor(flat_indices, dtype=torch.int64)
    ranking_tensor = ranking_tensor.unsqueeze(0).repeat(batch_size, 1)
    
    return ranking_tensor

def visualize_ranking(indices_grid, output_path, strategy_name):
    """Visualize the ranking strategy"""
    plt.figure(figsize=(10, 10))
    plt.imshow(indices_grid, cmap='viridis')
    plt.colorbar(label='Processing Order')
    plt.title(f'Patch Processing Order: {strategy_name}')
    
    # Add grid lines
    grid_h, grid_w = indices_grid.shape
    for i in range(grid_h):
        plt.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.5)
    for j in range(grid_w):
        plt.axvline(j - 0.5, color='white', linewidth=0.5, alpha=0.5)
    
    # Add rank numbers
    for i in range(grid_h):
        for j in range(grid_w):
            plt.text(j, i, f"{indices_grid[i, j]}", ha='center', va='center', 
                    color='white', fontsize=6, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def create_all_ranking_strategies(patch_scores, grid_size_h, grid_size_w, batch_size=1, output_dir="rankings"):
    """Create and save tensors for all ranking strategies"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Define all available strategies
    strategies = [
        "highest_first", 
        "lowest_first", 
        "center_outward", 
        "periphery_inward",
        "horizontal_scan",
        "vertical_scan"
    ]
    
    ranking_tensors = {}
    
    for strategy in strategies:
        # Create the ranking tensor
        ranking_tensor = create_ranking_tensor(
            patch_scores, grid_size_h, grid_size_w, batch_size, strategy
        )
        
        # Reshape for visualization
        indices_grid = ranking_tensor[0].reshape(grid_size_h, grid_size_w).numpy()
        
        # Visualize the ranking
        vis_path = os.path.join(output_dir, f"ranking_{strategy}.png")
        visualize_ranking(indices_grid, vis_path, strategy)
        
        # Save the tensor
        tensor_path = os.path.join(output_dir, f"ranking_{strategy}.pt")
        torch.save(ranking_tensor, tensor_path)
        
        ranking_tensors[strategy] = ranking_tensor
        
        print(f"Created and saved {strategy} ranking tensor to {tensor_path}")
    
    # Save all tensors in one file
    all_tensors_path = os.path.join(output_dir, "all_rankings.pkl")
    with open(all_tensors_path, 'wb') as f:
        pickle.dump(ranking_tensors, f)
    
    return ranking_tensors

def main():
    parser = argparse.ArgumentParser(description='Create custom ranking tensors based on bounding boxes')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset with bounding boxes')
    parser.add_argument('--output', type=str, default='rankings', help='Output directory for ranking tensors')
    parser.add_argument('--resolution', type=int, default=224, help='Image resolution')
    parser.add_argument('--patch-size', type=int, default=16, help='Patch size')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for tensors')
    
    args = parser.parse_args()
    
    resolution = (args.resolution, args.resolution)
    
    print(f"Generating bounding box heatmap for {args.dataset} at {resolution} resolution")
    heatmap, all_bboxes, image_sizes = generate_bbox_heatmap(args.dataset, resolution)
    
    if heatmap is None:
        print("Failed to generate heatmap. Exiting.")
        return
    
    # Compute patch scores
    patch_scores, grid_size_h, grid_size_w = compute_patch_scores(heatmap, args.patch_size)
    
    print(f"Grid size: {grid_size_h}×{grid_size_w} patches")
    print(f"Total patches: {grid_size_h * grid_size_w}")
    
    # Create and save tensors for all ranking strategies
    create_all_ranking_strategies(
        patch_scores, 
        grid_size_h, 
        grid_size_w, 
        args.batch_size, 
        args.output
    )
    
    # Save the heatmap for reference
    os.makedirs(args.output, exist_ok=True)
    np.save(os.path.join(args.output, 'bbox_heatmap.npy'), heatmap)
    
    print(f"All ranking tensors saved to {args.output}")
    print("To use them in your model:")
    print("  1. Load the tensor with: ranking = torch.load('rankings/ranking_highest_first.pt')")
    print("  2. Pass it to your model with: model(images, custom_rank=ranking)")

if __name__ == "__main__":
    main()