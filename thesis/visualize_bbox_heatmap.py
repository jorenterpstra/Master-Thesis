import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tqdm import tqdm
from pathlib import Path
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

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
    for xml_file in tqdm(bbox_files, desc="Processing bounding boxes"):
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
    
    # Create a ranking map: (row, col) -> (rank, score)
    patch_ranking = {}
    for rank, (i, j, score) in enumerate(patch_scores):
        patch_ranking[(i, j)] = (rank + 1, score)  # +1 for 1-based ranking
    
    return patch_ranking, patch_scores, grid_size_h, grid_size_w

def visualize_bbox_heatmap(heatmap, all_bboxes, output_dir, resolution=(224, 224), num_bboxes_to_show=100, patch_size=16):
    """Visualize the heatmap with bounding box overlays, patch grid lines, and patch rankings."""
    if heatmap is None:
        print("No heatmap data to visualize")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute patch scores and ranking
    patch_ranking, sorted_patches, grid_size_h, grid_size_w = compute_patch_scores(heatmap, patch_size)
    total_patches = grid_size_h * grid_size_w
    
    print(f"Image resolution: {resolution[0]}×{resolution[1]} pixels")
    print(f"Patch size: {patch_size}×{patch_size} pixels")
    print(f"Grid size: {grid_size_h}×{grid_size_w} patches ({total_patches} total patches)")
    
    # Create a directory for patch visualizations
    patches_dir = os.path.join(output_dir, 'patches')
    os.makedirs(patches_dir, exist_ok=True)
    
    # Save patch scores to a text file
    with open(os.path.join(output_dir, 'patch_ranking.txt'), 'w') as f:
        f.write("Rank,Row,Column,Score\n")
        for rank, (i, j, score) in enumerate(sorted_patches):
            f.write(f"{rank+1},{i},{j},{score:.6f}\n")
    
    # Plot the basic heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Normalized Frequency')
    plt.title(f'Bounding Box Heatmap ({grid_size_h}×{grid_size_w} grid of {patch_size}×{patch_size} patches)')
    
    # Add grid lines
    for i in range(1, grid_size_h):
        plt.axhline(y=i*patch_size, color='white', linewidth=0.5, alpha=0.7)
    for j in range(1, grid_size_w):
        plt.axvline(x=j*patch_size, color='white', linewidth=0.5, alpha=0.7)
    
    # Save the heatmap with grid lines
    plt.savefig(os.path.join(output_dir, 'bbox_heatmap_with_grid.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create the original heatmap with bounding boxes
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(heatmap, cmap='hot', interpolation='nearest')
    
    # Sample bboxes to show (to avoid cluttering the visualization)
    if len(all_bboxes) > num_bboxes_to_show:
        import random
        bboxes_to_show = random.sample(all_bboxes, num_bboxes_to_show)
    else:
        bboxes_to_show = all_bboxes
    
    # Draw sampled bounding boxes
    for xmin, ymin, xmax, ymax in bboxes_to_show:
        # Convert normalized coordinates to pixel coordinates for the visualization
        rect_x = xmin * resolution[1]
        rect_y = ymin * resolution[0]
        rect_width = (xmax - xmin) * resolution[1]
        rect_height = (ymax - ymin) * resolution[0]
        
        # Add rectangle patch
        rect = patches.Rectangle(
            (rect_x, rect_y), rect_width, rect_height, 
            linewidth=1, edgecolor='cyan', facecolor='none', alpha=0.5
        )
        ax.add_patch(rect)
    
    # Add grid lines to bounding box visualization
    for i in range(1, grid_size_h):
        ax.axhline(y=i*patch_size, color='white', linewidth=0.5, alpha=0.7)
    for j in range(1, grid_size_w):
        ax.axvline(x=j*patch_size, color='white', linewidth=0.5, alpha=0.7)
    
    plt.colorbar(label='Normalized Frequency')
    plt.title(f'Bounding Box Heatmap with {num_bboxes_to_show} Random Boxes')
    
    # Save the visualization with bounding boxes and grid lines
    plt.savefig(os.path.join(output_dir, 'bbox_heatmap_with_boxes_grid.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create histogram of bounding box sizes
    plt.figure(figsize=(10, 6))
    areas = [(xmax-xmin)*(ymax-ymin) for xmin, ymin, xmax, ymax in all_bboxes]
    plt.hist(areas, bins=50, alpha=0.75)
    plt.title('Distribution of Bounding Box Areas (Normalized)')
    plt.xlabel('Bounding Box Area (as fraction of image area)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'bbox_area_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create the patch interest heatmap
    patch_heatmap = np.zeros(resolution, dtype=np.float32)
    
    # For each bbox, add its contribution to the patch heatmap based on its area
    for xmin, ymin, xmax, ymax in all_bboxes:
        h_xmin = int(xmin * resolution[1])
        h_ymin = int(ymin * resolution[0])
        h_xmax = int(xmax * resolution[1])
        h_ymax = int(ymax * resolution[0])
        
        # Ensure valid coordinates
        h_xmin = max(0, min(h_xmin, resolution[1]-1))
        h_ymin = max(0, min(h_ymin, resolution[0]-1))
        h_xmax = max(0, min(h_xmax, resolution[1]-1))
        h_ymax = max(0, min(h_ymax, resolution[0]-1))
        
        # Add area contribution to patch heatmap
        area = (h_xmax - h_xmin) * (h_ymax - h_ymin)
        if area > 0:
            patch_heatmap[h_ymin:h_ymax+1, h_xmin:h_xmax+1] += 1 / area
    
    # Normalize patch heatmap
    if np.max(patch_heatmap) > 0:
        patch_heatmap = patch_heatmap / np.max(patch_heatmap)
    
    # Calculate patch scores for the patch interest heatmap
    patch_interest_ranking, sorted_interest_patches, _, _ = compute_patch_scores(patch_heatmap, patch_size)
    
    # Create visualizations with patch rankings displayed
    # 1. For the original heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(im, label='Normalized Frequency')
    plt.title(f'Bounding Box Heatmap with Patch Rankings ({patch_size}×{patch_size} patches)')
    
    # Add grid lines
    for i in range(1, grid_size_h):
        ax.axhline(y=i*patch_size, color='white', linewidth=0.5, alpha=0.7)
    for j in range(1, grid_size_w):
        ax.axvline(x=j*patch_size, color='white', linewidth=0.5, alpha=0.7)
    
    # Add patch rankings
    for i in range(grid_size_h):
        for j in range(grid_size_w):
            rank, score = patch_ranking.get((i, j), (0, 0))
            center_y = i * patch_size + patch_size // 2
            center_x = j * patch_size + patch_size // 2
            ax.text(center_x, center_y, f"{rank}\n{score:.3f}", ha='center', va='center', 
                    color='white', fontsize=7, fontweight='bold', 
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    plt.savefig(os.path.join(output_dir, 'bbox_heatmap_with_rankings.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. For the patch interest heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(patch_heatmap, cmap='viridis', interpolation='nearest')
    plt.colorbar(im, label='Normalized Patch Interest')
    plt.title(f'Patch Interest Heatmap with Rankings ({patch_size}×{patch_size} patches)')
    
    # Add grid lines
    for i in range(1, grid_size_h):
        ax.axhline(y=i*patch_size, color='white', linewidth=0.5, alpha=0.7)
    for j in range(1, grid_size_w):
        ax.axvline(x=j*patch_size, color='white', linewidth=0.5, alpha=0.7)
    
    # Add patch rankings
    for i in range(grid_size_h):
        for j in range(grid_size_w):
            rank, score = patch_interest_ranking.get((i, j), (0, 0))
            center_y = i * patch_size + patch_size // 2
            center_x = j * patch_size + patch_size // 2
            ax.text(center_x, center_y, f"{rank}\n{score:.3f}", ha='center', va='center', 
                    color='white', fontsize=7, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    plt.savefig(os.path.join(output_dir, 'patch_interest_heatmap_with_rankings.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a heatmap based on patch rankings (not values)
    rank_heatmap = np.zeros((grid_size_h, grid_size_w), dtype=np.float32)
    for i in range(grid_size_h):
        for j in range(grid_size_w):
            rank, _ = patch_ranking.get((i, j), (total_patches, 0))
            rank_heatmap[i, j] = total_patches - rank + 1  # Invert so highest rank is highest value
    
    # Normalize rank heatmap
    if np.max(rank_heatmap) > 0:
        rank_heatmap = rank_heatmap / np.max(rank_heatmap)
    
    # Plot rank heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(rank_heatmap, cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(label='Normalized Rank (higher = better)')
    plt.title(f'Patch Ranking Heatmap ({grid_size_h}×{grid_size_w} grid)')
    
    # Add grid lines
    for i in range(1, grid_size_h):
        plt.axhline(y=i, color='white', linewidth=0.5, alpha=0.7)
    for j in range(1, grid_size_w):
        plt.axvline(x=j, color='white', linewidth=0.5, alpha=0.7)
    
    # Add rank numbers
    for i in range(grid_size_h):
        for j in range(grid_size_w):
            rank, _ = patch_ranking.get((i, j), (0, 0))
            plt.text(j, i, f"{rank}", ha='center', va='center', color='black', fontweight='bold', fontsize=7)
    
    plt.savefig(os.path.join(output_dir, 'patch_rank_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the patch heatmap as a numpy array for later use
    np.save(os.path.join(output_dir, 'patch_interest_heatmap.npy'), patch_heatmap)
    
    # Save the patch rankings as numpy arrays for later use
    patch_ranks = np.zeros((grid_size_h, grid_size_w), dtype=np.int32)
    patch_scores = np.zeros((grid_size_h, grid_size_w), dtype=np.float32)
    
    for i in range(grid_size_h):
        for j in range(grid_size_w):
            rank, score = patch_ranking.get((i, j), (0, 0))
            patch_ranks[i, j] = rank
            patch_scores[i, j] = score
    
    np.save(os.path.join(output_dir, 'patch_ranks.npy'), patch_ranks)
    np.save(os.path.join(output_dir, 'patch_scores.npy'), patch_scores)
    
    # Generate separate patch visualizations for the top N patches
    top_n = min(25, total_patches)  # Top 25 or all if fewer
    top_patches = sorted_patches[:top_n]
    
    # Extract original-sized patches from the heatmap for visualization
    for idx, (i, j, score) in enumerate(top_patches):
        # Calculate patch boundaries
        y_start = i * patch_size
        y_end = (i + 1) * patch_size if i < grid_size_h - 1 else resolution[0]
        x_start = j * patch_size
        x_end = (j + 1) * patch_size if j < grid_size_w - 1 else resolution[1]
        
        # Extract the patch
        patch = heatmap[y_start:y_end, x_start:x_end]
        
        # Create a visualization of this patch
        plt.figure(figsize=(5, 5))
        plt.imshow(patch, cmap='hot', interpolation='nearest')
        plt.title(f'Patch Rank {idx+1} (Row {i}, Col {j}, Score: {score:.4f})')
        plt.colorbar()
        plt.savefig(os.path.join(patches_dir, f'patch_rank_{idx+1:02d}_r{i}c{j}.png'), dpi=200, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")
    print(f"Top patches saved to {patches_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate and visualize a heatmap of bounding boxes in Tiny ImageNet')
    parser.add_argument('--dataset', type=str, required=True, help='Path to Tiny ImageNet dataset')
    parser.add_argument('--output', type=str, default='bbox_heatmaps', help='Output directory for visualizations')
    parser.add_argument('--resolution', type=int, default=224, help='Resolution for heatmap generation')
    parser.add_argument('--boxes', type=int, default=100, help='Number of random boxes to show in visualization')
    parser.add_argument('--patch-size', type=int, default=16, help='Size of each patch (standard ViT uses 16×16)')
    
    args = parser.parse_args()
    
    resolution = (args.resolution, args.resolution)
    
    print(f"Generating bounding box heatmap for {args.dataset} at {resolution} resolution")
    heatmap, all_bboxes, image_sizes = generate_bbox_heatmap(args.dataset, resolution)
    
    if not all_bboxes:
        print("No bounding boxes found. Exiting.")
        return
    
    print(f"Found {len(all_bboxes)} bounding boxes in total")
    print(f"Average image size: {np.mean([w for w, h in image_sizes]):.1f}x{np.mean([h for w, h in image_sizes]):.1f}")
    
    visualize_bbox_heatmap(heatmap, all_bboxes, args.output, resolution, args.boxes, args.patch_size)

if __name__ == "__main__":
    main()

