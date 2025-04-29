#!/usr/bin/env python3
import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import cv2
from glob import glob
from pathlib import Path
from tqdm import tqdm


def load_rankings(ranking_file, image_filename=None):
    """
    Load rankings from a CSV file.
    
    Args:
        ranking_file (str): Path to the rankings CSV file
        image_filename (str, optional): If provided, only return rankings for this specific image
        
    Returns:
        dict: Mapping from image filename to ranking list, or a single ranking list if image_filename is provided
    """
    rankings = {}
    
    try:
        with open(ranking_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header row
            
            for row in reader:
                if not row:
                    continue
                
                filename = row[0]
                rankings_str = row[1]
                
                # Parse the rankings string into a list of integers
                try:
                    ranking_list = list(map(int, rankings_str.split(',')))
                    rankings[filename] = ranking_list
                except Exception as e:
                    print(f"Error parsing rankings for {filename}: {e}")
    except Exception as e:
        print(f"Error loading rankings file {ranking_file}: {e}")
        return {}
    
    if image_filename is not None:
        # Return just the rankings for the specified image
        return rankings.get(image_filename)
    
    return rankings


def create_ranking_overlay(image, rankings, patch_size=16, stride=16, num_levels=10, alpha=0.7):
    """
    Create a visualization of rankings overlaid on the original image.
    
    Args:
        image: PIL Image or path to image file
        rankings: List of patch indices sorted by importance (highest to lowest)
        patch_size (int): Size of each patch
        stride (int): Stride used in ranking generation
        num_levels (int): Number of ranking levels to display (for visual clarity)
        alpha (float): Transparency level for the overlay
        
    Returns:
        numpy array: Visualization image with ranking overlay
    """
    # Load image if a path is provided
    if isinstance(image, str):
        try:
            image = Image.open(image).convert('RGB')
            image = image.resize((224, 224), Image.LANCZOS)
        except Exception as e:
            print(f"Error loading image {image}: {e}")
            return None
    
    # Convert to numpy array if it's a PIL Image
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
        
    # Create a white overlay
    overlay = np.ones_like(image_np) * 255
    
    # Calculate patch grid dimensions
    height, width = image_np.shape[:2]
    num_patches_y = (height - patch_size) // stride + 1
    num_patches_x = (width - patch_size) // stride + 1
    total_patches = num_patches_y * num_patches_x
    
    # Create a colormap for ranking visualization
    # Using a colormap that goes from hot (important) to cold (less important)
    cmap = plt.cm.get_cmap('plasma', num_levels)
    
    # Divide rankings into levels
    level_size = max(1, len(rankings) // num_levels)
    
    # Draw patches by importance level
    for level in range(num_levels):
        start_idx = level * level_size
        end_idx = min((level + 1) * level_size, len(rankings))
        
        # Skip if we're out of rankings
        if start_idx >= len(rankings):
            break
            
        # Get color for this level
        color = np.array(cmap(level/num_levels)[:3]) * 255
        
        # Draw all patches in this level
        for rank_idx in range(start_idx, end_idx):
            if rank_idx >= len(rankings):
                break
                
            patch_idx = rankings[rank_idx]
            
            # Convert flat index to 2D coordinates
            y = (patch_idx // num_patches_x) * stride
            x = (patch_idx % num_patches_x) * stride
            
            # Ensure we're within image bounds
            if y + patch_size <= height and x + patch_size <= width:
                # Draw colored rectangle for this patch
                overlay[y:y+patch_size, x:x+patch_size] = color
    
    # Blend the overlay with the original image
    result = cv2.addWeighted(image_np, 1-alpha, overlay.astype(np.uint8), alpha, 0)
    
    return result


def create_ranking_heatmap(image, rankings, patch_size=16, stride=16):
    """
    Create a heatmap visualization of rankings.
    
    Args:
        image: PIL Image or path to image file
        rankings: List of patch indices sorted by importance (highest to lowest)
        patch_size (int): Size of each patch
        stride (int): Stride used in ranking generation
        
    Returns:
        numpy array: Heatmap image showing ranking importance
    """
    # Load image if a path is provided
    if isinstance(image, str):
        try:
            image = Image.open(image).convert('RGB')
            image = image.resize((224, 224), Image.LANCZOS)
        except Exception as e:
            print(f"Error loading image {image}: {e}")
            return None
    
    # Get image dimensions
    if isinstance(image, Image.Image):
        width, height = image.size
        image_np = np.array(image)
    else:
        height, width = image.shape[:2]
        image_np = image
    
    # Calculate patch grid dimensions
    num_patches_y = (height - patch_size) // stride + 1
    num_patches_x = (width - patch_size) // stride + 1
    total_patches = num_patches_y * num_patches_x
    
    # Create an empty heatmap
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Calculate importance value for each patch (normalized by ranking position)
    if len(rankings) == 0:
        return np.zeros_like(image_np)
        
    max_rank = len(rankings)
    
    for rank_idx, patch_idx in enumerate(rankings):
        # Normalize importance from 1.0 (most important) to 0.0 (least important)
        importance = 1.0 - (rank_idx / max_rank)
        
        # Convert flat index to 2D coordinates
        y = (patch_idx // num_patches_x) * stride
        x = (patch_idx % num_patches_x) * stride
        
        # Ensure we're within image bounds
        if y + patch_size <= height and x + patch_size <= width:
            # Add importance value to the heatmap
            heatmap[y:y+patch_size, x:x+patch_size] = importance
    
    # Normalize heatmap to 0-1 range
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    # Apply colormap to create RGB heatmap
    heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
    
    # Convert to uint8 for visualization
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    return heatmap_colored


def visualize_ranking(image_path, ranking_file, patch_size=16, stride=16, output_path=None, show=True):
    """
    Create and optionally save visualizations of rankings for an image.
    
    Args:
        image_path (str): Path to the image file
        ranking_file (str): Path to the rankings CSV file
        patch_size (int): Size of each patch
        stride (int): Stride used in ranking generation
        output_path (str, optional): If provided, save visualization to this path
        show (bool): Whether to display the visualization
        
    Returns:
        None
    """
    # Get image filename for lookup in rankings
    image_filename = os.path.basename(image_path)
    
    # Load rankings for this image
    rankings = load_rankings(ranking_file, image_filename)
    
    if rankings is None:
        print(f"No rankings found for {image_filename}")
        return
    
    # Load and resize image
    try:
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        image_resized = image.resize((224, 224), Image.LANCZOS)
        image_array = np.array(image_resized)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return
    
    # Create visualizations
    overlay = create_ranking_overlay(image_array, rankings, patch_size, stride)
    heatmap = create_ranking_heatmap(image_array, rankings, patch_size, stride)
    
    # Create combined visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Resized image
    axes[0, 1].imshow(image_array)
    axes[0, 1].set_title("Resized to 224x224")
    axes[0, 1].axis('off')
    
    # Ranking overlay
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Ranking Overlay")
    axes[1, 0].axis('off')
    
    # Ranking heatmap
    axes[1, 1].imshow(heatmap)
    axes[1, 1].set_title("Ranking Heatmap")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


def test_rankings_exist(class_folder, rankings_file):
    """
    Test that rankings exist for all images in a class folder.
    
    Args:
        class_folder (str): Path to the class folder
        rankings_file (str): Path to the rankings CSV file
        
    Returns:
        tuple: (success, total_images, found_rankings, missing_images)
    """
    # Get all images in the class folder
    image_files = []
    for ext in ('jpg', 'jpeg', 'png', 'JPEG', 'JPG', 'PNG'):
        image_files.extend(glob(os.path.join(class_folder, f'*.{ext}')))
    
    image_filenames = [os.path.basename(f) for f in image_files]
    total_images = len(image_filenames)
    
    # Load all rankings
    rankings = load_rankings(rankings_file)
    found_rankings = len(rankings)
    
    # Check for missing rankings
    missing_images = []
    for img_file in image_filenames:
        if img_file not in rankings:
            missing_images.append(img_file)
    
    success = len(missing_images) == 0
    
    return success, total_images, found_rankings, missing_images


def test_ranking_format(rankings_file, expected_length=None, patch_size=16, stride=16, image_size=224):
    """
    Test that rankings are correctly formatted.
    
    Args:
        rankings_file (str): Path to the rankings CSV file
        expected_length (int, optional): Expected length of each ranking list
        patch_size (int): Size of each patch
        stride (int): Stride used in ranking generation
        image_size (int): Size of the image (assumes square images)
        
    Returns:
        tuple: (success, issues, total_rankings)
    """
    rankings = load_rankings(rankings_file)
    total_rankings = len(rankings)
    
    issues = []
    
    # Calculate expected number of patches if not provided
    if expected_length is None:
        num_patches_per_dim = (image_size - patch_size) // stride + 1
        expected_length = num_patches_per_dim * num_patches_per_dim
    
    # Expected set of indices (0 to expected_length-1)
    expected_indices = set(range(expected_length))
    
    for filename, ranking_list in rankings.items():
        # Check if ranking is a list
        if not isinstance(ranking_list, list):
            issues.append(f"{filename}: Ranking is not a list")
            continue
        
        # Check if ranking contains integers
        if not all(isinstance(r, int) for r in ranking_list):
            issues.append(f"{filename}: Ranking contains non-integer values")
            continue
        
        # Check if ranking has expected length
        if len(ranking_list) != expected_length:
            issues.append(f"{filename}: Ranking length {len(ranking_list)} != expected {expected_length}")
            continue
        
        # Check for duplicates
        ranking_set = set(ranking_list)
        if len(ranking_list) != len(ranking_set):
            issues.append(f"{filename}: Ranking contains duplicate indices")
            continue
            
        # Check if all expected indices are present
        if ranking_set != expected_indices:
            missing = expected_indices - ranking_set
            unexpected = ranking_set - expected_indices
            issue_msg = f"{filename}: Ranking has "
            if missing:
                issue_msg += f"missing values: {sorted(missing)[:5]}"
                if len(missing) > 5:
                    issue_msg += f"... (and {len(missing)-5} more)"
            if unexpected:
                issue_msg += f"{' and ' if missing else ''}unexpected values: {sorted(unexpected)[:5]}"
                if len(unexpected) > 5:
                    issue_msg += f"... (and {len(unexpected)-5} more)"
            issues.append(issue_msg)
            continue
        
        # Check if values are in range 0 to expected_length-1
        min_val = min(ranking_list)
        max_val = max(ranking_list)
        if min_val < 0 or max_val >= expected_length:
            issues.append(f"{filename}: Ranking contains out-of-range values [{min_val}, {max_val}], expected [0, {expected_length-1}]")
            continue
        
    success = len(issues) == 0
    
    return success, issues, total_rankings


def run_ranking_tests(data_dir, rankings_dir, patch_size=16, stride=16):
    """
    Run tests on all ranking files for a dataset.
    
    Args:
        data_dir (str): Path to the dataset directory with class folders
        rankings_dir (str): Path to the directory containing ranking CSV files

        patch_size (int): Size of patches used in rankings
        stride (int): Stride used in ranking generation
        
    Returns:
        dict: Test results
    """
    # Get all class folders
    class_folders = [d for d in glob(os.path.join(data_dir, '*')) if os.path.isdir(d)]
    
    results = {
        'total_classes': len(class_folders),
        'classes_with_rankings': 0,
        'missing_ranking_files': [],
        'classes_with_issues': [],
        'overall_success': True
    }
    
    # Process each class
    for class_folder in tqdm(class_folders, desc="Testing class rankings"):
        class_name = os.path.basename(class_folder)
        ranking_file = os.path.join(rankings_dir, f"{class_name}_rankings.csv")
        
        # Check if ranking file exists
        if not os.path.exists(ranking_file):
            results['missing_ranking_files'].append(class_name)
            results['overall_success'] = False
            continue
        
        results['classes_with_rankings'] += 1
        
        # Test 1: Check all images have rankings
        exist_success, total_images, found_rankings, missing_images = test_rankings_exist(class_folder, ranking_file)
        
        # Test 2: Check ranking format (including unique range test)
        format_success, format_issues, _ = test_ranking_format(ranking_file, patch_size=patch_size, stride=stride)
        
        # Record issues for this class
        if not exist_success or not format_success:
            class_issues = {
                'class_name': class_name,
                'total_images': total_images,
                'found_rankings': found_rankings,
                'missing_images': missing_images[:5] + ['...'] if len(missing_images) > 5 else missing_images,
                'missing_count': len(missing_images),
                'format_issues': format_issues[:5] + ['...'] if len(format_issues) > 5 else format_issues,
                'format_issues_count': len(format_issues)
            }
            results['classes_with_issues'].append(class_issues)
            results['overall_success'] = False
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Visualize and test image rankings")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Visualization command
    viz_parser = subparsers.add_parser('visualize', help='Visualize rankings for an image')
    viz_parser.add_argument('--image', type=str, required=True,
                          help='Path to the image file')
    viz_parser.add_argument('--ranking_file', type=str, required=True,
                          help='Path to the rankings CSV file')
    viz_parser.add_argument('--output', type=str,
                          help='Path to save the visualization')
    viz_parser.add_argument('--patch_size', type=int, default=16,
                          help='Size of each patch')
    viz_parser.add_argument('--stride', type=int, default=16,
                          help='Stride used in ranking generation')
    viz_parser.add_argument('--no_show', action='store_true',
                          help='Do not display the visualization')
    
    # Sample visualization command (creates visualizations for random images)
    sample_parser = subparsers.add_parser('sample', help='Create visualizations for random images')
    sample_parser.add_argument('--data_dir', type=str, required=True,
                             help='Path to the dataset directory')
    sample_parser.add_argument('--rankings_dir', type=str, required=True,
                             help='Path to the rankings directory')
    sample_parser.add_argument('--output_dir', type=str, required=True,
                             help='Directory to save visualizations')
    sample_parser.add_argument('--num_samples', type=int, default=5,
                             help='Number of random images to visualize')
    sample_parser.add_argument('--patch_size', type=int, default=16,
                             help='Size of each patch')
    sample_parser.add_argument('--stride', type=int, default=16,
                             help='Stride used in ranking generation')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test ranking files')
    test_parser.add_argument('--data_dir', type=str, required=True,
                           help='Path to the dataset directory')
    test_parser.add_argument('--rankings_dir', type=str, required=True,
                           help='Path to the rankings directory')
    test_parser.add_argument('--report_file', type=str,
                           help='Path to save test results')
    test_parser.add_argument('--patch_size', type=int, default=16,
                           help='Size of each patch')
    test_parser.add_argument('--stride', type=int, default=16,
                           help='Stride used in ranking generation')
    
    args = parser.parse_args()
    
    # Handle visualization command
    if args.command == 'visualize':
        visualize_ranking(
            args.image,
            args.ranking_file,
            args.patch_size,
            args.stride,
            args.output,
            not args.no_show
        )
    
    # Handle sample visualization command
    elif args.command == 'sample':
        # Get all class folders
        class_folders = [d for d in glob(os.path.join(args.data_dir, '*')) if os.path.isdir(d)]
        
        # Process each class folder
        if not class_folders:
            print(f"No class folders found in {args.data_dir}")
            return
        image_paths = []
        
        for class_folder in class_folders:
            class_name = os.path.basename(class_folder)
            ranking_file = os.path.join(args.rankings_dir, f"{class_name}_rankings.csv")
            
            if not os.path.exists(ranking_file):
                print(f"No ranking file found for class {class_name}")
                continue
            
            # Find all images in this class
            image_files = []
            for ext in ('jpg', 'jpeg', 'png', 'JPEG', 'JPG', 'PNG'):
                image_files.extend(glob(os.path.join(class_folder, f'*.{ext}')))
            
            if not image_files:
                print(f"No images found in class {class_name}")
                continue
            
            # Select a random image
            image_paths.append(np.random.choice(image_files, args.num_samples , replace=False))
            
        for image_path in tqdm(image_paths, desc="Creating sample visualizations"):
            class_name = os.path.basename(os.path.dirname(image_path))
            output_path = os.path.join(args.output_dir, f"{class_name}_{os.path.basename(image_path)}_vis.png")
            print(f"Creating visualization for {image_path}")
            visualize_ranking(
                image_path,
                ranking_file,
                args.patch_size,
                args.stride,
                output_path,
                False  # Don't show inline
            )
    
    # Handle test command
    elif args.command == 'test':
        results = run_ranking_tests(args.data_dir, args.rankings_dir, 
                                    patch_size=args.patch_size, stride=args.stride)
        
        # Print summary results
        print("\n=== Ranking Test Results ===")
        print(f"Total classes: {results['total_classes']}")
        print(f"Classes with ranking files: {results['classes_with_rankings']}")
        print(f"Missing ranking files: {len(results['missing_ranking_files'])}")
        print(f"Classes with issues: {len(results['classes_with_issues'])}")
        print(f"Overall success: {results['overall_success']}")
        
        # Print details of missing files
        if results['missing_ranking_files']:
            print("\nMissing ranking files:")
            for class_name in results['missing_ranking_files'][:5]:
                print(f"  - {class_name}")
            if len(results['missing_ranking_files']) > 5:
                print(f"  - ... and {len(results['missing_ranking_files']) - 5} more")
        
        # Print details of classes with issues
        if results['classes_with_issues']:
            print("\nClasses with issues:")
            for issue in results['classes_with_issues'][:5]:
                print(f"  - {issue['class_name']}: {issue['missing_count']} missing rankings, {issue['format_issues_count']} format issues")
            if len(results['classes_with_issues']) > 5:
                print(f"  - ... and {len(results['classes_with_issues']) - 5} more classes with issues")
        
        # Save report if requested
        if args.report_file:
            from datetime import datetime
            import json
            
            # Add timestamp to results
            results['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Save as JSON
            os.makedirs(os.path.dirname(os.path.abspath(args.report_file)), exist_ok=True)
            with open(args.report_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nDetailed report saved to {args.report_file}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

