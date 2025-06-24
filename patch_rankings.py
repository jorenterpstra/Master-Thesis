"""
Script to extract patch rankings from all images in a dataset and save them to a CSV file.
The rankings are extracted from heatmaps using the built-in _ranking_from_heatmap method.
"""
import os
import argparse
import csv
import torch
from tqdm import tqdm
import sys

# Add parent directory to path to import from vim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vim.datasets import HeatmapImageFolder

def parse_args():
    parser = argparse.ArgumentParser(description='Extract patch rankings from heatmaps and save to CSV')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the ImageNet-style dataset directory')
    parser.add_argument('--heatmap-path', type=str, required=True,
                        help='Path to the heatmaps directory')
    parser.add_argument('--output-csv', type=str, required=True,
                        help='Path to save the output CSV file')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val'],
                        help='Dataset split to process (train or val)')
    parser.add_argument('--heatmap-extension', type=str, default='.JPEG',
                        help='File extension for heatmap files')
    
    return parser.parse_args()

def create_identity_transform():
    """Create a minimal transform that just converts to tensor without changing the data"""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    return A.Compose([
        ToTensorV2(always_apply=True),
    ])

def main():
    # Parse arguments
    args = parse_args()
    
    # Create a minimal dataset with identity transform
    identity_transform = create_identity_transform()
    
    # Find the appropriate data folder
    data_root = os.path.join(args.data_path, args.split)
    heatmap_root = os.path.join(args.heatmap_path, f"{args.split}_heat")
    
    print(f"Loading dataset from {data_root}")
    print(f"Loading heatmaps from {heatmap_root}")
    
    # Create dataset
    dataset = HeatmapImageFolder(
        root=data_root,
        heatmap_root=heatmap_root,
        transform=identity_transform,
        return_rankings=True,
        return_heatmap=False,
        heatmap_extension=args.heatmap_extension
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    
    # Open CSV file for writing
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Create header row: filename, patch_0, patch_1, ..., patch_195
        num_patches = 196  # 14x14 patches for 224x224 images with 16x16 patches
        header = ['filename'] + [f'patch_{i}' for i in range(num_patches)]
        writer.writerow(header)
        
        # Process all images and write to CSV
        for idx in tqdm(range(len(dataset)), desc="Processing images"):
            # Get the image path from the dataset
            path, _ = dataset.samples[idx]
            filename = os.path.basename(path)
            
            # Get the image, target, and ranking
            try:
                _, _, ranking = dataset[idx]
                
                # Convert ranking tensor to list and write to CSV
                ranking_list = ranking.cpu().numpy().tolist()
                writer.writerow([filename] + ranking_list)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    print(f"Rankings saved to {args.output_csv}")

if __name__ == '__main__':
    main()