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
import cv2

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
        # resize to 224x224, no normalization
        A.Resize(height=224, width=224, interpolation=cv2.INTER_CUBIC),
        ToTensorV2(),
    ], additional_targets={'heatmap': 'mask'})

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
        return_heatmap=True,
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
        summed_heatmap = torch.zeros((224, 224, 3), dtype=torch.float32)
        # Process all images and write to CSV
        for idx in tqdm(range(len(dataset)), desc="Processing images"):
            # Get the image path from the dataset
            path, _ = dataset.samples[idx]
            filename = os.path.basename(path)
            
            # Get the image, target, and ranking
            try:
                _, _, ranking, heatmap = dataset[idx]
                
                # Convert ranking tensor to list and write to CSV
                ranking_list = ranking.cpu().numpy().tolist()
                writer.writerow([filename] + ranking_list)
                # Accumulate heatmap
                summed_heatmap += heatmap.squeeze(0).cpu()
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    print(f"Rankings saved to {args.output_csv}")
    # Save the summed heatmap as  an image
    summed_heatmap = (summed_heatmap / summed_heatmap.max() * 255).byte().numpy()
    summed_heatmap = cv2.applyColorMap(summed_heatmap, cv2.COLORMAP_JET)
    heatmap_output_path = os.path.join(os.path.dirname(args.output_csv), 'summed_heatmap.png')
    cv2.imwrite(heatmap_output_path, summed_heatmap)
    print(f"Summed heatmap saved to {heatmap_output_path}")

if __name__ == '__main__':
    main()