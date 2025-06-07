import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from dataloader import ImageNetPatchRankLoader
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import torch
import xml.etree.ElementTree as ET
from utils import add_grid, create_patch_score_map
import os

def get_original_bboxes(xml_path):
    """Helper function to get unscaled bounding boxes"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    valid_boxes = []
    for obj in root.findall('object'):
        difficult = obj.find('difficult')
        if difficult is not None and int(difficult.text) == 1:
            continue
            
        bndbox = obj.find('bndbox')
        if bndbox is not None:
            try:
                xmin = max(0, float(bndbox.find('xmin').text))
                ymin = max(0, float(bndbox.find('ymin').text))
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                valid_boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
            except (AttributeError, ValueError):
                continue
    
    return valid_boxes

def visualize_sample(root_dir, split='train', idx=0, export_path=None):
    """
    Visualize a single sample with:
    1. Original image with original (unscaled) bounding boxes
    2. Transformed image (model input) with transformed bounding boxes
    3. Transformed image with patch score heatmap overlay
    
    Args:
        root_dir: Path to dataset root directory
        split: 'train' or 'val'
        idx: Index of sample to visualize
        export_path: Directory to export visualization images (None to skip export)
    
    Returns:
        If export_path is provided, returns tuple of (bbox_img_path, heatmap_img_path)
        Otherwise returns None
    """
    # Create datasets - one with transforms, one without
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset_raw = ImageNetPatchRankLoader(root_dir, split, transform=None, return_boxes=True)
    dataset_transformed = ImageNetPatchRankLoader(root_dir, split, transform=transform, return_boxes=True)
    
    # Get both versions
    image_raw, _, orig_boxes = dataset_raw[idx]
    image_transformed, scores, transformed_boxes = dataset_transformed[idx]
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
    
    # 1. Original image with original boxes
    draw = ImageDraw.Draw(image_raw)
    if orig_boxes:
        for bbox in orig_boxes:
            x, y, w, h = map(int, bbox)
            draw.rectangle([x, y, x+w, y+h], outline='red', width=2)
    
    ax1.imshow(image_raw)
    ax1.set_title(f'Original Input\n({image_raw.size[0]}x{image_raw.size[1]})')
    ax1.axis('off')
    
    # 2. Transformed image with transformed boxes
    img_display = image_transformed.clone()
    # Denormalize
    img_display = img_display * torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    img_display = img_display + torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    img_display = torch.clamp(img_display, 0, 1)
    
    # Convert to PIL for drawing
    transformed_img_pil = transforms.ToPILImage()(img_display)
    draw = ImageDraw.Draw(transformed_img_pil)
    
    if transformed_boxes:
        for bbox in transformed_boxes:
            x, y, w, h = map(int, bbox)
            if w > 0 and h > 0:  # Only draw valid boxes
                draw.rectangle([x, y, x+w, y+h], outline='blue', width=2)
    
    ax2.imshow(transformed_img_pil)
    add_grid(ax2, 224, 16, color='white', alpha=0.3)  # 224x224 image, 16x16 patches
    ax2.set_title('Model Input\n(224x224)')
    ax2.axis('off')
    
    # 3. Transformed image with patch scores overlay
    num_patches = int(np.sqrt(len(scores)))
    score_grid = scores.reshape(num_patches, num_patches)
    
    # Calculate patch boundaries for proper alignment
    patch_size = 224 // num_patches  # Calculate actual patch size in pixels
    
    # Create score map using utility function
    score_map = create_patch_score_map(scores, image_size=224)
    
    # Show transformed image
    ax3.imshow(transformed_img_pil)
    # Overlay score map with transparency
    heatmap = ax3.imshow(score_map, cmap='hot', alpha=0.5)
    add_grid(ax3, 224, patch_size, color='white', alpha=0.3)
    ax3.set_title(f'Patch Scores\n({num_patches}x{num_patches})')
    plt.colorbar(heatmap, ax=ax3)
    
    plt.tight_layout()
    
    # Export images for thesis report if path is provided
    bbox_img_path = None
    heatmap_img_path = None
    if export_path:
        # Create export directory if it doesn't exist
        os.makedirs(export_path, exist_ok=True)
        
        # Get sample identifier from the image path
        img_name = os.path.splitext(os.path.basename(dataset_raw.image_paths[idx]))[0]
        
        # Export the transformed image with bounding boxes (high resolution)
        bbox_img_path = os.path.join(export_path, f"{img_name}_bbox.png")
        bbox_fig = plt.figure(figsize=(8, 8), dpi=300)
        bbox_ax = bbox_fig.add_subplot(111)
        bbox_ax.imshow(transformed_img_pil)
        if transformed_boxes:
            # Draw bounding boxes on a separate figure for clean export
            draw_img = transformed_img_pil.copy()
            draw = ImageDraw.Draw(draw_img)
            for bbox in transformed_boxes:
                x, y, w, h = map(int, bbox)
                if w > 0 and h > 0:  # Only draw valid boxes
                    draw.rectangle([x, y, x+w, y+h], outline='blue', width=2)
            bbox_ax.imshow(draw_img)
        bbox_ax.axis('off')
        bbox_fig.tight_layout(pad=0)
        bbox_fig.savefig(bbox_img_path, bbox_inches='tight', pad_inches=0)
        plt.close(bbox_fig)
        
        # Export the heatmap visualization (high resolution)
        heatmap_img_path = os.path.join(export_path, f"{img_name}_heatmap.png")
        heat_fig = plt.figure(figsize=(8, 8), dpi=300)
        heat_ax = heat_fig.add_subplot(111)
        heat_ax.imshow(transformed_img_pil)
        heat = heat_ax.imshow(score_map, cmap='hot', alpha=0.5)
        heat_ax.axis('off')
        heat_fig.colorbar(heat)
        heat_fig.tight_layout(pad=0)
        heat_fig.savefig(heatmap_img_path, bbox_inches='tight', pad_inches=0)
        plt.close(heat_fig)
        
        print(f"Exported images to:\n{bbox_img_path}\n{heatmap_img_path}")
    
    plt.show()

    # Print statistics
    print(f"\nImage path: {dataset_raw.image_paths[idx]}")
    print(f"Original size: {image_raw.size}")
    print(f"\nScore statistics:")
    print(f"Min score: {scores.min():.3f}")
    print(f"Max score: {scores.max():.3f}")
    print(f"Mean score: {scores.mean():.3f}")
    print(f"Number of patches with score > 0.5: {(scores > 0.5).sum()}")
    
    return (bbox_img_path, heatmap_img_path) if export_path else None

if __name__ == "__main__":
    root_dir = Path("C:/Users/joren/Documents/_Uni/Master/Thesis/imagenet_subset")
    # Create an export directory for thesis figures
    export_path = Path("C:/Users/joren/Documents/_Uni/Master/Thesis/Master-Thesis/thesis_figures")
    
    for i in range(0, 230, 30):
        visualize_sample(root_dir, split='train', idx=i, export_path=export_path)

