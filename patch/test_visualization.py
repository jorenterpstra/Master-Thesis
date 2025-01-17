import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from patch_rank import ImageNetPatchRankLoader
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import torch
import xml.etree.ElementTree as ET
from utils import add_grid, create_patch_score_map

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

def visualize_sample(root_dir, split='train', idx=0):
    """
    Visualize a single sample with:
    1. Original image with original (unscaled) bounding boxes
    2. Transformed image (model input) with transformed bounding boxes
    3. Transformed image with patch score heatmap overlay
    """
    # Create datasets - one with transforms, one without
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset_raw = ImageNetPatchRankLoader(root_dir, split, transform=None)
    dataset_transformed = ImageNetPatchRankLoader(root_dir, split, transform=transform)
    
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
    plt.show()

    # Print statistics
    print(f"\nImage path: {dataset_raw.image_paths[idx]}")
    print(f"Original size: {image_raw.size}")
    print(f"\nScore statistics:")
    print(f"Min score: {scores.min():.3f}")
    print(f"Max score: {scores.max():.3f}")
    print(f"Mean score: {scores.mean():.3f}")
    print(f"Number of patches with score > 0.5: {(scores > 0.5).sum()}")

if __name__ == "__main__":
    root_dir = Path("C:/Users/joren/Documents/_Uni/Master/Thesis/imagenet_subset")
    
    for i in range(0, 230, 40):
        visualize_sample(root_dir, split='train', idx=i)

