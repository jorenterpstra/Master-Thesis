import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torch
from patch.dataloader import ImageNetPatchRankLoader
from tqdm import tqdm

def generate_gt_heatmap(image_shape, bboxes, normalize=True):
    """
    Generate a ground truth heatmap from bounding boxes
    
    Args:
        image_shape: tuple (height, width) of the image
        bboxes: list of bounding boxes in format [x, y, w, h]
        normalize: whether to normalize the heatmap to 0-1 range
    
    Returns:
        heatmap: numpy array of shape (height, width)
    """
    heatmap = np.zeros(image_shape[:2], dtype=np.float32)
    
    for bbox in bboxes:
        x, y, w, h = [int(val) for val in bbox]
        # Ensure coordinates are within image boundaries
        x = max(0, min(x, image_shape[1] - 1))
        y = max(0, min(y, image_shape[0] - 1))
        w = min(w, image_shape[1] - x)
        h = min(h, image_shape[0] - y)
        
        # Add 1.0 to the heatmap where the object is located
        heatmap[y:y+h, x:x+w] = 1.0
    
    if normalize and np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
        
    return heatmap

def generate_bing_heatmap(image, saliency, num_detections, normalize=True):
    """
    Generate a heatmap using BING objectness detector
    
    Args:
        image: OpenCV image (BGR format)
        saliency: OpenCV BING saliency detector
        num_detections: number of detections to use
        normalize: whether to normalize the heatmap to 0-1 range
    
    Returns:
        heatmap: numpy array of shape (height, width)
    """
    (success, saliencyMap) = saliency.computeSaliency(image)
    if not success:
        return np.zeros(image.shape[:2], dtype=np.float32)
    
    # Create an empty heatmap
    heatmap = np.zeros(image.shape[:2], dtype=np.float32)
    
    # Add each detection to the heatmap
    for i in range(0, min(num_detections, saliencyMap.shape[0])):
        # Extract the bounding box coordinates
        (startX, startY, endX, endY) = saliencyMap[i].flatten()
        startX, startY, endX, endY = int(startX), int(startY), int(endX), int(endY)
        
        # Ensure coordinates are within image boundaries
        startX = max(0, min(startX, image.shape[1] - 1))
        startY = max(0, min(startY, image.shape[0] - 1))
        endX = max(startX + 1, min(endX, image.shape[1]))
        endY = max(startY + 1, min(endY, image.shape[0]))
        
        # Higher weight for earlier detections
        weight = 1.0 - (i / num_detections) if num_detections > 0 else 0
        heatmap[startY:endY, startX:endX] += weight
    
    if normalize and np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
        
    return heatmap

def fully_vectorized_heatmap(img, saliencyMap, max_detections):
    """Fully vectorized heatmap generation without explicit loops"""
    height, width = img.shape[:2]
    num_boxes = min(saliencyMap.shape[0], max_detections)
    
    # Extract all boxes at once
    boxes = saliencyMap[:num_boxes].reshape(num_boxes, 4).astype(np.int32)
    
    # Clip to image boundaries
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width - 1)  # startX
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)  # startY
    boxes[:, 2] = np.clip(boxes[:, 2], boxes[:, 0] + 1, width)  # endX
    boxes[:, 3] = np.clip(boxes[:, 3], boxes[:, 1] + 1, height)  # endY
    
    # Calculate weights
    weights = 1.0 - (np.arange(num_boxes) / max_detections)
    
    # Create 3D mask array (num_boxes × height × width)
    masks = np.zeros((num_boxes, height, width), dtype=np.float32)
    
    # Create coordinate arrays once
    y_range = np.arange(height)
    x_range = np.arange(width)
    
    # Use broadcasting to create masks for all boxes at once
    Y, X = np.meshgrid(y_range, x_range, indexing='ij')
    
    # This is still a loop but much more efficient
    for i in range(num_boxes):
        masks[i] = ((X >= boxes[i, 0]) & (X < boxes[i, 2]) & 
                    (Y >= boxes[i, 1]) & (Y < boxes[i, 3]))
    
    # Apply weights and sum
    return np.sum(masks * weights[:, np.newaxis, np.newaxis], axis=0)

def calculate_iou(heatmap1, heatmap2, threshold=0.5):
    """
    Calculate Intersection over Union between two heatmaps
    
    Args:
        heatmap1, heatmap2: numpy arrays of the same shape
        threshold: threshold to binarize the heatmaps
    
    Returns:
        iou: Intersection over Union score
    """
    # Binarize the heatmaps using the threshold
    binary1 = (heatmap1 > threshold).astype(np.float32)
    binary2 = (heatmap2 > threshold).astype(np.float32)
    
    # Calculate intersection and union
    intersection = np.logical_and(binary1, binary2).sum()
    union = np.logical_or(binary1, binary2).sum()
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    return intersection / union

def calculate_mse(heatmap1, heatmap2):
    """
    Calculate Mean Squared Error between two heatmaps
    
    Args:
        heatmap1, heatmap2: numpy arrays of the same shape
    
    Returns:
        mse: Mean Squared Error
    """
    return np.mean((heatmap1 - heatmap2) ** 2)

def main():
    # Paths configuration
    data_root = Path("/storage/scratch/6403840/data/imagenet-tiny")
    model_path = "/storage/scratch/6403840/data/BING_models"
    output_dir = Path("/storage/scratch/6403840/data/bing_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.setLogLevel(3)
    
    # Parameters
    num_images = 50  # Number of images to evaluate
    detection_counts = [10, 50, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000]  # Different detection counts to try
    
    # Initialize the BING objectness saliency detector
    saliency = cv2.saliency.ObjectnessBING_create()
    saliency.setTrainingPath(model_path)
    
    # Create dataset
    dataset = ImageNetPatchRankLoader(
        root_dir=data_root,
        split='train',
        transform=None,  # We need the original untransformed images
        return_boxes=True  # Get bounding boxes as well
    )
    
    # Randomly sample images
    indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)
    
    # Track metrics for each detection count
    iou_scores = {count: [] for count in detection_counts}
    mse_scores = {count: [] for count in detection_counts}
    
    # Process images
    for idx in tqdm(indices, desc="Processing images"):
        # Get image and bounding boxes
        image, _, bboxes = dataset[idx]
        
        # Convert PIL Image to OpenCV format
        image_np = np.array(image)
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            # Handle grayscale images
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Generate ground truth heatmap from bounding boxes
        gt_heatmap = generate_gt_heatmap(image_np.shape, bboxes)
        (success, saliencyMap) = saliency.computeSaliency(image_cv)
        if not success:
            continue
        # Try different detection counts
        for count in detection_counts:
            # Generate BING heatmap
            bing_heatmap = fully_vectorized_heatmap(image_cv, saliencyMap, count)
            
            # Calculate metrics
            iou = calculate_iou(gt_heatmap, bing_heatmap)
            mse = calculate_mse(gt_heatmap, bing_heatmap)
            
            # Store metrics
            iou_scores[count].append(iou)
            mse_scores[count].append(mse)
            
        # Visualize results for the first few images
        if len(iou_scores[detection_counts[0]]) <= 5:
            # Create figure with multiple subplots
            fig, axes = plt.subplots(2, len(detection_counts) + 1, figsize=(16, 6))
            
            # Plot ground truth
            axes[0, 0].imshow(image)
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis('off')
            
            axes[1, 0].imshow(gt_heatmap, cmap='jet')
            axes[1, 0].set_title("Ground Truth")
            axes[1, 0].axis('off')
            
            # Plot BING heatmaps with different detection counts
            for i, count in enumerate(detection_counts, 1):
                if i < len(detection_counts) + 1:
                    bing_heatmap = generate_bing_heatmap(image_cv, saliency, count)
                    
                    # Create a colored overlay for visualization
                    bing_colored = cv2.applyColorMap((bing_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    bing_colored = cv2.cvtColor(bing_colored, cv2.COLOR_BGR2RGB)
                    
                    # Create a transparent overlay
                    overlay = image_np.copy()
                    alpha = 0.6
                    cv2.addWeighted(bing_colored, alpha, overlay, 1 - alpha, 0, overlay)
                    
                    axes[0, i].imshow(overlay)
                    axes[0, i].set_title(f"BING Overlay: {count}")
                    axes[0, i].axis('off')
                    
                    axes[1, i].imshow(bing_heatmap, cmap='jet')
                    axes[1, i].set_title(f"BING: {count}\nIoU: {iou_scores[count][-1]:.3f}, MSE: {mse_scores[count][-1]:.3f}")
                    axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"bing_comparison_image_{idx}.png")
            plt.close(fig)
    
    # Calculate average metrics
    avg_iou = {count: np.mean(scores) for count, scores in iou_scores.items()}
    avg_mse = {count: np.mean(scores) for count, scores in mse_scores.items()}
    
    # Find the best detection count
    best_iou_count = max(avg_iou.items(), key=lambda x: x[1])[0]
    best_mse_count = min(avg_mse.items(), key=lambda x: x[1])[0]
    
    print(f"Best detection count based on IoU: {best_iou_count} (IoU: {avg_iou[best_iou_count]:.3f})")
    print(f"Best detection count based on MSE: {best_mse_count} (MSE: {avg_mse[best_mse_count]:.3f})")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # IoU plot (higher is better)
    ax1.plot(detection_counts, [avg_iou[count] for count in detection_counts], 'o-')
    ax1.set_xlabel('Number of Detections')
    ax1.set_ylabel('Average IoU')
    ax1.set_title('IoU vs. Number of Detections')
    ax1.grid(True)
    ax1.axvline(x=best_iou_count, color='r', linestyle='--', label=f'Best: {best_iou_count}')
    ax1.legend()
    
    # MSE plot (lower is better)
    ax2.plot(detection_counts, [avg_mse[count] for count in detection_counts], 'o-')
    ax2.set_xlabel('Number of Detections')
    ax2.set_ylabel('Average MSE')
    ax2.set_title('MSE vs. Number of Detections')
    ax2.grid(True)
    ax2.axvline(x=best_mse_count, color='r', linestyle='--', label=f'Best: {best_mse_count}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "optimization_results.png")
    plt.close(fig)
    
    # Save numerical results
    with open(output_dir / "optimization_results.txt", "w") as f:
        f.write("Detection Count, Average IoU, Average MSE\n")
        for count in detection_counts:
            f.write(f"{count}, {avg_iou[count]:.5f}, {avg_mse[count]:.5f}\n")
        f.write(f"\nBest detection count based on IoU: {best_iou_count} (IoU: {avg_iou[best_iou_count]:.5f})\n")
        f.write(f"Best detection count based on MSE: {best_mse_count} (MSE: {avg_mse[best_mse_count]:.5f})\n")
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
