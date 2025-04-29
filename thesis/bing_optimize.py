import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torch
from patch.dataloader import ImageNetPatchRankLoader
from tqdm import tqdm
import sys
import os
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout output"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

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
    with suppress_stdout():
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

def fully_vectorized_heatmap(image, saliency, max_detections):
    """Fully vectorized heatmap generation without explicit loops"""
    with suppress_stdout():
        (success, saliencyMap) = saliency.computeSaliency(image)
    if not success:
        return np.zeros(image.shape[:2], dtype=np.float32)
    height, width = image.shape[:2]
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

def resize_bbox(bbox, original_size, target_size=(224, 224)):
    """
    Resize a bounding box from original image dimensions to target dimensions
    
    Args:
        bbox: [x, y, w, h] bounding box
        original_size: (width, height) of original image
        target_size: (width, height) of target image
        
    Returns:
        resized_bbox: [x, y, w, h] scaled to target size
    """
    x, y, w, h = bbox
    scale_x = target_size[0] / original_size[0]
    scale_y = target_size[1] / original_size[1]
    
    return [
        x * scale_x,
        y * scale_y,
        w * scale_x,
        h * scale_y
    ]

def get_balanced_sample_indices(dataset, num_images):
    """
    Get indices for a class-balanced sample of images
    
    Args:
        dataset: ImageNetPatchRankLoader dataset
        num_images: target number of images to sample
    
    Returns:
        List of indices for a balanced sample
    """
    # Get all image indices grouped by class
    class_to_indices = {}
    for idx in range(len(dataset)):
        _, class_id, _ = dataset[idx]
        if class_id not in class_to_indices:
            class_to_indices[class_id] = []
        class_to_indices[class_id].append(idx)
    
    # Calculate how many images to sample from each class
    num_classes = len(class_to_indices)
    images_per_class = max(1, num_images // num_classes)
    
    # Sample from each class
    balanced_indices = []
    for class_indices in class_to_indices.values():
        # Take either images_per_class or all available if less
        sample_size = min(images_per_class, len(class_indices))
        class_sample = np.random.choice(class_indices, sample_size, replace=False)
        balanced_indices.extend(class_sample)
    
    # If we need more images to reach target, sample randomly from remaining
    if len(balanced_indices) < num_images:
        remaining = num_images - len(balanced_indices)
        # Get all indices not already selected
        all_indices = set(range(len(dataset)))
        remaining_indices = list(all_indices - set(balanced_indices))
        if remaining_indices:
            extra_sample = np.random.choice(remaining_indices, 
                                           min(remaining, len(remaining_indices)), 
                                           replace=False)
            balanced_indices.extend(extra_sample)
    
    # If we have more than needed, subsample
    if len(balanced_indices) > num_images:
        balanced_indices = np.random.choice(balanced_indices, num_images, replace=False)
    
    return balanced_indices

def optimize_detection_count(dataset, indices, saliency, min_count=10, max_count=2000):
    """
    Use binary search to find optimal detection count more efficiently
    
    Args:
        dataset: ImageNetPatchRankLoader dataset
        indices: indices of images to process
        saliency: BING saliency detector
        min_count: minimum number of detections to consider
        max_count: maximum number of detections to consider
    
    Returns:
        optimal_count: optimal number of detections
        metrics: dictionary of metrics for the optimal count
    """
    # Start with a coarse grid search to determine promising regions
    grid_points = [10, 50, 200, 500, 1000, 2000]
    grid_points = [p for p in grid_points if min_count <= p <= max_count]
    
    iou_scores = {count: [] for count in grid_points}
    
    # Process images with grid points
    for idx in tqdm(indices[:20], desc="Initial grid search"):  # Use subset for initial search
        original_image, _, original_bboxes = dataset[idx]
        
        # Preprocessing steps (same as before)
        original_size = (original_image.width, original_image.height)
        image = original_image.resize((224, 224), Image.LANCZOS)
        bboxes = [resize_bbox(bbox, original_size) for bbox in original_bboxes]
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        gt_heatmap = generate_gt_heatmap((224, 224), bboxes)
        
        # Evaluate grid points
        for count in grid_points:
            bing_heatmap = generate_bing_heatmap(image_cv, saliency, count)
            iou = calculate_iou(gt_heatmap, bing_heatmap)
            iou_scores[count].append(iou)
    
    # Find best region based on average IoU
    avg_iou = {count: np.mean(scores) for count, scores in iou_scores.items()}
    sorted_counts = sorted(grid_points, key=lambda x: avg_iou[x], reverse=True)
    
    # Define search bounds based on grid results
    if len(sorted_counts) >= 3:
        # Use top 3 points to define region
        top_points = sorted_counts[:3]
        low_bound = max(min_count, min(top_points) // 2)
        high_bound = min(max_count, max(top_points) * 2)
    else:
        low_bound, high_bound = min_count, max_count
    
    # Binary search for optimal count
    search_precision = 50  # Minimum gap to continue searching
    while high_bound - low_bound > search_precision:
        mid1 = low_bound + (high_bound - low_bound) // 3
        mid2 = high_bound - (high_bound - low_bound) // 3
        
        # Evaluate at mid points
        mid1_iou, mid2_iou = [], []
        
        for idx in tqdm(indices, desc=f"Binary search [{low_bound}-{high_bound}]"):
            original_image, _, original_bboxes = dataset[idx]
            # Preprocessing (same as before)
            original_size = (original_image.width, original_image.height)
            image = original_image.resize((224, 224), Image.LANCZOS)
            bboxes = [resize_bbox(bbox, original_size) for bbox in original_bboxes]
            image_np = np.array(image)
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            gt_heatmap = generate_gt_heatmap((224, 224), bboxes)
            
            # Test mid points
            bing_heatmap1 = generate_bing_heatmap(image_cv, saliency, mid1)
            iou1 = calculate_iou(gt_heatmap, bing_heatmap1)
            mid1_iou.append(iou1)
            
            bing_heatmap2 = generate_bing_heatmap(image_cv, saliency, mid2)
            iou2 = calculate_iou(gt_heatmap, bing_heatmap2)
            mid2_iou.append(iou2)
        
        # Update search bounds
        if np.mean(mid1_iou) > np.mean(mid2_iou):
            high_bound = mid2
        else:
            low_bound = mid1
    
    # Final evaluation with more images at optimal point
    optimal_count = (low_bound + high_bound) // 2
    
    # Full evaluation at optimal point
    final_iou = []
    final_mse = []
    
    for idx in tqdm(indices, desc=f"Final evaluation at count={optimal_count}"):
        original_image, _, original_bboxes = dataset[idx]
        # Preprocessing (same as before)
        original_size = (original_image.width, original_image.height)
        image = original_image.resize((224, 224), Image.LANCZOS)
        bboxes = [resize_bbox(bbox, original_size) for bbox in original_bboxes]
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        gt_heatmap = generate_gt_heatmap((224, 224), bboxes)
        
        bing_heatmap = generate_bing_heatmap(image_cv, saliency, optimal_count)
        iou = calculate_iou(gt_heatmap, bing_heatmap)
        mse = calculate_mse(gt_heatmap, bing_heatmap)
        
        final_iou.append(iou)
        final_mse.append(mse)
    
    return optimal_count, {
        'iou_mean': np.mean(final_iou),
        'iou_std': np.std(final_iou),
        'mse_mean': np.mean(final_mse),
        'mse_std': np.std(final_mse)
    }

def main():
    # Paths configuration
    data_root = Path("/storage/scratch/6403840/data/imagenet-tiny")
    model_path = "/storage/scratch/6403840/data/BING_models"
    output_dir = Path("/storage/scratch/6403840/data/bing_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.setLogLevel(2)
    
    # Parameters
    num_images = 200  # Number of images to evaluate
    
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
    
    # Get class-balanced sample
    print("Selecting class-balanced sample of images...")
    indices = get_balanced_sample_indices(dataset, num_images)
    print(f"Selected {len(indices)} images across {len(set(dataset[idx][1] for idx in indices))} classes")
    
    # Find optimal detection count
    print("Starting adaptive optimization...")
    optimal_count, metrics = optimize_detection_count(
        dataset, indices, saliency, min_count=10, max_count=2000
    )
    
    print(f"\nResults:")
    print(f"Optimal detection count: {optimal_count}")
    print(f"IoU: {metrics['iou_mean']:.4f} ± {metrics['iou_std']:.4f}")
    print(f"MSE: {metrics['mse_mean']:.4f} ± {metrics['mse_std']:.4f}")

    # Save results to file
    with open(output_dir / "optimization_results.txt", "w") as f:
        f.write(f"Optimal detection count: {optimal_count}\n")
        f.write(f"IoU: {metrics['iou_mean']:.4f} ± {metrics['iou_std']:.4f}\n")
        f.write(f"MSE: {metrics['mse_mean']:.4f} ± {metrics['mse_std']:.4f}\n")

if __name__ == "__main__":
    main()
