#!/usr/bin/env python3
import cv2
import numpy as np
import os
import argparse
import csv
from glob import glob
from PIL import Image  # Add PIL for better image resizing

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

def sliding_window_scores(heatmap, patch_size, stride):
    """
    Compute patch scores for all patch positions in the heatmap using an integral image.
    The score for a patch is the sum over the heatmap in that region.
    
    Args:
        heatmap (np.ndarray): 2D array (H, W) of heatmap values.
        patch_size (int): Size (width and height) of each patch.
        stride (int): The stride with which patches are extracted.
    
    Returns:
        patch_scores (np.ndarray): Array of shape (num_patches_y, num_patches_x) with sums.
    """
    H, W = heatmap.shape
    # Compute the integral image. cv2.integral returns an image with one extra row/column.
    I = cv2.integral(heatmap)  # shape: (H+1, W+1)
    
    # Determine the number of patch positions along y and x.
    num_patches_y = (H - patch_size) // stride + 1
    num_patches_x = (W - patch_size) // stride + 1
    
    # Create grid of top-left patch coordinates (vectorized).
    ys = (np.arange(num_patches_y) * stride).reshape(-1, 1)
    xs = (np.arange(num_patches_x) * stride).reshape(1, -1)
    
    # For every patch starting at (y, x), the sum can be computed from the integral image.
    # Using vectorized slicing: sum = I[y+patch_size, x+patch_size] - I[y, x+patch_size] - I[y+patch_size, x] + I[y, x]
    # Use broadcasting to compute a full grid.
    sum1 = I[ys + patch_size, xs + patch_size]
    sum2 = I[ys, xs + patch_size]
    sum3 = I[ys + patch_size, xs]
    sum4 = I[ys, xs]
    
    patch_scores = sum1 - sum2 - sum3 + sum4  # shape: (num_patches_y, num_patches_x)
    return patch_scores

def generate_patch_ranking(image, patch_size, stride, num_bboxes=1000, training_path=None):
    """
    Generate a ranking (ordering) of patch indices based on BING saliency.
    This function attempts to use OpenCV's BING. If training_path is provided,
    it is used to initialize BING; otherwise, you might add a fallback.
    
    The ranking is computed by:
      - Running BING to obtain bounding boxes.
      - Creating a heatmap by "painting" each bbox (adding a constant).
      - Using a fast integral image method to compute per-patch sums.
      - Sorting patches by descending score.
      
    Args:
        image (np.ndarray): Input image array (H, W, C); colors can be BGR or RGB.
        patch_size (int): The patch (square) size.
        stride (int): The extraction stride.
        num_bboxes (int): Maximum number of boxes to use (if more are returned, take the first num_bboxes).
        training_path (str): Path to the BING training data. Required for BING.
        
    Returns:
        ranking (np.ndarray): 1D array of patch indices sorted by descending saliency.
                              The ranking is relative to the total number of patches.
    """
    # Initialize BING saliency detector if training_path provided.
    use_bing = training_path is not None
    if use_bing:
        bing_detector = cv2.saliency.ObjectnessBING_create()
        bing_detector.setTrainingPath(training_path)
    
        # Create heatmap from bounding boxes.
        heatmap = generate_bing_heatmap(image, bing_detector, num_bboxes)
    else:
        raise ValueError("BING training path is required to generate heatmap.")
    
    # Use an integral image to efficiently compute sliding window patch sums.
    patch_scores_grid = sliding_window_scores(heatmap, patch_size, stride)
    
    # Flatten the patch scores into a 1D array.
    patch_scores = patch_scores_grid.flatten()
    
    # Higher score means more "saliency". We want descending order.
    ranking = np.argsort(-patch_scores)
    
    return ranking

def resize_image(img_path, target_size=(224, 224)):
    """
    Read an image file and resize it to the target size.
    
    Args:
        img_path (str): Path to the image file
        target_size (tuple): Target size as (width, height)
        
    Returns:
        opencv_image: OpenCV compatible image in BGR format
    """
    try:
        # Read image using PIL (which handles various formats better)
        pil_image = Image.open(img_path).convert('RGB')
        
        # Get original size for potential future use
        original_size = (pil_image.width, pil_image.height)
        
        # Resize image to target size using LANCZOS resampling for better quality
        resized_image = pil_image.resize(target_size, Image.LANCZOS)
        
        # Convert PIL image to numpy array
        image_np = np.array(resized_image)
        
        # Convert RGB to BGR for OpenCV compatibility
        opencv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        return opencv_image
    except Exception as e:
        print(f"Error resizing image {img_path}: {e}")
        return None

def process_class_folder(class_folder, output_dir, patch_size, stride, training_path, num_bboxes, exts=('jpg','jpeg','png')):
    """
    Process all images in a single class folder: generate patch rankings using BING saliency,
    then write the rankings into a class-specific CSV file.
    
    Args:
        class_folder (str): Path to the class folder containing images
        output_dir (str): Directory to save the output CSV file
        patch_size (int): Patch size
        stride (int): Stride
        training_path (str): Path for BING training data
        num_bboxes (int): Maximum number of bounding boxes to consider
        exts (tuple): Image file extensions to consider
    
    Returns:
        str: Path to the created CSV file
    """
    # Extract class name from folder path
    class_name = os.path.basename(class_folder)
    
    # Create output CSV path
    output_csv = os.path.join(output_dir, f"{class_name}_rankings.csv")
    
    # Gather all image file paths in this class folder
    images = []
    for ext in exts:
        images.extend(glob(os.path.join(class_folder, f'*.{ext}')))
    images = sorted(images)
    
    print(f"Processing class {class_name}: Found {len(images)} images.")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header with clear column names
        csv_writer.writerow(["image_filename", "rankings"])
        
        for idx, img_path in enumerate(images):
            # Read and resize the image to 224x224
            image = resize_image(img_path, target_size=(224, 224))
            
            if image is None:
                print(f"Failed to load or resize image: {img_path}")
                continue
            
            # Generate the patch ranking using BING saliency
            try:
                ranking = generate_patch_ranking(image, patch_size, stride, 
                                                num_bboxes=num_bboxes, 
                                                training_path=training_path)
                # print(ranking)
                # Convert the ranking array to a list of integers
                ranking_list = ranking.tolist()
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
                
            # Extract just the filename without path for clearer identification
            filename = os.path.basename(img_path)
            
            # Convert ranking list to comma-separated string for more compact storage
            rankings_str = ','.join(map(str, ranking_list))
            csv_writer.writerow([filename, rankings_str])
            
            if idx % 10 == 0 and idx > 0:
                print(f"  - Processed {idx}/{len(images)} images in {class_name}...")
    
    print(f"Completed class {class_name}: Ranking CSV written to {output_csv}")
    return output_csv

def process_imagenet_dataset(data_dir, output_dir, patch_size, stride, training_path, num_bboxes):
    """
    Process the ImageNet-style dataset: identify all class folders and process each one.
    
    Args:
        data_dir (str): Root directory of the ImageNet-style dataset with class folders
        output_dir (str): Directory to save the output CSV files
        patch_size (int): Patch size
        stride (int): Stride
        training_path (str): Path for BING training data
        num_bboxes (int): Maximum number of bounding boxes to consider
    """
    # Get all subdirectories (class folders) in the data_dir
    class_folders = [d for d in glob(os.path.join(data_dir, '*')) if os.path.isdir(d)]
    
    if not class_folders:
        print(f"No class folders found in {data_dir}")
        return
    
    print(f"Found {len(class_folders)} class folders in {data_dir}")
    
    # Process each class folder
    output_csvs = []
    for i, class_folder in enumerate(class_folders):
        print(f"Processing class folder {i+1}/{len(class_folders)}: {class_folder}")
        csv_path = process_class_folder(
            class_folder, output_dir, patch_size, stride, training_path, num_bboxes
        )
        output_csvs.append(csv_path)
    
    # Create a metadata file that lists all class CSVs
    metadata_path = os.path.join(output_dir, "class_rankings_metadata.csv")
    with open(metadata_path, 'w', newline='') as meta_file:
        meta_writer = csv.writer(meta_file)
        meta_writer.writerow(["class_name", "class_path", "ranking_csv", "image_count"])
        
        for class_folder, csv_path in zip(class_folders, output_csvs):
            class_name = os.path.basename(class_folder)
            # Count images in this class
            image_count = 0
            for ext in ('jpg', 'jpeg', 'png'):
                image_count += len(glob(os.path.join(class_folder, f'*.{ext}')))
                
            meta_writer.writerow([
                class_name, 
                class_folder, 
                os.path.basename(csv_path),
                image_count
            ])
    
    # Create a summary file that contains overall dataset statistics
    summary_path = os.path.join(output_dir, "rankings_summary.txt")
    with open(summary_path, 'w') as summary_file:
        summary_file.write(f"ImageFolder Dataset Rankings Summary\n")
        summary_file.write(f"==============================\n")
        summary_file.write(f"Dataset path: {data_dir}\n")
        summary_file.write(f"Number of classes: {len(class_folders)}\n")
        summary_file.write(f"Patch size: {patch_size}\n")
        summary_file.write(f"Stride: {stride}\n")
        summary_file.write(f"BING training path: {training_path}\n")
        summary_file.write(f"Number of bounding boxes: {num_bboxes}\n")
        summary_file.write(f"Generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
    print(f"Completed processing all classes. Metadata CSV written to {metadata_path}")
    print(f"Summary information written to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate patch rankings using BING saliency.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing images (recursively searched).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the CSV files of patch rankings.")
    parser.add_argument("--patch_size", type=int, default=16,
                        help="Patch size (square).")
    parser.add_argument("--stride", type=int, default=16,
                        help="Stride for patch extraction.")
    parser.add_argument("--training_path", type=str, required=True,
                        help="Path to BING training data needed by ObjectnessBING.")
    parser.add_argument("--num_bboxes", type=int, default=500,
                        help="Maximum number of bounding boxes to consider.")
    
    args = parser.parse_args()
    
    process_imagenet_dataset(args.data_dir, args.output_dir, args.patch_size, args.stride,
                             args.training_path, args.num_bboxes)
