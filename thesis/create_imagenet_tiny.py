import os
import random
import shutil
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import hashlib
import sys
from collections import defaultdict

def compute_file_hash(file_path, chunk_size=8192):
    """Compute MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def find_all_files(directory, extensions):
    """Find all files with specified extensions in directory and subdirectories.
    Returns a dictionary mapping filenames to their full paths."""
    directory = Path(directory)
    file_dict = {}
    
    for root, _, files in os.walk(directory):
        for file in files:
            if extensions is None or any(file.lower().endswith(ext.lower()) for ext in extensions):
                file_dict[file] = os.path.join(root, file)
    return file_dict

def check_data_leakage(train_dir, val_dir, extensions=None):
    """Check for filename duplicates between training and validation sets."""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.JPEG']
        
    train_files = find_all_files(train_dir, extensions)
    val_files = find_all_files(val_dir, extensions)
    
    # Find duplicates (same filename in both sets)
    duplicates = set(train_files.keys()).intersection(set(val_files.keys()))
    
    return bool(duplicates), duplicates

def create_imagenet_tiny(
    source_imagenet_path,
    target_path,
    num_classes=200,
    num_train_images=100000,
    num_val_images=10000,
    enforce_no_leakage=True
):
    """
    Create a smaller version of ImageNet with specified parameters.
    
    Args:
        source_imagenet_path: Path to the full ImageNet dataset
        target_path: Path where the new dataset will be created
        num_classes: Number of classes to include
        num_train_images: Total number of training images
        num_val_images: Total number of validation images
        enforce_no_leakage: If True, strictly enforce no data leakage
    """
    source_path = Path(source_imagenet_path)
    target_path = Path(target_path)
    
    # Create target directories
    target_train_dir = target_path / 'train'
    target_val_dir = target_path / 'val'
    target_train_bbox_dir = target_path / 'train_bbox'
    target_val_bbox_dir = target_path / 'val_bbox'
    
    for directory in [target_path, target_train_dir, target_val_dir, target_train_bbox_dir, target_val_bbox_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Get all class folders in the training set
    print("Getting all class directories...")
    train_class_dirs = [d for d in (source_path / 'train').glob('*') if d.is_dir()]
    
    if len(train_class_dirs) < num_classes:
        raise ValueError(f"Source dataset contains only {len(train_class_dirs)} classes, but {num_classes} were requested")
    
    # Select random classes
    selected_classes = random.sample(train_class_dirs, num_classes)
    selected_class_names = [cls_dir.name for cls_dir in selected_classes]
    
    # Calculate images per class
    train_images_per_class = num_train_images // num_classes
    val_images_per_class = num_val_images // num_classes
    
    print(f"Creating ImageNet Tiny with {num_classes} classes")
    print(f"Training: {num_train_images} images total, ~{train_images_per_class} per class")
    print(f"Validation: {num_val_images} images total, ~{val_images_per_class} per class")
    
    # Save the selected class IDs for reference
    with open(target_path / "selected_classes.json", "w") as f:
        json.dump(selected_class_names, f)
    
    # Track progress
    total_train_copied = 0
    total_val_copied = 0
    
    # Track file hashes to prevent duplicates
    all_file_hashes = set()
    filename_registry = set()
    
    # First, collect all validation images to exclude from training
    print("Collecting validation images first to prevent data leakage...")
    val_images_to_copy = {}  # class_name -> [image_paths]
    val_has_subfolders = any((source_path / 'val').glob('*/'))
    
    # Dictionary to track all filenames by class
    all_filenames_by_class = defaultdict(set)
    
    # VALIDATION IMAGE COLLECTION
    if val_has_subfolders:
        # If validation images are organized in class folders
        for class_name in selected_class_names:
            val_class_dir = source_path / 'val' / class_name
            if val_class_dir.exists() and val_class_dir.is_dir():
                val_images = list(val_class_dir.glob('*.JPEG'))
                
                # Add all filenames to registry for this class
                for img_path in val_images:
                    all_filenames_by_class[class_name].add(img_path.name)
                
                # Select random validation images
                selected_val = random.sample(
                    val_images, 
                    min(val_images_per_class, len(val_images))
                )
                val_images_to_copy[class_name] = selected_val
    else:
        # Handle flat validation structure with mapping file
        # This code remains largely the same as the original, but we collect images to be used
        # Instead of copying them immediately
        
        # Check for validation class mapping files
        mapping_paths = [
            source_path / "val_annotations.txt",
            source_path / "val_map.txt",
            source_path / "val_labels.json"
        ]
        
        mapping_file = next((p for p in mapping_paths if p.exists()), None)
        
        if mapping_file:
            print(f"Using validation mapping file: {mapping_file}")
            
            # Load mapping based on file extension
            val_img_to_class = {}
            
            if mapping_file.suffix == '.json':
                with open(mapping_file) as f:
                    val_img_to_class = json.load(f)
            else:
                # Assume text file with image_name class_id format
                with open(mapping_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            img_name, class_id = parts[0], parts[1]
                            val_img_to_class[img_name] = class_id
            
            # Group validation images by class
            all_val_images_by_class = defaultdict(list)
            for img_path in (source_path / 'val').glob('*.JPEG'):
                img_name = img_path.name
                if img_name in val_img_to_class and val_img_to_class[img_name] in selected_class_names:
                    class_name = val_img_to_class[img_name]
                    all_val_images_by_class[class_name].append(img_path)
                    all_filenames_by_class[class_name].add(img_name)
            
            # Select random validation images from each class
            for class_name in selected_class_names:
                images = all_val_images_by_class.get(class_name, [])
                if images:
                    selected_val = random.sample(
                        images,
                        min(val_images_per_class, len(images))
                    )
                    val_images_to_copy[class_name] = selected_val
        else:
            print("Warning: No validation mapping file found. Using folder structure to infer classes.")
            # Simplified approach - distribute validation images evenly
            val_images = list((source_path / 'val').glob('*.JPEG'))
            
            # Track all validation filenames
            val_filenames = {img_path.name for img_path in val_images}
            
            # Distribute validation images evenly among selected classes
            images_per_class = {}
            for class_name in selected_class_names:
                images_per_class[class_name] = []
            
            for i, img_path in enumerate(val_images):
                class_idx = i % len(selected_class_names)
                class_name = selected_class_names[class_idx]
                images_per_class[class_name].append(img_path)
                all_filenames_by_class[class_name].add(img_path.name)
            
            # Select a subset for each class
            for class_name in selected_class_names:
                images = images_per_class[class_name]
                if images:
                    selected_val = random.sample(
                        images,
                        min(val_images_per_class // len(selected_class_names), len(images))
                    )
                    val_images_to_copy[class_name] = selected_val
    
    # TRAINING IMAGE SELECTION (excluding validation filenames)
    print("Processing training classes (avoiding validation images)...")
    train_images_to_copy = {}  # class_name -> [image_paths]
    
    for class_dir in tqdm(selected_classes, desc="Selecting training images"):
        class_name = class_dir.name
        
        # Get all training images for this class
        all_train_images = list(class_dir.glob('*.JPEG'))
        
        # Get validation filenames for this class to exclude
        validation_filenames = all_filenames_by_class.get(class_name, set())
        
        # Filter out training images that have the same filename as validation images
        filtered_train_images = [
            img for img in all_train_images 
            if img.name not in validation_filenames
        ]
        
        if not filtered_train_images:
            print(f"Warning: No training images available for class {class_name} after filtering out validation filenames")
            continue
            
        # Select random training images
        selected_train = random.sample(
            filtered_train_images,
            min(train_images_per_class, len(filtered_train_images))
        )
        
        train_images_to_copy[class_name] = selected_train
    
    # COPY VALIDATION IMAGES
    print("Copying validation images...")
    for class_name, images in tqdm(val_images_to_copy.items(), desc="Copying validation images"):
        if not images:
            continue
            
        # Create class directories if using subfolders
        if val_has_subfolders:
            target_val_class_dir = target_val_dir / class_name
            target_val_class_dir.mkdir(exist_ok=True)
            
            target_val_bbox_class_dir = target_val_bbox_dir / class_name
            target_val_bbox_class_dir.mkdir(exist_ok=True)
        
        for img_path in images:
            # Check for filename uniqueness
            if img_path.name in filename_registry:
                if enforce_no_leakage:
                    print(f"Warning: Duplicate filename detected: {img_path.name}. Skipping to prevent leakage.")
                    continue
            
            # For content-based duplication check
            if enforce_no_leakage:
                img_hash = compute_file_hash(img_path)
                if img_hash in all_file_hashes:
                    print(f"Warning: Duplicate image content detected: {img_path.name}. Skipping to prevent leakage.")
                    continue
                all_file_hashes.add(img_hash)
            
            # Add filename to registry
            filename_registry.add(img_path.name)
            
            # Determine target paths
            if val_has_subfolders:
                target_img_path = target_val_dir / class_name / img_path.name
                bbox_path = source_path / 'val_bbox' / class_name / f"{img_path.stem}.xml"
                target_bbox_path = target_val_bbox_dir / class_name / f"{img_path.stem}.xml"
            else:
                target_img_path = target_val_dir / img_path.name
                bbox_path = source_path / 'val_bbox' / f"{img_path.stem}.xml"
                target_bbox_path = target_val_bbox_dir / f"{img_path.stem}.xml"
            
            # Copy image
            shutil.copy2(img_path, target_img_path)
            total_val_copied += 1
            
            # Copy bbox if available
            if bbox_path.exists():
                shutil.copy2(bbox_path, target_bbox_path)
    
    # COPY TRAINING IMAGES
    print("Copying training images...")
    for class_name, images in tqdm(train_images_to_copy.items(), desc="Copying training images"):
        if not images:
            continue
        
        # Create target class directories
        target_train_class_dir = target_train_dir / class_name
        target_train_bbox_class_dir = target_train_bbox_dir / class_name
        
        target_train_class_dir.mkdir(exist_ok=True)
        target_train_bbox_class_dir.mkdir(exist_ok=True)
        
        for img_path in images:
            # Check for filename uniqueness
            if img_path.name in filename_registry:
                if enforce_no_leakage:
                    print(f"Warning: Duplicate filename detected: {img_path.name}. Skipping to prevent leakage.")
                    continue
            
            # For content-based duplication check
            if enforce_no_leakage:
                img_hash = compute_file_hash(img_path)
                if img_hash in all_file_hashes:
                    print(f"Warning: Duplicate image content detected: {img_path.name}. Skipping to prevent leakage.")
                    continue
                all_file_hashes.add(img_hash)
            
            # Add filename to registry
            filename_registry.add(img_path.name)
            
            # Copy image
            target_img_path = target_train_class_dir / img_path.name
            shutil.copy2(img_path, target_img_path)
            total_train_copied += 1
            
            # Copy corresponding bbox file if it exists
            bbox_path = source_path / 'train_bbox' / class_name / f"{img_path.stem}.xml"
            if bbox_path.exists():
                target_bbox_path = target_train_bbox_class_dir / f"{img_path.stem}.xml"
                shutil.copy2(bbox_path, target_bbox_path)
    
    # Verify no data leakage occurred
    if enforce_no_leakage:
        print("\nVerifying no data leakage...")
        has_duplicates, duplicates = check_data_leakage(target_train_dir, target_val_dir, ['.JPEG', '.jpeg'])
        
        if has_duplicates:
            print(f"ERROR: Data leakage detected! Found {len(duplicates)} duplicate filenames.")
            print("First few duplicates:")
            for dup in list(duplicates)[:5]:
                print(f"  {dup}")
            
            if "--force" not in sys.argv:
                print("\nTo keep the dataset despite leakage, run with --force flag.")
                print("Cleaning up created dataset...")
                shutil.rmtree(target_path)
                sys.exit(1)
        else:
            print("✅ Verification complete: No data leakage detected!")
    
    print(f"\nDataset creation complete!")
    print(f"Copied {total_train_copied} training images and {total_val_copied} validation images")
    print(f"Selected {num_classes} classes: {', '.join(selected_class_names[:5])}... (see selected_classes.json for complete list)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ImageNet Tiny dataset with 200 classes")
    parser.add_argument("--source", required=True, help="Path to full ImageNet dataset")
    parser.add_argument("--target", required=True, help="Path for the new ImageNet Tiny dataset")
    parser.add_argument("--classes", type=int, default=200, help="Number of classes to include")
    parser.add_argument("--train-images", type=int, default=100000, help="Number of training images")
    parser.add_argument("--val-images", type=int, default=10000, help="Number of validation images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-leakage-check", action="store_false", dest="enforce_no_leakage",
                       help="Skip strict leakage prevention checks (not recommended)")
    parser.add_argument("--force", action="store_true", help="Force dataset creation even if leakage is detected")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    print(args)
    
    create_imagenet_tiny(
        args.source,
        args.target,
        args.classes,
        args.train_images,
        args.val_images,
        args.enforce_no_leakage
    )
