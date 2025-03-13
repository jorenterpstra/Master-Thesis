import os
import random
import shutil
import argparse
from pathlib import Path
import json
from tqdm import tqdm

def create_imagenet_tiny(
    source_imagenet_path,
    target_path,
    num_classes=200,
    num_train_images=100000,
    num_val_images=10000,
):
    """
    Create a smaller version of ImageNet with specified parameters.
    
    Args:
        source_imagenet_path: Path to the full ImageNet dataset
        target_path: Path where the new dataset will be created
        num_classes: Number of classes to include
        num_train_images: Total number of training images
        num_val_images: Total number of validation images
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
    
    # Process training images by class
    for class_dir in tqdm(selected_classes, desc="Processing training classes"):
        class_name = class_dir.name
        
        # Create target class directories
        target_train_class_dir = target_train_dir / class_name
        target_train_bbox_class_dir = target_train_bbox_dir / class_name
        
        target_train_class_dir.mkdir(exist_ok=True)
        target_train_bbox_class_dir.mkdir(exist_ok=True)
        
        # Get all training images for this class
        train_images = list(class_dir.glob('*.JPEG'))
        
        # Select random training images
        selected_train_images = random.sample(
            train_images, 
            min(train_images_per_class, len(train_images))
        )
        
        # Copy training images and their bounding boxes
        for img_path in selected_train_images:
            # Copy image
            target_img_path = target_train_class_dir / img_path.name
            shutil.copy2(img_path, target_img_path)
            
            # Copy corresponding bbox file if it exists
            bbox_path = source_path / 'train_bbox' / class_name / f"{img_path.stem}.xml"
            if bbox_path.exists():
                target_bbox_path = target_train_bbox_class_dir / f"{img_path.stem}.xml"
                shutil.copy2(bbox_path, target_bbox_path)
            
            total_train_copied += 1
    
    # Process validation images
    print("Processing validation images...")
    
    # Check if validation images are in class subfolders (like training) or in a flat structure
    val_has_subfolders = any((source_path / 'val').glob('*/'))
    
    # Collect validation images for selected classes
    val_images_by_class = {cls: [] for cls in selected_class_names}
    
    if val_has_subfolders:
        # If validation images are organized in class folders
        for class_name in selected_class_names:
            val_class_dir = source_path / 'val' / class_name
            if val_class_dir.exists() and val_class_dir.is_dir():
                val_images = list(val_class_dir.glob('*.JPEG'))
                val_images_by_class[class_name].extend(val_images)
    else:
        # If validation images are in a flat structure with a mapping file
        # This assumes there's some way to know which class each validation image belongs to
        # For example, through a mapping file or naming convention
        
        # Check for validation class mapping files (common formats)
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
            for img_path in (source_path / 'val').glob('*.JPEG'):
                img_name = img_path.name
                if img_name in val_img_to_class and val_img_to_class[img_name] in selected_class_names:
                    class_name = val_img_to_class[img_name]
                    val_images_by_class[class_name].append(img_path)
        else:
            print("Warning: No validation mapping file found. Using folder structure to infer classes.")
            # Try to infer class from image name or another method
            # This is just a fallback approach and may not work for all datasets
            val_images = list((source_path / 'val').glob('*.JPEG'))
            
            # Distribute validation images evenly among selected classes
            # (This is a simplification and not ideal for a real scenario)
            for i, img_path in enumerate(val_images):
                class_idx = i % len(selected_class_names)
                class_name = selected_class_names[class_idx]
                val_images_by_class[class_name].append(img_path)
    
    # Copy validation images
    for class_name, images in tqdm(val_images_by_class.items(), desc="Copying validation images"):
        if not images:
            continue
            
        # Select random validation images for this class
        selected_val_images = random.sample(
            images,
            min(val_images_per_class, len(images))
        )
        
        # If validation images are in class subfolders, maintain that structure
        if val_has_subfolders:
            target_val_class_dir = target_val_dir / class_name
            target_val_class_dir.mkdir(exist_ok=True)
            
            for img_path in selected_val_images:
                target_img_path = target_val_class_dir / img_path.name
                shutil.copy2(img_path, target_img_path)
                
                # Copy bbox if available
                if val_has_subfolders:
                    bbox_path = source_path / 'val_bbox' / class_name / f"{img_path.stem}.xml"
                else:
                    bbox_path = source_path / 'val_bbox' / f"{img_path.stem}.xml"
                
                if bbox_path.exists():
                    if val_has_subfolders:
                        target_bbox_dir = target_val_bbox_dir / class_name
                        target_bbox_dir.mkdir(exist_ok=True)
                        target_bbox_path = target_bbox_dir / f"{img_path.stem}.xml"
                    else:
                        target_bbox_path = target_val_bbox_dir / f"{img_path.stem}.xml"
                    
                    shutil.copy2(bbox_path, target_bbox_path)
                
                total_val_copied += 1
        else:
            # For flat validation structure
            for img_path in selected_val_images:
                target_img_path = target_val_dir / img_path.name
                shutil.copy2(img_path, target_img_path)
                
                # Copy bbox if available
                bbox_path = source_path / 'val_bbox' / f"{img_path.stem}.xml"
                if bbox_path.exists():
                    target_bbox_path = target_val_bbox_dir / f"{img_path.stem}.xml"
                    shutil.copy2(bbox_path, target_bbox_path)
                
                total_val_copied += 1
    
    print(f"Dataset creation complete!")
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
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    print(args)
    create_imagenet_tiny(
        args.source,
        args.target,
        args.classes,
        args.train_images,
        args.val_images
    )
