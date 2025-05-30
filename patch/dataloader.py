import torch.nn as nn
import torch
import os
import tarfile
from pathlib import Path
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import xml.etree.ElementTree as ET

class BoundingBoxTransform:
    """Helper class to transform bounding boxes along with images"""
    @staticmethod
    def resize(bbox, orig_size, new_size):
        x, y, w, h = bbox
        scale_x = new_size[0] / orig_size[0]
        scale_y = new_size[1] / orig_size[1]
        return [x * scale_x, y * scale_y, w * scale_x, h * scale_y]
    
    @staticmethod
    def center_crop(bbox, img_size, crop_size):
        x, y, w, h = bbox
        left = (img_size[0] - crop_size[0]) // 2
        top = (img_size[1] - crop_size[1]) // 2
        
        # Adjust coordinates relative to crop
        x = x - left
        y = y - top
        
        # Clip to crop boundaries
        x = max(0, min(x, crop_size[0]))
        y = max(0, min(y, crop_size[1]))
        w = max(0, min(w, crop_size[0] - x))
        h = max(0, min(h, crop_size[1] - y))
        
        return [x, y, w, h]

def generate_scores(image_size, bboxes, patch_size=16):
    """
    Generate scores for each patch based on:
    1. Inside bounding box: Positive score based on proximity to center
    2. Outside bounding box: Negative score based on distance to nearest bbox CENTER
    Using vectorized operations for better performance.
    """
    if isinstance(image_size, tuple):
        image_size = min(image_size)

    # Ensure integer division for correct patch count
    num_patches = int(image_size // patch_size)  # e.g., 224 // 16 = 14
    num_boxes = len(bboxes)
    
    # Initialize scores tensor with correct dimensions
    scores = torch.zeros((num_patches, num_patches))
    
    # Create patch grid coordinates (use exact patch boundaries)
    patch_centers = torch.arange(num_patches) * patch_size + patch_size / 2
    
    # Create meshgrid of patch centers
    grid_y, grid_x = torch.meshgrid(patch_centers, patch_centers, indexing='ij')
    
    # Track which patches fall inside at least one bounding box
    has_intersection = torch.zeros((num_patches, num_patches), dtype=torch.bool)
    
    # Track distances from each patch to nearest bbox center
    min_distances_to_center = torch.ones((num_patches, num_patches)) * float('inf')
    
    # Process each bbox
    for bbox in bboxes:
        x, y, w, h = map(float, bbox)
        
        # Ensure bbox coordinates are within image bounds
        x = min(max(x, 0), image_size - 1)
        y = min(max(y, 0), image_size - 1)
        w = min(w, image_size - x)
        h = min(h, image_size - y)
        
        # Calculate bbox center
        center_x = x + w / 2
        center_y = y + h / 2
        
        # Calculate distances from patch centers to bbox center
        dx = (grid_x - center_x) / w if w > 0 else (grid_x - center_x)
        dy = (grid_y - center_y) / h if h > 0 else (grid_y - center_y)
        distances = torch.sqrt(dx**2 + dy**2)
        
        # Calculate distances from patch centers to bbox center (in pixels, not normalized)
        pixel_dx = grid_x - center_x
        pixel_dy = grid_y - center_y
        pixel_distances = torch.sqrt(pixel_dx**2 + pixel_dy**2)
        
        # Update minimum distances to any bbox center
        min_distances_to_center = torch.minimum(min_distances_to_center, pixel_distances)
        
        # Calculate Gaussian-like center weights with wider spread 
        # to create more distinctive values inside bounding boxes
        center_weights = torch.exp(-distances**2 / 0.45)  # Value of 0.45 provides good balance
        
        # Calculate intersection areas
        x_min = torch.maximum(grid_x - patch_size/2, torch.tensor(x))
        y_min = torch.maximum(grid_y - patch_size/2, torch.tensor(y))
        x_max = torch.minimum(grid_x + patch_size/2, torch.tensor(x + w))
        y_max = torch.minimum(grid_y + patch_size/2, torch.tensor(y + h))
        
        # Calculate areas
        widths = torch.maximum(x_max - x_min, torch.tensor(0.0))
        heights = torch.maximum(y_max - y_min, torch.tensor(0.0))
        intersections = widths * heights
        
        # Calculate intersection ratios
        intersection_ratios = intersections / (patch_size * patch_size)
        
        # Track patches that have intersection with any bounding box
        has_intersection = has_intersection | (intersection_ratios > 0)
        
        # Combine intersection ratios with center weights
        patch_scores = intersection_ratios * center_weights
        
        scores += patch_scores
    
    # Assign negative scores to patches outside all bounding boxes
    # The further from any center, the more negative the score
    outside_mask = ~has_intersection
    if outside_mask.any():
        # Normalize distances by image diagonal for consistent scaling
        diagonal = torch.sqrt(torch.tensor(image_size**2 + image_size**2, dtype=torch.float))
        normalized_distances = min_distances_to_center[outside_mask] / diagonal
        
        # Create negative scores that decrease as distance increases
        # Using a non-linear curve to create more differentiation
        negative_scores = -0.5 * normalized_distances**1.5  # Added exponent for more aggressive falloff
        
        scores[outside_mask] = negative_scores
    
    # Soft normalization as a compromise:
    # - Preserves more of the natural distribution
    # - Still keeps values in consistent ranges
    # - Adds small buffers to avoid exact boundary values
    if scores.min() < 0 and scores.max() > 0:
        # Handle positive scores - normalize to [0, 1] with buffer
        pos_mask = scores > 0
        if pos_mask.any():
            pos_min = scores[pos_mask].min()
            pos_max = scores[pos_mask].max()
            
            if pos_max > pos_min:  # Avoid division by zero
                # Calculate normalization with 10% buffers on each end
                buffer = 0.1 * (pos_max - pos_min)
                
                # Scale to range [0.05, 0.95] instead of [0, 1]
                scores[pos_mask] = 0.05 + 0.9 * (scores[pos_mask] - pos_min) / (pos_max - pos_min)
            else:
                # All positive scores are equal
                scores[pos_mask] = 0.5  # Set to middle of range
        
        # Handle negative scores - normalize to [-1, 0] with buffer
        neg_mask = scores < 0
        if neg_mask.any():
            neg_min = scores[neg_mask].min()
            neg_max = scores[neg_mask].max()
            
            if neg_min < neg_max:  # Avoid division by zero
                # Calculate normalization with 10% buffers on each end
                buffer = 0.1 * (neg_max - neg_min)
                
                # Scale to range [-0.95, -0.05] instead of [-1, 0]
                scores[neg_mask] = -0.95 + 0.9 * (scores[neg_mask] - neg_min) / (neg_max - neg_min)
            else:
                # All negative scores are equal
                scores[neg_mask] = -0.5  # Set to middle of range
    
    return scores.reshape(-1)

class ImageNetPatchRankLoader(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', transform=None, verbose=0, return_boxes=False):
        """
        Args:
            root_dir: Path to ImageNet dataset
            split: 'train' or 'val'
            transform: Optional transform to be applied on images
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.verbose = verbose
        self.return_boxes = return_boxes
        
        # Setup paths
        self.image_dir = self.root_dir / split
        self.bbox_dir = self.root_dir / f"{split}_bbox"
        
        # Get all image paths and their corresponding bbox files
        self.image_paths = []
        self.bbox_paths = []
        
        if split == 'train':
            # Training set: bbox files are in class subfolders
            for class_dir in self.image_dir.glob('*'):
                if not class_dir.is_dir():
                    continue
                    
                # Get corresponding bbox class directory
                bbox_class_dir = self.bbox_dir / class_dir.name
                
                # Find all images and their corresponding bbox files
                for img_path in class_dir.glob('*.JPEG'):
                    # Construct bbox path with same structure
                    bbox_path = bbox_class_dir / f"{img_path.stem}.xml"
                    
                    if bbox_path.exists():
                        self.image_paths.append(img_path)
                        self.bbox_paths.append(bbox_path)
        else:
            # Validation set: bbox files are all in the root bbox directory
            for img_path in self.image_dir.glob('**/*.JPEG'):
                # Construct bbox path directly in bbox directory
                bbox_path = self.bbox_dir / f"{img_path.stem}.xml"
                
                if bbox_path.exists():
                    self.image_paths.append(img_path)
                    self.bbox_paths.append(bbox_path)
                    
        print(f"Found {len(self.image_paths)} images with bounding box annotations in {split} set")

    def parse_bbox(self, xml_path, image_size):
        """
        Parse ImageNet VOC-style annotation XML file
        Args:
            xml_path: path to XML annotation file
            image_size: tuple (width, height) of the current image size
        Returns:
            list of bboxes: each [x, y, w, h] scaled to current image size
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find('size')
        if size is None:
            return None
        
        try:
            orig_width = float(size.find('width').text)
            orig_height = float(size.find('height').text)
        except (AttributeError, ValueError):
            return None

        # Calculate scale factors for both dimensions
        if isinstance(image_size, tuple):
            scale_x = image_size[0] / orig_width
            scale_y = image_size[1] / orig_height
        else:
            # If single size provided, maintain aspect ratio
            scale = min(image_size / orig_width, image_size / orig_height)
            scale_x = scale
            scale_y = scale

        valid_boxes = []
        objects = root.findall('object')
        
        for obj in objects:
            # Skip difficult objects
            difficult = obj.find('difficult')
            if difficult is not None and int(difficult.text) == 1:
                continue
                
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                try:
                    # Get coordinates
                    xmin = max(0, float(bndbox.find('xmin').text))
                    ymin = max(0, float(bndbox.find('ymin').text))
                    xmax = min(orig_width, float(bndbox.find('xmax').text))
                    ymax = min(orig_height, float(bndbox.find('ymax').text))
                    
                    # Convert to x, y, w, h format with proper scaling
                    valid_boxes.append([
                        xmin * scale_x,
                        ymin * scale_y,
                        (xmax - xmin) * scale_x,
                        (ymax - ymin) * scale_y
                    ])
                except (AttributeError, ValueError):
                    continue

        return valid_boxes if valid_boxes else None
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        bbox_path = self.bbox_paths[idx]
        if self.verbose > 1:
            print(f"\nLoading item {idx}")
            print(f"Image path: {img_path}")
            print(f"Bbox path: {bbox_path}")
        
        # Load image and original bboxes
        image = Image.open(img_path).convert('RGB')
        orig_size = image.size
        
        bboxes = self.parse_bbox(bbox_path, orig_size)
        if bboxes is None:
            print(f"Warning: No valid bounding boxes found for {img_path}. Using default bbox.")
            bboxes = [[0, 0, orig_size[0], orig_size[1]]]
        
        # Transform image and bboxes for model input
        if self.transform:
            # Keep track of intermediate sizes for box transformation
            transformed_bboxes = bboxes.copy()
            
            # Apply resize (256)
            resize_size = (256, 256)
            image = transforms.Resize(resize_size)(image)
            transformed_bboxes = [BoundingBoxTransform.resize(bbox, orig_size, resize_size) 
                                for bbox in transformed_bboxes]
            
            # Apply center crop (224)
            crop_size = (224, 224)
            image = transforms.CenterCrop(crop_size)(image)
            transformed_bboxes = [BoundingBoxTransform.center_crop(bbox, resize_size, crop_size) 
                                for bbox in transformed_bboxes]
            
            # Generate scores using transformed dimensions (224x224)
            scores = generate_scores(crop_size, transformed_bboxes)
            
            # Apply remaining transforms
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])(image)
            if self.return_boxes:
                return image, scores, transformed_bboxes
            return image, scores
        
        # If no transform, use original dimensions
        scores = generate_scores(min(orig_size), bboxes)
        if self.return_boxes:
            return image, scores, bboxes
        return image, scores

def get_patch_rank_loader(root_dir, split='train', batch_size=32, num_workers=4, **kwargs):
    """
    Create a DataLoader for the ImageNetPatchRank dataset
    The transforms are only applied to the input image, not for score calculation
    Args:
        root_dir: Path to ImageNet dataset
        split: 'train' or 'val'
        batch_size: number of samples per batch
        num_workers: number of workers for data loading
    Returns:
        loader: DataLoader that yields (images, scores, transformed_bboxes) where:
                images: tensor of shape [B, 3, 224, 224]  # BCHW format
                scores: tensor of shape [B, N] 
                transformed_bboxes: list of transformed bounding boxes
    """
    # Transforms only for the model input image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # PIL Image (HWC) -> Tensor (CHW)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageNetPatchRankLoader(
        root_dir=root_dir,
        split=split,
        transform=transform,
    )
    
    def custom_collate(batch):
        """Custom collate function to handle variable-sized tensors"""
        images = torch.stack([item[0] for item in batch])
        scores = torch.stack([item[1] for item in batch])
        
        # print(f"\nBatch collation:")
        # print(f"Images shape: {images.shape}")
        # print(f"Scores shape: {scores.shape}")
        
        return images, scores
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )
    
    return loader

def test_dataloader_shapes(root_dir, batch_size=4):
    """Test if dataloader returns consistent shapes across batches"""
    print("\nTesting dataloader batch shapes...")
    
    # Create small test loader
    loader = get_patch_rank_loader(
        root_dir, 
        split='train', 
        batch_size=batch_size, 
        num_workers=0
    )
    
    batch_shapes = []
    try:
        for i, (images, scores) in enumerate(loader):
                
            batch_info = {
                'batch': i,
                'images': images.shape,
                'scores': scores.shape,
            }
            batch_shapes.append(batch_info)
            
            print(f"\nBatch {i}:")
            print(f"- Images shape: {batch_info['images']}")
            print(f"- Scores shape: {batch_info['scores']}")
            
            # Verify shapes within batch
            B = images.shape[0]  # Batch size
            assert images.shape == (B, 3, 224, 224), f"Unexpected image shape: {images.shape}"
            assert scores.shape == (B, 196), f"Unexpected scores shape: {scores.shape}"
            
    except Exception as e:
        print(f"\nError during batch loading: {str(e)}")
        print("Last successful batch shapes:")
        for batch in batch_shapes:
            print(f"Batch {batch['batch']}:")
            print(f"- Images: {batch['images']}")
            print(f"- Scores: {batch['scores']}")
        raise
    
    print("\nAll batches processed successfully!")
    return batch_shapes
    

if __name__ == "__main__":
    data_root = Path("C:/Users/joren/Documents/_Uni/Master/Thesis/imagenet_subset")
    
    print("Testing dataloader consistency...")
    batch_shapes = test_dataloader_shapes(data_root)
    
    # Additional analysis of results
    print("\nShape consistency summary:")
    reference = batch_shapes[0]
    all_consistent = True
    
    for i, batch in enumerate(batch_shapes[1:], 1):
        if (batch['images'] != reference['images'] or 
            batch['scores'] != reference['scores']):
            print(f"Inconsistency in batch {i}:")
            print(f"Expected: {reference}")
            print(f"Got: {batch}")
            all_consistent = False
    
    if all_consistent:
        print("✓ All batch shapes are consistent!")
    else:
        print("✗ Found shape inconsistencies!")
