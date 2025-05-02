import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode, ToTensor
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
from pathlib import Path

from transforms import (DualTransforms, DualRandomResizedCrop, DualRandomHorizontalFlip, 
                      DualResize, DualCenterCrop, ColorJitter, RandomErasing, AutoAugment,
                      _has_unique_values, _preserve_ranking_values)
from datasets import RankedImageFolder

def setup_argparse():
    parser = argparse.ArgumentParser(description='Test dual transforms with RankedImageFolder')
    parser.add_argument('--data_path', type=str, default='path/to/data',
                        help='Path to image dataset (train folder)')
    parser.add_argument('--rankings_path', type=str, default='path/to/rankings',
                        help='Path to rankings directory')
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Batch size for testing')
    parser.add_argument('--use_global_ranking', action='store_true',
                        help='Use a global ranking for all images')
    parser.add_argument('--global_ranking_path', type=str, default=None,
                        help='Path to global ranking if use_global_ranking is True')
    parser.add_argument('--save_path', type=str, default='transform_test_results',
                        help='Path to save test results')
    return parser

def create_mock_ranking(size=196, importance_pattern="default"):
    """Create a mock ranking tensor with different patterns of importance."""
    if importance_pattern == "default":
        return torch.arange(size, dtype=torch.long)
    elif importance_pattern == "center":
        # Create center-focused ranking (center is most important)
        grid_size = int(np.sqrt(size))
        center_y, center_x = grid_size / 2, grid_size / 2
        
        distances = []
        for i in range(grid_size):
            for j in range(grid_size):
                dist = np.sqrt((i + 0.5 - center_y)**2 + (j + 0.5 - center_x)**2)
                patch_idx = i * grid_size + j
                distances.append((patch_idx, dist))
        
        # Sort by distance (closest first)
        distances.sort(key=lambda x: x[1])
        indices = [idx for idx, _ in distances]
        return torch.tensor(indices, dtype=torch.long)
    elif importance_pattern == "random":
        return torch.randperm(size, dtype=torch.long)
    else:
        raise ValueError(f"Unknown importance pattern: {importance_pattern}")

def visualize_image_with_ranking(image, ranking, filename, save_path):
    """Visualize an image with its ranking overlay for debugging."""
    # Convert image tensor to numpy for visualization
    if isinstance(image, torch.Tensor):
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    else:
        # Handle PIL Image
        to_tensor = ToTensor()
        img_tensor = to_tensor(image)
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    
    # Convert ranking to tensor if it's not already
    if not isinstance(ranking, torch.Tensor):
        ranking = torch.tensor(ranking)
    
    # Create a heatmap from the ranking
    plt.figure(figsize=(20, 10))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot ranking heatmap overlay
    plt.subplot(1, 2, 2)
    plt.imshow(img_np)
    plt.imshow(ranking.cpu().numpy()[0], alpha=0.6, cmap='viridis')
    plt.title('Ranking Overlay')
    plt.colorbar(label='Rank Order')
    plt.axis('off')
    
    # Save the visualization
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, filename))
    plt.close()

def check_tensor_uniqueness(tensor):
    """Check if all values in tensor are unique."""
    unique_values = torch.unique(tensor)
    total_values = tensor.numel()
    return len(unique_values) == total_values, len(unique_values), total_values

def test_preserve_ranking_values():
    """Test the _preserve_ranking_values function with different scenarios."""
    print("\n=== Testing _preserve_ranking_values function ===")
    
    # Test case 1: Tensor with no duplicates
    tensor1 = torch.arange(100).reshape(1, 10, 10)
    restored1 = _preserve_ranking_values(tensor1)
    is_identical = torch.all(restored1 == tensor1)
    print(f"Case 1 - No duplicates: Preserved identical values = {is_identical}")
    
    # Test case 2: Tensor with some duplicates (simulating a transform)
    tensor2 = torch.arange(100).reshape(1, 10, 10)
    # Create duplicates by copying some values
    tensor2[0, 2:4, 2:4] = tensor2[0, 0:2, 0:2]
    unique_before = torch.unique(tensor2).shape[0]
    
    restored2 = _preserve_ranking_values(tensor2)
    unique_after = torch.unique(restored2).shape[0]
    
    print(f"Case 2 - With duplicates: Unique values before={unique_before}, after={unique_after}")
    print(f"         Values are integers: {restored2.dtype == torch.long}")
    print(f"         All values unique: {len(torch.unique(restored2)) == restored2.numel()}")
    
    # Test case 3: Verify priority is maintained (lower values preserved)
    tensor3 = torch.ones(1, 5, 5, dtype=torch.long)  # All ones
    # Set some key positions with the value 1
    tensor3[0, 0, 0] = 1  # Most important position
    tensor3[0, 1, 1] = 1
    tensor3[0, 2, 2] = 1
    
    restored3 = _preserve_ranking_values(tensor3)
    
    # Check if first occurrence of 1 is preserved
    first_preserved = restored3[0, 0, 0] == 1
    print(f"Case 3 - Priority preservation: First occurrence preserved = {first_preserved}")
    print(f"         Other positions modified: {restored3[0, 1, 1] != 1 and restored3[0, 2, 2] != 1}")
    print(f"         All values are integers: {restored3.dtype == torch.long}")
    
    # Test case 4: Verify that ranking meaning is preserved as much as possible
    tensor4 = torch.zeros(1, 5, 5, dtype=torch.long)
    # Most important patches (values 0-3)
    tensor4[0, 0:2, 0:2] = torch.arange(4).reshape(2, 2)
    # Create some duplicates
    tensor4[0, 3:5, 3:5] = torch.arange(4).reshape(2, 2)
    
    restored4 = _preserve_ranking_values(tensor4)
    
    # Check if first occurrences are preserved
    first_preserved = torch.all(restored4[0, 0:2, 0:2] == tensor4[0, 0:2, 0:2])
    print(f"Case 4 - Meaning preservation: Important patches preserved = {first_preserved}")
    print(f"         Duplicates modified: {not torch.all(restored4[0, 3:5, 3:5] == tensor4[0, 3:5, 3:5])}")
    print(f"         All values are integers: {restored4.dtype == torch.long}")

class PILToTensorTransform:
    """
    Transform that converts PIL Images to tensors for use in DataLoader.
    This doesn't modify rankings and performs NO resizing.
    """
    def __init__(self):
        self.to_tensor = ToTensor()
    
    def __call__(self, img, ranking=None, **kwargs):
        if hasattr(img, 'convert'):  # Check if it's a PIL Image
            # Convert to tensor without any resizing
            img = self.to_tensor(img)
        
        if ranking is not None:
            return img, ranking
        return img

def test_transforms_with_batches(data_loader, transforms_dual, save_path):
    """Test transforms with batched data from data loader."""
    os.makedirs(save_path, exist_ok=True)
    
    # Get one batch
    for batch_idx, (images, targets, rankings, paths) in enumerate(data_loader):
        print(f"Original batch shape: Images {images.shape}, Rankings {rankings.shape}")
        
        # Check original rankings uniqueness
        batch_status = []
        for i in range(rankings.size(0)):
            is_unique, unique_count, total_count = check_tensor_uniqueness(rankings[i])
            batch_status.append((is_unique, unique_count, total_count))
            print(f"  Original ranking {i}: Unique={is_unique}, {unique_count}/{total_count} unique values")
        
        # Apply transforms to the batch - we need to do this individually since our transform accepts (img, ranking) pairs
        transformed_images = []
        transformed_rankings = []
        
        for i in range(images.size(0)):
            img, rank = transforms_dual(images[i], rankings[i])
            transformed_images.append(img)
            transformed_rankings.append(rank)
            
            # Check transformed ranking uniqueness
            is_unique, unique_count, total_count = check_tensor_uniqueness(rank)
            print(f"  Transformed ranking {i}: Unique={is_unique}, {unique_count}/{total_count} unique values")
            
            # Visualize original and transformed
            img_filename = f"batch{batch_idx}_img{i}_original.png"
            visualize_image_with_ranking(images[i], rankings[i], img_filename, save_path)
            
            trans_filename = f"batch{batch_idx}_img{i}_transformed.png"
            visualize_image_with_ranking(img, rank, trans_filename, save_path)
        
        # Stack back into batches
        transformed_images = torch.stack(transformed_images)
        transformed_rankings = torch.stack(transformed_rankings)
        
        print(f"Transformed batch shape: Images {transformed_images.shape}, Rankings {transformed_rankings.shape}")
        
        # Only process one batch for this test
        break

def test_transforms_individually(data_loader, save_path):
    """Test each transform individually to identify which might cause issues."""
    transforms_to_test = [
        ("RandomResizedCrop", DualRandomResizedCrop(size=224, interpolation=InterpolationMode.BILINEAR)),
        ("RandomHorizontalFlip", DualRandomHorizontalFlip(p=1.0)),  # p=1.0 to ensure it applies
        ("Resize", DualResize(size=224, interpolation=InterpolationMode.BILINEAR)),
        ("CenterCrop", DualCenterCrop(size=196)),
        ("ColorJitter", ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)),
        ("RandomErasing", RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3))), 
        ("AutoAugment_Rotate", lambda img, rank: AutoAugment()._rotate(img, rank, 1.0)),
        ("AutoAugment_ShearX", lambda img, rank: AutoAugment()._shear_x(img, rank, 1.0)),
        ("AutoAugment_ShearY", lambda img, rank: AutoAugment()._shear_y(img, rank, 1.0)),
        ("AutoAugment_TranslateX", lambda img, rank: AutoAugment()._translate_x(img, rank, 1.0)),
        ("AutoAugment_TranslateY", lambda img, rank: AutoAugment()._translate_y(img, rank, 1.0)),
        ("AutoAugment_Full", AutoAugment(policy_name='rand-m9-mstd0.5-inc1')),
    ]
    
    os.makedirs(save_path, exist_ok=True)
    
    # Get one image
    for batch_idx, (images, targets, rankings, paths) in enumerate(data_loader):
        # Just use the first image
        img, ranking = images[0], rankings[0]
        
        # Original uniqueness check
        is_unique, unique_count, total_count = check_tensor_uniqueness(ranking)
        print(f"Original: Unique={is_unique}, {unique_count}/{total_count} unique values")
        
        # Test each transform
        for name, transform in transforms_to_test:
            test_folder = os.path.join(save_path, name)
            os.makedirs(test_folder, exist_ok=True)
            
            # Apply transform
            try:
                transformed_img, transformed_ranking = transform(img.clone(), ranking.clone())
                
                # Check uniqueness
                is_unique, unique_count, total_count = check_tensor_uniqueness(transformed_ranking)
                print(f"{name}: Unique={is_unique}, {unique_count}/{total_count} unique values")
                
                # Visualize
                visualize_image_with_ranking(img, ranking, "original.png", test_folder)
                visualize_image_with_ranking(transformed_img, transformed_ranking, "transformed.png", test_folder)
                
                # If not unique, restore uniqueness and visualize again
                if not is_unique:
                    restored_ranking = _preserve_ranking_values(transformed_ranking)
                    is_unique, unique_count, total_count = check_tensor_uniqueness(restored_ranking)
                    print(f"{name} (Restored): Unique={is_unique}, {unique_count}/{total_count} unique values")
                    
                    # Check if we preserved integer values
                    is_int_type = restored_ranking.dtype == torch.long
                    print(f"{name} (Restored): Integer values preserved = {is_int_type}")
                    
                    # Visualize restored ranking
                    visualize_image_with_ranking(transformed_img, restored_ranking, "restored.png", test_folder)
                
                    # Analyze changes to verify ranking integrity
                    analyze_ranking_changes(ranking, transformed_ranking, restored_ranking, name, test_folder)
                
            except Exception as e:
                print(f"Error testing {name}: {e}")
        
        # Only process one image
        break

def analyze_ranking_changes(original, transformed, restored, transform_name, save_path):
    """Analyze how much the rankings change before and after restoration."""
    # Flatten tensors for analysis
    orig_flat = original.view(-1).cpu().numpy()
    trans_flat = transformed.view(-1).cpu().numpy()
    rest_flat = restored.view(-1).cpu().numpy()
    
    # Count preserved values (percentage of values that remain the same)
    preserved_count = np.sum(trans_flat == orig_flat)
    preserved_pct = preserved_count / len(orig_flat) * 100
    
    # Count preserved values after restoration
    preserved_after_count = np.sum(rest_flat == orig_flat)
    preserved_after_pct = preserved_after_count / len(orig_flat) * 100
    
    # Calculate Spearman rank correlation
    try:
        from scipy.stats import spearmanr
        corr_trans, _ = spearmanr(orig_flat, trans_flat)
        corr_rest, _ = spearmanr(orig_flat, rest_flat)
    except:
        corr_trans, corr_rest = 0, 0
    
    # Print analysis
    print(f"\nRanking analysis for {transform_name}:")
    print(f"  Preserved values: {preserved_pct:.1f}% ({preserved_count}/{len(orig_flat)})")
    print(f"  Preserved after restoration: {preserved_after_pct:.1f}% ({preserved_after_count}/{len(orig_flat)})")
    print(f"  Rank correlation: Original-Transformed={corr_trans:.3f}, Original-Restored={corr_rest:.3f}")
    
    # Create visualization of changes
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(np.arange(len(orig_flat)), orig_flat, alpha=0.5)
    plt.title('Original Rankings')
    plt.xlabel('Position')
    plt.ylabel('Rank Value')
    
    plt.subplot(1, 3, 2)
    plt.scatter(np.arange(len(trans_flat)), trans_flat, alpha=0.5)
    plt.title(f'Transformed Rankings\n(Preserved: {preserved_pct:.1f}%)')
    plt.xlabel('Position')
    plt.ylabel('Rank Value')
    
    plt.subplot(1, 3, 3)
    plt.scatter(np.arange(len(rest_flat)), rest_flat, alpha=0.5)
    plt.title(f'Restored Rankings\n(Preserved: {preserved_after_pct:.1f}%)')
    plt.xlabel('Position')
    plt.ylabel('Rank Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "ranking_analysis.png"))
    plt.close()

def test_sequential_transforms(data_loader, save_path):
    """Test sequences of transforms to verify they properly maintain ranking uniqueness."""
    print("\n=== Testing transform sequences ===")
    
    # Define transform sequences to test
    transform_sequences = [
        ("resize_then_crop", 
         [DualResize(size=256), DualCenterCrop(size=224)]),
        
        ("crop_then_flip", 
         [DualRandomResizedCrop(size=224), DualRandomHorizontalFlip(p=1.0)]),
        
        ("complex_sequence",
         [DualResize(size=256), DualCenterCrop(size=224), 
          DualRandomHorizontalFlip(p=1.0), AutoAugment(policy_name='rand-m9-mstd0.5')]),
    ]
    
    os.makedirs(save_path, exist_ok=True)
    
    # Get one image
    for batch_idx, (images, targets, rankings, paths) in enumerate(data_loader):
        # Just use the first image
        img, ranking = images[0], rankings[0]
        
        # Test each sequence
        for name, transforms_list in transform_sequences:
            print(f"\nTesting transform sequence: {name}")
            
            sequence_folder = os.path.join(save_path, name)
            os.makedirs(sequence_folder, exist_ok=True)
            
            # Visualize original
            visualize_image_with_ranking(img, ranking, "0_original.png", sequence_folder)
            
            # Apply transforms sequentially
            current_img, current_ranking = img.clone(), ranking.clone()
            
            for i, transform in enumerate(transforms_list, 1):
                try:
                    # Apply transform
                    current_img, current_ranking = transform(current_img, current_ranking)
                    
                    # Check uniqueness
                    is_unique, unique_count, total_count = check_tensor_uniqueness(current_ranking)
                    print(f"  Step {i}: Unique={is_unique}, {unique_count}/{total_count} unique values")
                    
                    # If not unique, restore uniqueness
                    if not is_unique:
                        current_ranking = _preserve_ranking_values(current_ranking)
                        is_unique, unique_count, total_count = check_tensor_uniqueness(current_ranking)
                        print(f"  Step {i} (Restored): Unique={is_unique}, {unique_count}/{total_count} unique values")
                    
                    # Visualize intermediate result
                    visualize_image_with_ranking(
                        current_img, current_ranking, 
                        f"{i}_{transform.__class__.__name__}.png", 
                        sequence_folder
                    )
                    
                except Exception as e:
                    print(f"  Error in step {i}: {e}")
                    break
            
            # Analyze final result compared to original
            analyze_ranking_changes(
                ranking, current_ranking, current_ranking,
                f"{name}_final", sequence_folder
            )
        
        # Only process one image
        break

def test_ranking_pattern_preservation(data_loader, save_path):
    """Test if different initial ranking patterns are preserved after transforms."""
    print("\n=== Testing ranking pattern preservation ===")
    
    # Create different ranking patterns
    patterns = {
        "default": create_mock_ranking(196, "default"),
        "center": create_mock_ranking(196, "center"),
        "random": create_mock_ranking(196, "random"),
    }
    
    # Define transforms to test
    transforms_to_test = [
        ("resize_crop", DualRandomResizedCrop(size=224)),
        ("flip", DualRandomHorizontalFlip(p=1.0)),
        ("rotate", lambda img, rank: AutoAugment()._rotate(img, rank, 0.5)),
    ]
    
    os.makedirs(save_path, exist_ok=True)
    
    # Get one image
    for batch_idx, (images, targets, rankings, paths) in enumerate(data_loader):
        # Just use the first image and repeat tests with different ranking patterns
        img = images[0]
        
        for pattern_name, pattern_ranking in patterns.items():
            pattern_folder = os.path.join(save_path, pattern_name)
            os.makedirs(pattern_folder, exist_ok=True)
            
            # Reshape pattern to match expected shape and ensure it's on the same device as image
            shaped_ranking = pattern_ranking.reshape(1, 14, 14).to(img.device)
            
            # Visualize original
            visualize_image_with_ranking(img, shaped_ranking, "original.png", pattern_folder)
            
            # Test transforms
            for transform_name, transform in transforms_to_test:
                test_folder = os.path.join(pattern_folder, transform_name)
                os.makedirs(test_folder, exist_ok=True)
                
                try:
                    # Apply transform
                    trans_img, trans_ranking = transform(img.clone(), shaped_ranking.clone())
                    
                    # Check uniqueness
                    is_unique, unique_count, total_count = check_tensor_uniqueness(trans_ranking)
                    print(f"{pattern_name}-{transform_name}: Unique={is_unique}, {unique_count}/{total_count}")
                    
                    # Visualize
                    visualize_image_with_ranking(trans_img, trans_ranking, "transformed.png", test_folder)
                    
                    # If not unique, restore and visualize again
                    if not is_unique:
                        restored = _preserve_ranking_values(trans_ranking)
                        visualize_image_with_ranking(trans_img, restored, "restored.png", test_folder)
                        
                        # Analyze pattern preservation
                        analyze_ranking_changes(shaped_ranking, trans_ranking, restored, 
                                              f"{pattern_name}-{transform_name}", test_folder)
                    else:
                        # Still analyze changes
                        analyze_ranking_changes(shaped_ranking, trans_ranking, trans_ranking, 
                                              f"{pattern_name}-{transform_name}", test_folder)
                
                except Exception as e:
                    print(f"Error testing {pattern_name}-{transform_name}: {e}")
        
        # Only process one image
        break

def test_full_pipeline_with_dataloader(args):
    """Test the full pipeline with dataloader and batch processing."""
    # Setup save paths
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    
    # First, test the _preserve_ranking_values function independently
    test_preserve_ranking_values()
    
    # Load a global ranking if specified
    global_ranking = None
    if args.use_global_ranking:
        if args.global_ranking_path:
            try:
                global_ranking = torch.load(args.global_ranking_path)
                print(f"Using global ranking from {args.global_ranking_path}")
            except Exception as e:
                print(f"Error loading global ranking: {e}")
                global_ranking = create_mock_ranking()
                print("Using mock global ranking instead")
        else:
            global_ranking = create_mock_ranking()
            print("Using mock global ranking (no path provided)")
    
    # Create dual transforms
    transforms_dual = DualTransforms(
        size=224,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation=InterpolationMode.BILINEAR,
        re_prob=0.25,
        re_mode='pixel',
        re_count=1
    )
    
    # Create basic transform to convert PIL images to tensors
    basic_transform = PILToTensorTransform()
    
    # Create dataset
    dataset = RankedImageFolder(
        root=args.data_path,
        rankings_dir=args.rankings_path,
        transform=basic_transform,  # Add basic transform to convert PIL to tensor
        random_rankings=False,
        global_ranking=global_ranking,
        return_path=True
    )
    
    # Create dataloader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    print(f"Dataset size: {len(dataset)} images")
    
    # Run specific test sets
    print("\n=== Running transform tests ===")
    
    # Test individual transforms
    test_transforms_individually(data_loader, save_path / "individual_tests")
    
    # Test transform sequences
    test_sequential_transforms(data_loader, save_path / "sequential_tests")
    
    # Test with different ranking patterns
    test_ranking_pattern_preservation(data_loader, save_path / "pattern_tests")
    
    # Test transforms with batches
    test_transforms_with_batches(data_loader, transforms_dual, save_path / "batch_tests")
    
    print("\nTransform testing complete! Check the output directories for visualizations.")

if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Run the test
    test_full_pipeline_with_dataloader(args)