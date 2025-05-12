import os
import PIL.ImageShow
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import cv2
import argparse
from pathlib import Path
import albumentations as A
from torchvision.datasets.folder import ImageFolder, default_loader, IMG_EXTENSIONS

from transforms import build_transform, AlbumentationsRandAugment, _RAND_INCREASING_TRANSFORMS
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from datasets import build_dataset, HeatmapImageFolder

def denormalize(tensor, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    """Convert normalized tensor back to displayable image"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)

def tensor_to_numpy(tensor):
    """Convert torch tensor to numpy array for display"""
    if tensor.ndim == 4:  # Batch of images
        return tensor.permute(0, 2, 3, 1).cpu().numpy()
    else:  # Single image
        return tensor.permute(1, 2, 0).cpu().numpy()

def create_sample_data(out_dir="sample_data", num_classes=3, imgs_per_class=5):
    """Create a sample dataset with images and heatmaps for testing"""
    images_root = os.path.join(out_dir, "images")
    heatmaps_root = os.path.join(out_dir, "heatmaps")
    
    # Create class directories
    for cls_idx in range(num_classes):
        cls_name = f"class_{cls_idx}"
        os.makedirs(os.path.join(images_root, cls_name), exist_ok=True)
        os.makedirs(os.path.join(heatmaps_root, cls_name), exist_ok=True)
        
        # Create sample images and heatmaps
        for img_idx in range(imgs_per_class):
            img_name = f"image_{img_idx}"
            
            # Create a sample image with patterns
            img = np.ones((224, 224, 3), dtype=np.uint8) * 128
            
            # Add some patterns based on class and index for visual distinction
            cv2.rectangle(img, (50, 50), (150, 150), (200, 100 + cls_idx*30, 100), -1)
            cv2.circle(img, (160, 160), 40 + img_idx*5, (100, 200, 100 + img_idx*30), -1)
            
            # Add a grid pattern
            for i in range(0, 224, 32):
                cv2.line(img, (0, i), (223, i), (200, 200, 200), 1)
                cv2.line(img, (i, 0), (i, 223), (200, 200, 200), 1)
            
            # Create corresponding heatmap (14x14 for a 224x224 image with patch_size=16)
            heatmap = np.zeros((14, 14), dtype=np.float32)
            
            # Add a pattern to heatmap (different for each image)
            cx, cy = img_idx % 4 + 5, cls_idx * 3 + 3  # Center coords vary by image and class
            for i in range(14):
                for j in range(14):
                    # Distance-based pattern
                    dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                    heatmap[i, j] = max(0, 1 - dist/8)
            
            # Save the image
            img_path = os.path.join(images_root, cls_name, f"{img_name}.jpg")
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # Resize heatmap to match image size (224x224)
            heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
            
            # Convert heatmap to an image (scale to 0-255 and convert to uint8)
            heatmap_img = (heatmap_resized * 255).astype(np.uint8)
            
            # Save the heatmap as an image instead of .npy
            heatmap_path = os.path.join(heatmaps_root, cls_name, f"{img_name}.png")
            cv2.imwrite(heatmap_path, heatmap_img)
    
    print(f"Created sample dataset with {num_classes} classes and {imgs_per_class} images each")
    return images_root, heatmaps_root

def visualize_ranking(ranking, num_patches_per_dim=14):
    """Convert a flat ranking to a 2D visualization grid"""
    ranking_np = ranking.numpy() if isinstance(ranking, torch.Tensor) else ranking
    ranks_display_grid = np.zeros((num_patches_per_dim, num_patches_per_dim), dtype=np.float32)
    
    for rank_order, patch_flat_idx in enumerate(ranking_np):
        patch_y = patch_flat_idx // num_patches_per_dim
        patch_x = patch_flat_idx % num_patches_per_dim
        if 0 <= patch_y < num_patches_per_dim and 0 <= patch_x < num_patches_per_dim:
            ranks_display_grid[patch_y, patch_x] = rank_order
            
    return ranks_display_grid

def test_transforms_with_dataset(images_root, heatmaps_root, output_dir="transform_results"):
    """Test transforms with the dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Define test configurations
    test_configs = [
        {"name": "basic", "args": {"input_size": 224, "color_jitter": 0, "aa": None, "reprob": 0, "recount": 1, "remode": "const", "eval_crop_ratio": 1.0}},
        {"name": "color_jitter", "args": {"input_size": 224, "color_jitter": 0.4, "aa": None, "reprob": 0, "recount": 1, "remode": "const", "eval_crop_ratio": 1.0}},
        {"name": "rand_augment", "args": {"input_size": 224, "color_jitter": 0, "aa": "rand-m9-mstd0.5", "reprob": 0, "recount": 1, "remode": "const", "eval_crop_ratio": 1.0}},
        {"name": "full_augment", "args": {"input_size": 224, "color_jitter": 0.4, "aa": "rand-m9-mstd0.5", "reprob": 0.25, "recount": 1, "remode": "pixel", "eval_crop_ratio": 1.0}}
    ]
    
    # Create dataset with both heatmap and rankings
    dataset = HeatmapImageFolder(images_root, heatmaps_root, transform=None, 
                               return_heatmap=True, return_rankings=True)
    
    # Test each transform configuration
    for config in test_configs:
        print(f"Testing {config['name']} transform...")
        
        # Create args namespace
        args = argparse.Namespace(**config["args"])
        
        # Create transforms for training and validation
        train_transform = build_transform(is_train=True, args=args)
        val_transform = build_transform(is_train=False, args=args)
        
        # Apply to some samples
        num_samples = min(3, len(dataset))
        
        fig, axes = plt.subplots(num_samples, 6, figsize=(24, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Get a sample - correct order: image, target, ranking, heatmap
            image, target, ranking, heatmap = dataset[i]
            
            # Convert original image for display
            if isinstance(image, torch.Tensor):
                orig_img_disp = denormalize(image)
                orig_img_disp = tensor_to_numpy(orig_img_disp)
            else:
                orig_img_disp = image
            
            # Convert ranking to visualization grid
            orig_ranking_disp = visualize_ranking(ranking)
            
            # Convert original heatmap for display
            if isinstance(heatmap, torch.Tensor):
                orig_heatmap_disp = heatmap.permute(1, 2, 0).numpy()
            else:
                orig_heatmap_disp = heatmap
            
            # Apply train transform
            dataset.transform = train_transform
            train_img, train_target, train_ranking, train_heatmap = dataset[i]
            
            # Apply val transform
            dataset.transform = val_transform
            val_img, val_target, val_ranking, val_heatmap = dataset[i]
            
            # Convert train tensors for display
            if isinstance(train_img, torch.Tensor):
                train_img_disp = denormalize(train_img)
                train_img_disp = tensor_to_numpy(train_img_disp)
            else:
                train_img_disp = train_img
            
            # Convert train ranking to visualization grid
            train_ranking_disp = visualize_ranking(train_ranking)
            
            # Convert train heatmap for display
            if isinstance(train_heatmap, torch.Tensor):
                train_heatmap_disp = train_heatmap.squeeze().numpy()
            else:
                train_heatmap_disp = train_heatmap
            
            # Plot
            axes[i, 0].imshow(orig_img_disp)
            axes[i, 0].set_title(f"Original Image {i+1}\nTarget: {target}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(orig_ranking_disp, cmap='viridis')
            axes[i, 1].set_title(f"Original Ranking {i+1}")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(orig_heatmap_disp, cmap='viridis')
            axes[i, 2].set_title(f"Original Heatmap {i+1}")
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(train_img_disp)
            axes[i, 3].set_title(f"Train Transform\nTarget: {train_target}")
            axes[i, 3].axis('off')
            
            axes[i, 4].imshow(train_ranking_disp, cmap='viridis')
            axes[i, 4].set_title(f"Train Ranking")
            axes[i, 4].axis('off')
            
            axes[i, 5].imshow(train_heatmap_disp, cmap='viridis')
            axes[i, 5].set_title(f"Train Heatmap")
            axes[i, 5].axis('off')

            # Reset dataset transform
            dataset.transform = None
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{config['name']}_transforms.png"))
        plt.close(fig)
        
    
    print(f"Transform tests completed. Results saved to {output_dir}.")

def test_rand_augment_variations(images_root, heatmaps_root, output_dir="rand_augment_results"):
    """Test different RandAugment configurations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset with both heatmap and rankings
    dataset = HeatmapImageFolder(images_root, heatmaps_root, transform=None, 
                               return_heatmap=True, return_rankings=True)
    
    # Test sample image with multiple augment applications
    sample_idx = 143
    img, target, ranking, heatmap = dataset[sample_idx]
    
    # Convert ranking to visualization grid
    ranking_disp = visualize_ranking(ranking)
    
    # Show original for reference
    PIL.ImageShow.show(img)
    plt.figure(figsize=(6, 6))
    plt.imshow(ranking_disp, cmap='viridis')
    plt.title('Original Ranking')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "original_ranking.png"))
    plt.close()
    PIL.ImageShow.show(heatmap)
    
    # Convert to numpy for albumentations
    if not isinstance(img, np.ndarray):
        img_np = np.array(img)
    else:
        img_np = img
    
    # Test different RandAugment configs
    rand_augment_configs = [
        {"name": "m5_std0", "magnitude": 5, "std": 0},
        {"name": "m9_std0.5", "magnitude": 9, "std": 0.5},
        {"name": "m15_std0.5", "magnitude": 15, "std": 0.5}
    ]
    
    # Apply each transform multiple times to see variety
    num_repeats = 5
    
    for config in rand_augment_configs:
        print(f"Testing RandAugment with {config['name']}...")
        
        # Create RandAugment transform
        rand_augment = AlbumentationsRandAugment(
            num_ops=2,
            magnitude=config["magnitude"],
            std=config["std"]
        )
        
        fig, axes = plt.subplots(num_repeats, 2, figsize=(8, 4*num_repeats))
        
        # Show original first
        fig.suptitle(f"RandAugment {config['name']}", fontsize=16)
        
        # Apply transform multiple times
        for i in range(num_repeats):
            # Force apply the transform
            result = rand_augment(image=img_np.copy(), heatmap=heatmap.copy(), force_apply=True)
            aug_img = result["image"]
            aug_heatmap = result["heatmap"]
            
            # Plot
            axes[i, 0].imshow(aug_img)
            axes[i, 0].set_title(f"Image Sample {i+1}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(aug_heatmap, cmap='viridis')
            axes[i, 1].set_title(f"Heatmap Sample {i+1}")
            axes[i, 1].axis('off')
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(os.path.join(output_dir, f"{config['name']}_variations.png"))
        plt.close(fig)
    
    print(f"RandAugment variation tests completed. Results saved to {output_dir}.")

def test_ImageFolder_compatibility():
    """Test that transforms work with regular ImageFolder too"""
    # Create sample data
    data_dir, _ = create_sample_data("image_folder_test", num_classes=2, imgs_per_class=3)
    
    # Create args
    args = argparse.Namespace(
        input_size=224,
        color_jitter=0.4,
        aa="rand-m9-mstd0.5",
        reprob=0.25,
        recount=1,
        eval_crop_ratio=1.0
    )
    
    # Build transform
    transform = build_transform(is_train=True, args=args)
    
    # Create regular ImageFolder dataset
    dataset = ImageFolder(data_dir, transform=transform)
    
    # Test loading a few samples
    for i in range(min(3, len(dataset))):
        # This should work without errors
        img, target = dataset[i]
        print(f"Successfully loaded image {i} with target {target}")
        if isinstance(img, torch.Tensor):
            print(f"Image shape: {img.shape}, dtype: {img.dtype}")
        else:
            print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    
    print("ImageFolder compatibility test passed!")

def test_single_image(image_path, heatmap_path, args, output_dir="single_image_results"):
    """Test all transforms on one image (and optional heatmap) and save visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    # load image and heatmap
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = None
    if heatmap_path:
        hm = cv2.imread(heatmap_path, cv2.IMREAD_UNCHANGED)
        if hm is None:
            raise ValueError(f"Could not load heatmap from {heatmap_path}")
        heatmap = hm if hm.ndim==2 else hm[:,:,0]
    # reuse same configs
    test_configs = [
        {"name": "basic",     "args": {"input_size":224,"color_jitter":0,"aa":None,"reprob":0,"recount":1,"remode":"const","eval_crop_ratio":1.0}},
        {"name": "color_jitter","args":{"input_size":224,"color_jitter":0.4,"aa":None,"reprob":0,"recount":1,"remode":"const","eval_crop_ratio":1.0}},
        {"name": "rand_augment","args":{"input_size":224,"color_jitter":0,"aa":"rand-m9-mstd0.5","reprob":0,"recount":1,"remode":"const","eval_crop_ratio":1.0}},
        {"name": "full_augment","args":{"input_size":224,"color_jitter":0.4,"aa":"rand-m9-mstd0.5","reprob":0.25,"recount":1,"remode":"pixel","eval_crop_ratio":1.0}},
    ]
    for cfg in test_configs:
        cfg_args = argparse.Namespace(**cfg["args"])
        train_t = build_transform(is_train=True,  args=cfg_args)
        val_t   = build_transform(is_train=False, args=cfg_args)
        # apply train
        if heatmap is not None:
            tr = train_t(image=img.copy(), heatmap=heatmap.copy())
            vr = val_t(image=img.copy(), heatmap=heatmap.copy())
            tr_img, tr_hm = tr["image"], tr["heatmap"]
            vr_img, vr_hm = vr["image"], vr["heatmap"]
        else:
            tr_img = train_t(image=img.copy())["image"]
            vr_img = val_t(image=img.copy())["image"]
            tr_hm = vr_hm = None
        # convert tensors for display
        def prep(x):
            if isinstance(x, torch.Tensor):
                return tensor_to_numpy(denormalize(x))
            return x
        tr_img, vr_img = prep(tr_img), prep(vr_img)
        # plot
        cols = 3 if heatmap is None else 3
        rows = 1 if heatmap is None else 2
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols,4*rows))
        ax = axes if rows==1 else axes.flatten()
        ax[0].imshow(img);     ax[0].set_title("Original");   ax[0].axis("off")
        ax[1].imshow(tr_img);  ax[1].set_title("Train");      ax[1].axis("off")
        ax[2].imshow(vr_img);  ax[2].set_title("Val");        ax[2].axis("off")
        if heatmap is not None:
            ax[3].imshow(heatmap, cmap="viridis");    ax[3].set_title("Orig HM");      ax[3].axis("off")
            ax[4].imshow(tr_hm,    cmap="viridis");    ax[4].set_title("Train HM");     ax[4].axis("off")
            ax[5].imshow(vr_hm,    cmap="viridis");    ax[5].set_title("Val HM");       ax[5].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"single_{cfg['name']}.png"))
        plt.close(fig)

def test_individual_transforms(image_path, heatmap_path=None, output_dir="individual_transforms_results"):
    """Test each transform in _RAND_INCREASING_TRANSFORMS with different magnitude levels."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Import timm transforms directly
    from timm.data.auto_augment import (
        auto_contrast, equalize, invert, posterize, solarize, solarize_add,
        color, contrast, brightness, sharpness, shear_x, shear_y,
        translate_x_rel, translate_y_rel, rotate, _LEVEL_DENOM
    )
    from timm.data.auto_augment import (
        _rotate_level_to_arg as timm_rotate_level_to_arg,
        _shear_level_to_arg as timm_shear_level_to_arg,
        _translate_rel_level_to_arg as timm_translate_rel_level_to_arg,
        _posterize_increasing_level_to_arg as timm_posterize_increasing_level_to_arg,
        _solarize_increasing_level_to_arg as timm_solarize_increasing_level_to_arg,
        _solarize_add_level_to_arg as timm_solarize_add_level_to_arg,
        _enhance_increasing_level_to_arg as timm_enhance_increasing_level_to_arg,
    )
    
    # Load image and optional heatmap
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create PIL image for timm transforms
    pil_img = Image.fromarray(img)
    
    heatmap = None
    if heatmap_path:
        hm = cv2.imread(heatmap_path, cv2.IMREAD_UNCHANGED)
        if hm is None:
            raise ValueError(f"Could not load heatmap from {heatmap_path}")
        heatmap = hm if hm.ndim == 2 else hm[:,:,0]
    
    # Map transform names to timm functions
    timm_transforms = {
        'AutoContrast': lambda img, level: auto_contrast(img),
        'Equalize': lambda img, level: equalize(img),
        'Invert': lambda img, level: invert(img),
        'Rotate': lambda img, level: rotate(img, *timm_rotate_level_to_arg(level, {})),
        'PosterizeIncreasing': lambda img, level: posterize(img, *timm_posterize_increasing_level_to_arg(level, {})),
        'SolarizeIncreasing': lambda img, level: solarize(img, *timm_solarize_increasing_level_to_arg(level, {})),
        'SolarizeAdd': lambda img, level: solarize_add(img, *timm_solarize_add_level_to_arg(level, {})),
        'ColorIncreasing': lambda img, level: color(img, *timm_enhance_increasing_level_to_arg(level, {})),
        'ContrastIncreasing': lambda img, level: contrast(img, *timm_enhance_increasing_level_to_arg(level, {})),
        'BrightnessIncreasing': lambda img, level: brightness(img, *timm_enhance_increasing_level_to_arg(level, {})),
        'SharpnessIncreasing': lambda img, level: sharpness(img, *timm_enhance_increasing_level_to_arg(level, {})),
        'ShearX': lambda img, level: shear_x(img, *timm_shear_level_to_arg(level, {})),
        'ShearY': lambda img, level: shear_y(img, *timm_shear_level_to_arg(level, {})),
        'TranslateXRel': lambda img, level: translate_x_rel(img, *timm_translate_rel_level_to_arg(level, {})),
        'TranslateYRel': lambda img, level: translate_y_rel(img, *timm_translate_rel_level_to_arg(level, {})),
    }
    # Test magnitude levels
    magnitude_levels = [1, 5, 9]
    
    # Create base RandAugment to access its transforms
    base_augmenter = AlbumentationsRandAugment(num_ops=1, magnitude=9)
    
    # Test each transform with different magnitude levels
    for transform_name in _RAND_INCREASING_TRANSFORMS:
        if transform_name not in base_augmenter.transforms:
            print(f"Warning: Transform {transform_name} not implemented in AlbumentationsRandAugment")
            continue
            
        if transform_name not in timm_transforms:
            print(f"Warning: Transform {transform_name} not mapped to timm implementation")
            continue
            
        print(f"Testing {transform_name}...")
        
        transform_info = base_augmenter.transforms[transform_name]
        is_spatial = transform_info['spatial']
        
        # Create a 2x4 grid: top row for Albumentations, bottom row for timm
        # First column is original image, next 3 are for different magnitudes
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Display original images (same for both rows)
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("Original (Albumentation)")
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(img)
        axes[1, 0].set_title("Original (timm)")
        axes[1, 0].axis('off')
        
        # Apply transforms with different magnitude levels
        for i, magnitude in enumerate(magnitude_levels):
            # Albumentations transform
            albumentation_transform = transform_info['op'](magnitude)
            
            # Apply albumentation transform
            data = {'image': img.copy()}
            if heatmap is not None and is_spatial:
                data['mask'] = heatmap.copy()
                
            result = albumentation_transform(**data)
            album_img = result['image']
            
            # Get transform parameters for Albumentations
            album_params = ""
            try:
                # Try to get transform parameters if available
                if hasattr(albumentation_transform, "get_transform_init_args"):
                    params = albumentation_transform.get_transform_init_args()
                    param_str = ", ".join(f"{k}={v}" for k, v in params.items() if k != "always_apply")
                    if param_str:
                        album_params = f"\n({param_str})"
            except:
                pass
            
            # Display albumentation transformed image
            axes[0, i+1].imshow(album_img)
            axes[0, i+1].set_title(f"Album {transform_name}\nMagnitude {magnitude}{album_params}")
            axes[0, i+1].axis('off')
            
            # Apply timm transform
            timm_transform = timm_transforms[transform_name]
            
            # Get parameters for timm transforms
            timm_params = ""
            try:
                if transform_name == 'Rotate':
                    args = timm_rotate_level_to_arg(magnitude, {})
                    timm_params = f"\n(degrees={args[0]})"
                elif transform_name in ['ShearX', 'ShearY']:
                    args = timm_shear_level_to_arg(magnitude, {})
                    timm_params = f"\n(factor={args[0]:.2f})"
                elif transform_name in ['TranslateXRel', 'TranslateYRel']:
                    args = timm_translate_rel_level_to_arg(magnitude, {})
                    timm_params = f"\n(offset={args[0]:.2f})"
                elif transform_name == 'PosterizeIncreasing':
                    args = timm_posterize_increasing_level_to_arg(magnitude, {})
                    timm_params = f"\n(bits={args[0]})"
                elif transform_name == 'SolarizeIncreasing':
                    args = timm_solarize_increasing_level_to_arg(magnitude, {})
                    timm_params = f"\n(thresh={args[0]})"
                elif transform_name == 'SolarizeAdd':
                    args = timm_solarize_add_level_to_arg(magnitude, {})
                    timm_params = f"\n(add={args[0]})"
                elif transform_name in ['ColorIncreasing', 'ContrastIncreasing', 'BrightnessIncreasing', 'SharpnessIncreasing']:
                    args = timm_enhance_increasing_level_to_arg(magnitude, {})
                    timm_params = f"\n(factor={args[0]:.2f})"
            except:
                pass
            
            timm_img = np.array(timm_transform(pil_img, magnitude))
            
            # Display timm transformed image
            axes[1, i+1].imshow(timm_img)
            axes[1, i+1].set_title(f"Timm {transform_name}\nMagnitude {magnitude}{timm_params}")
            axes[1, i+1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{transform_name}_comparison.png"))
        plt.close(fig)
    
    print(f"Individual transform tests completed. Results saved to {output_dir}.")

def test_dataloader_pipeline(images_root, heatmaps_root, output_dir="dataloader_results", batch_size=4, num_batches=3):
    """Test transforms in a dataloader pipeline setting with batched data."""
    os.makedirs(output_dir, exist_ok=True)
    
    
    # Define standard transform configuration
    args = argparse.Namespace(
        input_size=224,
        color_jitter=0.4,
        aa="rand-m9-mstd0.5",
        reprob=0.25,
        recount=1,
        remode="pixel",
        eval_crop_ratio=0.875
    )
    
    # Build transforms
    train_transform = build_transform(is_train=True, args=args)
    val_transform = build_transform(is_train=False, args=args)
    
    # Create dataset with heatmaps and rankings
    dataset = HeatmapImageFolder(images_root, heatmaps_root, transform=train_transform, 
                               return_heatmap=True, return_rankings=True)
    valset = HeatmapImageFolder(images_root, heatmaps_root, transform=val_transform,
                                 return_heatmap=True, return_rankings=True)
    # Create a DataLoader for the dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for easier debugging
        pin_memory=True,
        drop_last=True
    )

    valloader = torch.utils.data.DataLoader(
        valset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 for easier debugging
        pin_memory=True,
        drop_last=True
    )
    
    # Process a few batches
    for batch_idx, (images, targets, rankings, heatmaps) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
            
        print(f"Processing batch {batch_idx+1}/{num_batches}")
        
        # Create ranking visualizations
        orig_ranking_grids = []
        for i in range(batch_size):
            ranking_grid = visualize_ranking(rankings[i])
            orig_ranking_grids.append(ranking_grid)
        
        # Plot results
        fig, axes = plt.subplots(batch_size, 3, figsize=(20, 4*batch_size))
        
        for i in range(batch_size):
            # Original image
            orig_img = denormalize(images[i]) if isinstance(images[i], torch.Tensor) else orig_images[i]
            orig_img_np = tensor_to_numpy(orig_img) if isinstance(orig_img, torch.Tensor) else orig_img
            
            # Original heatmap
            orig_hm = heatmaps[i]
            orig_hm_np = orig_hm.squeeze().cpu().numpy() if isinstance(orig_hm, torch.Tensor) else orig_hm
            
            # Original ranking visualization
            orig_ranking_vis = orig_ranking_grids[i]
            
            # Plot
            axes[i, 0].imshow(orig_img_np)
            axes[i, 0].set_title(f"Original Image\nTarget: {targets[i].item()}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(orig_hm_np, cmap='viridis')
            axes[i, 1].set_title("Original Heatmap")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(orig_ranking_vis, cmap='viridis')
            axes[i, 2].set_title("Original Ranking")
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"batch_{batch_idx+1}_pipeline.png"))
        plt.show()
        plt.close(fig)

    for batch_idx, (images, targets, rankings, heatmaps) in enumerate(valloader):
        if batch_idx >= num_batches:
            break
            
        print(f"Processing batch {batch_idx+1}/{num_batches}")
        
        # Create ranking visualizations
        orig_ranking_grids = []
        for i in range(batch_size):
            ranking_grid = visualize_ranking(rankings[i])
            orig_ranking_grids.append(ranking_grid)
        
        # Plot results
        fig, axes = plt.subplots(batch_size, 3, figsize=(20, 4*batch_size))
        
        for i in range(batch_size):
            # Original image
            orig_img = denormalize(images[i]) if isinstance(images[i], torch.Tensor) else orig_images[i]
            orig_img_np = tensor_to_numpy(orig_img) if isinstance(orig_img, torch.Tensor) else orig_img
            
            # Original heatmap
            orig_hm = heatmaps[i]
            orig_hm_np = orig_hm.squeeze().cpu().numpy() if isinstance(orig_hm, torch.Tensor) else orig_hm
            
            # Original ranking visualization
            orig_ranking_vis = orig_ranking_grids[i]
            
            # Plot
            axes[i, 0].imshow(orig_img_np)
            axes[i, 0].set_title(f"Original Image\nTarget: {targets[i].item()}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(orig_hm_np, cmap='viridis')
            axes[i, 1].set_title("Original Heatmap")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(orig_ranking_vis, cmap='viridis')
            axes[i, 2].set_title("Original Ranking")
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
    
    print(f"DataLoader pipeline test completed. Results saved to {output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir",
                        default=r"C:\Users\joren\Documents\_Uni\Master\Thesis\imagenet_subset\train")
    parser.add_argument("--heatmaps_dir",
                        default=r"C:\Users\joren\Documents\_Uni\Master\Thesis\imagenet_subset\train_heat")
    parser.add_argument("--dataset",
                        action="store_true",
                        help="Test on full dataset")
    parser.add_argument("--pipeline",
                        action="store_true",
                        default=True,
                        help="Test dataloader pipeline")
    parser.add_argument("--image",
                        default=r"C:\Users\joren\Documents\_Uni\Master\Thesis\imagenet_subset\train\n02500267\n02500267_6113.JPEG",
                        help="Path to single image to test")
    parser.add_argument("--heatmap",
                        default=r"C:\Users\joren\Documents\_Uni\Master\Thesis\imagenet_subset\train_heat\n02500267\n02500267_6113.JPEG",
                        help="Corresponding heatmap for the single image")
    parser.add_argument("--output_dir",      default=r"transforms\transform_results")
    parser.add_argument("--rand_output_dir", default=r"transforms\rand_augment_results")
    parser.add_argument("--single_output_dir", default=r"transforms\single_image_results")
    parser.add_argument("--individual_output_dir", default=r"transforms\individual_transforms_results",
                        help="Directory to save individual transform test results")
    parser.add_argument("--dataloader_output_dir", default=r"transforms\dataloader_results",
                        help="Directory to save dataloader pipeline test results") 
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for dataloader test")
    parser.add_argument("--num_batches", type=int, default=3,
                        help="Number of batches to process in dataloader test")
    args = parser.parse_args()

    if args.dataset:
        # test on full dataset
        test_transforms_with_dataset(args.images_dir, args.heatmaps_dir,
                                    output_dir=args.output_dir)
        test_rand_augment_variations(args.images_dir, args.heatmaps_dir,
                                    output_dir=args.rand_output_dir)
        # Test dataloader pipeline
    if args.pipeline:
        test_dataloader_pipeline(args.images_dir, args.heatmaps_dir,
                               output_dir=args.dataloader_output_dir,
                               batch_size=args.batch_size,
                               num_batches=args.num_batches)

    # optional single-image test
    if args.image:
        test_single_image(args.image, args.heatmap,
                          args=args, output_dir=args.single_output_dir)

    # Add test for individual transforms
    if args.image:
        test_individual_transforms(args.image, args.heatmap, 
                                  output_dir=args.individual_output_dir)

    print("All tests completed successfully!")