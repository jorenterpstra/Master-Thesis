import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import argparse
from pathlib import Path
import albumentations as A
from torchvision.datasets.folder import ImageFolder, default_loader, IMG_EXTENSIONS

from transforms import build_transform, AlbumentationsRandAugment
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

def test_transforms_with_dataset(images_root, heatmaps_root, output_dir="transform_results"):
    """Test transforms with the dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Define test configurations
    test_configs = [
        {"name": "basic", "args": {"input_size": 224, "color_jitter": 0, "aa": None, "reprob": 0, "eval_crop_ratio": 1.0}},
        {"name": "color_jitter", "args": {"input_size": 224, "color_jitter": 0.4, "aa": None, "reprob": 0, "eval_crop_ratio": 1.0}},
        {"name": "rand_augment", "args": {"input_size": 224, "color_jitter": 0, "aa": "rand-m9-mstd0.5", "reprob": 0, "eval_crop_ratio": 1.0}},
        {"name": "full_augment", "args": {"input_size": 224, "color_jitter": 0.4, "aa": "rand-m9-mstd0.5", "reprob": 0.25, "recount": 1, "eval_crop_ratio": 1.0}}
    ]
    
    # Create dataset
    dataset = HeatmapImageFolder(images_root, heatmaps_root, transform=None, return_path=True)
    
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
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Get a sample
            image, heatmap, _, path = dataset[i]
            
            # convert original image for display
            if isinstance(image, torch.Tensor):
                orig_img_disp = denormalize(image)
                orig_img_disp = tensor_to_numpy(orig_img_disp)
            else:
                orig_img_disp = image
            
            # convert original heatmap for display
            if isinstance(heatmap, torch.Tensor):
                orig_heatmap_disp = heatmap.squeeze().numpy()
            else:
                orig_heatmap_disp = heatmap
            
            # Apply train transform
            dataset.transform = train_transform
            train_img, train_heatmap, _, _ = dataset[i]
            
            # Apply val transform
            dataset.transform = val_transform
            val_img, val_heatmap, _, _ = dataset[i]
            
            # Convert tensors for display if needed
            if isinstance(train_img, torch.Tensor):
                train_img_disp = denormalize(train_img)
                train_img_disp = tensor_to_numpy(train_img_disp)
            else:
                train_img_disp = train_img
                
            if isinstance(val_img, torch.Tensor):
                val_img_disp = denormalize(val_img)
                val_img_disp = tensor_to_numpy(val_img_disp)
            else:
                val_img_disp = val_img
                
            if isinstance(train_heatmap, torch.Tensor):
                train_heatmap_disp = train_heatmap.squeeze().numpy()
            else:
                train_heatmap_disp = train_heatmap
                
            if isinstance(val_heatmap, torch.Tensor):
                val_heatmap_disp = val_heatmap.squeeze().numpy()
            else:
                val_heatmap_disp = val_heatmap
            
            # Plot
            axes[i, 0].imshow(orig_img_disp)
            axes[i, 0].set_title(f"Original Image {i+1}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(orig_heatmap_disp, cmap='viridis')
            axes[i, 1].set_title(f"Original Heatmap {i+1}")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(train_img_disp)
            axes[i, 2].set_title(f"Train Transform")
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(train_heatmap_disp, cmap='viridis')
            axes[i, 3].set_title(f"Train Heatmap")
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{config['name']}_transforms.png"))
        plt.close(fig)
        
        # Reset dataset transform
        dataset.transform = None
    
    print(f"Transform tests completed. Results saved to {output_dir}.")

def test_rand_augment_variations(images_root, heatmaps_root, output_dir="rand_augment_results"):
    """Test different RandAugment configurations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset
    dataset = HeatmapImageFolder(images_root, heatmaps_root, transform=None, return_path=True)
    
    # Test sample image with multiple augment applications
    sample_idx = 0
    img, heatmap, _, _ = dataset[sample_idx]
    
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
        {"name": "basic",     "args": {"input_size":224,"color_jitter":0,"aa":None,"reprob":0,"eval_crop_ratio":1.0}},
        {"name": "color_jitter","args":{"input_size":224,"color_jitter":0.4,"aa":None,"reprob":0,"eval_crop_ratio":1.0}},
        {"name": "rand_augment","args":{"input_size":224,"color_jitter":0,"aa":"rand-m9-mstd0.5","reprob":0,"eval_crop_ratio":1.0}},
        {"name": "full_augment","args":{"input_size":224,"color_jitter":0.4,"aa":"rand-m9-mstd0.5","reprob":0.25,"recount":1,"eval_crop_ratio":1.0}},
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir",
                        default=r"C:\Users\joren\Documents\_Uni\Master\Thesis\imagenet_subset\train")
    parser.add_argument("--heatmaps_dir",
                        default=r"C:\Users\joren\Documents\_Uni\Master\Thesis\imagenet_subset\train_heat")
    parser.add_argument("--image",
                        default=r"C:\Users\joren\Documents\_Uni\Master\Thesis\imagenet_subset\train\n02500267\n02500267_6113.JPEG",
                        help="Path to single image to test")
    parser.add_argument("--heatmap",
                        default=r"C:\Users\joren\Documents\_Uni\Master\Thesis\imagenet_subset\train_heat\n02500267\n02500267_6113.JPEG",
                        help="Corresponding heatmap for the single image")
    parser.add_argument("--output_dir",      default="transform_results")
    parser.add_argument("--rand_output_dir", default="rand_augment_results")
    parser.add_argument("--single_output_dir", default="single_image_results")
    args = parser.parse_args()

    # test on full dataset
    test_transforms_with_dataset(args.images_dir, args.heatmaps_dir,
                                 output_dir=args.output_dir)
    test_rand_augment_variations(args.images_dir, args.heatmaps_dir,
                                 output_dir=args.rand_output_dir)

    # optional single-image test
    if args.image:
        test_single_image(args.image, args.heatmap,
                          args=args, output_dir=args.single_output_dir)

    print("All tests completed successfully!")