import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path

def load_and_preprocess_image(image_path, size=224):
    """Load and preprocess image to tensor"""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image)

def extract_patches(x, patch_size=16):
    """Extract patches using the same logic as the model"""
    B, C, H, W = x.unsqueeze(0).shape
    P = patch_size
    
    # # Unfold into patches [B, C, P, P, num_patches, num_patches]
    # patches = x.unsqueeze(0).unfold(2, P, P).unfold(3, P, P)
    
    # # Reshape to [B*num_patches*num_patches, C, P, P]
    # patches = patches.permute(0, 4, 5, 1, 2, 3).contiguous()
    # patches = patches.view(-1, C, P, P)
    print(x.shape)
    # out = x.unfold(-2, P, P).unfold(-1, P, P)
    # out = out.contiguous().view(B, C, -1, P, P).permute(0,2,1,4,3)
    # out = out.contiguous().view(-1, C, P, P)
    patches = x.unsqueeze(0).unfold(2, P, P).unfold(3, P, P)  # [B, C, H//P, W//P, P, P]
    patches = patches.permute(0, 2, 3, 1, 4, 5)   # [B, H//P, W//P, C, P, P]
    patches = patches.reshape(-1, C, P, P)        # [B*H//P*W//P, C, P, P]
    print(patches.shape)
    return patches

def visualize_patches(image_path, patch_size=16, max_patches=None):
    """Visualize original image and its patches"""
    # Load and process image
    image_tensor = load_and_preprocess_image(image_path)
    patches = extract_patches(image_tensor, patch_size)
    
    # Convert tensors to displayable format
    image_np = image_tensor.permute(1, 2, 0).numpy()
    patches_np = patches.permute(0, 2, 3, 1).numpy()
    
    num_patches = patches.shape[0]
    if max_patches is None:
        max_patches = num_patches
    
    # Create figure
    fig = plt.figure(figsize=(15, 8))
    
    # Plot original image
    ax1 = plt.subplot(121)
    ax1.imshow(image_np)
    ax1.set_title('Original Image')
    
    # Add grid to show patch boundaries
    num_grid_lines = 224 // patch_size
    for i in range(num_grid_lines + 1):
        ax1.axhline(y=i * patch_size, color='white', alpha=0.3)
        ax1.axvline(x=i * patch_size, color='white', alpha=0.3)
    
    # Plot patches
    ax2 = plt.subplot(122)
    n = int(np.ceil(np.sqrt(max_patches)))
    grid_size = patch_size * n
    patch_grid = np.zeros((grid_size, grid_size, 3))
    
    for idx in range(min(max_patches, num_patches)):
        i, j = idx // n, idx % n
        y_start = i * patch_size
        y_end = (i + 1) * patch_size
        x_start = j * patch_size
        x_end = (j + 1) * patch_size
        patch_grid[y_start:y_end, x_start:x_end] = patches_np[idx]
    
    ax2.imshow(patch_grid)
    ax2.set_title(f'Extracted Patches ({patch_size}x{patch_size})')
    
    # Add grid to show patch boundaries
    for i in range(n + 1):
        ax2.axhline(y=i * patch_size - 0.5, color='white', alpha=0.3)
        ax2.axvline(x=i * patch_size - 0.5, color='white', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_test_pattern(size=224, method='checkerboard'):
    """Create a synthetic test image with recognizable patterns"""
    # Create a tensor with distinct patterns
    test_image = torch.zeros(3, size, size)
    
    if method == 'checkerboard':
        # Red channel: horizontal gradient
        test_image[0] = torch.linspace(0, 1, size).repeat(size, 1)
        # Green channel: vertical gradient
        test_image[1] = torch.linspace(0, 1, size).repeat(size, 1).t()
        # Blue channel: checkerboard pattern
        checkerboard = torch.zeros(size, size)
        square_size = size // 8
        for i in range(0, size, square_size * 2):
            for j in range(0, size, square_size * 2):
                checkerboard[i:i+square_size, j:j+square_size] = 1
                checkerboard[i+square_size:i+2*square_size, j+square_size:j+2*square_size] = 1
        test_image[2] = checkerboard
    elif method == 'range':
        # Adding sequential values that continue from previous ones
        test_image[0] = torch.arange(size * size).view(size, size)
    
    return test_image

def test_synthetic(image_size = 224, patch_size = 16, method='range'):
    """Test patch extraction with synthetic pattern"""
    # Create synthetic test image
    test_image = create_test_pattern(size=image_size, method=method)
    
    # Extract and visualize patches
    patches = extract_patches(test_image, patch_size=patch_size)
    
    # Convert tensors to displayable format
    image_np = test_image.permute(1, 2, 0).numpy()
    patches_np = patches.permute(0, 2, 3, 1).numpy()
    
    # Create figure
    fig = plt.figure(figsize=(15, 8))
    
    # Plot original image
    ax1 = plt.subplot(121)
    ax1.imshow(image_np)
    ax1.set_title('Synthetic Test Pattern')
    
    # Add grid to show patch boundaries
    num_grid_lines = image_size // patch_size
    for i in range(num_grid_lines + 1):
        ax1.axhline(y=i * patch_size, color='white', alpha=0.3)
        ax1.axvline(x=i * patch_size, color='white', alpha=0.3)
    
    # Plot patches
    ax2 = plt.subplot(122)
    n = int(np.ceil(np.sqrt(len(patches))))
    grid_size = patch_size * n
    patch_grid = np.zeros((grid_size, grid_size, 3))
    
    for idx in range(len(patches)):
        i, j = idx // n, idx % n
        y_start = i * patch_size
        y_end = (i + 1) * patch_size
        x_start = j * patch_size
        x_end = (j + 1) * patch_size
        patch_grid[y_start:y_end, x_start:x_end] = patches_np[idx]
    
    ax2.imshow(patch_grid)
    ax2.set_title(f'Extracted Patches ({patch_size}x{patch_size})')
    
    # Add grid to show patch boundaries
    for i in range(n + 1):
        ax2.axhline(y=i * patch_size - 0.5, color='white', alpha=0.3)
        ax2.axvline(x=i * patch_size - 0.5, color='white', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def test_unfolding():
    """Run tests with different patch sizes and images"""
    # Test with a sample image
    image_path = r"C:\Users\joren\Documents\_Uni\Master\Thesis\imagenet_subset\train\n02749479\n02749479_5745.JPEG"  # Replace with actual path
    patch_sizes = [16, 32]
    
    for patch_size in patch_sizes:
        print(f"\nTesting patch size: {patch_size}x{patch_size}")
        visualize_patches(image_path, patch_size=patch_size)

if __name__ == "__main__":
    print("Testing with small synthetic pattern (4x4, 2x2 patches)...")
    test_synthetic(image_size=4, patch_size=2)

    print("\nTesting with larger synthetic pattern (16x16, 4x4 patches)...")
    test_synthetic(image_size=16, patch_size=4)
    
    print("\nTesting with regular synthetic pattern (224x224, 16x16 patches)...")
    test_synthetic(image_size=224, patch_size=16)
    
    print("\nTesting with real image...")
    test_unfolding()
