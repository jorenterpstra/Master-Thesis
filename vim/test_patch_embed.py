import torch
import torch.nn as nn
import numpy as np
from custom_patch_embed import PatchEmbedCustom

def test_different_batch_sizes():
    """Test PatchEmbedCustom with different batch sizes."""
    print("\n=== Testing different batch sizes ===")
    custom_order = [1, 0, 2, 3]  # Simple order for a 2x2 grid
    patch_embed = PatchEmbedCustom(img_size=4, patch_size=2, stride=2, 
                                  in_chans=3, embed_dim=8, patch_order=custom_order)
    
    # Test with batch sizes 1, 2, and 4
    for batch_size in [1, 2, 4]:
        x = torch.ones((batch_size, 3, 4, 4))
        output, output_reordered = patch_embed(x)
        print(f"Batch size {batch_size}: Output shape = {output.shape}, Reordered shape = {output_reordered.shape}")
        assert output.shape == (batch_size, 4, 8), f"Expected shape ({batch_size}, 4, 8) but got {output.shape}"
        assert output_reordered.shape == output.shape, "Reordered output should have the same shape"

def test_different_image_sizes():
    """Test PatchEmbedCustom with different image sizes."""
    print("\n=== Testing different image sizes ===")
    
    for img_size in [8, 16, 32]:
        # Calculate patches in grid (non-overlapping)
        patch_size = 4
        grid_size = img_size // patch_size
        num_patches = grid_size * grid_size
        custom_order = list(range(num_patches))
        np.random.shuffle(custom_order)  # Randomize order for testing
        
        patch_embed = PatchEmbedCustom(img_size=img_size, patch_size=patch_size, stride=patch_size, 
                                      in_chans=3, embed_dim=16, patch_order=custom_order)
        
        x = torch.randn((2, 3, img_size, img_size))
        output, output_reordered = patch_embed(x)
        
        print(f"Image size {img_size}x{img_size}: Grid = {grid_size}x{grid_size}, Patches = {num_patches}")
        print(f"Output shape = {output.shape}")
        assert output.shape == (2, num_patches, 16), f"Expected shape (2, {num_patches}, 16) but got {output.shape}"

def test_different_patch_configs():
    """Test PatchEmbedCustom with different patch sizes (non-overlapping)."""
    print("\n=== Testing different patch configurations ===")
    
    configs = [
        {"img_size": 16, "patch_size": 4},
        {"img_size": 32, "patch_size": 8},
        {"img_size": 32, "patch_size": 4},
        {"img_size": 64, "patch_size": 16},
    ]
    
    for config in configs:
        img_size = config["img_size"]
        patch_size = config["patch_size"]
        stride = patch_size  # Non-overlapping
        
        grid_size = img_size // patch_size
        num_patches = grid_size * grid_size
        
        custom_order = list(range(num_patches))
        np.random.shuffle(custom_order)  # Random order for testing
        
        patch_embed = PatchEmbedCustom(img_size=img_size, patch_size=patch_size, stride=stride,
                                      in_chans=3, embed_dim=16, patch_order=custom_order)
        
        x = torch.randn((1, 3, img_size, img_size))
        output, output_reordered = patch_embed(x)
        
        print(f"Config: {config} → Grid: {grid_size}x{grid_size}, Patches: {num_patches}")
        print(f"Output shape = {output.shape}")
        assert output.shape == (1, num_patches, 16), f"Expected shape (1, {num_patches}, 16) but got {output.shape}"

def test_common_pattern_orders():
    """Test PatchEmbedCustom with common patterns for patch ordering."""
    print("\n=== Testing common patch ordering patterns ===")
    
    img_size = 16
    patch_size = 4
    grid_size = img_size // patch_size
    num_patches = grid_size * grid_size  # 16 patches for 16x16 image with 4x4 patches
    
    # Test different ordering patterns
    patterns = {
        "Identity": list(range(num_patches)),
        "Reverse": list(reversed(range(num_patches))),
        "Spiral inward": [0, 1, 2, 3, 7, 11, 15, 14, 13, 12, 8, 4, 5, 6, 10, 9],
        "Zigzag": [0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12],
        "Columns-first": [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15],
    }
    
    # Create a test image where each patch has a unique value
    # This will help us verify that reordering works correctly
    x = torch.zeros((1, 1, img_size, img_size))
    for i in range(grid_size):
        for j in range(grid_size):
            patch_idx = i * grid_size + j
            # Set a single value for the entire patch
            x[0, 0, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = patch_idx
    
    for pattern_name, order in patterns.items():
        patch_embed = PatchEmbedCustom(img_size=img_size, patch_size=patch_size, stride=patch_size,
                                      in_chans=1, embed_dim=4, patch_order=order)
        
        output, output_reordered = patch_embed(x)
        
        print(f"\nPattern: {pattern_name}")
        print(f"Order: {order}")
        
        # Verify that patches were reordered correctly by checking patch values
        is_correct = True
        for i in range(num_patches):
            # The convolution in PatchEmbedCustom has weights=1.0, so each patch value
            # gets multiplied by the patch size squared (sum of all 1s in the 4x4 kernel)
            expected_idx = order[i]
            expected_value = expected_idx * (patch_size * patch_size)
            actual_value = float(output_reordered[0, i].mean())
            
            is_correct = is_correct and (abs(actual_value - expected_value) < 0.01)
            print(f"Position {i}: Expected value {expected_value}, Got {actual_value}")
        
        assert is_correct, f"Pattern '{pattern_name}' did not reorder patches correctly"

def test_verify_patch_order_with_distinct_patches():
    """Test to verify patch ordering using visually distinct patches."""
    print("\n=== Testing patch ordering with distinct patches ===")
    
    img_size = 8
    patch_size = 4
    grid_size = img_size // patch_size
    num_patches = grid_size * grid_size  # 4 patches
    
    # Define a few test orders
    test_orders = [
        [0, 1, 2, 3],  # Identity (original order)
        [3, 2, 1, 0],  # Reverse
        [0, 2, 1, 3],  # Custom order 1
        [2, 0, 3, 1],  # Custom order 2
    ]
    
    for order_idx, custom_order in enumerate(test_orders):
        # Create an input where each patch has a distinct pattern
        x = torch.zeros((1, 3, img_size, img_size))
        
        # Create distinct patterns for each patch
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx == 0:  # Top-left: all ones
                    x[0, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = 1.0
                elif idx == 1:  # Top-right: horizontal gradient
                    for k in range(patch_size):
                        x[0, :, i*patch_size:(i+1)*patch_size, j*patch_size+k] = k / float(patch_size)
                elif idx == 2:  # Bottom-left: vertical gradient
                    for k in range(patch_size):
                        x[0, :, i*patch_size+k, j*patch_size:(j+1)*patch_size] = k / float(patch_size)
                else:  # Bottom-right: checkerboard
                    for k in range(patch_size):
                        for l in range(patch_size):
                            x[0, :, i*patch_size+k, j*patch_size+l] = (k + l) % 2
        
        patch_embed = PatchEmbedCustom(img_size=img_size, patch_size=patch_size, stride=patch_size,
                                      in_chans=3, embed_dim=8, patch_order=custom_order)
        
        output, output_reordered = patch_embed(x)
        
        print(f"\nTesting order: {custom_order}")
        
        # Compute mean values for each patch to identify them
        original_means = [output[0, i].mean().item() for i in range(num_patches)]
        reordered_means = [output_reordered[0, i].mean().item() for i in range(num_patches)]
        
        print("Original patch means:", [f"{val:.4f}" for val in original_means])
        print("Reordered patch means:", [f"{val:.4f}" for val in reordered_means])
        
        # Verify reordering
        for i in range(num_patches):
            expected_idx = custom_order[i]
            expected_mean = original_means[expected_idx]
            actual_mean = reordered_means[i]
            assert abs(expected_mean - actual_mean) < 1e-6, \
                f"Patch at position {i} should have value from original patch {expected_idx}"
        
        print("Ordering verified successfully ✓")

def test_same_ordering_across_batch():
    """Test that the same ordering is applied to all images in a batch."""
    print("\n=== Testing same ordering across batch ===")
    
    img_size = 8
    patch_size = 4
    num_patches = 4  # 2x2 grid
    batch_size = 3
    custom_order = [3, 1, 0, 2]  # Custom ordering
    
    patch_embed = PatchEmbedCustom(img_size=img_size, patch_size=patch_size, stride=patch_size,
                                  in_chans=1, embed_dim=4, patch_order=custom_order)
    
    # Create a batch where each image has different values in its patches
    x = torch.zeros((batch_size, 1, img_size, img_size))
    for b in range(batch_size):
        for i in range(2):
            for j in range(2):
                idx = i * 2 + j
                # Each batch element has different pattern: b+10 + idx ensures unique values
                x[b, 0, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = b*10 + idx
    
    output, output_reordered = patch_embed(x)
    
    # For each batch element, verify the patch reordering
    for b in range(batch_size):
        print(f"\nBatch element {b}:")
        print("Original patches:", [output[b, i, 0].item() for i in range(num_patches)])
        print("Reordered patches:", [output_reordered[b, i, 0].item() for i in range(num_patches)])
        
        # Verify that each batch element was reordered correctly
        for i in range(num_patches):
            expected_idx = custom_order[i]
            # Use allclose instead of isclose to get a single boolean result
            assert torch.allclose(output_reordered[b, i], output[b, expected_idx]), \
                f"Batch {b}, position {i} should have value from original patch {expected_idx}"

if __name__ == "__main__":
    test_different_batch_sizes()
    test_different_image_sizes()
    test_different_patch_configs()
    test_common_pattern_orders()
    test_verify_patch_order_with_distinct_patches()
    test_same_ordering_across_batch()
