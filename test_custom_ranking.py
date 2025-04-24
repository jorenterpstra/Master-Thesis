#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse
import random

def create_test_image_batch(batch_size=2, img_size=224, channels=3):
    """Create a batch of test images with different patterns for easy visualization"""
    batch = []
    for i in range(batch_size):
        # Create a base image with a unique pattern for each sample
        img = np.zeros((img_size, img_size, channels), dtype=np.float32)
        
        # Add a distinctive pattern based on the index
        if i % 3 == 0:
            # Horizontal stripes
            for y in range(img_size):
                if (y // 20) % 2 == 0:
                    img[y, :, 0] = 1.0  # Red channel
        elif i % 3 == 1:
            # Vertical stripes
            for x in range(img_size):
                if (x // 20) % 2 == 0:
                    img[: ,x, 1] = 1.0  # Green channel
        else:
            # Diagonal pattern
            for y in range(img_size):
                for x in range(img_size):
                    if ((x + y) // 20) % 2 == 0:
                        img[y, x, 2] = 1.0  # Blue channel
        
        # Convert to PyTorch tensor and change to [C, H, W] format
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        batch.append(img_tensor)
    
    # Stack to create batch
    return torch.stack(batch)

def create_custom_ranking(batch_size=2, num_patches=196, ranking_type="default"):
    """Create custom rankings for testing"""
    if ranking_type == "default":
        # Default ranking is just 0 to num_patches-1
        rank = torch.arange(num_patches).unsqueeze(0).repeat(batch_size, 1)
    elif ranking_type == "reverse":
        # Reverse ranking
        rank = torch.arange(num_patches-1, -1, -1).unsqueeze(0).repeat(batch_size, 1)
    elif ranking_type == "random":
        # Random ranking (different for each image)
        rank = torch.zeros(batch_size, num_patches, dtype=torch.long)
        for i in range(batch_size):
            rank[i] = torch.randperm(num_patches)
    elif ranking_type == "center_first":
        # Center patches first, then outward
        center_x = int(np.sqrt(num_patches) / 2)
        center_y = center_x
        grid_size = int(np.sqrt(num_patches))
        
        distances = []
        for i in range(grid_size):
            for j in range(grid_size):
                dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                distances.append((i * grid_size + j, dist))
        
        # Sort by distance (closest first)
        distances.sort(key=lambda x: x[1])
        indices = [idx for idx, _ in distances]
        rank = torch.tensor(indices).unsqueeze(0).repeat(batch_size, 1)
    elif ranking_type == "mixed":
        # Different ranking for each image in batch
        rank = torch.zeros(batch_size, num_patches, dtype=torch.long)
        rank[0] = torch.arange(num_patches)  # First image: default order
        for i in range(1, batch_size):
            rank[i] = torch.randperm(num_patches)  # Rest: random order
    else:
        raise ValueError(f"Unknown ranking type: {ranking_type}")
    
    return rank

def create_patch_embeddings(images, patch_size=16, embed_dim=192):
    """
    Convert images to patch embeddings similar to how the model would do it
    This simulates the patch_embed step of Vision Transformers/Mamba
    """
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by patch size"
    
    # Number of patches in height and width
    grid_h, grid_w = H // patch_size, W // patch_size
    num_patches = grid_h * grid_w
    
    # Create unique embedding for each patch position (for easy tracking)
    patch_embeddings = torch.zeros((B, num_patches, embed_dim))
    
    # Create embeddings with recognizable patterns for visualization and verification
    for b in range(B):  # For each image in batch
        for i in range(grid_h):
            for j in range(grid_w):
                # Calculate patch index
                patch_idx = i * grid_w + j
                
                # Create a unique pattern for this patch
                # We'll use the patch position to create a recognizable pattern
                # First half of embedding: image index marker
                patch_embeddings[b, patch_idx, 0:embed_dim//3] = b + 1
                
                # Second third: row position marker (i)
                patch_embeddings[b, patch_idx, embed_dim//3:2*embed_dim//3] = i + 1
                
                # Last third: column position marker (j)
                patch_embeddings[b, patch_idx, 2*embed_dim//3:] = j + 1
    
    return patch_embeddings

def apply_custom_ranking(x, custom_rank):
    """
    Apply custom ranking to reorder patches - this is the exact code 
    from VisionMamba that we want to test
    """
    B, P, C = x.shape  # batch size, number of patches, embedding dimension
    
    # Ensure custom_rank has the right shape
    assert custom_rank.shape[0] == B and custom_rank.shape[1] == P, \
        f"Custom rank shape {custom_rank.shape} doesn't match expected ({B}, {P})"
    
    # Expand custom_rank to match embedding dimension 
    idx = custom_rank.unsqueeze(-1).expand(-1, -1, C)
    
    # Apply the reordering using torch.gather
    reordered_x = torch.gather(x, dim=1, index=idx)
    
    return reordered_x

def visualize_patch_embeddings(embeddings, grid_size, save_path, title):
    """Visualize patch embeddings to verify correct ordering"""
    B, P, C = embeddings.shape
    
    fig, axs = plt.subplots(B, 3, figsize=(15, 5 * B))
    if B == 1:
        axs = axs.reshape(1, -1)  # Handle case with only 1 image
        
    for b in range(B):
        # Extract the different components of our embeddings
        batch_markers = embeddings[b, :, 0].reshape(grid_size, grid_size).detach().numpy()
        row_markers = embeddings[b, :, C//3].reshape(grid_size, grid_size).detach().numpy()
        col_markers = embeddings[b, :, 2*C//3].reshape(grid_size, grid_size).detach().numpy()
        
        # Plot each component
        im0 = axs[b, 0].imshow(batch_markers)
        axs[b, 0].set_title(f"Image {b} - Batch Markers")
        plt.colorbar(im0, ax=axs[b, 0])
        
        im1 = axs[b, 1].imshow(row_markers)
        axs[b, 1].set_title(f"Image {b} - Row Position")
        plt.colorbar(im1, ax=axs[b, 1])
        
        im2 = axs[b, 2].imshow(col_markers)
        axs[b, 2].set_title(f"Image {b} - Column Position")
        plt.colorbar(im2, ax=axs[b, 2])
        
        # Add grid lines
        for ax in axs[b]:
            for i in range(grid_size):
                ax.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.5)
                ax.axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.5)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def visualize_spatial_ranking(custom_rank, grid_size, save_path, title):
    """Visualize the ranking order spatially alongside the original patch indices."""
    B, P = custom_rank.shape
    
    # Create 2 columns per image: Original Index | Spatial Rank
    fig, axs = plt.subplots(B, 2, figsize=(16, 8 * B), squeeze=False) # Ensure axs is always 2D
        
    for b in range(B):
        # --- Plot 1: Original Patch Indices --- 
        original_indices_grid = np.arange(P).reshape(grid_size, grid_size)
        
        ax_orig = axs[b, 0]
        im_orig = ax_orig.imshow(original_indices_grid, cmap='gray')
        ax_orig.set_title(f"Image {b} - Original Patch Indices")
        plt.colorbar(im_orig, ax=ax_orig, label="Original Index (0 to {P-1})")
        
        # Add grid lines and numbers for original indices
        for i in range(grid_size):
            ax_orig.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.5)
            ax_orig.axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.5)
        for r in range(grid_size):
            for c in range(grid_size):
                idx_val = original_indices_grid[r, c]
                ax_orig.text(c, r, f"{int(idx_val)}", ha='center', va='center', 
                             color='red', fontsize=6, fontweight='bold')

        # --- Plot 2: Spatial Ranking Order --- 
        spatial_rank_grid = np.zeros((grid_size, grid_size))
        for original_patch_idx in range(P):
            rank = custom_rank[b, original_patch_idx].item()
            row = original_patch_idx // grid_size
            col = original_patch_idx % grid_size
            spatial_rank_grid[row, col] = rank
            
        ax_rank = axs[b, 1]
        im_rank = ax_rank.imshow(spatial_rank_grid, cmap='viridis')
        ax_rank.set_title(f"Image {b} - Spatial Ranking Order")
        plt.colorbar(im_rank, ax=ax_rank, label="Processing Order (Rank)")
        
        # Add grid lines and numbers for ranks
        for i in range(grid_size):
            ax_rank.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.5)
            ax_rank.axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.5)
        for r in range(grid_size):
            for c in range(grid_size):
                rank_val = spatial_rank_grid[r, c]
                ax_rank.text(c, r, f"{int(rank_val)}", ha='center', va='center', 
                             color='white', fontsize=6, fontweight='bold')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def visualize_patches(image_tensor, patch_size=16, ranking=None, filename="patch_visualization.png", title=None):
    """Visualize image with optional patch ranking overlay"""
    # Convert tensor to numpy array for visualization
    if isinstance(image_tensor, torch.Tensor):
        image = image_tensor.permute(1, 2, 0).numpy()
        
        # Ensure values are between 0 and 1
        if image.max() > 1.0:
            image = image / 255.0
    else:
        image = image_tensor
    
    grid_size = image.shape[0] // patch_size
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # Draw grid lines for patches
    for i in range(0, image.shape[0], patch_size):
        plt.axhline(y=i, color='white', linestyle='-', alpha=0.3)
        plt.axvline(x=i, color='white', linestyle='-', alpha=0.3)
    
    # If ranking is provided, overlay the ranking numbers
    if ranking is not None:
        for i, rank in enumerate(ranking):
            y = (i // grid_size) * patch_size + patch_size // 2
            x = (i % grid_size) * patch_size + patch_size // 2
            plt.text(x, y, str(rank.item()), color='white', ha='center', va='center',
                     bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="white", alpha=0.7))
    
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def test_reordering_logic(save_dir='./reordering_tests'):
    """Test the core reordering logic that would be used in VisionMamba"""
    print("\n=== Testing Patch Reordering Logic ===")
    
    # Create directory for results
    os.makedirs(save_dir, exist_ok=True)
    
    # Parameters
    batch_size = 2
    img_size = 224
    patch_size = 16
    grid_size = img_size // patch_size
    num_patches = (img_size // patch_size) ** 2
    embed_dim = 192
    
    # Create test batch and patch embeddings
    images = create_test_image_batch(batch_size, img_size)
    patch_embeddings = create_patch_embeddings(images, patch_size, embed_dim)
    
    # Visualize original patch embeddings
    visualize_patch_embeddings(
        patch_embeddings, 
        grid_size, 
        os.path.join(save_dir, "original_embeddings.png"),
        "Original Patch Embeddings"
    )
    
    # Create a variety of rankings to test
    ranking_types = ["default", "reverse", "center_first", "random", "mixed"]
    
    for ranking_type in ranking_types:
        print(f"\nTesting ranking type: {ranking_type}")
        
        # Create custom ranking
        custom_rank = create_custom_ranking(batch_size, num_patches, ranking_type)
        
        # Apply reordering
        reordered_embeddings = apply_custom_ranking(patch_embeddings, custom_rank)
        
        # Visualize the reordered sequence embeddings
        visualize_patch_embeddings(
            reordered_embeddings, 
            grid_size, 
            os.path.join(save_dir, f"{ranking_type}_reordered_sequence.png"), # Renamed for clarity
            f"Reordered Sequence Embeddings ({ranking_type})"
        )
        
        # --- Add call to new visualization ---
        visualize_spatial_ranking(
            custom_rank,
            grid_size,
            os.path.join(save_dir, f"{ranking_type}_spatial_ranking.png"),
            f"Spatial Ranking Order ({ranking_type})"
        )
        # --- End added call ---
        
        # ---- Validation logic ----
        # The issue is we were mixing up how to check reordering
        # The actual validation requires a different approach:
        
        # For each batch and position in reordered embeddings
        for b in range(batch_size):
            # We need to verify that each position in the reordered sequence 
            # contains the patch that should be there according to custom_rank
            for pos in range(num_patches):
                # This position should contain the patch indicated by custom_rank[b, pos]
                original_patch_idx = custom_rank[b, pos].item()
                
                # Get the expected batch, row, and col markers
                expected_batch_marker = b + 1
                expected_row_marker = (original_patch_idx // grid_size) + 1
                expected_col_marker = (original_patch_idx % grid_size) + 1
                
                # Check that the markers match what we expect
                actual_batch_marker = reordered_embeddings[b, pos, 0].item()
                actual_row_marker = reordered_embeddings[b, pos, embed_dim//3].item()
                actual_col_marker = reordered_embeddings[b, pos, 2*embed_dim//3].item()
                
                assert abs(actual_batch_marker - expected_batch_marker) < 1e-5, \
                    f"Batch {b}, pos {pos}, rank[b,pos]={original_patch_idx}: batch marker incorrect"
                
                assert abs(actual_row_marker - expected_row_marker) < 1e-5, \
                    f"Batch {b}, pos {pos}, rank[b,pos]={original_patch_idx}: row marker incorrect"
                
                assert abs(actual_col_marker - expected_col_marker) < 1e-5, \
                    f"Batch {b}, pos {pos}, rank[b,pos]={original_patch_idx}: col marker incorrect"
        
        print(f"✓ {ranking_type} ranking validation passed")
    
    # The test is: Instead of properly applying the patch reordering, 
    # did we accidentally just shuffle the ranking indices themselves?
    # To test this, let's create a deliberately wrong application
    wrong_custom_rank = create_custom_ranking(batch_size, num_patches, "reverse")
    
    # THIS IS WRONG - we're gathering from the custom_rank instead of using it as indices
    # This simulates a common mistake when implementing patch reordering
    try:
        wrong_idx = wrong_custom_rank.unsqueeze(-1).expand(-1, -1, embed_dim) 
        incorrect_reordering = torch.gather(wrong_custom_rank.unsqueeze(-1).expand(-1, -1, embed_dim), 
                                          dim=1, index=wrong_idx)
        
        print("❌ Incorrect reordering logic didn't raise an error as expected")
    except Exception as e:
        print(f"✓ Caught expected error in incorrect reordering logic: {str(e)}")
    
    print("\nPatch reordering logic tests completed!")
    return True

def test_with_real_data_from_file(ranking_file='rankings/ranking_highest_first.pt', save_dir='./test_results_real'):
    """Test with real rankings loaded from file"""
    if not os.path.exists(ranking_file):
        print(f"Warning: Ranking file {ranking_file} not found. Skipping real data test.")
        return False
    
    print(f"\n=== Testing with Real Rankings from {ranking_file} ===")
    
    # Create directory for results
    os.makedirs(save_dir, exist_ok=True)
    
    # Load real ranking tensor
    real_ranking = torch.load(ranking_file)
    print(f"Loaded ranking tensor shape: {real_ranking.shape}")
    
    # Parameters
    batch_size = real_ranking.shape[0]
    img_size = 224
    patch_size = 16
    grid_size = img_size // patch_size
    num_patches = grid_size * grid_size
    embed_dim = 192
    
    # Create test images and patch embeddings
    images = create_test_image_batch(batch_size, img_size)
    patch_embeddings = create_patch_embeddings(images, patch_size, embed_dim)
    
    # Visualize original embeddings
    visualize_patch_embeddings(
        patch_embeddings, 
        grid_size, 
        os.path.join(save_dir, "real_original_embeddings.png"),
        "Original Embeddings (Real Ranking Test)"
    )
    
    # Visualize rankings (spatial layout)
    # --- Use new visualization function --- 
    visualize_spatial_ranking(
        real_ranking,
        grid_size,
        os.path.join(save_dir, "real_spatial_ranking.png"),
        f"Spatial Ranking Order (from {Path(ranking_file).name})"
    )
    # --- End change ---
    
    # Apply reordering
    reordered_embeddings = apply_custom_ranking(patch_embeddings, real_ranking)
    
    # Visualize the reordered sequence embeddings
    visualize_patch_embeddings(
        reordered_embeddings, 
        grid_size, 
        os.path.join(save_dir, "real_reordered_sequence.png"), # Renamed for clarity
        "Reordered Sequence Embeddings (Real Ranking)"
    )
    
    print("Real ranking reordering test completed!")
    return True

def test_small_grid_ranking(save_dir='./small_grid_tests'):
    """
    Test ranking on a small grid (4x4 or 6x6) for easier visualization and debugging.
    This is especially useful for center-first ordering where we need to verify
    that the center patches are indeed processed first.
    """
    print("\n=== Testing Ranking with Small Grid ===")
    
    # Create directory for results
    os.makedirs(save_dir, exist_ok=True)
    
    # Parameters - using a small grid for clarity
    batch_size = 1
    grid_size = 6  # 6x6 grid for easier visualization
    num_patches = grid_size * grid_size
    embed_dim = 192
    
    # Create patch embeddings directly (without actual images)
    patch_embeddings = torch.zeros((batch_size, num_patches, embed_dim))
    
    # Populate embeddings with recognizable patterns
    for b in range(batch_size):
        for i in range(grid_size):
            for j in range(grid_size):
                patch_idx = i * grid_size + j
                
                # First third: batch marker (b+1)
                patch_embeddings[b, patch_idx, 0:embed_dim//3] = b + 1
                
                # Second third: row position marker (i+1)
                patch_embeddings[b, patch_idx, embed_dim//3:2*embed_dim//3] = i + 1
                
                # Last third: column position marker (j+1)
                patch_embeddings[b, patch_idx, 2*embed_dim//3:] = j + 1
    
    # Create center-first ranking specifically for this grid size
    center_y, center_x = grid_size / 2, grid_size / 2
    
    # For even grid sizes, we need to find the geometric center
    # which might be between grid cells
    
    # Calculate distances from center for each patch
    distances = []
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate distance from the center point to the center of this cell
            # Adding 0.5 to get the center of the cell
            dist = np.sqrt((i + 0.5 - center_y)**2 + (j + 0.5 - center_x)**2)
            patch_idx = i * grid_size + j
            distances.append((patch_idx, dist))
    
    # Sort by distance (closest first)
    distances.sort(key=lambda x: x[1])
    
    # Create ranking tensor
    indices = torch.tensor([idx for idx, _ in distances], dtype=torch.long)
    custom_rank = indices.unsqueeze(0)  # Add batch dimension
    
    # Print the complete distance-to-rank mapping for verification
    print("\nDistance-to-Rank Mapping:")
    for idx, (patch_idx, dist) in enumerate(distances):
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        print(f"Rank {idx}: Patch {patch_idx} (row={row}, col={col}, dist={dist:.3f})")
    
    # Apply reordering
    reordered_embeddings = apply_custom_ranking(patch_embeddings, custom_rank)
    
    # Create detailed visualizations of original and reordered patches
    
    # 1. Create matrix showing original indices (0-35 in a 6x6 grid)
    original_indices = np.arange(num_patches).reshape(grid_size, grid_size)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(original_indices, cmap='viridis')
    plt.colorbar(label='Original Index')
    plt.title('Original Patch Indices')
    
    # Add grid lines
    for i in range(grid_size):
        plt.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
        plt.axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
    
    # Add indices as text
    for i in range(grid_size):
        for j in range(grid_size):
            plt.text(j, i, f"{original_indices[i, j]}", ha='center', va='center', 
                    color='white', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "small_grid_original_indices.png"), dpi=200)
    plt.close()
    
    # 2. Create matrix showing the rank assigned to each original position
    rank_matrix = np.zeros((grid_size, grid_size), dtype=int)
    for idx, rank in enumerate(custom_rank[0]):
        original_idx = rank.item()
        row = original_idx // grid_size
        col = original_idx % grid_size
        rank_matrix[row, col] = idx
    
    plt.figure(figsize=(8, 8))
    plt.imshow(rank_matrix, cmap='viridis')
    plt.colorbar(label='Rank (Processing Order)')
    plt.title('Center-First Ranks Assigned to Each Patch')
    
    # Add grid lines
    for i in range(grid_size):
        plt.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
        plt.axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
    
    # Add ranks as text
    for i in range(grid_size):
        for j in range(grid_size):
            plt.text(j, i, f"{rank_matrix[i, j]}", ha='center', va='center', 
                    color='white', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "small_grid_ranks.png"), dpi=200)
    plt.close()
    
    # 3. Alternative visualization: Where each rank will pull from
    position_by_rank = np.zeros((grid_size, grid_size), dtype=int)
    for pos, original_idx in enumerate(custom_rank[0]):
        original_idx = original_idx.item()
        row = pos // grid_size  # Position in the output sequence
        col = pos % grid_size
        position_by_rank[row, col] = original_idx  # Which patch goes here
    
    plt.figure(figsize=(8, 8))
    plt.imshow(position_by_rank, cmap='viridis')
    plt.colorbar(label='Original Patch Index')
    plt.title('Original Patch at Each Position After Reordering')
    
    # Add grid lines
    for i in range(grid_size):
        plt.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
        plt.axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
    
    # Add original indices as text
    for i in range(grid_size):
        for j in range(grid_size):
            plt.text(j, i, f"{position_by_rank[i, j]}", ha='center', va='center', 
                    color='white', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "small_grid_reordered.png"), dpi=200)
    plt.close()
    
    # 4. Create a composite visualization
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))
    
    # Original indices
    im0 = axs[0].imshow(original_indices, cmap='viridis')
    axs[0].set_title('Original Patch Indices')
    plt.colorbar(im0, ax=axs[0], label='Patch Index')
    
    # Grid lines and text for original indices
    for i in range(grid_size):
        axs[0].axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
        axs[0].axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
    for i in range(grid_size):
        for j in range(grid_size):
            axs[0].text(j, i, f"{original_indices[i, j]}", ha='center', va='center', 
                      color='white', fontsize=12, fontweight='bold')
    
    # Rank matrix (spatial layout)
    im1 = axs[1].imshow(rank_matrix, cmap='plasma')
    axs[1].set_title('Rank Assigned to Each Position')
    plt.colorbar(im1, ax=axs[1], label='Rank')
    
    # Grid lines and text for rank matrix
    for i in range(grid_size):
        axs[1].axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
        axs[1].axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
    for i in range(grid_size):
        for j in range(grid_size):
            axs[1].text(j, i, f"{rank_matrix[i, j]}", ha='center', va='center', 
                      color='white', fontsize=12, fontweight='bold')
    
    # Where patches go in the new sequence
    sequence_order = np.zeros((grid_size, grid_size), dtype=int)
    for rank in range(num_patches):
        original_idx = custom_rank[0, rank].item()
        row = original_idx // grid_size
        col = original_idx % grid_size
        sequence_order[row, col] = rank  # At original position (row,col), what's the sequence position?
        
    im2 = axs[2].imshow(sequence_order, cmap='viridis')
    axs[2].set_title('Processing Sequence Position for Each Patch')
    plt.colorbar(im2, ax=axs[2], label='Sequence Position')
    
    # Grid lines and text for sequence order
    for i in range(grid_size):
        axs[2].axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
        axs[2].axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
    for i in range(grid_size):
        for j in range(grid_size):
            axs[2].text(j, i, f"{sequence_order[i, j]}", ha='center', va='center', 
                      color='white', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "small_grid_composite.png"), dpi=300)
    plt.close()
    
    # Verify the reordering with assertions
    # For each position in the reordered sequence, check that it has the right patch
    for pos in range(num_patches):
        # This position should contain the patch indicated by custom_rank[0, pos]
        original_patch_idx = custom_rank[0, pos].item()
        
        # Get the expected row and col markers
        expected_row_marker = (original_patch_idx // grid_size) + 1
        expected_col_marker = (original_patch_idx % grid_size) + 1
        
        # Check that the markers match what we expect
        actual_row_marker = reordered_embeddings[0, pos, embed_dim//3].item()
        actual_col_marker = reordered_embeddings[0, pos, 2*embed_dim//3].item()
        
        assert abs(actual_row_marker - expected_row_marker) < 1e-5, \
            f"Position {pos}, original_idx={original_patch_idx}: row marker incorrect. Expected {expected_row_marker}, got {actual_row_marker}"
        
        assert abs(actual_col_marker - expected_col_marker) < 1e-5, \
            f"Position {pos}, original_idx={original_patch_idx}: col marker incorrect. Expected {expected_col_marker}, got {actual_col_marker}"
    
    print("✓ Small grid reordering validation passed")
    return True

def test_large_grid_ranking(save_dir='./large_grid_tests'):
    """
    Test ranking on the standard larger grid (14x14) with the same detailed visualization
    as the small grid test to identify any issues with center-first ranking.
    """
    print("\n=== Testing Ranking with Large Grid ===")
    
    # Create directory for results
    os.makedirs(save_dir, exist_ok=True)
    
    # Parameters - using the standard larger grid size
    batch_size = 1
    grid_size = 14  # 14x14 grid (standard for 224x224 images with 16x16 patches)
    num_patches = grid_size * grid_size
    embed_dim = 192
    
    # Create patch embeddings directly (without actual images)
    patch_embeddings = torch.zeros((batch_size, num_patches, embed_dim))
    
    # Populate embeddings with recognizable patterns
    for b in range(batch_size):
        for i in range(grid_size):
            for j in range(grid_size):
                patch_idx = i * grid_size + j
                
                # First third: batch marker (b+1)
                patch_embeddings[b, patch_idx, 0:embed_dim//3] = b + 1
                
                # Second third: row position marker (i+1)
                patch_embeddings[b, patch_idx, embed_dim//3:2*embed_dim//3] = i + 1
                
                # Last third: column position marker (j+1)
                patch_embeddings[b, patch_idx, 2*embed_dim//3:] = j + 1
    
    # Create center-first ranking for this grid size
    center_y, center_x = grid_size / 2, grid_size / 2
    
    # Calculate distances from center for each patch
    distances = []
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate distance from the center point to the center of this cell
            # Adding 0.5 to get the center of the cell
            dist = np.sqrt((i + 0.5 - center_y)**2 + (j + 0.5 - center_x)**2)
            patch_idx = i * grid_size + j
            distances.append((patch_idx, dist))
    
    # Sort by distance (closest first)
    distances.sort(key=lambda x: x[1])
    
    # Create ranking tensor
    indices = torch.tensor([idx for idx, _ in distances], dtype=torch.long)
    custom_rank = indices.unsqueeze(0)  # Add batch dimension
    
    # Print the first few distance-to-rank mappings for verification
    print("\nFirst 10 Distance-to-Rank Mappings:")
    for idx, (patch_idx, dist) in enumerate(distances[:10]):
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        print(f"Rank {idx}: Patch {patch_idx} (row={row}, col={col}, dist={dist:.3f})")
    
    # Also print the specific patches mentioned (90 and 105)
    print("\nChecking specific patches of interest:")
    for patch_idx in [90, 105]:
        # Find this patch in the distances list
        for rank, (idx, dist) in enumerate(distances):
            if idx == patch_idx:
                row = idx // grid_size
                col = idx % grid_size
                print(f"Patch {patch_idx} (row={row}, col={col}) -> Rank {rank}, dist={dist:.3f}")
                break
    
    # Apply reordering
    reordered_embeddings = apply_custom_ranking(patch_embeddings, custom_rank)
    
    # Create detailed visualizations of original and reordered patches
    
    # 1. Create matrix showing original indices
    original_indices = np.arange(num_patches).reshape(grid_size, grid_size)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(original_indices, cmap='viridis')
    plt.colorbar(label='Original Index')
    plt.title('Original Patch Indices')
    
    # Add grid lines
    for i in range(grid_size):
        plt.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
        plt.axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
    
    # Add indices as text but only for specific regions of interest
    # Center region
    center_region_size = 4  # Show indices for a 4x4 region in the center
    start_i = grid_size // 2 - center_region_size // 2
    start_j = grid_size // 2 - center_region_size // 2
    for i in range(start_i, start_i + center_region_size):
        for j in range(start_j, start_j + center_region_size):
            plt.text(j, i, f"{original_indices[i, j]}", ha='center', va='center', 
                    color='white', fontsize=9, fontweight='bold')
    
    # Mark patch 90 and 105 with a different color
    for patch_idx in [90, 105]:
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        plt.text(col, row, f"{patch_idx}", ha='center', va='center', 
                color='red', fontsize=12, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "large_grid_original_indices.png"), dpi=200)
    plt.close()
    
    # 2. Create matrix showing the rank assigned to each original position
    rank_matrix = np.zeros((grid_size, grid_size), dtype=int)
    for idx, rank in enumerate(custom_rank[0]):
        original_idx = rank.item()
        row = original_idx // grid_size
        col = original_idx % grid_size
        rank_matrix[row, col] = idx
    
    plt.figure(figsize=(10, 10))
    plt.imshow(rank_matrix, cmap='viridis')
    plt.colorbar(label='Rank (Processing Order)')
    plt.title('Center-First Ranks Assigned to Each Patch')
    
    # Add grid lines
    for i in range(grid_size):
        plt.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
        plt.axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
    
    # Add ranks as text but only for center region
    for i in range(start_i, start_i + center_region_size):
        for j in range(start_j, start_j + center_region_size):
            plt.text(j, i, f"{rank_matrix[i, j]}", ha='center', va='center', 
                    color='white', fontsize=9, fontweight='bold')
    
    # Mark patch 90 and 105 with a different color
    for patch_idx in [90, 105]:
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        plt.text(col, row, f"{rank_matrix[row, col]}", ha='center', va='center', 
                color='red', fontsize=12, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "large_grid_ranks.png"), dpi=200)
    plt.close()
    
    # 3. Alternative visualization: Where each rank will pull from
    position_by_rank = np.zeros((grid_size, grid_size), dtype=int)
    for pos, original_idx in enumerate(custom_rank[0]):
        original_idx = original_idx.item()
        row = pos // grid_size  # Position in the output sequence
        col = pos % grid_size
        position_by_rank[row, col] = original_idx  # Which patch goes here
    
    plt.figure(figsize=(10, 10))
    plt.imshow(position_by_rank, cmap='viridis')
    plt.colorbar(label='Original Patch Index')
    plt.title('Original Patch at Each Position After Reordering')
    
    # Add grid lines
    for i in range(grid_size):
        plt.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
        plt.axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
    
    # Add original indices as text for the first few positions
    for i in range(2):  # First two rows only
        for j in range(grid_size):
            pos = i * grid_size + j
            if pos < 10:  # Only first 10 positions
                plt.text(j, i, f"{position_by_rank[i, j]}", ha='center', va='center', 
                        color='white', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "large_grid_reordered.png"), dpi=200)
    plt.close()
    
    # 4. Create a composite visualization of all three
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))
    
    # Original indices
    im0 = axs[0].imshow(original_indices, cmap='viridis')
    axs[0].set_title('Original Patch Indices')
    plt.colorbar(im0, ax=axs[0], label='Patch Index')
    
    # Grid lines
    for i in range(grid_size):
        axs[0].axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
        axs[0].axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
    
    # Mark patches 90 and 105
    for patch_idx in [90, 105]:
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        axs[0].text(col, row, f"{patch_idx}", ha='center', va='center', 
                  color='red', fontsize=12, fontweight='bold',
                  bbox=dict(facecolor='black', alpha=0.7))
    
    # Rank matrix
    im1 = axs[1].imshow(rank_matrix, cmap='plasma')
    axs[1].set_title('Rank Assigned to Each Position')
    plt.colorbar(im1, ax=axs[1], label='Rank')
    
    # Grid lines
    for i in range(grid_size):
        axs[1].axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
        axs[1].axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
    
    # Mark patches 90 and 105
    for patch_idx in [90, 105]:
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        axs[1].text(col, row, f"{rank_matrix[row, col]}", ha='center', va='center', 
                  color='red', fontsize=12, fontweight='bold',
                  bbox=dict(facecolor='black', alpha=0.7))
    
    # Sequence order
    sequence_order = np.zeros((grid_size, grid_size), dtype=int)
    for rank in range(num_patches):
        original_idx = custom_rank[0, rank].item()
        row = original_idx // grid_size
        col = original_idx % grid_size
        sequence_order[row, col] = rank  # At original position (row,col), what's the sequence position?
        
    im2 = axs[2].imshow(sequence_order, cmap='viridis')
    axs[2].set_title('Processing Sequence Position for Each Patch')
    plt.colorbar(im2, ax=axs[2], label='Sequence Position')
    
    # Grid lines
    for i in range(grid_size):
        axs[2].axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
        axs[2].axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
    
    # Mark patches 90 and 105
    for patch_idx in [90, 105]:
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        axs[2].text(col, row, f"{sequence_order[row, col]}", ha='center', va='center', 
                  color='red', fontsize=12, fontweight='bold',
                  bbox=dict(facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "large_grid_composite.png"), dpi=300)
    plt.close()
    
    # 5. Addition: Create a separate composite image just of the three non-composite images
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))
    
    # Load the three individual images
    img1 = plt.imread(os.path.join(save_dir, "large_grid_original_indices.png"))
    img2 = plt.imread(os.path.join(save_dir, "large_grid_ranks.png"))
    img3 = plt.imread(os.path.join(save_dir, "large_grid_reordered.png"))
    
    # Display the three images
    axs[0].imshow(img1)
    axs[0].set_title('Original Patch Indices')
    axs[0].axis('off')
    
    axs[1].imshow(img2)
    axs[1].set_title('Center-First Ranks Assigned')
    axs[1].axis('off')
    
    axs[2].imshow(img3)
    axs[2].set_title('Original Patch at Each Position')
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "large_grid_three_images.png"), dpi=300)
    plt.close()
    
    # Verify the reordering with assertions
    # For each position in the reordered sequence, check that it has the right patch
    for pos in range(num_patches):
        # This position should contain the patch indicated by custom_rank[0, pos]
        original_patch_idx = custom_rank[0, pos].item()
        
        # Get the expected row and col markers
        expected_row_marker = (original_patch_idx // grid_size) + 1
        expected_col_marker = (original_patch_idx % grid_size) + 1
        
        # Check that the markers match what we expect
        actual_row_marker = reordered_embeddings[0, pos, embed_dim//3].item()
        actual_col_marker = reordered_embeddings[0, pos, 2*embed_dim//3].item()
        
        assert abs(actual_row_marker - expected_row_marker) < 1e-5, \
            f"Position {pos}, original_idx={original_patch_idx}: row marker incorrect. Expected {expected_row_marker}, got {actual_row_marker}"
        
        assert abs(actual_col_marker - expected_col_marker) < 1e-5, \
            f"Position {pos}, original_idx={original_patch_idx}: col marker incorrect. Expected {expected_col_marker}, got {actual_col_marker}"
    
    print("✓ Large grid reordering validation passed")
    return True

# Update the main function to include the new test
def main():
    parser = argparse.ArgumentParser(description='Test patch reordering functionality')
    parser.add_argument('--save-dir', type=str, default='./reordering_test_results', help='Directory to save test results')
    parser.add_argument('--ranking-file', type=str, default='rankings/ranking_highest_first.pt', help='Path to real ranking file for testing')
    parser.add_argument('--small-grid', action='store_true', help='Run small grid test for detailed visualization')
    parser.add_argument('--large-grid', action='store_true', help='Run large grid test for detailed visualization')
    
    args = parser.parse_args()
    
    print("Starting patch reordering tests...")
    
    # Run requested test(s)
    if args.small_grid:
        small_grid_test_passed = test_small_grid_ranking(os.path.join(args.save_dir, 'small_grid'))
        print("\n=== Test Summary ===")
        print(f"Small grid test: {'PASSED' if small_grid_test_passed else 'FAILED'}")
    
    if args.large_grid:
        large_grid_test_passed = test_large_grid_ranking(os.path.join(args.save_dir, 'large_grid'))
        print("\n=== Test Summary ===")
        print(f"Large grid test: {'PASSED' if large_grid_test_passed else 'FAILED'}")
    
    if not (args.small_grid or args.large_grid):
        # Run the regular tests
        reordering_test_passed = test_reordering_logic(os.path.join(args.save_dir, 'reordering'))
        real_data_test_passed = test_with_real_data_from_file(args.ranking_file, os.path.join(args.save_dir, 'real_data'))
        
        # Print summary
        print("\n=== Test Summary ===")
        print(f"Patch reordering test: {'PASSED' if reordering_test_passed else 'FAILED'}")
        print(f"Real data test: {'PASSED' if real_data_test_passed else 'SKIPPED/FAILED'}")
        
        if reordering_test_passed:
            print("\nOverall: PASSED - Patch reordering logic works as expected!")
            print(f"Test visualizations saved to {args.save_dir}")
        else:
            print("\nOverall: FAILED - Issues detected with patch reordering functionality")

if __name__ == "__main__":
    main()