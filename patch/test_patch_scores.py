import torch
import matplotlib.pyplot as plt
import numpy as np
from patch.dataloader import generate_scores

def visualize_patch_calculation(image_size=224, patch_size=16, bboxes=None):
    """
    Visualize patch score calculation for one or more bboxes with detailed information
    """
    if bboxes is None:
        bboxes = [[
            image_size // 8,      # x
            image_size // 8,      # y
            image_size * 3 // 4,  # width - make it larger by default
            image_size * 3 // 4   # height
        ]]

    # Calculate scores
    scores = generate_scores(image_size, bboxes, patch_size)
    num_patches = image_size // patch_size
    score_grid = scores.reshape(num_patches, num_patches)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Plot 1: Image with bbox and patch grid
    ax1.set_title(f"Image with {len(bboxes)} Bounding Box(es) and Patch Grid")
    
    # Draw patch grid
    for i in range(num_patches + 1):
        ax1.axhline(y=i * patch_size, color='gray', alpha=0.5)
        ax1.axvline(x=i * patch_size, color='gray', alpha=0.5)
    
    # Draw each bbox with a different color
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        color = colors[i % len(colors)]
        rect = plt.Rectangle((x, y), w, h, fill=False, color=color, linewidth=2, label=f'Box {i+1}')
        ax1.add_patch(rect)
        
        # Plot bbox center
        center_x = x + w/2
        center_y = y + h/2
        ax1.plot(center_x, center_y, '+', color=color, markersize=10)

    if len(bboxes) > 1:
        ax1.legend()

    ax1.set_xlim(0, image_size)
    ax1.set_ylim(image_size, 0)
    
    # Plot 2: Heatmap of scores
    im = ax2.imshow(score_grid, cmap='hot')
    ax2.set_title(f"Patch Scores ({num_patches}x{num_patches})")
    plt.colorbar(im, ax=ax2)

    # Add patch grid to heatmap
    for i in range(num_patches + 1):
        ax2.axhline(y=i - 0.5, color='white', alpha=0.3)
        ax2.axvline(x=i - 0.5, color='white', alpha=0.3)

    # Print statistics
    print("\nScore Statistics:")
    print(f"Min score: {scores.min():.3f}")
    print(f"Max score: {scores.max():.3f}")
    print(f"Mean score: {scores.mean():.3f}")
    print(f"Patches with score > 0.5: {(scores > 0.5).sum()}/{len(scores)}")
    print(f"Patches with score > 0.8: {(scores > 0.8).sum()}/{len(scores)}")
    
    plt.tight_layout()
    plt.show()

def test_edge_cases():
    """Test various edge cases for patch score calculation"""
    image_size = 224
    patch_size = 16
    
    test_cases = [
        {
            'name': 'Full Image Box',
            'bbox': [0, 0, 224, 224]  # Entire image
        },
        {
            'name': 'Large Center Box',
            'bbox': [32, 32, 160, 160]  # Large box (~70% of image)
        },
        {
            'name': 'Wide Box',
            'bbox': [16, 96, 192, 32]  # Wide horizontal box
        },
        {
            'name': 'Tall Box',
            'bbox': [96, 16, 32, 192]  # Tall vertical box
        },
        {
            'name': 'Quarter Boxes',  # Test multiple boxes
            'bbox': [
                [0, 0, 112, 112],      # Top-left quarter
                [112, 0, 112, 112],    # Top-right quarter
                [0, 112, 112, 112],    # Bottom-left quarter
                [112, 112, 112, 112]   # Bottom-right quarter
            ]
        },
        {
            'name': 'Overlapping Boxes',  # Test score accumulation
            'bbox': [
                [48, 48, 128, 128],    # Center box
                [32, 32, 160, 160],    # Larger box around it
            ]
        },
    ]
    
    for case in test_cases:
        print(f"\nTesting: {case['name']}")
        if isinstance(case['bbox'][0], list):
            # Multiple bboxes case
            visualize_patch_calculation(image_size, patch_size, case['bbox'])
        else:
            # Single bbox case
            visualize_patch_calculation(image_size, patch_size, [case['bbox']])

if __name__ == "__main__":
    print("Testing default case (large center box):")
    visualize_patch_calculation()
    
    print("\nTesting edge cases:")
    test_edge_cases()
