import torch
from tqdm import tqdm
import numpy as np
import cv2


def _ranking_from_heatmap(self, heatmap):
        """Convert a 2D or 3D heatmap into a flat ranking tensor of 196 patch-scores."""
        # If RGB or multi-channel, collapse to single channel by averaging
        current_heatmap = heatmap
        if isinstance(current_heatmap, torch.Tensor):
            # Permute CHW to HWC if it's a tensor, then convert to numpy
            if current_heatmap.ndim == 3 and current_heatmap.shape[0] in [1, 3, 4]: # Basic check for CHW
                current_heatmap = current_heatmap.permute(1, 2, 0)
            current_heatmap = current_heatmap.cpu().numpy()

        # Now current_heatmap is a numpy array
        if current_heatmap.ndim == 3:
            if current_heatmap.shape[2] == 1: # Handle (H, W, 1)
                current_heatmap = current_heatmap.squeeze(axis=2)
            else: # Assuming (H, W, C), average over channels
                current_heatmap = current_heatmap.mean(axis=2)
    
        # Now current_heatmap should be a 2D (H, W) array
        h, w = current_heatmap.shape
        
        # We expect a 224×224 heatmap and an output of 14×14 patches
        patch_size = 16  # Standard ViT patch size
        stride = 16      # Non-overlapping patches
        
        num_patches_h = (h - patch_size) // stride + 1
        num_patches_w = (w - patch_size) // stride + 1
        total_patches = num_patches_h * num_patches_w
        
        # Use integral image for efficient patch sum computation
        integral_img = cv2.integral(current_heatmap.astype(np.float32))
        
        # Precompute coordinates
        y_coords = np.arange(num_patches_h) * stride
        x_coords = np.arange(num_patches_w) * stride
        
        y_start = y_coords.reshape(-1, 1)
        x_start = x_coords.reshape(1, -1)
        y_end = y_start + patch_size
        x_end = x_start + patch_size
        
        # Using integral image formula: sum = tl + br - tr - bl
        top_left = integral_img[y_start, x_start]
        top_right = integral_img[y_start, x_end]
        bottom_left = integral_img[y_end, x_start] 
        bottom_right = integral_img[y_end, x_end]
        
        patch_scores = bottom_right - top_right - bottom_left + top_left
        
        # For flood-fill like tie-breaking, create a distance map from highest-scoring patch
        flat_scores = patch_scores.flatten()
        max_idx = np.argmax(flat_scores)
        max_y, max_x = max_idx // num_patches_w, max_idx % num_patches_w
        
        # Create a distance map from the max score patch
        y_grid, x_grid = np.mgrid[:num_patches_h, :num_patches_w]
        distance_map = np.sqrt((y_grid - max_y)**2 + (x_grid - max_x)**2)
        
        # Scale the distance map to a very small value so it only affects ties
        # This ensures that patches closer to the highest-scored one will be ranked higher
        # when there are ties (similar to flood-fill behavior)
        epsilon = np.finfo(np.float32).eps
        max_score = np.max(patch_scores)
        tie_breaking_scores = patch_scores - (distance_map * epsilon * max_score)
        
        # Get the ranking
        flat_tie_breaking_scores = tie_breaking_scores.flatten()
        ranking = np.argsort(-flat_tie_breaking_scores)
        
        return torch.from_numpy(ranking).long()

if __name__ == "__main__":
    dummy_heatmap_32x32 = torch.zeros((1, 16, 16), dtype=torch.float32)
    # Set more pixels since the heatmap is 16x16 (bigger image)
    dummy_heatmap_32x32[0, 2, 2] = 1.0    # High value
    dummy_heatmap_32x32[0, 0, 0] = 0.5    # Medium value
    dummy_heatmap_32x32[0, 1, 2] = 0.3    # Lower value
    dummy_heatmap_32x32[0, 7, 7] = 0.8    # Bottom-right-ish
    dummy_heatmap_32x32[0, 4, 4] = 0.6    # Center-ish
    dummy_heatmap_32x32[0, 6, 1] = 0.7    # Near bottom-left
    dummy_heatmap_32x32[0, 3, 7] = 0.4    # Right edge
    dummy_heatmap_32x32[0, 10, 10] = 0.9  # Farther bottom-right
    dummy_heatmap_32x32[0, 12, 3] = 0.65  # Lower left
    dummy_heatmap_32x32[0, 8, 14] = 0.55  # Far right
    dummy_heatmap_32x32[0, 15, 15] = 0.95 # Bottom-right corner
    dummy_heatmap_32x32[0, 13, 8] = 0.75  # Lower center
    dummy_heatmap_32x32[0, 5, 12] = 0.45  # Upper right
    dummy_heatmap_32x32[0, 9, 5] = 0.35   # Center left
    dummy_heatmap_32x32[0, 11, 1] = 0.5   # Lower left edge
    dummy_heatmap_32x32[0, 14, 7] = 0.6   # Bottom center
    dummy_heatmap_32x32[0, 2, 6] = 0.5
    # Test the function with a dummy heatmap
    ranking = _ranking_from_heatmap(None, dummy_heatmap_32x32)
    print("Ranking from dummy heatmap (32x32):", ranking)
    #plot the heatmap
    import matplotlib.pyplot as plt
    plt.imshow(dummy_heatmap_32x32.squeeze().numpy(), cmap='hot', interpolation='nearest')
    #show patch lines, adjust for bordering
    for i in range(0, 16, 2):
        plt.axhline(y=i-.5, color='white', linewidth=0.5)
        plt.axvline(x=i-.5, color='white', linewidth=0.5)
    # write the patch numbers, one per patch of 2x2
    for i in range(8):
        for j in range(8):
            plt.text(j*2+0.5, i*2+0.5, str(i*8+j), color='white', ha='center', va='center')
    plt.colorbar()
    plt.title("Dummy Heatmap (32x32)")
    plt.show()