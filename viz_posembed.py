import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from PIL import Image
import torchvision.transforms as transforms
from vim.models_mamba import vim_extra_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2

def load_model_and_get_pos_embed(checkpoint_path):
    """
    Load a pretrained Vision Mamba model and extract position embeddings
    """
    print(f"Loading model from: {checkpoint_path}")
    # Initialize model on CPU
    model = vim_extra_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(num_classes=200)
    
    # Load pretrained weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    print("Model loaded successfully!")
    
    # Check if model has position embedding
    if not hasattr(model, 'pos_embed'):
        raise AttributeError("Model does not have position embedding attribute 'pos_embed'")
    
    # Extract position embedding
    pos_embed = model.pos_embed.detach().cpu()
    print(f"Position embedding shape: {pos_embed.shape}")
    
    return pos_embed, model

def get_random_val_image(val_dir):
    """Load a random image from the validation directory"""
    # Get all synset directories
    synset_dirs = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
    
    # Select a random synset
    random_synset = random.choice(synset_dirs)
    synset_path = os.path.join(val_dir, random_synset)
    
    # Get all images in this synset
    images = [f for f in os.listdir(synset_path) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Select a random image
    random_img = random.choice(images)
    img_path = os.path.join(synset_path, random_img)
    
    # Load the image
    img = Image.open(img_path).convert('RGB')
    
    # Preprocess the image for the model
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    return img_tensor, img, img_path

def visualize_pos_embedding_pca(pos_embed, patch_size=16, grid_size=14):
    """Visualize position embeddings using PCA"""
    # Check if pos_embed includes class tokens
    num_patches = grid_size * grid_size
    has_cls_token = pos_embed.shape[1] > num_patches
    
    # Extract only the patch position embeddings (skip class token if present)
    if has_cls_token:
        patch_pos_embed = pos_embed[0, 1:, :].reshape(grid_size, grid_size, -1)
    else:
        patch_pos_embed = pos_embed[0, :, :].reshape(grid_size, grid_size, -1)
    
    # Convert to numpy for visualization
    patch_pos_embed = patch_pos_embed.numpy()
    
    # Use PCA to reduce dimensions for visualization
    from sklearn.decomposition import PCA
    
    # Flatten spatial dimensions
    embed_dim = patch_pos_embed.shape[-1]
    flattened = patch_pos_embed.reshape(-1, embed_dim)
    
    # Apply PCA to get 3 principal components
    pca = PCA(n_components=3)
    pos_embed_pca = pca.fit_transform(flattened)
    
    # Reshape back to spatial grid
    pos_embed_pca = pos_embed_pca.reshape(grid_size, grid_size, 3)
    
    # Normalize to [0,1] for visualization
    pos_embed_pca = (pos_embed_pca - pos_embed_pca.min()) / (pos_embed_pca.max() - pos_embed_pca.min())
    
    return pos_embed_pca

def visualize_pos_embedding_cosine(pos_embed, patch_size=16, grid_size=14):
    """Visualize position embeddings using cosine similarity for every second patch (even indices)"""
    # Convert to numpy for cosine similarity calculation
    patch_pos_embed = pos_embed.numpy()
    # Flatten spatial dimensions
    embed_dim = patch_pos_embed.shape[-1]
    flattened = patch_pos_embed.reshape(-1, embed_dim)
    # Select only even-indexed patches
    even_indices = np.arange(0, grid_size * grid_size, 2)
    flattened_even = flattened[even_indices]
    # Calculate cosine similarity between each selected patch and all patches
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_sim = cosine_similarity(flattened_even, flattened_even)
    # Reshape for visualization: [num_even_patches, grid_size, grid_size]
    cosine_sim = cosine_sim.reshape(len(even_indices), grid_size, grid_size)
    # Normalize cosine similarity values to [0, 1]
    cosine_sim = (cosine_sim - cosine_sim.min()) / (cosine_sim.max() - cosine_sim.min())
    return cosine_sim


def plot_cosine_similarity(cosine_sim, even_indices, grid_size=7):
    """
    Plot cosine similarity for every second patch (even indices) as a grid of subplots.
    Each subplot shows the similarity of one selected patch to all others.
    """
    import matplotlib.pyplot as plt

    # Ensure cosine_sim is a numpy array
    if not isinstance(cosine_sim, np.ndarray):
        cosine_sim = np.array(cosine_sim)
    num_patches = cosine_sim.shape[0]
    # Determine subplot grid size (try to make it as square as possible)
    n_cols = int(np.ceil(np.sqrt(num_patches)))
    n_rows = int(np.ceil(num_patches / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = np.array(axes).reshape(n_rows, n_cols)
    vmin, vmax = -1, 1  # For cosine similarity

    for idx in range(num_patches):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        sim_map = cosine_sim[idx]
        im = ax.imshow(sim_map, cmap='viridis', vmin=0, vmax=1)
        patch_idx = even_indices[idx]
        i, j = divmod(patch_idx, grid_size)
        ax.set_title(f"Patch ({i},{j})", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    # Hide unused subplots
    for idx in range(num_patches, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis('off')
    # Add a colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Cosine similarity')
    fig.suptitle("Position embedding similarity (even-indexed patches)", fontsize=16)
    plt.tight_layout()
    plt.savefig("posembed_cosine_similarity_even.png")
    plt.show()
    

def main(checkpoint_path, val_dir):
    # Load model and extract position embeddings
    pos_embed, model = load_model_and_get_pos_embed(checkpoint_path)
    cls_pos = pos_embed.shape[0] // 2 + 1 # class token is at the middle of the position embeddings
    x_before = pos_embed[:, :cls_pos, :]      # [B, cls_pos, C]
    x_cls    = pos_embed[:, cls_pos:cls_pos+1, :]  # [B, 1, C]
    x_after  = pos_embed[:, cls_pos+1:, :]    # [B, P - cls_pos, C]

    pos_embeds = torch.cat((x_before, x_after), dim=1)  # [B, P, C]
    
    print("Without class token", pos_embeds.shape)
    # Ensure position embedding is on CPU
    pos_embeds = pos_embeds.detach().cpu()
    # Visualize position embeddings
    pos_embed_rgb = visualize_pos_embedding_pca(pos_embeds)
    pos_embed_cosine = visualize_pos_embedding_cosine(pos_embeds)
    plot_cosine_similarity(pos_embed_cosine, np.arange(0, pos_embeds.shape[1], 2))
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # # Original image
    # axes[0].imshow(original_img)
    # axes[0].set_title(f"Random Image: {os.path.basename(img_path)}")
    # axes[0].axis('off')
    
    # Position embedding visualization
    axes[0].imshow(pos_embed_rgb)
    axes[0].set_title("Position Embedding\n(PCA visualization)")
    axes[0].axis('off')
    
    plt.tight_layout()
    plt.savefig("posembed_visualization.png")
    plt.show()

if __name__ == "__main__":
    # Default paths - update these to your actual paths
    default_checkpoint = r"/storage/scratch/6403840/Master-Thesis/vim/output/vim_extra_tiny_custom_transforms_heatmap/best_checkpoint.pth"
    default_val_dir = r"/storage/scratch/6403840/data/imagenet-tiny/val"
    
    # Get paths from command line arguments or use defaults
    import sys
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else default_checkpoint
    val_dir = sys.argv[2] if len(sys.argv) > 2 else default_val_dir
    
    main(checkpoint_path, val_dir)