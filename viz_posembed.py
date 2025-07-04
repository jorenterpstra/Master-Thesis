import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from PIL import Image
import scipy.stats
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
    
    # Convert to numpy for visualization
    patch_pos_embed = pos_embed.numpy()
    
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

def visualize_pos_embedding_cosine(pos_embed, grid_size=14):
    """Visualize position embeddings using cosine similarity for all patches"""
    # Convert to numpy for cosine similarity calculation
    patch_pos_embed = pos_embed.numpy()
    # Flatten spatial dimensions
    embed_dim = patch_pos_embed.shape[-1]
    flattened = patch_pos_embed.reshape(-1, embed_dim)
    # Calculate cosine similarity between all patches
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_sim = cosine_similarity(flattened, flattened)
    # Reshape for visualization: [num_patches, grid_size, grid_size]
    num_patches = grid_size * grid_size
    cosine_sim = cosine_sim.reshape(num_patches, grid_size, grid_size)
    return cosine_sim

def plot_cosine_similarity(cosine_sim, grid_size=14):
    """
    Plot cosine similarity for all patches as a grid of subplots.
    Each subplot shows the similarity of one patch to all others.
    """
    import matplotlib.pyplot as plt

    # Ensure cosine_sim is a numpy array
    if not isinstance(cosine_sim, np.ndarray):
        cosine_sim = np.array(cosine_sim)
    num_patches = cosine_sim.shape[0]
    print("cosine_sim shape:", cosine_sim.shape)
    # Determine subplot grid size (try to make it as square as possible)
    n_cols = grid_size
    n_rows = grid_size

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx in range(num_patches):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        sim_map = cosine_sim[idx]
        im = ax.imshow(sim_map, cmap='magma', vmin=-1, vmax=1)
        # make the patch with the highest similarity red
        # max_idx = np.unravel_index(np.argmax(sim_map), sim_map.shape)
        # ax.add_patch(plt.Rectangle((max_idx[1]-0.5, max_idx[0]-0.5), 1, 1, edgecolor='none', facecolor='black', lw=2))
        
        # Hide ticks but keep axis labels if needed
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Only set labels at the bottom row and leftmost column
        if row == n_rows - 1:
            ax.set_xlabel(f"col {col}", fontsize=9, labelpad=6)
        else:
            ax.set_xlabel("")
        if col == 0:
            ax.set_ylabel(f"row {row}", fontsize=9, labelpad=6)
        else:
            ax.set_ylabel("")

    # Hide unused subplots (shouldn't be any, but just in case)
    for idx in range(num_patches, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis('off')
    # Add a colorbar
    fig.subplots_adjust(right=0.88, wspace=0.1, hspace=0.2)
    # Adjust colorbar position to be closer to the plots
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Cosine similarity')
    # Move suptitle closer to the subplots
    fig.suptitle("Position embedding similarity", fontsize=18, y=0.91)
    plt.savefig("posembed_cosine_similarity_all.png", dpi=300)
    plt.show()

def plot_pearson(pos_embeds):
    # pos_embeds: [1, num_patches, dim]
    pos_embeds_np = pos_embeds.squeeze(0).numpy()  # [num_patches, dim]
    grid_size = int(np.sqrt(pos_embeds_np.shape[0]))

    coords = np.array([(i, j) for i in range(grid_size) for j in range(grid_size)])  # [num_patches, 2]
    x_coords = coords[:, 1]
    y_coords = coords[:, 0]

    cor_x = []
    cor_y = []
    for d in range(pos_embeds_np.shape[1]):
        cor_x.append(abs(scipy.stats.pearsonr(pos_embeds_np[:, d], x_coords)[0]))
        cor_y.append(abs(scipy.stats.pearsonr(pos_embeds_np[:, d], y_coords)[0]))
    cor_x = np.array(cor_x)
    cor_y = np.array(cor_y)

    plt.figure(figsize=(10,4))
    plt.plot(cor_x, label='|corr with x|')
    plt.plot(cor_y, label='|corr with y|')
    plt.xlabel('Embedding dimension')
    plt.ylabel('Absolute Pearson correlation')
    plt.title('Correlation of position embedding dimensions with spatial coordinates')
    plt.legend()
    plt.tight_layout()
    plt.savefig("posembed_pearson_correlation.png", dpi=300)
    plt.show()

def plot_tsne_pos_embeds(pos_embeds, grid_size=14, perplexity=30):
    """
    Visualize position embeddings using t-SNE, colored by spatial location.
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # pos_embeds: [1, num_patches, dim] or [num_patches, dim]
    if pos_embeds.ndim == 3:
        pos_embeds_np = pos_embeds.squeeze(0).numpy()
    else:
        pos_embeds_np = pos_embeds.numpy()
    num_patches = pos_embeds_np.shape[0]

    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_coords = tsne.fit_transform(pos_embeds_np)

    # Get grid coordinates for coloring
    coords = np.array([(i, j) for i in range(grid_size) for j in range(grid_size)])[:num_patches]
    x_coords = coords[:, 1]
    y_coords = coords[:, 0]

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=x_coords + y_coords * grid_size, cmap='viridis', s=40)
    plt.title("t-SNE of Position Embeddings (colored by patch index)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(scatter, label="Patch index (row*grid + col)")
    plt.tight_layout()
    plt.savefig("posembed_tsne.png", dpi=300)
    plt.show()

def main(checkpoint_path, val_dir):
    # Load model and extract position embeddings
    pos_embed, model = load_model_and_get_pos_embed(checkpoint_path)
    cls_pos = pos_embed.shape[1] // 2 # class token is at the middle of the position embeddings
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
    # Plot Pearson correlation
    plot_pearson(pos_embeds)
    plot_cosine_similarity(pos_embed_cosine)
    plot_tsne_pos_embeds(pos_embeds)
    
    # Plot results
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    
    # # Original image
    # axes[0].imshow(original_img)
    # axes[0].set_title(f"Random Image: {os.path.basename(img_path)}")
    # axes[0].axis('off')
    
    # Position embedding visualization
    axes.imshow(pos_embed_rgb)
    axes.set_title("Position Embedding\n(PCA visualization)")
    axes.axis('off')
    
    plt.tight_layout()
    plt.savefig("posembed_visualization.png")
    plt.show()

if __name__ == "__main__":
    # Default paths - update these to your actual paths
    default_checkpoint = r"/storage/scratch/6403840/Master-Thesis/vim/output/vim_extra_tiny_custom_transforms_baseline2/best_checkpoint.pth"
    default_val_dir = r"/storage/scratch/6403840/data/imagenet-tiny/val"
    
    # Get paths from command line arguments or use defaults
    import sys
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else default_checkpoint
    val_dir = sys.argv[2] if len(sys.argv) > 2 else default_val_dir
    
    main(checkpoint_path, val_dir)