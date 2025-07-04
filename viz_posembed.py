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
    """Visualize position embeddings using cosine similarity"""
    # pos embed does not include class tokens
    num_patches = grid_size * grid_size
    has_cls_token = pos_embed.shape[1] > num_patches
    # Extract only the patch position embeddings (skip class token if present)
    if has_cls_token:
        patch_pos_embed = pos_embed[0, 1:, :].reshape(grid_size, grid_size, -1)
    else:
        patch_pos_embed = pos_embed[0, :, :].reshape(grid_size, grid_size, -1)
    # Convert to numpy for visualization
    patch_pos_embed = patch_pos_embed.numpy()
    # Compute cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    flattened = patch_pos_embed.reshape(-1, patch_pos_embed.shape[-1])
    cosine_sim = cosine_similarity(flattened)
    # Reshape back to spatial grid
    cosine_sim = cosine_sim.reshape(grid_size, grid_size, grid_size, grid_size)
    # Normalize to [0,1] for visualization
    cosine_sim = (cosine_sim - cosine_sim.min()) / (cosine_sim.max() - cosine_sim.min())
    return cosine_sim

def main(checkpoint_path, val_dir):
    # Load model and extract position embeddings
    pos_embed, model = load_model_and_get_pos_embed(checkpoint_path)
    cls_pos = pos_embed.shape[0] // 2 # class token is at the middle of the position embeddings
    x_before = pos_embed[:, :cls_pos, :]      # [B, cls_pos, C]
    x_cls    = pos_embed[:, cls_pos:cls_pos+1, :]  # [B, 1, C]
    x_after  = pos_embed[:, cls_pos+1:, :]    # [B, P - cls_pos, C]

    pos_embed = torch.cat((x_before, x_after), dim=1)  # [B, P, C]
    
    # Ensure position embedding is on CPU
    pos_embed = pos_embed.detach().cpu()
    
    
    # Get a random validation image
    img_tensor, original_img, img_path = get_random_val_image(val_dir)
    
    # Visualize position embeddings
    pos_embed_rgb = visualize_pos_embedding_pca(pos_embed)
    pos_embed_cosine = visualize_pos_embedding_cosine(pos_embed)
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title(f"Random Image: {os.path.basename(img_path)}")
    axes[0].axis('off')
    
    # Position embedding visualization
    axes[1].imshow(pos_embed_rgb)
    axes[1].set_title("Position Embedding\n(PCA visualization)")
    axes[1].axis('off')

    # # Cosine similarity visualization
    # axes[2].imshow(pos_embed_cosine.mean(axis=2), cmap='viridis')
    # axes[2].set_title("Position Embedding\n(Cosine Similarity)")
    # axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("posembed_visualization.png")
    plt.show()
    
    # Return the position embedding for further analysis
    return pos_embed, img_tensor, img_path

if __name__ == "__main__":
    # Default paths - update these to your actual paths
    default_checkpoint = r"/storage/scratch/6403840/Master-Thesis/vim/output/vim_extra_tiny_custom_transforms_heatmap/best_checkpoint.pth"
    default_val_dir = r"/storage/scratch/6403840/data/imagenet-tiny/val"
    
    # Get paths from command line arguments or use defaults
    import sys
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else default_checkpoint
    val_dir = sys.argv[2] if len(sys.argv) > 2 else default_val_dir
    
    pos_embed, img_tensor, img_path = main(checkpoint_path, val_dir)
    print(f"Position embedding shape: {pos_embed.shape}")
    print(f"Sample image path: {img_path}")