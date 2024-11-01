import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional

from timm.models.layers import DropPath, to_2tuple
    
class PatchEmbedCustom(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, 
                 norm_layer=None, flatten=True, patch_order=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.patch_order = patch_order if patch_order is not None else list(range(self.num_patches))

        with torch.no_grad():
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
            self.proj.weight.fill_(1.0)
            self.proj.bias.fill_(0.0)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        print(x.shape, x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        # Reorder patches to global order
        #TODO check if this is correct
        patch_order = torch.tensor(self.patch_order).expand(B, -1)  # Shape (B, N)
        y = torch.gather(x, dim=1, index=patch_order.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        
        return x, y
    
def test_patch_embed():
    # Define custom patch order
    custom_order = [15, 2, 0, 1, 3, 5, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    # Initialize PatchEmbed with custom order
    patch_embed = PatchEmbedCustom(img_size=16, patch_size=4, stride=4, in_chans=3, 
                                   embed_dim=2, patch_order=custom_order)
    
    # Create a dummy input image tensor (batch size 1, 3 channels, 32x32 image)
    # Create a dummy tensor of zeros
    dummy_tensor = torch.zeros((1, 3, 16, 16))
    # Fill the first layer with a range counting up
    dummy_tensor[0] = torch.arange(16 * 16).view(16, 16)
    # sum the upper left patch of the dummy tensor
    dummy_sums = []
    dummy_proj_sums = []
    for i in range(0, 16, 4):
        for j in range(0, 16, 4):
            patch_sum = dummy_tensor[0, :, i:i+4, j:j+4].sum()
            patch_proj_sum = patch_embed.proj(dummy_tensor)[0, :, i:i+4, j:j+4].sum()
            dummy_sums.append(patch_sum)
            dummy_proj_sums.append(patch_proj_sum)
    print("Dummy sums:", dummy_sums)
        
    # Pass the input tensor through the PatchEmbed instance
    output, output_n = patch_embed(dummy_tensor)
    
    # Print the shape of the output tensor
    print("Output shape:", output.shape)
    print("Output_n shape:", output_n.shape)
    
    # Verify the order of patches
    # For simplicity, we will print the indices of the patches in the output tensor
    # In a real test, you would compare the output tensor to an expected tensor
    print(output[:,:,:], output_n[:,:,:])

# Run the test
test_patch_embed()