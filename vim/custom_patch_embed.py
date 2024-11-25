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
        self.patch_order = torch.tensor(patch_order, dtype=torch.long) if patch_order is not None else torch.arange(self.num_patches)

        self.order_predictor = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, self.num_patches, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, self.num_patches)),
            nn.Flatten(),
            nn.Softmax(dim=-1)
        )
        with torch.no_grad():
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
            self.proj.weight.fill_(1.0)
            self.proj.bias.fill_(0.0)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.embed_dim = embed_dim

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        #patch_order = self.order_predictor(x).argsort(dim=-1).astype(torch.long)
        x = self.proj(x)
        
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        # Reorder patches to global order
        y = torch.gather(x, 1, self.patch_order.unsqueeze(0).unsqueeze(-1).expand(B, -1, x.size(-1)))
        
        return x, y
    
def test_patch_embed():
    # Define custom patch order
    custom_order_16 = [1, 0, 2, 15, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 3]
    # Initialize PatchEmbed with custom order
    img_size = 8
    patch_embed = PatchEmbedCustom(img_size=img_size, patch_size=2, stride=2, in_chans=3, 
                                   embed_dim=1, patch_order=custom_order_16)
    
    # Create a dummy input image tensor (batch size 1, 3 channels, 32x32 image)
    # Create a dummy tensor of zeros
    dummy_tensor = torch.zeros((2, 3, 8, 8))
    # Fill the first layer with a range counting up
    dummy_tensor[0, 0] = torch.arange(img_size * img_size).view(img_size, img_size)
    # Fill the second layer with a range counting down
    # dummy_tensor[0, 1] = torch.arange(img_size * img_size - 1, -1, -1).view(img_size, img_size)
    # dummy_tensor[0, 2] = torch.arange(img_size * img_size).view(img_size, img_size)
    # # Repeat for the second dummy tensor but reverse the order of the layers
    dummy_tensor[1, 0] = torch.arange(img_size * img_size - 1, -1, -1).view(img_size, img_size)
    # dummy_tensor[1, 1] = torch.arange(img_size * img_size).view(img_size, img_size)
    # dummy_tensor[1, 2] = torch.arange(img_size * img_size - 1, -1, -1).view(img_size, img_size)
    # sum the upper left patch of the dummy tensor
    dummy_sums = []
    dummy_proj_sums = []
    for i in range(0, 8, 2):
        for j in range(0, 8, 2):
            patch_sum = dummy_tensor[0, :, i:i+2, j:j+2].sum()
            patch_proj_sum = patch_embed.proj(dummy_tensor)[0, :, i:i+2, j:j+2].sum()
            dummy_sums.append(patch_sum)
            dummy_proj_sums.append(patch_proj_sum)
    print("Dummy sums:", dummy_sums)
        
    # Pass the input tensor through the PatchEmbed instance
    output, output_n = patch_embed(dummy_tensor)
    
    # Verify the order of patches
    # For simplicity, we will print the indices of the patches in the output tensor
    # In a real test, you would compare the output tensor to an expected tensor
    for i in range(16):
        print(f"Initial order index {i}: {output[0, i, :].detach().numpy()} -> " +
              f"Custom order index {custom_order_16[i]}: {output_n[0, i, :].detach().numpy()}")
    print()
    for i in range(16):
        print(f"Initial order index {i}: {output[1, i, :].detach().numpy()} -> " +
              f"Custom order index {custom_order_16[i]}: {output_n[1, i, :].detach().numpy()}")

# Run the test
test_patch_embed()