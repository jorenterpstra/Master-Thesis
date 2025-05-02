import random
import math
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from typing import Tuple, Optional, Dict, Any, Union, List, Callable

class DualTransform:
    """Base class for transforms that can be applied to both image and ranking tensors."""
    
    def __call__(self, img: torch.Tensor, ranking: Optional[torch.Tensor] = None, **kwargs):
        """Apply the transform to both image and ranking tensor."""
        raise NotImplementedError("Subclasses must implement __call__")

class DualRandomResizedCrop(DualTransform):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3/4, 4/3), interpolation=InterpolationMode.BILINEAR):
        self.size = size if isinstance(size, (list, tuple)) else (size, size)
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        
    def __call__(self, img, ranking=None, **kwargs):
        # Get crop parameters
        height, width = img.shape[-2:]
        area = height * width
        
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
            
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            
            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                break
        else:
            # Fallback to central crop
            in_ratio = width / height
            if in_ratio < min(self.ratio):
                w = width
                h = int(w / min(self.ratio))
            elif in_ratio > max(self.ratio):
                h = height
                w = int(h * max(self.ratio))
            else:
                w = width
                h = height
            i = (height - h) // 2
            j = (width - w) // 2
        
        # Apply crop and resize to both tensors
        img = img[..., i:i+h, j:j+w]
        img = F.interpolate(img.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False).squeeze(0)
        
        if ranking is not None:
            ranking = ranking[..., i:i+h, j:j+w]
            # Use nearest interpolation for ranking to preserve unique values
            ranking = F.interpolate(ranking.unsqueeze(0), size=self.size, mode='nearest').squeeze(0)
            return img, ranking
        
        return img

class DualRandomHorizontalFlip(DualTransform):
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img, ranking=None, **kwargs):
        if random.random() < self.p:
            img = torch.flip(img, [-1])
            if ranking is not None:
                ranking = torch.flip(ranking, [-1])
        
        if ranking is not None:
            return img, ranking
        return img

class ColorJitter(DualTransform):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        
    def __call__(self, img, ranking=None, **kwargs):
        factors = []
        if self.brightness:
            factors.append(lambda x: TF.adjust_brightness(x, random.uniform(1-self.brightness, 1+self.brightness)))
        if self.contrast:
            factors.append(lambda x: TF.adjust_contrast(x, random.uniform(1-self.contrast, 1+self.contrast)))
        if self.saturation:
            factors.append(lambda x: TF.adjust_saturation(x, random.uniform(1-self.saturation, 1+self.saturation)))
        if self.hue:
            factors.append(lambda x: TF.adjust_hue(x, random.uniform(-self.hue, self.hue)))
        
        random.shuffle(factors)
        for factor in factors:
            img = factor(img)
        
        if ranking is not None:
            return img, ranking
        return img

class RandomErasing(DualTransform):
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False, mode='const'):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace
        self.mode = mode  # 'const', 'rand', 'pixel'
        
    def __call__(self, img, ranking=None, **kwargs):
        if random.random() >= self.p:
            if ranking is not None:
                return img, ranking
            return img
            
        # Erasing only applies to the image, not the ranking tensor
        img_h, img_w = img.shape[-2:]
        area = img_h * img_w
        
        for _ in range(10):
            erase_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            
            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            
            if h < img_h and w < img_w:
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)
                
                if not self.inplace:
                    img = img.clone()
                
                if self.mode == 'const':
                    img[..., i:i+h, j:j+w] = self.value
                elif self.mode == 'rand':
                    img[..., i:i+h, j:j+w] = torch.empty_like(img[..., i:i+h, j:j+w]).normal_()
                elif self.mode == 'pixel':
                    # Random pixels from the image
                    flattened = img.view(img.size(0), -1)
                    perm = torch.randperm(flattened.size(1))
                    idx = perm[:h*w]
                    random_pixels = flattened[:, idx].view(img.size(0), h, w)
                    img[..., i:i+h, j:j+w] = random_pixels
                
                break
        
        if ranking is not None:
            return img, ranking
        return img

class AutoAugment(DualTransform):
    """
    AutoAugment implementation compatible with dual image-ranking tensor processing.
    """
    def __init__(self, policy_name='rand-m9-mstd0.5-inc1', interpolation=InterpolationMode.BILINEAR):
        self.policy_name = policy_name
        self.interpolation = interpolation
        
        # Parse policy parameters (for RandAugment)
        if 'rand' in policy_name:
            parts = policy_name.split('-')
            self.magnitude = int(parts[1].replace('m', ''))
            self.std = float(parts[2].replace('mstd', '')) if len(parts) > 2 else 0
            self.inc = int(parts[3].replace('inc', '')) if len(parts) > 3 else 0
            self.num_ops = 2  # Default for RandAugment
        else:
            # For other policies like imagenet/cifar10, etc.
            self.magnitude = 10
            self.std = 0
            self.inc = 0
            self.num_ops = 2
    
    def _get_magnitude(self):
        """Get magnitude with random variation based on config."""
        m = self.magnitude
        if self.std > 0:
            m = random.gauss(m, self.std)
        return max(0, min(m, 10)) / 10  # Normalize to [0, 1]
    
    def __call__(self, img, ranking=None, **kwargs):
        # Operations pool (spatial operations that affect both img and ranking)
        spatial_ops = [
            self._rotate,
            self._shear_x,
            self._shear_y,
            self._translate_x,
            self._translate_y,
            self._auto_contrast,  # These don't affect ranking but are part of RandAugment
            self._equalize,
            self._posterize,
            self._solarize,
            self._color,
            self._contrast,
            self._brightness,
            self._sharpness,
        ]
        
        # For RandAugment, select random operations
        ops = random.choices(spatial_ops, k=self.num_ops)
        
        # Apply operations
        for op in ops:
            magnitude = self._get_magnitude()
            img, ranking = op(img, ranking, magnitude)
        
        if ranking is not None:
            return img, ranking
        return img
    
    # Individual operations
    def _rotate(self, img, ranking, magnitude):
        degrees = 30.0 * magnitude
        angle = random.uniform(-degrees, degrees)
        
        img = TF.rotate(img, angle, self.interpolation)
        if ranking is not None:
            # Use nearest neighbor interpolation for ranking tensor to preserve unique values
            ranking = TF.rotate(ranking, angle, InterpolationMode.NEAREST)
        
        return img, ranking
    
    def _shear_x(self, img, ranking, magnitude):
        shear = 0.3 * magnitude
        shear_factor = random.uniform(-shear, shear)
        
        img = TF.affine(img, angle=0, translate=[0, 0], scale=1.0, 
                      shear=[shear_factor, 0], interpolation=self.interpolation)
        
        if ranking is not None:
            # Use nearest neighbor interpolation for ranking tensor to preserve unique values
            ranking = TF.affine(ranking, angle=0, translate=[0, 0], scale=1.0, 
                             shear=[shear_factor, 0], interpolation=InterpolationMode.NEAREST)
        
        return img, ranking
    
    def _shear_y(self, img, ranking, magnitude):
        shear = 0.3 * magnitude
        shear_factor = random.uniform(-shear, shear)
        
        img = TF.affine(img, angle=0, translate=[0, 0], scale=1.0, 
                      shear=[0, shear_factor], interpolation=self.interpolation)
        
        if ranking is not None:
            # Use nearest neighbor interpolation for ranking tensor to preserve unique values
            ranking = TF.affine(ranking, angle=0, translate=[0, 0], scale=1.0, 
                             shear=[0, shear_factor], interpolation=InterpolationMode.NEAREST)
        
        return img, ranking
    
    def _translate_x(self, img, ranking, magnitude):
        _, h, w = img.shape
        translate_ratio = 0.45 * magnitude
        max_dx = int(translate_ratio * w)
        tx = random.randint(-max_dx, max_dx)
        
        img = TF.affine(img, angle=0, translate=[tx, 0], scale=1.0, 
                      shear=[0, 0], interpolation=self.interpolation)
        
        if ranking is not None:
            # Use nearest neighbor interpolation for ranking tensor to preserve unique values
            ranking = TF.affine(ranking, angle=0, translate=[tx, 0], scale=1.0, 
                             shear=[0, 0], interpolation=InterpolationMode.NEAREST)
        
        return img, ranking
    
    def _translate_y(self, img, ranking, magnitude):
        _, h, w = img.shape
        translate_ratio = 0.45 * magnitude
        max_dy = int(translate_ratio * h)
        ty = random.randint(-max_dy, max_dy)
        
        img = TF.affine(img, angle=0, translate=[0, ty], scale=1.0, 
                      shear=[0, 0], interpolation=self.interpolation)
        
        if ranking is not None:
            # Use nearest neighbor interpolation for ranking tensor to preserve unique values
            ranking = TF.affine(ranking, angle=0, translate=[0, ty], scale=1.0, 
                             shear=[0, 0], interpolation=InterpolationMode.NEAREST)
        
        return img, ranking
        
    # Non-spatial operations (only affect image, not ranking)
    def _auto_contrast(self, img, ranking, magnitude):
        # Auto contrast doesn't use magnitude
        img = TF.autocontrast(img)
        return img, ranking
    
    def _equalize(self, img, ranking, magnitude):
        # Equalize doesn't use magnitude
        img = TF.equalize(img)
        return img, ranking
    
    def _posterize(self, img, ranking, magnitude):
        # Posterize reduces bits, 4 to 8 is a good range
        bits = int(8 - (4 * magnitude))
        img = TF.posterize(img, bits)
        return img, ranking
    
    def _solarize(self, img, ranking, magnitude):
        # Solarize inverts all pixels above threshold
        threshold = 1.0 - magnitude
        img = TF.solarize(img, threshold)
        return img, ranking
    
    def _color(self, img, ranking, magnitude):
        # Adjust color balance
        factor = 1.0 + (0.9 * magnitude)
        img = TF.adjust_saturation(img, factor)
        return img, ranking
    
    def _contrast(self, img, ranking, magnitude):
        # Adjust contrast
        factor = 1.0 + (0.9 * magnitude)
        img = TF.adjust_contrast(img, factor)
        return img, ranking
    
    def _brightness(self, img, ranking, magnitude):
        # Adjust brightness
        factor = 1.0 + (0.9 * magnitude)
        img = TF.adjust_brightness(img, factor)
        return img, ranking
    
    def _sharpness(self, img, ranking, magnitude):
        # Adjust sharpness
        factor = 1.0 + (0.9 * magnitude)
        img = TF.adjust_sharpness(img, factor)
        return img, ranking

class DualCenterCrop(DualTransform):
    """Center crop transform that can be applied to both image and ranking tensors."""
    
    def __init__(self, size):
        """
        Args:
            size (int or tuple): Desired output size of the crop.
                If size is an int, a square crop is made.
        """
        self.size = size if isinstance(size, (list, tuple)) else (size, size)
        
    def __call__(self, img, ranking=None, **kwargs):
        """
        Args:
            img (Tensor): Image to be center cropped
            ranking (Tensor, optional): Ranking tensor to be center cropped
            
        Returns:
            Tensor or tuple: Cropped image, or (cropped image, cropped ranking)
        """
        if isinstance(img, torch.Tensor):
            # For tensor inputs
            _, h, w = img.shape
            th, tw = self.size
            
            i = (h - th) // 2
            j = (w - tw) // 2
            
            img = img[:, i:i+th, j:j+tw]
            
            if ranking is not None:
                ranking = ranking[:, i:i+th, j:j+tw]
                return img, ranking
            return img
        else:
            # For PIL inputs
            img = TF.center_crop(img, self.size)
            if ranking is not None:
                # Handle PIL to tensor conversion if needed
                if not isinstance(ranking, torch.Tensor):
                    raise ValueError("Ranking should be a tensor even when image is PIL")
                    
                # Apply center crop to ranking
                _, h, w = ranking.shape
                th, tw = self.size
                
                i = (h - th) // 2
                j = (w - tw) // 2
                
                ranking = ranking[:, i:i+th, j:j+tw]
                return img, ranking
            return img

class DualResize(DualTransform):
    """Resize transform that can be applied to both image and ranking tensors."""
    
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        """
        Args:
            size (int or tuple): Desired output size. 
                If size is a sequence like (h, w), output size will be matched to this.
                If size is an int, smaller edge of the image will be matched to this number.
            interpolation (InterpolationMode): Desired interpolation mode for image.
                Ranking tensors always use NEAREST to preserve unique values.
        """
        self.size = size if isinstance(size, (list, tuple)) else (size, size)
        self.interpolation = interpolation
        
    def __call__(self, img, ranking=None, **kwargs):
        """
        Args:
            img (Tensor or PIL Image): Image to be resized
            ranking (Tensor, optional): Ranking tensor to be resized
            
        Returns:
            Tensor or tuple: Resized image, or (resized image, resized ranking)
        """
        if isinstance(img, torch.Tensor):
            # For tensor inputs, use F.interpolate
            if len(img.shape) == 3:  # Add batch dimension if missing
                img = F.interpolate(img.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False).squeeze(0)
            else:
                img = F.interpolate(img, size=self.size, mode='bilinear', align_corners=False)
            
            if ranking is not None:
                # Always use nearest neighbor interpolation for ranking tensors
                if len(ranking.shape) == 3:  # Add batch dimension if missing
                    ranking = F.interpolate(ranking.unsqueeze(0), size=self.size, mode='nearest').squeeze(0)
                else:
                    ranking = F.interpolate(ranking, size=self.size, mode='nearest')
                return img, ranking
            return img
        else:
            # For PIL inputs
            img = TF.resize(img, self.size, interpolation=self.interpolation)
            
            if ranking is not None:
                # Handle ranking tensor resize
                if len(ranking.shape) == 3:  # Add batch dimension if missing
                    ranking = F.interpolate(ranking.unsqueeze(0), size=self.size, mode='nearest').squeeze(0)
                else:
                    ranking = F.interpolate(ranking, size=self.size, mode='nearest')
                return img, ranking
            return img

class DualTransforms:
    """
    Collection of transforms that match timm's create_transforms with 
    support for simultaneous image and ranking tensor transformations.
    Ensures ranking tensor undergoes the same geometric transformations as images,
    while preserving the meaning of ranking values.
    """
    def __init__(
        self, 
        size=224,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation=InterpolationMode.BILINEAR,
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
    ):
        self.size = size
        self.transforms = []
        
        # Add random resized crop
        self.transforms.append(DualRandomResizedCrop(size=size, interpolation=interpolation))
        
        # Add horizontal flip
        self.transforms.append(DualRandomHorizontalFlip(p=0.5))
        
        # Add color jitter if specified
        if color_jitter > 0:
            if isinstance(color_jitter, (list, tuple)):
                # Individual values for brightness, contrast, etc.
                self.transforms.append(ColorJitter(*color_jitter))
            else:
                # Single value for all parameters
                self.transforms.append(ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter,
                    hue=color_jitter/2
                ))
        
        # Add auto augment if specified
        if auto_augment:
            self.transforms.append(AutoAugment(
                policy_name=auto_augment,
                interpolation=interpolation
            ))
        
        # Add random erasing if specified
        if re_prob > 0:
            self.transforms.append(RandomErasing(
                p=re_prob,
                mode=re_mode,
                value=0,
                inplace=False
            ))
    
    def __call__(self, img, ranking=None):
        """
        Apply all transforms to the image and ranking tensor.
        
        Args:
            img: Image tensor [C, H, W]
            ranking: Optional ranking tensor [C, H, W]
            
        Returns:
            Transformed image or (transformed image, transformed ranking)
        """
        for t in self.transforms:
            if ranking is not None:
                img, ranking = t(img, ranking)
                
                # For spatial transforms that might introduce duplicate values,
                # we need to ensure uniqueness is maintained without completely
                # reassigning values. We want transformed rankings to be
                # similar to the original ranking where possible.
                if isinstance(t, (DualRandomResizedCrop, DualResize, DualCenterCrop, 
                               DualRandomHorizontalFlip, AutoAugment)):
                    # Check uniqueness
                    if not _has_unique_values(ranking):
                        # We want to preserve the ranking values where possible,
                        # only making minimal adjustments to ensure uniqueness
                        ranking = _preserve_ranking_values(ranking)
            else:
                img = t(img)
        
        if ranking is not None:
            return img, ranking
        return img

def _has_unique_values(tensor):
    """Check if tensor has the expected number of unique values."""
    unique_count = len(torch.unique(tensor))
    expected_count = tensor.shape[-1] * tensor.shape[-2]
    return unique_count >= expected_count * 0.99  # Allow small tolerance

def _preserve_ranking_values(tensor):
    """
    Preserve original ranking values as much as possible while ensuring uniqueness.
    
    This function makes minimal adjustments to ensure uniqueness:
    1. Identifies duplicated values
    2. Modifies duplicated values to ensure uniqueness while preserving ordering
    3. Prioritizes preserving the most interesting patches (lower ranking values)
    
    Args:
        tensor (torch.Tensor): The ranking tensor that needs uniqueness enforcement
        
    Returns:
        torch.Tensor: A tensor with unique integer values that maintain the
                    relative importance of the original values
    """
    # Flatten tensor for processing
    shape = tensor.shape
    flat_tensor = tensor.reshape(-1)
    
    # Get unique values and their counts
    unique_values, counts = torch.unique(flat_tensor, return_counts=True)
    
    # If all values are already unique, return unchanged
    if (counts == 1).all():
        return tensor
        
    # Identify duplicated values
    duplicate_mask = counts > 1
    duplicate_values = unique_values[duplicate_mask]
    
    # Create result tensor - start with a copy
    result = flat_tensor.clone()
    
    # Process each duplicated value
    for duplicate in duplicate_values:
        # Find all positions where this duplicate appears
        positions = (flat_tensor == duplicate).nonzero().squeeze(-1)
        
        # Sort positions by priority (if we have any heuristic for importance)
        # For now, we'll just keep the first occurrence as is and modify the others
        
        # Keep first occurrence unchanged
        # Shift others to available integer slots that don't conflict
        for i, pos in enumerate(positions[1:], 1):
            # Start searching from the original value
            new_value = duplicate
            
            # Find the next available integer value that's not used
            while (result == new_value).any():
                new_value += 1
                
            # Assign the new unique value
            result[pos] = new_value
    
    # Reshape to original dimensions
    return result.reshape(shape)

class ToTensorWithRanking(DualTransform):
    """Convert a PIL Image or numpy.ndarray to tensor and leave ranking as is."""
    
    def __call__(self, img, ranking=None, **kwargs):
        if not isinstance(img, torch.Tensor):
            img = TF.to_tensor(img)
            
        if ranking is not None:
            return img, ranking
        return img
        
class NormalizeWithRanking(DualTransform):
    """Normalize a tensor image with mean and standard deviation and leave ranking as is."""
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, img, ranking=None, **kwargs):
        img = TF.normalize(img, self.mean, self.std)
        
        if ranking is not None:
            return img, ranking
        return img