import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import BasicTransform, ImageOnlyTransform
import numpy as np
import torch
import random
import cv2
from functools import partial

from PIL import Image, ImageEnhance, ImageOps, ImageFilter

_RAND_INCREASING_TRANSFORMS = [
    'AutoContrast',
    'Equalize',
    'Invert',
    'Rotate',
    'PosterizeIncreasing',
    'SolarizeIncreasing',
    'SolarizeAdd',
    'ColorIncreasing',
    'ContrastIncreasing',
    'BrightnessIncreasing',
    'SharpnessIncreasing',
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',
]

# Categorize transforms as spatial (affecting layout) or non-spatial (affecting only appearance)
_SPATIAL_TRANSFORMS = {
    'Rotate', 'ShearX', 'ShearY', 'TranslateXRel', 'TranslateYRel'
}

_LEVEL_DENOM = 10

def _randomly_negate(v):
    """With 50% prob, negate the value"""
    return -v if random.random() > 0.5 else v

def _enhance_increasing_level_to_arg(level, _hparams=None):
    # the 'no change' level is 1.0, moving away from that towards 0. or 2.0 increases the enhancement blend
    # range [0.1, 1.9] if level <= _LEVEL_DENOM
    level = (level / _LEVEL_DENOM) * .9
    level = max(0.1, 1.0 + _randomly_negate(level))  # keep it >= 0.1
    return level,

def _rotate_level_to_arg(level):
    # range [-30, 30]
    level = (level / _LEVEL_DENOM) * 30.
    level = _randomly_negate(level)
    return level

def _shear_level_to_arg(level):
    # range [-0.3, 0.3]
    level = (level / _LEVEL_DENOM) * 0.3
    level = _randomly_negate(level)
    return level

def _translate_rel_level_to_arg(level, translate_pct=0.45):
    # default range [-0.45, 0.45]
    level = (level / _LEVEL_DENOM) * translate_pct
    level = _randomly_negate(level)
    return level

def _posterize_increasing_level_to_arg(level):
    # Range [4, 0], 'keep 4 down to 0 MSB of original image'
    # intensity increases with level
    level = 4 - int((level / _LEVEL_DENOM) * 4)
    return max(1, level)  # Ensure at least 1 bit

def _solarize_increasing_level_to_arg(level):
    # range [256, 0]
    # intensity increases with level
    level = 256 - int((level / _LEVEL_DENOM) * 256)
    return level

def _solarize_add_level_to_arg(level):
    # range [0, 110]
    return min(128, int((level / _LEVEL_DENOM) * 110))

def SolarizeAdd(img, add_value):
    """Implement SolarizeAdd using numpy operations"""
    img = np.clip(img + add_value, 0, 255).astype(np.uint8)
    threshold = 128
    return np.where(img < threshold, img, 255 - img)

def apply_solarize_add(img, add_value, **kwargs):
    """Function version of SolarizeAdd for multiprocessing compatibility"""
    img = np.clip(img + add_value, 0, 255).astype(np.uint8)
    threshold = 128
    return np.where(img < threshold, img, 255 - img)

class AlbumentationsRandAugment(BasicTransform):
    """Implementation of RandAugment using albumentations transforms"""
    
    def __init__(self, num_ops=2, magnitude=9, std=0.5, p=1.0):
        # Remove always_apply parameter as it's not valid
        super().__init__(p=p)
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.std = std
        self.transforms = self._create_transforms()
        # Cache for selected operations
        self._selected_ops = []
        # For storing intermediate results
        self._cached_result = {}
    
    @property
    def targets(self):
        # Both mask and heatmap will use the same handling
        return {"image": self.apply, "mask": self.apply_to_mask}
    
    def get_params_dependent_on_targets(self, params):
        return {}
    
    def get_transform_init_args_names(self):
        return ("num_ops", "magnitude", "std")
    
    def apply(self, img, **params):
        # Process image and cache transforms for mask/heatmap
        selected_ops = random.choices(list(_RAND_INCREASING_TRANSFORMS), k=self.num_ops)
        self._selected_ops = selected_ops
        
        # Ensure img is numpy array (handle PIL Image)
        if not isinstance(img, np.ndarray):
            img = np.array(img)
            
        result = img.copy()
        
        # Apply transforms to image
        for op_name in selected_ops:
            if op_name in self.transforms:
                level = self._sample_level()
                transform_info = self.transforms[op_name]
                transform = transform_info['op'](level)
                
                if transform_info['spatial']:
                    # Store for mask/heatmap
                    self._cached_result[op_name] = {
                        'transform': transform,
                        'level': level
                    }
                
                # Apply to image
                result = transform(image=result)["image"]
        
        return result
    
    def apply_to_mask(self, mask, **params):
        # Ensure mask is numpy array (handle PIL Image)
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
            
        result = mask.copy()
        
        for op_name in self._selected_ops:
            if op_name in self.transforms and self.transforms[op_name]['spatial']:
                # Use the same transform as was applied to the image
                if op_name in self._cached_result:
                    transform = self._cached_result[op_name]['transform']
                    result = transform(image=result)["image"]
        
        return result

    def _sample_level(self):
        """Sample magnitude with Gaussian noise"""
        magnitude = self.magnitude
        if self.std > 0:
            if self.std == float('inf'):
                magnitude = random.uniform(0, magnitude)
            else:
                magnitude = random.gauss(magnitude, self.std)
        return max(0, min(magnitude, 10))
    
    def _create_transforms(self):
        """Create mapping of transform names to their albumentations implementations"""
        transforms = {}
        
        # SPATIAL TRANSFORMS (affecting both image and mask) - using Affine
        
        # Rotate: range [-30, 30] degrees
        transforms['Rotate'] = {
            'op': lambda level: A.Affine(
                rotate=_rotate_level_to_arg(level),
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0
            ),
            'spatial': True
        }
        
        # ShearX: range [-0.3, 0.3]
        transforms['ShearX'] = {
            'op': lambda level: A.Affine(
                shear={'x': _shear_level_to_arg(level), 'y': 0},
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0
            ),
            'spatial': True
        }
        
        # ShearY: range [-0.3, 0.3]
        transforms['ShearY'] = {
            'op': lambda level: A.Affine(
                shear={'x': 0, 'y': _shear_level_to_arg(level)},
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0
            ),
            'spatial': True
        }
        
        # TranslateXRel: range [-0.45, 0.45] of image width
        transforms['TranslateXRel'] = {
            'op': lambda level: A.Affine(
                translate_percent={'x': _translate_rel_level_to_arg(level), 'y': 0},
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0
            ),
            'spatial': True
        }
        
        # TranslateYRel: range [-0.45, 0.45] of image height
        transforms['TranslateYRel'] = {
            'op': lambda level: A.Affine(
                translate_percent={'x': 0, 'y': _translate_rel_level_to_arg(level)},
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0
            ),
            'spatial': True
        }
        
        # IMAGE-ONLY TRANSFORMS (affecting only appearance)
        
        # AutoContrast
        transforms['AutoContrast'] = {
            'op': lambda level: A.CLAHE(clip_limit=4, p=1.0),
            'spatial': False
        }
        
        # Equalize
        transforms['Equalize'] = {
            'op': lambda level: A.Equalize(p=1.0),
            'spatial': False
        }
        
        # Invert
        transforms['Invert'] = {
            'op': lambda level: A.InvertImg(p=1.0),
            'spatial': False
        }
        
        # PosterizeIncreasing: intensity increases with level
        transforms['PosterizeIncreasing'] = {
            'op': lambda level: A.Posterize(
                num_bits=_posterize_increasing_level_to_arg(level),
                p=1.0
            ),
            'spatial': False
        }
        
        # SolarizeIncreasing: fix threshold parameter
        transforms['SolarizeIncreasing'] = {
            'op': lambda level: A.Solarize(
                p=1.0
            ),
            'spatial': False
        }
        
        # SolarizeAdd: use partial instead of lambda
        transforms['SolarizeAdd'] = {
            'op': lambda level: A.Lambda(
                image=partial(apply_solarize_add, add_value=_solarize_add_level_to_arg(level)),
                p=1.0
            ),
            'spatial': False
        }
        
        # ColorIncreasing - Fix sat_shift_limit to be a proper range
        transforms['ColorIncreasing'] = {
            'op': lambda level: A.HueSaturationValue(
                hue_shift_limit=0,  # No hue change
                # Create a proper range for saturation from 0 to magnitude-based value
                sat_shift_limit=(-30 * level / _LEVEL_DENOM, 30 * level / _LEVEL_DENOM),
                val_shift_limit=0,  # No value change
                p=1.0
            ),
            'spatial': False
        }
        
        # ContrastIncreasing - Fix contrast_limit to be a proper range within [-1.0, 1.0]
        transforms['ContrastIncreasing'] = {
            'op': lambda level: A.RandomBrightnessContrast(
                brightness_limit=0,  # No brightness change
                # Create a proper range for contrast from 0 to magnitude-based value
                contrast_limit=(0, min(0.8, level / _LEVEL_DENOM * 0.8)),
                p=1.0
            ),
            'spatial': False
        }
        
        # BrightnessIncreasing - Fix brightness_limit to be a proper range within [-1.0, 1.0]
        transforms['BrightnessIncreasing'] = {
            'op': lambda level: A.RandomBrightnessContrast(
                # Create a proper range for brightness from 0 to magnitude-based value
                brightness_limit=(0, min(0.8, level / _LEVEL_DENOM * 0.8)),
                contrast_limit=0,  # No contrast change
                p=1.0
            ),
            'spatial': False
        }
        
        # SharpnessIncreasing - Fix lightness to ensure positive values
        transforms['SharpnessIncreasing'] = {
            'op': lambda level: A.Sharpen(
                alpha=(0.2, 0.5),
                # Ensure lightness is in valid range (must be non-negative)
                lightness=(0.1, min(1.0, 0.1 + level / _LEVEL_DENOM * 0.9)),
                p=1.0
            ),
            'spatial': False
        }
        
        return transforms

    def __call__(self, force_apply=False, **data):
        if self.p < 1 and random.random() > self.p and not force_apply:
            return data
        
        # Make a copy to avoid modifying input
        params = self.get_params()
        
        # Process image first
        if "image" in data:
            data["image"] = self.apply(data["image"], **{**params, **data})
        
        # Process mask/heatmap with the same spatial transforms
        for key in ["mask", "heatmap"]:
            if key in data:
                data[key] = self.apply_to_mask(data[key], **{**params, **data})
        
        return data

def build_transform(is_train, args):
    """Create an albumentations transform pipeline for images and heatmaps"""
    resize_im = args.input_size > 32
    
    if is_train:
        # Build training transforms
        transform_list = []
        
        # Basic spatial transforms
        if resize_im:
            transform_list.append(
                A.RandomResizedCrop(
                    size=(args.input_size, args.input_size),
                    scale=(0.08, 1.0),
                    interpolation=cv2.INTER_CUBIC
                )
            )
        else:
            transform_list.append(
                A.PadIfNeeded(
                    min_height=args.input_size + 8,
                    min_width=args.input_size + 8,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0
                )
            )
            transform_list.append(
                A.RandomCrop(
                    height=args.input_size,
                    width=args.input_size
                )
            )
        
        # Add horizontal flip
        transform_list.append(A.HorizontalFlip(p=0.5))
        
        # Color jitter (only affects image, not heatmap)
        if args.color_jitter > 0:
            transform_list.append(
                A.ColorJitter(
                    brightness=0.4 * args.color_jitter,
                    contrast=0.4 * args.color_jitter,
                    saturation=0.4 * args.color_jitter,
                    hue=0.1 * args.color_jitter,
                    p=0.8
                )
            )
        
        # RandAugment
        if args.aa and args.aa.startswith('rand'):
            # Parse parameters from the policy string
            magnitude = 9  # Default m9
            std = 0.5      # Default mstd0.5
            num_ops = 2    # Default operations
            
            # Extract values if available in the policy string
            if 'm' in args.aa:
                magnitude_str = args.aa.split('-')[1][1:]
                if magnitude_str.isdigit():
                    magnitude = int(magnitude_str)
                    
            if 'mstd' in args.aa:
                mstd_parts = [p for p in args.aa.split('-') if p.startswith('mstd')]
                if mstd_parts:
                    std = float(mstd_parts[0][4:])
            
            if 'n' in args.aa:
                n_parts = [p for p in args.aa.split('-') if p.startswith('n') and len(p) > 1]
                if n_parts and n_parts[0][1:].isdigit():
                    num_ops = int(n_parts[0][1:])
            
            # Add the RandAugment transform
            transform_list.append(AlbumentationsRandAugment(num_ops=num_ops, magnitude=magnitude, std=std))
        
        # Normalization and conversion to tensor
        
        # Random erasing (only applied to image tensor, not heatmap)
        if args.reprob > 0:
            transform_list.append(
                A.CoarseDropout(num_holes_range=(1, 1),
                    fill='random',
                    p=args.reprob,
                    # CoarseDropout doesn't take mask_fill_value, so we fix that
                    # mask_fill_value is handled through additional_targets in Compose
                )
            )
        
        transform_list.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        transform_list.append(ToTensorV2())
        # Compose all transforms - specify that heatmap should be treated as a mask
        transform = A.Compose(transform_list, additional_targets={'heatmap': 'mask'})
        return transform
    
    else:
        # Validation transforms
        transform_list = []
        
        if resize_im:
            size = int(args.input_size / args.eval_crop_ratio)
            transform_list.append(A.Resize(height=size, width=size, interpolation=cv2.INTER_CUBIC))
            transform_list.append(A.CenterCrop(height=args.input_size, width=args.input_size))
            
        transform_list.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        transform_list.append(ToTensorV2())
        
        transform = A.Compose(transform_list, additional_targets={'heatmap': 'mask'})
        return transform
