# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import csv
from glob import glob
from pathlib import Path
import torch
import pickle
import logging
import numpy as np
import cv2

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader, IMG_EXTENSIONS

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from transforms import build_transform




class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


class RankedImageFolder(ImageFolder):
    """
    A custom dataset that extends ImageFolder to include patch rankings.
    """
    def __init__(self, root, rankings_dir, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, random_rankings=False,
                 cache_rankings=True, log_level="INFO", return_path=False, global_ranking=None):
        # No transform tracking needed since we removed spatial transforms
        super(RankedImageFolder, self).__init__(root, transform=transform,
                                              target_transform=target_transform,
                                              loader=loader, is_valid_file=is_valid_file)
        
        self.rankings_dir = rankings_dir
        self.random_rankings = random_rankings
        self.cache_rankings = cache_rankings
        self.global_ranking = global_ranking  # Global ranking parameter

        # Setup logger
        self.logger = logging.getLogger("RankedImageFolder")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Patch/grid parameters
        self.patch_size = 16
        self.stride = 16
        self.image_size = 224
        self.num_patches_per_dim = (self.image_size - self.patch_size) // self.stride + 1
        self.total_patches = self.num_patches_per_dim * self.num_patches_per_dim

        # Rankings dict: {image_path: tensor}
        self.rankings = {}
        
        # Skip loading individual rankings if global ranking is provided
        if self.global_ranking is not None:
            self._log(f"Using global ranking for all images. Skipping individual rankings loading.", "info")
            return
            
        # Rankings cache file
        self.rankings_cache_file = os.path.join(self.rankings_dir, "rankings_cache.pkl")

        if not self.random_rankings:
            if self.cache_rankings and os.path.exists(self.rankings_cache_file):
                self.logger.info(f"Loading rankings from cache: {self.rankings_cache_file}")
                with open(self.rankings_cache_file, "rb") as f:
                    self.rankings = pickle.load(f)
                self.logger.info(f"Loaded {len(self.rankings)} rankings from cache.")
            else:
                self._load_all_rankings()
                if self.cache_rankings:
                    self.logger.info(f"Caching rankings to {self.rankings_cache_file}")
                    with open(self.rankings_cache_file, "wb") as f:
                        pickle.dump(self.rankings, f)
        self.return_path = return_path

    def _log(self, msg, level="info"):
        if level == "info":
            self.logger.info(msg)
        elif level == "warning":
            self.logger.warning(msg)
        elif level == "error":
            self.logger.error(msg)
        else:
            self.logger.debug(msg)

    def _load_all_rankings(self):
        """Load rankings from CSV files for all classes."""
        self._log(f"Loading rankings from {self.rankings_dir}", "info")
        class_names = [d.name for d in os.scandir(self.root) if os.path.isdir(os.path.join(self.root, d.name))]
        self.filename_to_path = {}
        self.filename_only_to_paths = {}

        for path, target in self.samples:
            filename = os.path.basename(path)
            class_name = os.path.basename(os.path.dirname(path))
            class_prefixed_key = f"{class_name}/{filename}"
            self.filename_to_path[class_prefixed_key] = path
            if filename not in self.filename_only_to_paths:
                self.filename_only_to_paths[filename] = []
            self.filename_only_to_paths[filename].append((path, class_name))

        duplicate_filenames = {f: paths for f, paths in self.filename_only_to_paths.items() if len(paths) > 1}
        if duplicate_filenames:
            self._log(f"Found {len(duplicate_filenames)} filenames that appear in multiple classes.", "warning")
            for filename, paths in list(duplicate_filenames.items())[:3]:
                classes = [p[1] for p in paths]
                self._log(f"  '{filename}' appears in classes: {classes}", "warning")
            if len(duplicate_filenames) > 3:
                self._log(f"  ...and {len(duplicate_filenames) - 3} more.", "warning")

        total_loaded = 0
        missing_rankings = 0
        classes_with_missing = set()

        for class_name in class_names:
            ranking_file = os.path.join(self.rankings_dir, f"{class_name}_rankings.csv")
            if not os.path.exists(ranking_file):
                self._log(f"No ranking file found for class {class_name}", "warning")
                continue
            class_rank_heat_out = False
            class_loaded_count = 0
            with open(ranking_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) < 2:
                        continue
                    image_filename = row[0]
                    rankings_str = row[1]
                    class_prefixed_key = f"{class_name}/{image_filename}"
                    path = self.filename_to_path.get(class_prefixed_key)
                    if path is not None:
                        try:
                            ranking = torch.tensor(list(map(int, rankings_str.split(','))), dtype=torch.long)
                            self.rankings[path] = ranking
                            total_loaded += 1
                            class_loaded_count += 1
                            class_rank_heat_out = True
                        except Exception as e:
                            self._log(f"Error parsing ranking for {image_filename} in {class_name}: {e}", "error")
                    else:
                        all_matches = self.filename_only_to_paths.get(image_filename, [])
                        if all_matches:
                            other_classes = [p[1] for p in all_matches]
                            self._log(f"'{image_filename}' from {class_name}'s rankings found in other classes: {other_classes}", "warning")
                            missing_rankings += 1
                        else:
                            missing_rankings += 1
                            if missing_rankings <= 10:
                                self._log(f"No image file matches '{image_filename}' from {class_name}'s rankings", "warning")
            if not class_rank_heat_out:
                classes_with_missing.add(class_name)
            else:
                self._log(f"  Class {class_name}: Loaded {class_loaded_count} rankings", "info")

        self._log(f"Loaded rankings for {total_loaded} images across {len(class_names) - len(classes_with_missing)} classes.", "info")
        self._log(f"Missing rankings: {missing_rankings}", "info")
        coverage = total_loaded / len(self.samples) if self.samples else 0
        if coverage < 0.5 and not self.random_rankings:
            self._log(f"Only found rankings for {total_loaded}/{len(self.samples)} images ({coverage:.1%})", "warning")
            self._log(f"Consider using random rankings instead by setting random_rankings=True", "warning")
        else:
            self._log(f"Ranking coverage: {coverage:.1%}", "info")

    def __getitem__(self, index):
        """
        Returns (image, target, ranking, path)
        """
        path, target = self.samples[index]
        image = self.loader(path)
        
        # Prepare ranking
        # Use global ranking if provided (this will be the fastest path)
        if self.global_ranking is not None:
            ranking = self.global_ranking.clone()  # Clone to avoid modifying the original
        # Otherwise, use image-specific ranking
        elif path in self.rankings:
            ranking = self.rankings[path].clone()  # Clone to avoid modifying the original
        else:
            # Create sequential ranking if not found
            ranking = torch.arange(self.total_patches, dtype=torch.long)
            # Only log warnings if we're not deliberately using random or global rankings
            if not self.random_rankings:
                filename = os.path.basename(path)
                class_name = os.path.basename(os.path.dirname(path))
                if hasattr(self, '_warning_count') and self._warning_count < 10:
                    self._log(f"No ranking found for {filename} in class {class_name}", "warning")
                    self._warning_count += 1
                elif not hasattr(self, '_warning_count'):
                    self._warning_count = 1
                    self._log(f"No ranking found for {filename} in class {class_name}", "warning")
        
        # Reshape ranking to [1, H, W] format for transforms
        ranking = ranking.view(1, self.num_patches_per_dim, self.num_patches_per_dim)
        
        # Apply transforms
        if self.transform is not None:
            # Check if transform is our DualTransform
            if hasattr(self.transform, '__call__') and 'ranking' in self.transform.__call__.__code__.co_varnames:
                # This is a DualTransform - pass both image and ranking
                image, ranking = self.transform(image, ranking)
            else:
                # Standard transform only for image
                image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.return_path:
            return image, target, ranking, path
        else:
            return image, target, ranking
        

class HeatmapImageFolder(ImageFolder):
    """Dataset that loads images and their corresponding heatmaps from parallel directories"""
    
    def __init__(self, root, heatmap_root, transform=None, target_transform=None,
                 loader=default_loader, heatmap_extension='.png', return_path=False):
        super().__init__(root, transform=None, target_transform=target_transform, loader=loader)
        self.heatmap_root = heatmap_root
        self.heatmap_extension = heatmap_extension
        self.transform = transform
        self.return_path = return_path
        self.heatmap_loader = loader  # Use the same loader for both image and heatmap
        
        # Verify heatmaps exist for some random samples
        self._verify_heatmaps(10)
    
    def _verify_heatmaps(self, num_samples):
        """Verify that heatmap files exist for some sample images"""
        import random
        indices = random.sample(range(len(self.samples)), min(num_samples, len(self.samples)))
        missing = 0
        
        for idx in indices:
            path, _ = self.samples[idx]
            heatmap_path = self._get_heatmap_path(path)
            if not os.path.exists(heatmap_path):
                missing += 1
                print(f"Warning: Missing heatmap for {path} at {heatmap_path}")
        
        if missing > 0:
            print(f"Warning: {missing}/{num_samples} sample heatmaps not found.")
        else:
            print(f"Verified: {num_samples} sample heatmaps exist.")
    
    def _get_heatmap_path(self, image_path):
        """Convert image path to corresponding heatmap path"""
        rel_path = os.path.relpath(image_path, self.root)
        base_path = os.path.splitext(rel_path)[0]
        heatmap_path = os.path.join(self.heatmap_root, base_path + self.heatmap_extension)
        return heatmap_path
    
    def __getitem__(self, index):
        """Get image, heatmap, and target"""
        path, target = self.samples[index]
        
        # Load image
        image = self.loader(path)
        
        # Load heatmap as an image
        heatmap_path = self._get_heatmap_path(path)
        heatmap = self.heatmap_loader(heatmap_path)
        
        # Convert both to numpy for albumentations if needed
        if not isinstance(image, np.ndarray):
            image_np = np.array(image)
        else:
            image_np = image
            
        if not isinstance(heatmap, np.ndarray):
            heatmap_np = np.array(heatmap)
        else:
            heatmap_np = heatmap
            
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image_np, heatmap=heatmap_np)
            image = transformed['image']
            heatmap = transformed['heatmap']
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.return_path:
            return image, heatmap, target, path
        else:
            return image, heatmap, target
        


def build_dataset(is_train, args):
    
    transform = build_transform(is_train, args)
    rank_heat_out = False  # Default to no rankings or heatmaps

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 200
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'IMNET_RANK':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        rankings_dir = os.path.join(args.rankings_path, 'train' if is_train else 'val')
        global_ranking = None
        
        # Load global ranking if specified
        if hasattr(args, 'global_ranking_path') and args.global_ranking_path:
            try:
                global_ranking = torch.load(args.global_ranking_path)
                print(f"Using global ranking from {args.global_ranking_path}")
            except Exception as e:
                print(f"Error loading global ranking from {args.global_ranking_path}: {e}")
                print("Falling back to individual image rankings")
                
        dataset = RankedImageFolder(
            root, rankings_dir, transform=transform, 
            random_rankings=getattr(args, 'random_rankings', False),
            global_ranking=global_ranking
        )
        nb_classes = 200
        rank_heat_out = True  # RankedImageFolder always returns rankings

    elif args.data_set == 'IMNET_HEATMAP':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        heatmap_root = os.path.join(args.heatmap_path, 'train' if is_train else 'val')
        dataset = HeatmapImageFolder(
            root, heatmap_root, transform=transform,
            return_path=getattr(args, 'return_path', False)
        )
        nb_classes = 200
        rank_heat_out = True
    
    # Set ranking flag in args for easy access across the codebase
    if not hasattr(args, 'rank_heat_out'):
        args.rank_heat_out = {}
    args.rank_heat_out['train' if is_train else 'val'] = rank_heat_out

    return dataset, nb_classes


# ==== Unused Code, replaced by build_transform ====
# def build_transform(is_train, args):
#     resize_im = args.input_size > 32
#     if is_train:
#         # this should always dispatch to transforms_imagenet_train
#         transform = create_transform(
#             input_size=args.input_size,
#             is_training=True,
#             color_jitter=args.color_jitter,
#             auto_augment=args.aa,
#             interpolation=args.train_interpolation,
#             re_prob=args.reprob,
#             re_mode=args.remode,
#             re_count=args.recount,
#         )
#         if not resize_im:
#             # replace RandomResizedCropAndInterpolation with
#             # RandomCrop
#             transform.transforms[0] = transforms.RandomCrop(
#                 args.input_size, padding=4)
#         return transform

#     t = []
#     if resize_im:
#         size = int(args.input_size / args.eval_crop_ratio)
#         t.append(
#             transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
#         )
#         t.append(transforms.CenterCrop(args.input_size))

#     t.append(transforms.ToTensor())
#     t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
#     return transforms.Compose(t)
