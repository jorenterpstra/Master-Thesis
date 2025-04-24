# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import csv
from glob import glob
from pathlib import Path
import torch

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader, IMG_EXTENSIONS

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


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
    
    Args:
        root (string): Root directory path for images.
        rankings_dir (string): Directory containing ranking CSV files.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and checks if the file is a valid file.
        random_rankings (bool, optional): If True, use random rankings instead of loaded ones.
    """
    def __init__(self, root, rankings_dir, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, random_rankings=False):
        super(RankedImageFolder, self).__init__(root, transform=transform,
                                              target_transform=target_transform,
                                              loader=loader, is_valid_file=is_valid_file)
        
        self.rankings_dir = rankings_dir
        self.random_rankings = random_rankings
        
        # Calculate number of patches for a 224x224 image with 16x16 patches and stride 16
        self.patch_size = 16
        self.stride = 16
        self.image_size = 224
        self.num_patches_per_dim = (self.image_size - self.patch_size) // self.stride + 1
        self.total_patches = self.num_patches_per_dim * self.num_patches_per_dim
        
        # Dictionary to store rankings: {image_path: ranking_array}
        self.rankings = {}
        
        # Load rankings for all classes
        if not self.random_rankings:
            self._load_all_rankings()
    
    def _load_all_rankings(self):
        """Load rankings from CSV files for all classes."""
        print(f"Loading rankings from {self.rankings_dir}")
                
        # Get all class names from the folder structure
        class_names = [d.name for d in os.scandir(self.root) if os.path.isdir(os.path.join(self.root, d.name))]
        
        # Build comprehensive mapping from filenames to paths
        # Include class in the key to handle potential filename collisions across classes
        self.filename_to_path = {}
        self.filename_only_to_paths = {}  # For debugging and collision detection
        
        for path, target in self.samples:
            filename = os.path.basename(path)
            class_name = os.path.basename(os.path.dirname(path))
            
            # Store with class prefix for unique lookup
            class_prefixed_key = f"{class_name}/{filename}"
            self.filename_to_path[class_prefixed_key] = path
            
            # Also store without class for debugging
            if filename not in self.filename_only_to_paths:
                self.filename_only_to_paths[filename] = []
            self.filename_only_to_paths[filename].append((path, class_name))
        
        # Check for duplicate filenames across classes (for warning purposes)
        duplicate_filenames = {f: paths for f, paths in self.filename_only_to_paths.items() if len(paths) > 1}
        if duplicate_filenames:
            print(f"Warning: Found {len(duplicate_filenames)} filenames that appear in multiple classes.")
            for filename, paths in list(duplicate_filenames.items())[:3]:  # Show first 3 examples
                classes = [p[1] for p in paths]
                print(f"  '{filename}' appears in classes: {classes}")
            if len(duplicate_filenames) > 3:
                print(f"  ...and {len(duplicate_filenames) - 3} more.")
        
        # For each class, load its rankings
        total_loaded = 0
        missing_rankings = 0
        classes_with_missing = set()
        
        for class_name in class_names:
            # Look for ranking file
            ranking_file = os.path.join(self.rankings_dir, f"{class_name}_rankings.csv")
            
            if not os.path.exists(ranking_file):
                print(f"Warning: No ranking file found for class {class_name}")
                continue
            
            # Keep track of which files in this class had rankings
            class_has_rankings = False
            class_loaded_count = 0
            
            # Load rankings for this class
            with open(ranking_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header row
                
                for row in reader:
                    if len(row) < 2:
                        continue
                        
                    image_filename = row[0]
                    rankings_str = row[1]
                    
                    # Try both with and without class prefix
                    class_prefixed_key = f"{class_name}/{image_filename}"
                    path = self.filename_to_path.get(class_prefixed_key)
                    
                    if path is not None:
                        # Found a match with class prefix
                        try:
                            # Convert rankings string directly to tensor instead of numpy array
                            ranking = torch.tensor(list(map(int, rankings_str.split(','))), dtype=torch.long)
                            self.rankings[path] = ranking
                            total_loaded += 1
                            class_loaded_count += 1
                            class_has_rankings = True
                        except Exception as e:
                            print(f"Error parsing ranking for {image_filename} in {class_name}: {e}")
                    else:
                        # No match found - try looking across all classes for debugging
                        all_matches = self.filename_only_to_paths.get(image_filename, [])
                        
                        if all_matches:
                            # Found matches in other classes
                            other_classes = [p[1] for p in all_matches]
                            print(f"Warning: '{image_filename}' from {class_name}'s rankings found in other classes: {other_classes}")
                            missing_rankings += 1
                        else:
                            # No matches anywhere
                            missing_rankings += 1
                            if missing_rankings <= 10:
                                print(f"Warning: No image file matches '{image_filename}' from {class_name}'s rankings")
            
            if not class_has_rankings:
                classes_with_missing.add(class_name)
            else:
                print(f"  Class {class_name}: Loaded {class_loaded_count} rankings")
        
        print(f"Loaded rankings for {total_loaded} images across {len(class_names) - len(classes_with_missing)} classes.")
        print(f"Missing rankings: {missing_rankings}")
        
        # Verify that we have rankings for most images
        coverage = total_loaded / len(self.samples) if self.samples else 0
        if coverage < 0.5 and not self.random_rankings:
            print(f"Warning: Only found rankings for {total_loaded}/{len(self.samples)} images ({coverage:.1%})")
            print(f"Consider using random rankings instead by setting random_rankings=True")
        else:
            print(f"Ranking coverage: {coverage:.1%}")
    
    def __getitem__(self, index):
        """
        Override the __getitem__ method to return (image, target, ranking) triplets.
        
        Args:
            index (int): Index of the sample to fetch
            
        Returns:
            tuple: (image, target, ranking) where ranking is a tensor of patch indices
        """
        # Get path and target using ImageFolder's internal structure
        path, target = self.samples[index]
        filename = os.path.basename(path)
        class_name = os.path.basename(os.path.dirname(path))
        
        # Load the image
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # Get ranking for this image
        if path in self.rankings:
            ranking = self.rankings[path]
        else:
            # If no ranking exists, create default one
            # sequential ordering as fallback (directly as tensor)
            ranking = torch.arange(self.total_patches, dtype=torch.long)
            # This can flood the output, so limit the warnings
            if hasattr(self, '_warning_count') and self._warning_count < 10:
                print(f"Warning: No ranking found for {filename} in class {class_name} (path: {path})")
                self._warning_count += 1
            elif not hasattr(self, '_warning_count'):
                self._warning_count = 1
                print(f"Warning: No ranking found for {filename} in class {class_name} (path: {path})")
        
        return image, target, ranking


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

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
        rankings_dir = os.path.join(args.rankings_dir, 'train' if is_train else 'val')
        dataset = RankedImageFolder(root, rankings_dir, transform=transform)
        nb_classes = 200

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
