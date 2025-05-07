import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import csv
import pickle
from tqdm import tqdm
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import cv2

class HeatmapGenerator:
    """
    Class for generating heatmaps from images using various techniques.
    These heatmaps can be used to derive image-specific rankings later.
    """
    def __init__(self, method='gradient', model_name='resnet18', device='cuda', 
                 bing_training_path=None, num_bboxes=1000):
        """
        Initialize the heatmap generator.
        
        Args:
            method (str): Method to generate heatmap. Options:
                - 'gradient': Uses gradient information to generate saliency maps
                - 'cam': Class Activation Mapping
                - 'center_outward': Simple heatmap that diminishes from center to edges
                - 'edge_detection': Uses edge detection to highlight important areas
                - 'bing': Uses BING objectness detection to highlight important objects
            model_name (str): Name of the model to use (for gradient and cam methods)
            device (str): Device to use for computation
            bing_training_path (str): Path to BING training data (required for 'bing' method)
            num_bboxes (int): Number of bounding boxes to use for BING heatmap
        """
        self.method = method
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.bing_training_path = bing_training_path
        self.num_bboxes = num_bboxes
        
        # Initialize model for gradient-based methods
        if method in ['gradient', 'cam']:
            if model_name == 'resnet18':
                self.model = models.resnet18(pretrained=True)
            elif model_name == 'resnet50':
                self.model = models.resnet50(pretrained=True)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # For CAM, we need to register hooks
            if method == 'cam':
                # Register hook to get activations
                self.activations = None
                self.gradients = None
                
                def forward_hook(module, input, output):
                    self.activations = output.detach()
                
                def backward_hook(module, grad_input, grad_output):
                    self.gradients = grad_output[0].detach()
                
                # Get the last convolutional layer
                if model_name == 'resnet18' or model_name == 'resnet50':
                    target_layer = self.model.layer4[-1].conv2
                    
                target_layer.register_forward_hook(forward_hook)
                target_layer.register_full_backward_hook(backward_hook)
        
        # Initialize BING detector if needed
        if method == 'bing':
            if bing_training_path is None:
                raise ValueError("BING training path is required for BING method")
            # We'll initialize the detector on demand to avoid issues
        
        # Standard image preprocessing (not used for BING method)
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def generate(self, image_path):
        """
        Generate a heatmap for the given image.
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            torch.Tensor: A heatmap tensor with shape [1, H, W]
        """
        if self.method == 'gradient':
            return self._generate_gradient_heatmap(image_path)
        elif self.method == 'cam':
            return self._generate_cam_heatmap(image_path)
        elif self.method == 'center_outward':
            return self._generate_center_outward_heatmap(image_path)
        elif self.method == 'edge_detection':
            return self._generate_edge_heatmap(image_path)
        elif self.method == 'bing':
            return self._generate_bing_heatmap(image_path)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def _generate_gradient_heatmap(self, image_path):
        """Generate heatmap using gradient-based saliency."""
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop is the predicted class
        one_hot = torch.zeros_like(output)
        one_hot[0, pred_class] = 1
        
        # Backward pass
        output.backward(gradient=one_hot)
        
        # Get gradients with respect to input
        gradients = input_tensor.grad.data.abs()
        
        # Average over channels 
        heatmap = gradients.mean(dim=1, keepdim=True)
        
        # Normalize between 0 and 1
        heatmap = F.interpolate(heatmap, size=(224, 224), mode='bilinear', align_corners=False)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap.detach().cpu()
    
    def _generate_cam_heatmap(self, image_path):
        """Generate heatmap using Class Activation Mapping."""
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Forward pass
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop is the predicted class
        one_hot = torch.zeros_like(output)
        one_hot[0, pred_class] = 1
        
        # Backward pass
        output.backward(gradient=one_hot)
        
        # Create CAM
        # Get pooled gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3], keepdim=True)
        
        # Weight activation maps by gradients
        heatmap = self.activations * pooled_gradients
        
        # Average over channels and apply ReLU
        heatmap = torch.mean(heatmap, dim=1, keepdim=True)
        heatmap = F.relu(heatmap)
        
        # Normalize between 0 and 1
        heatmap = F.interpolate(heatmap, size=(224, 224), mode='bilinear', align_corners=False)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap.detach().cpu()
    
    def _generate_center_outward_heatmap(self, image_path):
        """Generate a simple center-outward heatmap."""
        # Create a radial gradient from center (highest) to edges (lowest)
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        
        # Resize to standard size
        if width != 224 or height != 224:
            image = transforms.Resize((224, 224))(image)
            width, height = 224, 224
        
        # Generate coordinates
        x = torch.linspace(-1, 1, width)
        y = torch.linspace(-1, 1, height)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # Calculate distance from center
        distance = torch.sqrt(xx.pow(2) + yy.pow(2))
        
        # Create heatmap (1 at center, 0 at edges)
        heatmap = 1 - (distance / distance.max())
        
        # Reshape to [1, 224, 224]
        heatmap = heatmap.unsqueeze(0)
        
        return heatmap
    
    def _generate_edge_heatmap(self, image_path):
        """Generate heatmap based on edge detection."""
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize to standard size
        if image.size[0] != 224 or image.size[1] != 224:
            image = transforms.Resize((224, 224))(image)
        
        # Convert to tensor
        img_tensor = transforms.ToTensor()(image)
        
        # Convert to grayscale
        gray = 0.2989 * img_tensor[0] + 0.5870 * img_tensor[1] + 0.1140 * img_tensor[2]
        gray = gray.unsqueeze(0).unsqueeze(0)
        
        # Define Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], 
                               [-2, 0, 2], 
                               [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        sobel_y = torch.tensor([[-1, -2, -1], 
                               [0, 0, 0], 
                               [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Apply Sobel filters
        edge_x = F.conv2d(gray, sobel_x, padding=1)
        edge_y = F.conv2d(gray, sobel_y, padding=1)
        
        # Compute edge magnitude
        edge_mag = torch.sqrt(edge_x.pow(2) + edge_y.pow(2))
        
        # Normalize
        heatmap = edge_mag / (edge_mag.max() + 1e-8)
        
        return heatmap
    
    def _generate_bing_heatmap(self, image_path):
        """Generate heatmap using BING objectness detection."""
        # Initialize BING saliency detector on demand
        bing_detector = cv2.saliency.ObjectnessBING_create()
        bing_detector.setTrainingPath(self.bing_training_path)
        
        # Load image in BGR via OpenCV (full size, no resizing)
        image_cv = cv2.imread(image_path)
        
        # Compute saliency (returns list of bboxes)
        success, saliencyMap = bing_detector.computeSaliency(image_cv)
        if not success or saliencyMap is None:
            h, w = image_cv.shape[:2]
            return torch.zeros(1, h, w, dtype=torch.float32)
        
        # Build heatmap at original image size
        h, w = image_cv.shape[:2]
        heatmap_np = np.zeros((h, w), dtype=np.float32)
        for i in range(min(self.num_bboxes, saliencyMap.shape[0])):
            x1, y1, x2, y2 = saliencyMap[i].astype(int)
            x1, y1 = np.clip([x1, y1], 0, [w-1, h-1])
            x2, y2 = np.clip([x2, y2], [x1+1, y1+1], [w, h])
            weight = 1.0 - i / self.num_bboxes
            heatmap_np[y1:y2, x1:x2] += weight
        
        # Normalize to [0,1]
        if heatmap_np.max() > 0:
            heatmap_np /= heatmap_np.max()
        
        # Return as [1,H,W] tensor
        return torch.from_numpy(heatmap_np).unsqueeze(0)
    
    def visualize(self, heatmap, original_image_path, save_path=None):
        """
        Visualize the heatmap overlaid on the original image.
        
        Args:
            heatmap (torch.Tensor): Generated heatmap with shape [1, H, W]
            original_image_path (str): Path to the original image
            save_path (str, optional): Path to save the visualization
        """
        # Load original image
        image = Image.open(original_image_path).convert('RGB')
        
        # Resize original image if needed
        if image.size[0] != 224 or image.size[1] != 224:
            image = transforms.Resize((224, 224))(image)
        
        # Convert image to numpy array
        img_np = np.array(image)
        
        # Convert heatmap to numpy array
        heatmap_np = heatmap.squeeze().numpy()
        
        # Create figure
        plt.figure(figsize=(12, 4))
        
        # Plot original image
        plt.subplot(1, 3, 1)
        plt.imshow(img_np)
        plt.title('Original Image')
        plt.axis('off')
        
        # Plot heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap_np, cmap='hot')
        plt.title('Heatmap')
        plt.axis('off')
        plt.colorbar()
        
        # Plot overlay
        plt.subplot(1, 3, 3)
        plt.imshow(img_np)
        plt.imshow(heatmap_np, cmap='hot', alpha=0.6)
        plt.title('Overlay')
        plt.axis('off')
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.tight_layout()
            plt.show()
        plt.close()


def process_dataset(root_dir, output_dir, method='gradient', model_name='resnet18', 
                    device='cuda', visualize=False, limit=None, bing_training_path=None, 
                    num_bboxes=1000, save_as_images=True):
    """
    Generate heatmaps for all images in a dataset directory structure.
    
    Args:
        root_dir (str): Root directory of the dataset
        output_dir (str): Directory to save output heatmaps
        method (str): Method to use for generating heatmaps
        model_name (str): Model name for gradient-based methods
        device (str): Device to use
        visualize (bool): Whether to save visualization images
        limit (int, optional): Limit the number of images to process per class
        bing_training_path (str, optional): Path to BING training data (required for 'bing' method)
        num_bboxes (int): Number of bounding boxes to use for BING heatmap
        save_as_images (bool): Whether to save heatmaps as PNG images instead of tensors
    """
    # Initialize heatmap generator
    generator = HeatmapGenerator(
        method=method, 
        model_name=model_name, 
        device=device,
        bing_training_path=bing_training_path,
        num_bboxes=num_bboxes
    )
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    heatmaps_dir = os.path.join(output_dir, 'heatmaps')
    os.makedirs(heatmaps_dir, exist_ok=True)
    
    if visualize:
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
    
    # Get class directories
    class_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    # Dictionary to store heatmaps
    all_heatmaps = {}
    
    # Process each class
    for class_name in tqdm(class_dirs, desc="Processing classes"):
        class_dir = os.path.join(root_dir, class_name)
        
        # Get image files
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if limit:
            image_files = image_files[:limit]
        
        # Create class output directory
        class_heatmap_dir = os.path.join(heatmaps_dir, class_name)
        os.makedirs(class_heatmap_dir, exist_ok=True)
        
        if visualize:
            class_vis_dir = os.path.join(vis_dir, class_name)
            os.makedirs(class_vis_dir, exist_ok=True)
        
        # Process each image
        for img_file in tqdm(image_files, desc=f"Processing {class_name}", leave=False):
            img_path = os.path.join(class_dir, img_file)
            
            try:
                # Generate heatmap
                heatmap = generator.generate(img_path)
                
                # Save heatmap
                base_filename = os.path.splitext(img_file)[0]
                if save_as_images:
                    # Save as PNG image
                    heatmap_np = (heatmap.squeeze().numpy() * 255).astype(np.uint8)
                    heatmap_path = os.path.join(class_heatmap_dir, f"{base_filename}.png")
                    cv2.imwrite(heatmap_path, heatmap_np)
                else:
                    # Save as tensor
                    heatmap_path = os.path.join(class_heatmap_dir, f"{base_filename}.pt")
                    torch.save(heatmap, heatmap_path)
                
                # Store in dictionary
                all_heatmaps[img_path] = heatmap
                
                # Visualize if requested
                if visualize:
                    vis_path = os.path.join(class_vis_dir, f"{base_filename}_vis.png")
                    generator.visualize(heatmap, img_path, save_path=vis_path)
            
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Save all heatmaps to a single file
    pickle_path = os.path.join(output_dir, f"{method}_heatmaps.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(all_heatmaps, f)
    
    print(f"Processed {len(all_heatmaps)} images. Heatmaps saved to {pickle_path}")
    return all_heatmaps


def convert_heatmaps_to_rankings(heatmaps_dict, output_dir):
    """
    Convert heatmaps to rankings and save as CSV files.
    
    Args:
        heatmaps_dict (dict): Dictionary mapping image paths to heatmap tensors
        output_dir (str): Directory to save output rankings
    """
    from transforms import heatmap_to_ranking
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by class
    class_images = {}
    for img_path, heatmap in heatmaps_dict.items():
        class_name = os.path.basename(os.path.dirname(img_path))
        if class_name not in class_images:
            class_images[class_name] = []
        
        # Convert heatmap to ranking
        ranking = heatmap_to_ranking(heatmap)
        
        # Flatten ranking to 1D tensor
        ranking_flat = ranking.view(-1)
        
        # Store image info
        img_file = os.path.basename(img_path)
        class_images[class_name].append((img_file, ranking_flat))
    
    # Save rankings by class
    for class_name, images in class_images.items():
        output_file = os.path.join(output_dir, f"{class_name}_rankings.csv")
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image', 'ranking'])
            
            for img_file, ranking in images:
                # Convert ranking tensor to comma-separated string
                ranking_str = ','.join(map(str, ranking.tolist()))
                writer.writerow([img_file, ranking_str])
    
    print(f"Rankings saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate heatmaps for image dataset')
    parser.add_argument('--root', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--method', type=str, default='gradient', 
                        choices=['gradient', 'cam', 'center_outward', 'edge_detection', 'bing'],
                        help='Method to generate heatmaps')
    parser.add_argument('--model', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet50'],
                        help='Model to use for gradient-based methods')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--visualize', action='store_true', help='Save visualizations')
    parser.add_argument('--limit', type=int, default=None, help='Limit images per class')
    parser.add_argument('--generate_rankings', action='store_true', help='Generate rankings from heatmaps')
    parser.add_argument('--bing_training_path', type=str, default=None, help='Path to BING training data')
    parser.add_argument('--num_bboxes', type=int, default=1000, help='Number of bounding boxes for BING method')
    args = parser.parse_args()
    
    # Generate heatmaps
    heatmaps = process_dataset(
        root_dir=args.root,
        output_dir=args.output,
        method=args.method,
        model_name=args.model,
        device=args.device,
        visualize=args.visualize,
        limit=args.limit,
        bing_training_path=args.bing_training_path,
        num_bboxes=args.num_bboxes
    )
    
    # Convert to rankings if requested
    if args.generate_rankings:
        rankings_dir = os.path.join(args.output, 'rankings')
        convert_heatmaps_to_rankings(heatmaps, rankings_dir)