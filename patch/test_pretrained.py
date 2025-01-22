import torch
import torchvision.models as models
from patch.dataloader import get_patch_rank_loader
from pathlib import Path
import torch.nn as nn
from models import ResNetPatchScorer, BagNetPatchScorer
import torch.utils.model_zoo as model_zoo
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import urllib.request

class PretrainedModelTester:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.bagnet_url = "https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet32-2ddd53ed.pth.tar"
        
        # Add ImageNet class names for better readability
        self.class_names = {
            'dog': range(151, 269),  # ImageNet dog class range
            'cat': range(281, 286),  # ImageNet cat class range
            # Add more class ranges as needed
        }
    
    def load_original_bagnet(self):
        """Load original BagNet model from local disk or download if missing."""
        local_dir = Path.home() / ".cache" / "bagnet"
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / "bagnet32-2ddd53ed.pth.tar"

        # Check if local file exists, else download
        if not local_path.exists():
            print("[INFO] Downloading BagNet weights...")
            urllib.request.urlretrieve(self.bagnet_url, local_path)

        # Now load from the local file
        state_dict = torch.load(local_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # Create a temporary model with BagNet architecture
        model = torch.nn.Sequential(
            # Initial conv layer
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Main layers (matching BagNet33 architecture)
            *[nn.Sequential(
                nn.Conv2d(64 if i == 0 else 64 * (2 ** (i-1)), 
                         64 * (2 ** i), 
                         kernel_size=3, stride=2 if i > 0 else 1, 
                         padding=1, bias=False),
                nn.BatchNorm2d(64 * (2 ** i)),
                nn.ReLU(inplace=True)
            ) for i in range(4)],
            
            # Classification head
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1000)
        )
        
        # Load compatible weights
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() 
                          if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        # Log missing/unexpected keys for debugging
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("[DEBUG] Missing keys:", missing)
        print("[DEBUG] Unexpected keys:", unexpected)
        
        return model
    
    def create_models(self):
        models_dict = {}
        
        # Original ResNet50
        resnet_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        resnet_ckpt = resnet_dir / "resnet50-0676ba61.pth"

        if not resnet_ckpt.exists():
            print("[INFO] ResNet50 weights not found locally; PyTorch will download automatically.")
        models_dict['resnet_original'] = models.resnet50(pretrained=True)
        
        # Original BagNet
        try:
            models_dict['bagnet_original'] = self.load_original_bagnet()
            print("Successfully loaded original BagNet")
        except Exception as e:
            print(f"Failed to load original BagNet: {e}")
        
        # Our modified models
        models_dict['resnet_ours'] = ResNetPatchScorer(pretrained=True, num_patches=14)
        models_dict['bagnet_ours'] = BagNetPatchScorer(pretrained=True, num_patches=14)
        
        # If available, load custom checkpoints for 'resnet_ours' or 'bagnet_ours'
        # and debug any missing/unexpected keys
        try:
            custom_checkpoint_path = Path("path/to/resnet_ours_checkpoint.pth")
            if custom_checkpoint_path.exists():
                custom_state = torch.load(custom_checkpoint_path, map_location=self.device)
                missing, unexpected = models_dict['resnet_ours'].load_state_dict(
                    custom_state['model_state_dict'], strict=False
                )
                print("[DEBUG] ResNet Ours - Missing:", missing)
                print("[DEBUG] ResNet Ours - Unexpected:", unexpected)
        except Exception as e:
            print("[DEBUG] Could not load resnet_ours checkpoint:", e)

        try:
            custom_checkpoint_path = Path("path/to/bagnet_ours_checkpoint.pth")
            if custom_checkpoint_path.exists():
                custom_state = torch.load(custom_checkpoint_path, map_location=self.device)
                missing, unexpected = models_dict['bagnet_ours'].load_state_dict(
                    custom_state['model_state_dict'], strict=False
                )
                print("[DEBUG] BagNet Ours - Missing:", missing)
                print("[DEBUG] BagNet Ours - Unexpected:", unexpected)
        except Exception as e:
            print("[DEBUG] Could not load bagnet_ours checkpoint:", e)
        
        # Move all models to device and eval mode
        for name, model in models_dict.items():
            models_dict[name] = model.eval().to(self.device)
        
        return models_dict
    
    def show_image(self, image_tensor):
        """Convert normalized tensor to displayable image"""
        # Clone the tensor and move to CPU
        img = image_tensor.cpu().clone()
        
        # Remove batch dimension if present
        if img.dim() == 4:
            img = img[0]
            
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        img = img * std + mean
        
        # Clamp values to valid range
        img = torch.clamp(img, 0, 1)
        
        # Convert to PIL Image for display
        img = transforms.ToPILImage()(img)
        return img
    
    def test_models(self, data_loader, num_samples=10):
        models = self.create_models()
        results = {name: [] for name in models.keys()}
        
        print("\nTesting model predictions:")
        print("-" * 80)
        
        with torch.no_grad():
            for i, (images, _, _) in enumerate(data_loader):
                if i >= num_samples:
                    break
                    
                images = images.to(self.device)
                
                # Create figure for this image
                fig = plt.figure(figsize=(15, 5))
                
                # Show original image
                ax1 = plt.subplot(1, 2, 1)
                img_show = self.show_image(images[0])
                ax1.imshow(img_show)
                ax1.axis('off')
                ax1.set_title(f'Image {i+1}')
                
                # Show predictions
                ax2 = plt.subplot(1, 2, 2)
                ax2.axis('off')
                predictions_text = []
                
                # Test each model
                for name, model in models.items():
                    outputs = model(images)
                    
                    # Add original classification head back for testing
                    if name == 'resnet_ours':
                        outputs = nn.Linear(2048, 1000).to(self.device)(
                            nn.AdaptiveAvgPool2d(1)(model.backbone(images)).squeeze(-1).squeeze(-1)
                        )
                    elif name == 'bagnet_ours':
                        outputs = nn.Linear(512, 1000).to(self.device)(
                            nn.AdaptiveAvgPool2d(1)(model.features(images)).squeeze(-1).squeeze(-1)
                        )
                    
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    _, predicted = outputs.max(1)
                    pred_class = predicted.item()
                    
                    # Determine class type
                    class_type = 'unknown'
                    prob_value = probs[0, pred_class].item()
                    
                    for class_name, class_range in self.class_names.items():
                        if pred_class in class_range:
                            class_type = class_name
                            break
                    
                    # Store results
                    results[name].append({
                        'class_id': pred_class,
                        'class_type': class_type,
                        'probability': prob_value
                    })
                    
                    # Add prediction text
                    predictions_text.append(
                        f"{name}:\nClass {pred_class} ({class_type})\n"
                        f"Confidence: {prob_value:.2%}\n"
                    )
                
                # Show predictions
                ax2.text(0.05, 0.95, '\n'.join(predictions_text),
                        transform=ax2.transAxes,
                        verticalalignment='top',
                        fontfamily='monospace')
                ax2.set_title('Model Predictions')
                
                plt.tight_layout()
                plt.show()
                
                print(f"\nImage {i+1} Predictions:")
                print("-" * 40)
                print('\n'.join(predictions_text))
        
        # Print summary
        print("\nOverall Results:")
        print("-" * 80)
        for name in results:
            correct_dogs = sum(1 for r in results[name] if r['class_type'] == 'dog')
            print(f"{name:20s}: {correct_dogs}/{num_samples} dog classifications")
            print(f"{'':20s}  Average confidence: "
                  f"{sum(r['probability'] for r in results[name])/len(results[name]):.2%}")


if __name__ == "__main__":
    data_root = Path("C:/Users/joren/Documents/_Uni/Master/Thesis/imagenet_subset")
    loader = get_patch_rank_loader(data_root, split='train', batch_size=1)
    
    tester = PretrainedModelTester()
    tester.test_models(loader, num_samples=10)
