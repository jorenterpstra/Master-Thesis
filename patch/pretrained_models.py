import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from utils import compare_model_states

def load_partial_weights(model, state_dict, prefix=''):
    """Enhanced weight loading with key remapping and better error handling"""
    model_keys = model.state_dict()
    filtered = {}
    
    # Try different key mappings
    for k, v in state_dict.items():
        # Remove prefix if exists
        if k.startswith(prefix):
            k = k[len(prefix):]
            
        # Try different key variations
        possible_keys = [
            k,
            k.replace('module.', ''),
            'module.' + k,
            k.split('.')[-1]
        ]
        
        for possible_key in possible_keys:
            if possible_key in model_keys:
                if v.shape == model_keys[possible_key].shape:
                    filtered[possible_key] = v
                    break
                else:
                    print(f"[WARNING] Shape mismatch for {k}: {v.shape} vs {model_keys[possible_key].shape}")

    if not filtered:
        raise ValueError("No compatible weights found!")

    # Load weights
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[INFO] Loaded {len(filtered)} weights:")
    print(f" - Missing keys: {len(missing)}")
    print(f" - Unexpected keys: {len(unexpected)}")
    return len(filtered) > 0

def load_weights_by_shape(model, state_dict):
    """Load weights by matching shapes, ignoring names"""
    model_state = model.state_dict()
    
    # Get all weights from state dict that are for conv/bn layers
    src_weights = [(k, v) for k, v in state_dict.items() 
                  if any(t in k for t in ['conv', 'bn', 'downsample'])]
    
    # Get all target parameter names that need weights
    dst_weights = [(k, v) for k, v in model_state.items() 
                  if any(t in k for t in ['conv', 'bn', 'downsample'])]
    
    # Match weights by shape
    matched = {}
    src_used = set()
    
    for dst_name, dst_param in dst_weights:
        # Find first unused source weight with matching shape
        for src_name, src_param in src_weights:
            if src_name not in src_used and src_param.shape == dst_param.shape:
                matched[dst_name] = src_param
                src_used.add(src_name)
                print(f"Matched {src_name} -> {dst_name}")
                break
    
    # Load matched weights
    model.load_state_dict(matched, strict=False)
    print(f"[INFO] Loaded {len(matched)}/{len(dst_weights)} weight tensors by shape matching")
    return len(matched) > 0

class ResNetPatchScorer(nn.Module):
    """Modified ResNet-50 with FC head for patch score prediction"""
    def __init__(self, pretrained=True, num_patches=14):
        super().__init__()
        # Load pretrained ResNet-50 backbone
        resnet = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avg pool and fc
        
        # Freeze backbone weights
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Calculate input features for FC layers
        self.feature_dims = 2048 * 7 * 7  # ResNet outputs 2048 channels at 7x7 spatial dims
        
        # FC prediction head
        self.score_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dims, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_patches * num_patches)  # Output one score per patch
        )
        
        # Initialize FC layers
        for m in self.score_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if pretrained:
            try:
                # Load official ResNet50 weights
                state_dict = models.resnet50(pretrained=True).state_dict()
                success = load_partial_weights(self.backbone, state_dict)
                if success:
                    print("[INFO] Successfully loaded ResNet50 weights")
                    # Freeze backbone weights
                    for param in self.backbone.parameters():
                        param.requires_grad = False
            except Exception as e:
                print(f"[ERROR] Failed to load ResNet weights: {e}")
                print("[DEBUG] Using random initialization")

    def forward(self, x):
        # Extract features using frozen backbone
        with torch.set_grad_enabled(not self.training):
            features = self.backbone(x)  # [B, 2048, 7, 7]
        
        # Generate patch scores using FC head
        scores = self.score_head(features)  # [B, num_patches²]
        
        return scores
        
    def unfreeze_backbone(self, from_layer=None):
        """Optionally unfreeze backbone layers for fine-tuning"""
        if from_layer is None:
            # Unfreeze all backbone layers
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Unfreeze layers from specified point
            layers = list(self.backbone.children())
            for layer in layers[from_layer:]:
                for param in layer.parameters():
                    param.requires_grad = True

class BagNetPatchScorer(nn.Module):
    """BagNet-style architecture using pretrained BagNet33 weights"""
    def __init__(self, pretrained=True, num_patches=14):
        super().__init__()
        self.num_patches = num_patches
        self.inplanes = 64
        
        # Feature extraction layers matching BagNet33
        self.features = nn.Sequential(
            # Initial conv with 3x3 kernel (reduced from 7x7 in ResNet)
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Main layers
            self._make_layer(64, 64, 3, stride=1),    # Layer 1
            self._make_layer(64, 128, 4, stride=2),   # Layer 2
            self._make_layer(128, 256, 6, stride=2),  # Layer 3
            self._make_layer(256, 512, 3, stride=2),  # Layer 4
        )
        
        # Scoring head
        self.score_head = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1)
        )

        if pretrained:
            try:
                # URL for pretrained BagNet33 weights
                url = "https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet32-2ddd53ed.pth.tar"
                
                state_dict = model_zoo.load_url(url, map_location='cpu')
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                # Try loading weights by shape
                success = load_weights_by_shape(self.features, state_dict)
                
                if success:
                    print("[INFO] Successfully loaded BagNet weights by shape matching")
                    # Freeze pretrained layers
                    for param in self.features.parameters():
                        param.requires_grad = False
                else:
                    print("[WARNING] Failed to load weights by shape matching")
                    
            except Exception as e:
                print(f"[ERROR] Failed to load BagNet weights: {e}")
                print("[DEBUG] Using random initialization")
        
        # Initialize scoring head
        self._initialize_head()

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        layers = []
        
        # First block might have stride
        layers.append(nn.Conv2d(inplanes, planes, kernel_size=3, 
                              stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(planes, planes, kernel_size=3,
                                  stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)

    def _initialize_head(self):
        """Initialize scoring head weights"""
        for m in self.score_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def unfreeze_features(self, from_layer=None):
        """Optionally unfreeze feature layers"""
        if from_layer is None:
            # Unfreeze all feature layers
            for param in self.features.parameters():
                param.requires_grad = True
        else:
            # Unfreeze from specific layer onwards
            layers = list(self.features.children())
            for layer in layers[from_layer:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, x):
        # Extract features using BagNet backbone
        features = self.features(x)
        
        # Generate patch scores
        scores = self.score_head(features)
        
        # Interpolate to desired patch grid size if needed
        if scores.shape[-1] != self.num_patches:
            scores = F.interpolate(scores, size=(self.num_patches, self.num_patches), 
                                 mode='bilinear', align_corners=False)
        
        # Return flattened scores
        return scores.reshape(x.size(0), -1)

if __name__ == "__main__":
    # Test models
    x = torch.randn(5, 3, 224, 224)
    
    print("Testing ResNetPatchScorer...")
    model = ResNetPatchScorer(pretrained=False)
    out = model(x)
    print(f"Output shape: {out.shape}")
    
    print("\nTesting BagNetPatchScorer...")
    model = BagNetPatchScorer()
    out = model(x)
    print(f"Output shape: {out.shape}")
