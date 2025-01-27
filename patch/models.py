import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchsummary import summary
from torchvision.models import resnet50, ResNet50_Weights
from pathlib import Path

class PatchEmbeddingScorer(nn.Module):
    """Model that extracts features from non-overlapping patches using shared weights.
    
    Each patch is processed independently to preserve spatial locality:
    1. Local Feature Extraction:
       - Splits image into non-overlapping 16x16 patches
       - Each patch is processed independently through CNN layers
       - No information leakage between patches
       - Captures local texture and structure information
       
    2. Score Generation:
       - Each patch produces its own score independently
       - Maintains spatial locality of features
    """
    def __init__(self, patch_size=16, hidden_dim=512, num_patches=14):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        # Enhanced feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()  # Output: [B*P*P, 512]
        )
        
        # Enhanced FC head
        self.score_head = nn.Sequential(
            nn.Linear(512, hidden_dim), # layer 1
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2), # layer 2
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), # layer 3
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1) # Output: [B*P*P, 1]
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        
        patches = x.unfold(2, P, P).unfold(3, P, P)  # [B, C, H//P, W//P, P, P]
        patches = patches.permute(0, 2, 3, 1, 4, 5)   # [B, H//P, W//P, C, P, P]
        patches = patches.reshape(-1, C, P, P)        # [B*H//P*W//P, C, P, P]
        
        # Extract features from all patches in parallel
        features = self.feature_extractor(patches)  # [B*P*P, 512]
        
        # Generate scores
        scores = self.score_head(features)  # [B*P*P, 1]
        
        # Reshape back to [B, num_patches*num_patches]
        scores = scores.view(B, -1)
        
        return scores

class ResNetScorer(nn.Module):
    """Model that uses pretrained ResNet to extract features from patches.
    
    Architecture:
    1. Split image into patches
    2. Process each patch through pretrained ResNet backbone
    3. Generate patch scores through custom head
    """
    def __init__(self, patch_size=16, hidden_dim=512, num_patches=14, freeze_backbone=True):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        # Load pretrained ResNet
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove final layers
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Calculate feature dimension
        self.feature_dim = 2048  # ResNet50's final conv layer channels
        
        # Custom scoring head
        self.score_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._initialize_head()
    
    def _initialize_head(self):
        """Initialize the custom head layers"""
        for m in self.score_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        
        # Split into patches
        patches = x.unfold(2, P, P).unfold(3, P, P)  # [B, C, H//P, W//P, P, P]
        patches = patches.permute(0, 2, 3, 1, 4, 5)   # [B, H//P, W//P, C, P, P]
        patches = patches.reshape(-1, C, P, P)        # [B*H//P*W//P, C, P, P]
        
        # Extract features using ResNet backbone
        features = self.backbone(patches)  # [B*P*P, 2048, H', W']
        
        # Generate scores
        scores = self.score_head(features)  # [B*P*P, 1]
        
        # Reshape back to [B, num_patches*num_patches]
        scores = scores.view(B, -1)
        
        return scores

class GlobalResNetScorer(nn.Module):
    """Model that processes full image and directly generates all patch scores.
    
    Architecture:
    1. Process full image through ResNet backbone
    2. Global average pooling of features
    3. Direct prediction of all patch scores through MLP
    """
    def __init__(self, patch_size=16, hidden_dim=512, num_patches=14, freeze_backbone=True):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_total_patches = num_patches * num_patches  # e.g., 14*14 = 196
        
        # Load pretrained ResNet but remove final layers
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Global feature processing
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        # Scoring head that directly predicts all patch scores
        self.score_head = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, self.num_total_patches)  # Direct prediction of all patch scores
        )
        
        self._initialize_head()
    
    def _initialize_head(self):
        """Initialize the scoring head layers"""
        for m in self.score_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Process full image through backbone
        features = self.backbone(x)  # [B, 2048, H/32, W/32]
        
        # Global pooling and flatten
        features = self.global_pool(features)  # [B, 2048, 1, 1]
        features = self.flatten(features)      # [B, 2048]
        
        # Generate all patch scores directly
        scores = self.score_head(features)     # [B, num_patches*num_patches]
        
        return scores

def get_model(model_name='patch', **kwargs):
    """Factory function for creating models with default configurations"""
    models = {
        'patch': PatchEmbeddingScorer,
        'resnet': ResNetScorer,
        'global_resnet': GlobalResNetScorer
    }
    
    # Default optimizer configurations per model
    default_optimizers = {
        'patch': {
            'name': 'sgd',
            'lr': 0.001,
            'momentum': 0.9,
            'weight_decay': 1e-4
        },
        'resnet': {
            'name': 'sgd',
            'lr': 0.001,
            'momentum': 0.9,
            'weight_decay': 1e-4
        },
        'global_resnet': {
            'name': 'adamw',
            'lr': 0.0001,
            'weight_decay': 0.01,
            'betas': (0.9, 0.999)
        }
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")
    
    model = models[model_name](**kwargs)
    model.default_optimizer = default_optimizers[model_name]
    
    return model

def find_latest_checkpoint(base_dir, model_name=None):
    """Find the latest checkpoint in a directory with date-time subdirectories.
    
    Args:
        base_dir: Base directory containing date-time subdirectories
        model_name: Optional model name to include in search pattern
    
    Returns:
        Path: Path to the latest checkpoint
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    
    # Find all date-time directories
    dt_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not dt_dirs:
        raise FileNotFoundError(f"No subdirectories found in {base_dir}")
    
    # Sort by directory name (datetime format ensures chronological order)
    dt_dirs.sort()
    latest_dir = dt_dirs[-1]
    
    # Look for best_model.pth in the latest directory
    checkpoint_path = latest_dir / 'best_model.pth'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found in latest directory: {latest_dir}")
    
    return checkpoint_path

def load_checkpoint(checkpoint_path, model_name='patch', **kwargs):
    """Load a model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file or directory containing date-time subdirs
        model_name: Type of model to load ('patch', 'resnet', or 'global_resnet')
        **kwargs: Additional arguments to pass to model constructor
    
    Returns:
        model: Loaded model with weights
        checkpoint: Dictionary containing checkpoint info
    """
    if not isinstance(checkpoint_path, (str, Path)):
        raise ValueError("checkpoint_path must be a string or Path object")
    
    checkpoint_path = Path(checkpoint_path)
    
    # If checkpoint_path is a directory, find the latest checkpoint
    if checkpoint_path.is_dir():
        try:
            checkpoint_path = find_latest_checkpoint(checkpoint_path, model_name)
            print(f"Using latest checkpoint: {checkpoint_path}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find checkpoint in directory: {e}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Initialize model
    model = get_model(model_name, **kwargs)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if it exists (from DDP training)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load weights
    try:
        model.load_state_dict(state_dict)
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        print(f"Warning: Error loading checkpoint: {e}")
        print("Attempting to load with strict=False...")
        model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded checkpoint with some missing or unexpected keys")
    
    return model, checkpoint

if __name__ == "__main__":
    from training_loop import TrainingConfig, train_model
    from dataloader import get_patch_rank_loader
    from torch.utils.data import random_split
    from pathlib import Path
    
    def test_training(model_name='patch'):
        """Test if model can learn patch scoring using real data"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Setup data
        data_root = Path("C:/Users/joren/Documents/_Uni/Master/Thesis/imagenet_subset")
        
        # First test data loading
        print("\nTesting data loading...")
        test_loader = get_patch_rank_loader(data_root, split='train', batch_size=2, num_workers=0)
        
        # Split dataset
        full_dataset = test_loader.dataset
        num_total = len(full_dataset)
        num_train = int(0.8 * num_total)
        num_val = num_total - num_train
        
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [num_train, num_val],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders with smaller batch size for testing
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=8,
            pin_memory=True,
            num_workers=4
        )
        
        print(f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        # Initialize model
        model = get_model(model_name, patch_size=16, hidden_dim=512, num_patches=14).to(device)
        
        # Use model's default optimizer settings
        optimizer_config = getattr(model, 'default_optimizer', {
            'name': 'sgd',
            'lr': 0.0001,
            'momentum': 0.9,
            'weight_decay': 1e-4
        })
        
        # Training configuration using model-specific optimizer
        config = TrainingConfig(
            verbose=2,
            save_dir=f'test_runs/{model_name}_test',
            plot_every=1,
            save_best=True,
            num_epochs=2,
            batch_size=8,
            optimizer=optimizer_config,
            loss={
                'alpha': 1.0,
                'beta': 0.0
            },
            scheduler=None
        )
        
        # Train model
        print(f"\nStarting training with {optimizer_config['name'].upper()}...")
        tracker = train_model(model, train_loader, val_loader, config)
        
        return model, tracker
    
    for model_name in ['patch', 'resnet', 'global_resnet']:
        print(f"\nTesting {model_name} model...")
        model = get_model(model_name)
        summary(model, (3, 224, 224))

    # test_training('global_resnet')
    
    def test_loading():
        """Test loading models from checkpoints"""
        print("\nTesting model loading...")
        
        # First train and save a model
        model, tracker = test_training('global_resnet')
        
        # Try loading from the runs directory
        try:
            loaded_model, checkpoint = load_checkpoint(
                'test_runs\global_resnet_test',  # Will find latest checkpoint automatically
                model_name='global_resnet',
                patch_size=16,
                hidden_dim=512,
                num_patches=14
            )
            
            # Verify the loaded model
            loaded_model.eval()
            model.eval()
            
            # Compare original and loaded model outputs
            x = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                y1 = model(x)
                y2 = loaded_model(x)
            
            print(f"Max difference in outputs: {(y1 - y2).abs().max().item()}")
            assert torch.allclose(y1, y2, atol=1e-6), "Loaded model produces different outputs!"
            print("Model loading test passed!")
            
        except FileNotFoundError as e:
            print(f"Test loading failed: {e}")
    
    # Run loading test
    test_loading()

