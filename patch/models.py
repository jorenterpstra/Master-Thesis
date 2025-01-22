import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

class SimplePatchScorer(nn.Module):
    """Simplified model for debugging patch ordering"""
    def __init__(self, patch_size=16, num_patches=14):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        # Simple linear transformation without non-linearities
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),  # Flatten the 16x16x3 patch
            nn.Linear(patch_size * patch_size * 3, 1)  # Direct mapping to score
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        
        patches = x.unfold(2, P, P).unfold(3, P, P)
        patches = patches.permute(0, 4, 5, 1, 2, 3).contiguous()
        patches = patches.view(-1, C, P, P)
        
        # Process each patch
        scores = self.feature_extractor(patches)
        return scores.view(B, -1)

class SimpleUnsharedPatchScorer(nn.Module):
    """Simplified unshared model for debugging patch ordering"""
    def __init__(self, patch_size=16, image_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = image_size // patch_size
        self.total_patches = self.num_patches * self.num_patches
        
        # Create separate linear layers for each patch
        self.extractors = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(patch_size * patch_size * 3, 1)
            ) for _ in range(self.total_patches)
        ])
    
    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        
        patches = x.unfold(2, P, P).unfold(3, P, P)
        patches = patches.permute(0, 4, 5, 1, 2, 3).contiguous()
        patches = patches.view(B, self.total_patches, C, P, P)
        
        # Process patches
        scores = []
        for i in range(self.total_patches):
            patch = patches[:, i]  # [B, C, P, P]
            score = self.extractors[i](patch)
            scores.append(score)
        
        return torch.cat(scores, dim=1)

def get_model(model_name='patch', **kwargs):
    """Factory function for creating models"""
    models = {
        'patch': PatchEmbeddingScorer,
        'simple': SimplePatchScorer
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")
    return models[model_name](**kwargs)

if __name__ == "__main__":
    from training_loop import TrainingConfig, train_model
    from dataloader import get_patch_rank_loader
    from torch.utils.data import random_split
    from pathlib import Path
    
    def test_training():
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
        model = PatchEmbeddingScorer().to(device)
        
        # Training configuration with SGD only
        config = TrainingConfig(
            verbose=2,
            save_dir='test_runs/sgd_test',
            plot_every=1,
            save_best=True,
            num_epochs=20,  
            batch_size=8,
            optimizer={
                'name': 'sgd',
                'lr': 0.0001,
                'momentum': 0.9,
                'weight_decay': 1e-4
            },
            loss={
                'alpha': 1.0,
                'beta': 0.0
            },
            scheduler=None 
        )
        
        # Train model
        print("\nStarting training with SGD...")
        tracker = train_model(model, train_loader, val_loader, config)
        
        return model, tracker
    
    # Run test
    print("Testing model training with SGD optimizer...")
    trained_model, tracker = test_training()


