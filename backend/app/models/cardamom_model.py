"""
Cardamom disease classification model.
"""
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path


class CardamomClassifier(nn.Module):
    """
    EfficientNetV2-S based classifier for cardamom leaf disease detection.
    """
    
    def __init__(self, num_classes=3):
        super(CardamomClassifier, self).__init__()
        
        # Load EfficientNetV2-S backbone
        self.backbone = models.efficientnet_v2_s(weights=None)
        
        # Replace classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)


def load_model(device: torch.device, model_path: str = "models/cardamom_model.pt") -> nn.Module:
    """
    Load the cardamom classification model.
    
    Args:
        device: PyTorch device (cuda/cpu)
        model_path: Path to model weights
        
    Returns:
        Loaded model
    """
    model = CardamomClassifier(num_classes=3)
    
    # Try to load weights if available
    model_file = Path(model_path)
    if model_file.exists():
        try:
            state_dict = torch.load(model_file, map_location=device)
            model.load_state_dict(state_dict)
            print(f"✓ Loaded trained model weights from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model weights: {e}")
            print("Using randomly initialized weights")
    else:
        print(f"Model file not found at {model_path}")
        print("Using randomly initialized weights (placeholder model)")
    
    model = model.to(device)
    return model