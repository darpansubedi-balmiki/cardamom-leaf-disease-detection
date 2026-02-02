"""Placeholder CNN model for cardamom leaf disease classification."""
import torch
import torch.nn as nn
import os
from typing import Optional


class CardamomClassifier(nn.Module):
    """Simple CNN classifier with 4 conv blocks for 3-class classification.
    
    Classes:
        0: Colletotrichum Blight
        1: Phyllosticta Leaf Spot
        2: Healthy
    """
    
    def __init__(self):
        super(CardamomClassifier, self).__init__()
        
        # Block 1: 3 -> 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 224 -> 112
        )
        
        # Block 2: 32 -> 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 112 -> 56
        )
        
        # Block 3: 64 -> 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 56 -> 28
        )
        
        # Block 4: 128 -> 256 (last conv layer for Grad-CAM)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 28 -> 14
        )
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 3)  # 3 classes
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Logits of shape (batch_size, 3)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_model(model_path: Optional[str] = None, device: Optional[str] = None) -> CardamomClassifier:
    """Load the cardamom classifier model.
    
    Args:
        model_path: Path to saved model weights. If None, uses default path.
        device: Device to load model on ('cuda' or 'cpu'). If None, auto-detects.
        
    Returns:
        Loaded model on the specified device.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_path is None:
        # Default path relative to backend directory
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'cardamom_model.pt')
    
    model = CardamomClassifier()
    
    # Try to load trained weights if they exist
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print("Successfully loaded trained weights")
        except Exception as e:
            print(f"Warning: Could not load trained weights: {e}")
            print("Using random initialization instead")
    else:
        print(f"No trained model found at {model_path}")
        print("Using random initialization (placeholder model)")
    
    model = model.to(device)
    model.eval()
    
    return model


# Class names for prediction output
CLASS_NAMES = [
    "Colletotrichum Blight",
    "Phyllosticta Leaf Spot",
    "Healthy"
]
