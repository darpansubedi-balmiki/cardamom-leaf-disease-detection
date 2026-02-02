"""Image preprocessing utilities for model input."""
import torch
from PIL import Image
from torchvision import transforms
from typing import Tuple


# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_preprocessing_transform(image_size: int = 224) -> transforms.Compose:
    """Get the preprocessing transform for model input.
    
    Args:
        image_size: Target image size (default: 224)
        
    Returns:
        Composed transform that resizes and normalizes images
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def preprocess_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Preprocess a PIL Image for model inference.
    
    Args:
        image: PIL Image to preprocess
        image_size: Target size for the image (default: 224)
        
    Returns:
        Preprocessed tensor of shape (1, 3, image_size, image_size)
    """
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply preprocessing
    transform = get_preprocessing_transform(image_size)
    tensor = transform(image)
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize an ImageNet-normalized tensor back to [0, 1] range.
    
    Args:
        tensor: Normalized tensor
        
    Returns:
        Denormalized tensor in [0, 1] range
    """
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    
    # Move to same device as input tensor
    mean = mean.to(tensor.device)
    std = std.to(tensor.device)
    
    return tensor * std + mean
