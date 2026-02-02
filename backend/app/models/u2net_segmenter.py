"""Placeholder U2-Net segmentation model for background removal."""
from PIL import Image
from typing import Optional


def load_u2net(model_path: Optional[str] = None, device: Optional[str] = None) -> None:
    """Load U2-Net model for background removal.
    
    TODO: Implement actual U2-Net loading when trained model is available.
    For now, this is a placeholder that returns None.
    
    Args:
        model_path: Path to U2-Net weights
        device: Device to load model on ('cuda' or 'cpu')
        
    Returns:
        None (placeholder)
    """
    print("U2-Net background removal is not implemented yet (placeholder)")
    return None


def apply_background_removal(image: Image.Image, model=None) -> Image.Image:
    """Apply background removal to an image using U2-Net.
    
    TODO: Implement actual background removal when U2-Net model is available.
    For now, this just returns the original image unchanged.
    
    Args:
        image: PIL Image to process
        model: U2-Net model instance (unused in placeholder)
        
    Returns:
        Original image unchanged (placeholder behavior)
    """
    # TODO: Implement U2-Net-based background removal
    # For now, just return the original image
    return image
