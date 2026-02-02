"""Utilities for overlaying heatmaps on images."""
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from typing import Tuple


def overlay_heatmap_on_image(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> bytes:
    """Overlay a heatmap on an image and return as PNG bytes.
    
    Args:
        image: Original PIL Image
        heatmap: Heatmap as numpy array with values in [0, 1]
        alpha: Blending factor for overlay (0.0 to 1.0)
        colormap: OpenCV colormap to use (default: COLORMAP_JET)
        
    Returns:
        PNG image bytes with heatmap overlay
    """
    # Convert PIL image to numpy array (RGB)
    img_array = np.array(image.convert('RGB'))
    img_height, img_width = img_array.shape[:2]
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (img_width, img_height))
    
    # Convert heatmap to uint8 [0, 255]
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Convert from BGR to RGB (OpenCV uses BGR)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend image and heatmap
    overlayed = cv2.addWeighted(img_array, 1 - alpha, heatmap_colored, alpha, 0)
    
    # Convert back to PIL Image
    overlayed_pil = Image.fromarray(overlayed)
    
    # Convert to PNG bytes
    buffer = BytesIO()
    overlayed_pil.save(buffer, format='PNG')
    png_bytes = buffer.getvalue()
    
    return png_bytes


def encode_image_to_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string.
    
    Args:
        image_bytes: Image bytes (e.g., PNG)
        
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(image_bytes).decode('utf-8')


def overlay_and_encode(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4
) -> str:
    """Overlay heatmap on image and return as base64 encoded PNG.
    
    Args:
        image: Original PIL Image
        heatmap: Heatmap as numpy array with values in [0, 1]
        alpha: Blending factor for overlay
        
    Returns:
        Base64 encoded PNG string
    """
    png_bytes = overlay_heatmap_on_image(image, heatmap, alpha)
    return encode_image_to_base64(png_bytes)
