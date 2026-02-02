"""Grad-CAM implementation for visualizing CNN activations."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class GradCAM:
    """Grad-CAM: Gradient-weighted Class Activation Mapping.
    
    Generates visual explanations for CNN predictions by highlighting
    important regions in the input image.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """Initialize Grad-CAM.
        
        Args:
            model: The CNN model to analyze
            target_layer: The convolutional layer to hook into (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Forward hook to save activations."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook to save gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generate Class Activation Map for the target class.
        
        Args:
            input_tensor: Input tensor of shape (1, 3, H, W)
            target_class: Target class index. If None, uses predicted class.
            
        Returns:
            CAM heatmap as numpy array with values in [0, 1]
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # If no target class specified, use the predicted class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Create one-hot encoded target
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        
        # Apply ReLU to focus on positive influence
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def remove_hooks(self):
        """Remove registered hooks."""
        self.forward_handle.remove()
        self.backward_handle.remove()


def generate_gradcam_heatmap(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: nn.Module,
    target_class: Optional[int] = None
) -> np.ndarray:
    """Generate Grad-CAM heatmap for a model prediction.
    
    Args:
        model: The CNN model
        input_tensor: Input tensor of shape (1, 3, H, W)
        target_layer: Target convolutional layer for Grad-CAM
        target_class: Target class index (None for predicted class)
        
    Returns:
        Heatmap as numpy array with values in [0, 1]
    """
    gradcam = GradCAM(model, target_layer)
    try:
        heatmap = gradcam.generate_cam(input_tensor, target_class)
    finally:
        gradcam.remove_hooks()
    
    return heatmap
