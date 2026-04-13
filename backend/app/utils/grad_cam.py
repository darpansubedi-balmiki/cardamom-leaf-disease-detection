"""Grad-CAM and Grad-CAM++ implementations for visualizing CNN activations."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


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


class GradCAMPlusPlus:
    """Grad-CAM++: Improved Gradient-weighted Class Activation Mapping.

    Produces sharper, more accurate localization than standard Grad-CAM by
    weighting each gradient value with a second-order Taylor-series coefficient,
    reducing the influence of diffuse background gradients.

    Reference: Chattopadhay et al., "Grad-CAM++: Improved Visual Explanations
    for Deep Convolutional Networks", WACV 2018.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        self._fwd = target_layer.register_forward_hook(self._save_activation)
        self._bwd = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Grad-CAM++ heatmap.

        Args:
            input_tensor: Input tensor of shape (1, 3, H, W).
            target_class: Target class index. Defaults to predicted class.

        Returns:
            Heatmap numpy array with values in [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        grads = self.gradients   # (1, C, H, W)
        acts = self.activations  # (1, C, H, W)

        # Grad-CAM++ alpha weights
        grads_sq = grads ** 2
        grads_cu = grads ** 3
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)          # (1, C, 1, 1)
        alpha_denom = 2 * grads_sq + grads_cu * sum_acts       # (1, C, H, W)
        alpha_denom = torch.where(
            alpha_denom != 0,
            alpha_denom,
            torch.ones_like(alpha_denom),
        )
        alpha = grads_sq / alpha_denom                         # (1, C, H, W)

        weights = (alpha * F.relu(grads)).mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        cam = (weights * acts).sum(dim=1, keepdim=True)        # (1, 1, H, W)
        cam = F.relu(cam)

        cam_np: np.ndarray = cam.squeeze().cpu().numpy()
        cam_np = cam_np - cam_np.min()
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()

        return cam_np

    def remove_hooks(self) -> None:
        self._fwd.remove()
        self._bwd.remove()


def generate_gradcam_heatmap(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: nn.Module,
    target_class: Optional[int] = None,
    method: str = "gradcam",
) -> np.ndarray:
    """Generate a Grad-CAM or Grad-CAM++ heatmap for a model prediction.

    Args:
        model: The CNN model.
        input_tensor: Input tensor of shape (1, 3, H, W).
        target_layer: Target convolutional layer.
        target_class: Target class index (None for predicted class).
        method: ``"gradcam"`` (default) or ``"gradcam++"`` for sharper maps.

    Returns:
        Heatmap as numpy array with values in [0, 1].
    """
    if method == "gradcam++":
        cam_obj: GradCAMPlusPlus | GradCAM = GradCAMPlusPlus(model, target_layer)
    else:
        cam_obj = GradCAM(model, target_layer)

    try:
        heatmap = cam_obj.generate_cam(input_tensor, target_class)
    finally:
        cam_obj.remove_hooks()

    return heatmap

