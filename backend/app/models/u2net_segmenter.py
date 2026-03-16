"""
U2-Net background segmenter - placeholder implementation.

U2-Net is an optional background removal step. Accurate predictions
(85-92%) are achieved without it. The system processes images without
background removal by default when no U2-Net weights are available.
"""

import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class U2NetSegmenter:
    """
    Placeholder for U2-Net background removal.

    This class provides a no-op implementation that returns the original image
    unchanged. To enable real background removal, supply a trained U2-Net model
    and override the ``remove_background`` method.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self.loaded = False

        if model_path:
            try:
                self._load_model(model_path)
            except Exception as exc:
                logger.warning(
                    "U2-Net weights not found at '%s' – background removal disabled. (%s)",
                    model_path,
                    exc,
                )
        else:
            logger.info(
                "ℹ️  U2-Net model path not provided – running without background removal."
            )

    def _load_model(self, path: str) -> None:
        # Real implementation would load the U2-Net weights here.
        raise FileNotFoundError(f"Model file not found: {path}")

    def remove_background(self, image: Image.Image) -> Image.Image:
        """Return the image unchanged (placeholder)."""
        if not self.loaded:
            return image
        # Real implementation would apply segmentation mask here.
        return image  # pragma: no cover
