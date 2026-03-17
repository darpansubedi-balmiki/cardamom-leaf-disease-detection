"""
Placeholder U2-Net segmentation model for background removal.

The FastAPI app expects a U2NetSegmenter class with:
- __init__(model_path: Optional[str] = None, device: Optional[str] = None)
- remove_background(image: PIL.Image.Image) -> PIL.Image.Image

This placeholder keeps API startup working without real U2-Net weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PIL import Image


@dataclass
class U2NetSegmenter:
    model_path: Optional[str] = None
    device: Optional[str] = None
    _model: object | None = None

    def __post_init__(self) -> None:
        # Placeholder: no real model is loaded.
        print("Loading U2-Net model...")
        print("U2-Net background removal is not implemented yet (placeholder)")
        self._model = None
        print("U2-Net model loaded (placeholder)")

    def remove_background(self, image: Image.Image) -> Image.Image:
        # Placeholder behavior: return image unchanged.
        return image