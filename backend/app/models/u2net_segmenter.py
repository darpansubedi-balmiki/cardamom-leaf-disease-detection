"""
Background removal using rembg library.
Falls back to passthrough if rembg is not installed.
"""
from __future__ import annotations

import logging
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

try:
    from rembg import remove as rembg_remove
    _REMBG_AVAILABLE = True
except (ImportError, SystemExit, Exception):
    _REMBG_AVAILABLE = False
    logger.warning("rembg not installed or not usable – background removal disabled.")


class U2NetSegmenter:
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None) -> None:
        self.model_path = model_path
        self.device = device
        if _REMBG_AVAILABLE:
            logger.info("✓ Background removal ready (rembg)")
        else:
            logger.warning("⚠️ Background removal disabled (rembg not installed)")

    def remove_background(self, image: Image.Image) -> Image.Image:
        if not _REMBG_AVAILABLE:
            return image
        try:
            result = rembg_remove(image.convert("RGB"))
            if result.mode == "RGBA":
                black_bg = Image.new("RGB", result.size, (0, 0, 0))
                black_bg.paste(result, mask=result.split()[3])
                return black_bg
            return result
        except Exception as exc:
            logger.warning("Background removal failed: %s – returning original image", exc)
            return image