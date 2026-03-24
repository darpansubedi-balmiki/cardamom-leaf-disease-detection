"""
U2-Net segmentation model for background removal.

Uses ``rembg`` (which bundles a U²-Net model) when available.  Falls back to
returning the image unchanged when rembg is not installed, so the API keeps
working without the optional dependency.

The FastAPI app expects a U2NetSegmenter class with:
- __init__(model_path: Optional[str] = None, device: Optional[str] = None)
- remove_background(image: PIL.Image.Image) -> PIL.Image.Image
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional rembg import
# ---------------------------------------------------------------------------

try:
    import rembg as _rembg

    _REMBG_AVAILABLE = True
except ImportError:  # pragma: no cover
    _rembg = None  # type: ignore[assignment]
    _REMBG_AVAILABLE = False


def _composite_on_black(rgba_image: Image.Image) -> Image.Image:
    """Composite an RGBA image onto a solid black background, returning RGB."""
    rgba = rgba_image.convert("RGBA")
    background = Image.new("RGB", rgba.size, (0, 0, 0))
    r, g, b, a = rgba.split()
    background.paste(Image.merge("RGB", (r, g, b)), mask=a)
    return background


@dataclass
class U2NetSegmenter:
    model_path: Optional[str] = None
    device: Optional[str] = None
    _session: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if _REMBG_AVAILABLE:
            self._session = _rembg.new_session()
            logger.info("U2-Net background removal ready (rembg).")
        else:
            self._session = None
            logger.warning(
                "rembg is not installed — background removal is disabled. "
                "Install it with: pip install rembg"
            )

    def remove_background(self, image: Image.Image) -> Image.Image:
        """Remove the background from *image* and composite onto black.

        Returns the original image unchanged if rembg is not available.
        """
        if not _REMBG_AVAILABLE or self._session is None:
            return image

        try:
            rgba_result: Image.Image = _rembg.remove(image, session=self._session)
            return _composite_on_black(rgba_result)
        except Exception:
            logger.exception("rembg failed to process image; returning original.")
            return image