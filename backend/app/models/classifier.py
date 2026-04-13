"""
Disease classifier using EfficientNetV2-S backbone (torchvision).

The model is loaded from a checkpoint file when one is present.  If no
checkpoint is found the model runs with *random* (untrained) weights and
predictions will be meaningless – this is intentional to keep the server
start-up working in development.

Inference contract
------------------
``predict(image)`` returns a :class:`PredictionResult` containing:

- ``top_class``      – name of the class with the highest probability, or
                       ``"Uncertain"`` when ``top_probability < confidence_threshold``.
- ``top_probability`` – probability of the top class (0–1).
- ``is_uncertain``   – ``True`` when the prediction is below the threshold.
- ``top_k``          – list of (class_name, probability) for the K most
                       probable classes (default K=3).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_USE_TTA: bool = False  # Test-Time Augmentation off by default

CLASS_NAMES: list[str] = [
    "Colletotrichum Blight",
    "Healthy",
    "Other",
    "Phyllosticta Leaf Spot",
]

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.60
DEFAULT_TOP_K: int = 3

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# TTA augmentation variants (applied on top of the base transform)
_TTA_AUGMENTS = [
    transforms.Compose([]),  # identity
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.RandomRotation(degrees=(15, 15)),
    transforms.RandomRotation(degrees=(-15, -15)),
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TopKPrediction:
    class_name: str
    probability: float  # 0–1


@dataclass
class PredictionResult:
    top_class: str
    top_probability: float  # 0–1
    is_uncertain: bool
    top_k: list[TopKPrediction] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

# Use the same preprocessing pipeline as training (direct Resize to 224×224)
# so that train/inference transforms are consistent.
_preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class DiseaseClassifier:
    """EfficientNetV2-S disease classifier with top-k and uncertainty output."""

    def __init__(
        self,
        model_path: str | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        device: str | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.top_k = min(top_k, len(CLASS_NAMES))
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._model = self._build_model()

        if model_path and os.path.isfile(model_path):
            self._load_weights(model_path)
        else:
            if model_path:
                logger.warning(
                    "⚠️  Model checkpoint not found at '%s'. "
                    "Using untrained weights – predictions will be random.",
                    model_path,
                )
            else:
                logger.warning(
                    "⚠️  No model checkpoint provided. "
                    "Using untrained weights – predictions will be random."
                )

        self._model.eval()
        self._model.to(self.device)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> torch.nn.Module:
        # Must match the architecture used in train.py
        model = models.efficientnet_v2_s(weights=None)

        num_features = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(512, len(CLASS_NAMES)),
        )
        return model

    def _load_weights(self, path: str) -> None:
        try:
            state = torch.load(path, map_location=self.device)
            # Accept both raw state-dicts and checkpoint dicts.
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self._model.load_state_dict(state)
            logger.info("✓  Loaded model weights from '%s'.", path)
        except Exception as exc:
            logger.error("Failed to load model weights from '%s': %s", path, exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image: Image.Image, use_tta: bool = False) -> PredictionResult:
        """Run inference on a PIL image and return a :class:`PredictionResult`.

        Args:
            image:    PIL image to classify.
            use_tta:  When True, runs Test-Time Augmentation (5 flips/rotations)
                      and averages the softmax probabilities before deciding the
                      top class.  Slightly slower but more stable on borderline
                      inputs.
        """
        rgb = image.convert("RGB")

        if use_tta:
            probs_list = []
            for aug in _TTA_AUGMENTS:
                augmented = aug(rgb)
                tensor = _preprocess(augmented).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self._model(tensor)
                    probs_list.append(F.softmax(logits, dim=1).squeeze(0).cpu().numpy())
            probs_np: np.ndarray = np.mean(probs_list, axis=0)
        else:
            tensor = _preprocess(rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self._model(tensor)
                probs_np = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Top-k indices sorted by descending probability
        top_k_count = min(self.top_k, len(CLASS_NAMES))
        top_indices = np.argsort(probs_np)[::-1][:top_k_count]

        top_k_list = [
            TopKPrediction(
                class_name=CLASS_NAMES[i],
                probability=float(probs_np[i]),
            )
            for i in top_indices
        ]

        top_probability = float(probs_np[top_indices[0]])
        is_uncertain = top_probability < self.confidence_threshold
        top_class = "Uncertain" if is_uncertain else CLASS_NAMES[top_indices[0]]

        return PredictionResult(
            top_class=top_class,
            top_probability=top_probability,
            is_uncertain=is_uncertain,
            top_k=top_k_list,
        )
