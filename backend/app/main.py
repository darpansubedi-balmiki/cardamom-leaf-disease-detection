"""
FastAPI application for cardamom leaf disease detection.

Endpoints
---------
POST /predict
    Accepts a multipart image upload and returns predicted disease class
    with probability plus a Grad-CAM heatmap overlay (base64 PNG).

GET /health
    Returns service health status.
"""

from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from .models.classifier import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_TOP_K,
    DiseaseClassifier,
)
from .models.u2net_segmenter import U2NetSegmenter
from .schemas import PredictionResponse
from .utils.grad_cam import generate_gradcam_heatmap
from .utils.image_preprocess import preprocess_image
from .utils.overlay import overlay_and_encode

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model instances (loaded once at startup)
# ---------------------------------------------------------------------------

_classifier: DiseaseClassifier | None = None
_segmenter: U2NetSegmenter | None = None


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    global _classifier, _segmenter

    print("=" * 60)
    print("  Cardamom Leaf Disease Detection API – Starting up")
    print("=" * 60)

    model_path = os.environ.get("MODEL_PATH", "models/cardamom_model.pt")
    u2net_path = os.environ.get("U2NET_PATH")

    _segmenter = U2NetSegmenter(model_path=u2net_path)
    print("  ℹ️   U2-Net background removal: placeholder (disabled)")

    confidence_threshold = float(
        os.environ.get("CONFIDENCE_THRESHOLD", DEFAULT_CONFIDENCE_THRESHOLD)
    )
    top_k = int(os.environ.get("TOP_K", DEFAULT_TOP_K))

    _classifier = DiseaseClassifier(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        top_k=top_k,
    )
    print(f"  ✓   Classifier ready  (threshold={confidence_threshold}, top_k={top_k})")
    print("=" * 60)

    yield

    # Shutdown: nothing to clean up for now.


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Cardamom Leaf Disease Detection API",
    description=(
        "Classifies cardamom leaf images into Colletotrichum Blight, "
        "Phyllosticta Leaf Spot, or Healthy. Returns prediction confidence and "
        "a Grad-CAM heatmap overlay."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Response schema (health)
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Leaf image (JPEG/PNG)"),
    confidence_threshold: float = Form(
        DEFAULT_CONFIDENCE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Override the confidence threshold for this request.",
    ),
    top_k: int = Form(
        DEFAULT_TOP_K,
        ge=1,
        le=10,
        description="(Currently unused) Number of top predictions to return.",
    ),
) -> PredictionResponse:
    """
    Predict disease class for an uploaded leaf image and generate a Grad-CAM heatmap.

    Notes:
    - Returns fields the current frontend expects: class_name, confidence, heatmap.
    - The classifier still internally supports top-k/uncertainty, but this endpoint
      returns a single class + confidence for UI compatibility.
    """
    if _classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Validate content type
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG or PNG/WebP.",
        )

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {exc}") from exc

    # Optional background removal (no-op for placeholder segmenter)
    if _segmenter is not None:
        image = _segmenter.remove_background(image)

    # Per-request threshold / top_k override (kept for compatibility)
    _classifier.confidence_threshold = float(confidence_threshold)
    _classifier.top_k = min(int(top_k), 10)

    # Run prediction via classifier (gets class name using its CLASS_NAMES mapping)
    result = _classifier.predict(image)

    # Prepare tensor for inference + Grad-CAM
    input_tensor = preprocess_image(image).to(_classifier.device)

    # Forward pass to get predicted class index + confidence
    with torch.no_grad():
        logits = _classifier._model(input_tensor)  # noqa: SLF001 (private access for now)
        probs = F.softmax(logits, dim=1)
        predicted_class = int(probs.argmax(dim=1).item())
        confidence = float(probs[0, predicted_class].item())

    # Choose a target conv layer for EfficientNetV2-S
    model = _classifier._model  # noqa: SLF001
    target_layer = None

    # Prefer the last feature block's conv
    try:
        for module in model.features[-1].modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
    except Exception:
        target_layer = None

    # Fallback: last Conv2d anywhere
    if target_layer is None:
        for _, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module

    if target_layer is None:
        raise HTTPException(
            status_code=500,
            detail="Could not locate a Conv2d layer for Grad-CAM.",
        )

    # Grad-CAM heatmap + overlay as base64 png
    heatmap = generate_gradcam_heatmap(
        model,
        input_tensor,
        target_layer,
        target_class=predicted_class,
    )
    heatmap_base64 = overlay_and_encode(image, heatmap, alpha=0.4)

    return PredictionResponse(
        class_name=result.top_class,
        confidence=confidence,
        heatmap=heatmap_base64,
        model_trained=True,
        warning=None,
    )