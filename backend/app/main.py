"""
FastAPI application for cardamom leaf disease detection.

Endpoints
---------
POST /predict
    Accepts a multipart image upload and returns the top-k predicted disease
    classes with probabilities plus an uncertainty flag.  When
    ``include_severity=true`` is included in the form data the response also
    contains a Grad-CAM heatmap overlay and a heuristic severity estimate.
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
from typing import AsyncGenerator, Optional

import torch
import torch.nn.functional as F
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
from .utils.grad_cam import generate_gradcam_heatmap
from .utils.image_preprocess import preprocess_image
from .utils.overlay import overlay_and_encode
from .utils.severity import (
    SeverityResult,
    compute_severity_from_heatmap,
    get_heatmap_threshold,
    get_stage_thresholds,
)
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


class TopKItem(BaseModel):
    class_name: str
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability (0–1)")
    probability_pct: float = Field(
        ..., ge=0.0, le=100.0, description="Probability as percentage"
    )


class PredictResponse(BaseModel):
    top_class: str = Field(
        ...,
        description=(
            'Predicted disease class, or "Uncertain" when confidence is below threshold.'
        ),
    )
    top_probability: float = Field(..., ge=0.0, le=1.0)
    top_probability_pct: float = Field(..., ge=0.0, le=100.0)
    is_uncertain: bool = Field(
        ...,
        description="True when the top-1 probability is below the confidence threshold.",
    )
    confidence_threshold: float = Field(
        ..., description="The threshold used to determine uncertainty."
    )
    top_k: list[TopKItem] = Field(..., description="Top-k predictions with probabilities.")
    # -----------------------------------------------------------------------
    # Severity fields – present only when include_severity=true was requested
    # -----------------------------------------------------------------------
    heatmap: Optional[str] = Field(
        None,
        description="Base64-encoded PNG of the Grad-CAM heatmap overlay.",
    )
    severity_stage: Optional[int] = Field(
        None,
        ge=0,
        le=4,
        description="Severity stage 0–4 (0=healthy, 1=mild … 4=very severe).",
    )
    severity_percent: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Estimated percentage of leaf area affected (0–100).",
    )
    severity_method: str = Field(
        "none",
        description=(
            'Method used to compute severity. One of "none", "heuristic", '
            '"ordinal_model", "segmentation".'
        ),
    )


class HealthResponse(BaseModel):
    status: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prediction_to_response(
    result: PredictionResult,
    threshold: float,
    heatmap_b64: Optional[str] = None,
    severity: Optional[SeverityResult] = None,
) -> PredictResponse:
    return PredictResponse(
        top_class=result.top_class,
        top_probability=result.top_probability,
        top_probability_pct=round(result.top_probability * 100, 2),
        is_uncertain=result.is_uncertain,
        confidence_threshold=threshold,
        top_k=[
            TopKItem(
                class_name=item.class_name,
                probability=item.probability,
                probability_pct=round(item.probability * 100, 2),
            )
            for item in result.top_k
        ],
        heatmap=heatmap_b64,
        severity_stage=severity.severity_stage if severity else None,
        severity_percent=severity.severity_percent if severity else None,
        severity_method=severity.severity_method if severity else "none",
    )


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
    include_severity: bool = Form(
        False,
        description=(
            "When true, compute a heuristic severity estimate from the Grad-CAM "
            "heatmap and include severity_stage, severity_percent, severity_method, "
            "and the heatmap overlay in the response."
        ),
    ),
    severity_heatmap_threshold: Optional[float] = Form(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Override the heatmap binarisation threshold used for severity "
            "estimation (0–1).  Defaults to the SEVERITY_HEATMAP_THRESHOLD env var "
            f"(default {get_heatmap_threshold():.2f})."
        ),
    ),
) -> PredictResponse:
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

    # ------------------------------------------------------------------
    # Severity + heatmap (only when requested)
    # ------------------------------------------------------------------
    heatmap_b64: Optional[str] = None
    severity: Optional[SeverityResult] = None

    if include_severity:
        input_tensor = preprocess_image(image).to(_classifier.device)

        # Forward pass to determine predicted class index
        with torch.no_grad():
            logits = _classifier._model(input_tensor)  # noqa: SLF001
            probs = F.softmax(logits, dim=1)
            predicted_class = int(probs.argmax(dim=1).item())

        # Locate target Conv2d for Grad-CAM
        model = _classifier._model  # noqa: SLF001
        target_layer = None
        try:
            for module in model.features[-1].modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
        except Exception:
            target_layer = None

        if target_layer is None:
            for _, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module

        if target_layer is None:
            raise HTTPException(
                status_code=500,
                detail="Could not locate a Conv2d layer for Grad-CAM.",
            )

        heatmap_np = generate_gradcam_heatmap(
            model,
            input_tensor,
            target_layer,
            target_class=predicted_class,
        )
        heatmap_b64 = overlay_and_encode(image, heatmap_np, alpha=0.4)

        # Heuristic severity from heatmap
        ht = severity_heatmap_threshold if severity_heatmap_threshold is not None else get_heatmap_threshold()
        severity = compute_severity_from_heatmap(
            heatmap_np,
            threshold=ht,
            thresholds=get_stage_thresholds(),
        )

    return _prediction_to_response(result, confidence_threshold, heatmap_b64, severity)
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