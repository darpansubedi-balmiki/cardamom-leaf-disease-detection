"""
FastAPI application for cardamom leaf disease detection.

Endpoints
---------
POST /predict
    Accepts a multipart image upload and returns the top-k predicted disease
    classes with probabilities plus an uncertainty flag.

GET /health
    Returns service health status.
"""

from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from .models.classifier import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_TOP_K,
    DiseaseClassifier,
    PredictionResult,
)
from .models.u2net_segmenter import U2NetSegmenter

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

    model_path = os.environ.get("MODEL_PATH", "models/efficientnet_v2_s.pth")
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
        "Phyllosticta Leaf Spot, or Healthy. Returns top-k predictions with "
        "probabilities and an uncertainty flag when confidence is low."
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
# Pydantic response schemas
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


class HealthResponse(BaseModel):
    status: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prediction_to_response(
    result: PredictionResult, threshold: float
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
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
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
        description="Number of top predictions to return.",
    ),
) -> PredictResponse:
    if _classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Validate content type
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG or PNG.",
        )

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {exc}") from exc

    # Optional background removal (no-op when U2-Net is not loaded)
    if _segmenter is not None:
        image = _segmenter.remove_background(image)

    # Per-request threshold / top_k override
    _classifier.confidence_threshold = confidence_threshold
    _classifier.top_k = min(top_k, 10)

    result = _classifier.predict(image)

    return _prediction_to_response(result, confidence_threshold)
