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

"""FastAPI application for cardamom leaf disease detection."""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn.functional as F
from io import BytesIO
import traceback

from app.schemas import HealthResponse, PredictionResponse
from app.models.cardamom_model import load_model
from app.models.u2net_segmenter import load_u2net, apply_background_removal
from app.utils.image_preprocess import preprocess_image
from app.utils.grad_cam import generate_gradcam_heatmap
from app.utils.overlay import overlay_and_encode


# Initialize FastAPI app
app = FastAPI(
    title="Cardamom Leaf Disease Detection API",
    description="API for detecting diseases in cardamom leaves using deep learning",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default port
        "http://localhost:3000",  # Alternative React port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class names
CLASS_NAMES = ["Colletotrichum Blight", "Phyllosticta Leaf Spot", "Healthy"]

# Global variables for models
classifier_model = None
u2net_model = None
device = None
model_is_trained = False  # Track if using trained or placeholder model


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global classifier_model, u2net_model, device, model_is_trained
    
    print("Starting up application...")
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load classifier model
    print("Loading classifier model...")
    classifier_model, model_is_trained = load_model(device=device)
    print("Classifier model loaded successfully")
    
    # Load U2-Net model (placeholder for now)
    print("Loading U2-Net model...")
    u2net_model = load_u2net(device=device)
    print("U2-Net model loaded (placeholder)")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.
    
    Returns:
        Status message indicating the API is running
    """
    model_status = "trained" if model_is_trained else "untrained (placeholder)"
    return HealthResponse(status="ok", model_status=model_status)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Predict disease class for an uploaded cardamom leaf image.
    
    Args:
        file: Uploaded image file (multipart/form-data)
        
    Returns:
        Prediction result with class name, confidence, and Grad-CAM heatmap
        
    Raises:
        HTTPException: If image processing or prediction fails
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        # Read image file
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        
        # Apply background removal (placeholder - just passes through)
        image = apply_background_removal(image, u2net_model)
        
        # Preprocess image for model
        input_tensor = preprocess_image(image)
        input_tensor = input_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            output = classifier_model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = probabilities.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Get class name
        class_name = CLASS_NAMES[predicted_class]
        
        # Generate Grad-CAM heatmap
        # Target the last convolutional layer (conv4)
        # Get the last convolutional layer from EfficientNetV2
        # EfficientNetV2 structure: backbone.features[-1] is the last conv block
        target_layer = None
        for module in classifier_model.backbone.features[-1].modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module

        if target_layer is None:
            # Fallback: find any last Conv2d layer
            for name, module in classifier_model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    
        heatmap = generate_gradcam_heatmap(
            classifier_model,
            input_tensor,
            target_layer,
            target_class=predicted_class
        )
        
        # Overlay heatmap on original image and encode as base64
        heatmap_base64 = overlay_and_encode(image, heatmap, alpha=0.4)
        
        # Add warning if using untrained model
        warning = None
        if not model_is_trained:
            warning = (
                "⚠️ UNTRAINED MODEL: This prediction uses a placeholder model with random weights. "
                "Predictions are not accurate. Please train the model with real data for production use. "
                "See MODEL_TRAINING.md for instructions."
            )
        
        return PredictionResponse(
            class_name=class_name,
            confidence=confidence,
            heatmap=heatmap_base64,
            model_trained=model_is_trained,
            warning=warning
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
