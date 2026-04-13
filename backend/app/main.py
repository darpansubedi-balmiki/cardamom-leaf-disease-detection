"""
FastAPI application for cardamom leaf disease detection.

Endpoints
---------
POST /predict
    Accepts a multipart image upload and returns the top-k predicted disease
    classes with probabilities plus an uncertainty flag.

    When include_severity=true is included in the form data the response also
    contains a Grad-CAM heatmap overlay and a heuristic severity estimate.

POST /predict/batch
    Accepts up to 10 images and returns a list of predictions.

GET /health
    Returns service health status, model load state, and device info.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncGenerator, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from .models.classifier import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_TOP_K,
    DEFAULT_USE_TTA,
    DiseaseClassifier,
    PredictionResult,
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

# Structured prediction logger — writes one JSON line per request to
# predictions.log (or the path set in PREDICTION_LOG_PATH)
_pred_log_path = os.environ.get("PREDICTION_LOG_PATH", "predictions.log")
_pred_logger = logging.getLogger("predictions")
_pred_logger.setLevel(logging.INFO)
_pred_handler = logging.FileHandler(_pred_log_path, encoding="utf-8")
_pred_handler.setFormatter(logging.Formatter("%(message)s"))
_pred_logger.addHandler(_pred_handler)
_pred_logger.propagate = False  # don't forward to root logger


def _log_prediction(
    *,
    filename: str,
    top_class: str,
    top_probability: float,
    is_uncertain: bool,
    include_severity: bool,
    latency_ms: float,
    use_tta: bool,
    cam_method: str,
) -> None:
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file": filename,
        "top_class": top_class,
        "top_prob": round(top_probability, 4),
        "uncertain": is_uncertain,
        "severity": include_severity,
        "tta": use_tta,
        "cam": cam_method,
        "latency_ms": round(latency_ms, 1),
    }
    _pred_logger.info(json.dumps(record, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Thread pool for off-loading blocking inference
# ---------------------------------------------------------------------------
# asyncio.to_thread() is used instead of an explicit ThreadPoolExecutor so
# the thread pool is managed by the event loop and survives across
# test-client instantiations.

# ---------------------------------------------------------------------------
# Global model instances (loaded once at startup)
# ---------------------------------------------------------------------------

_classifier: DiseaseClassifier | None = None
_segmenter: U2NetSegmenter | None = None
_model_metadata: dict = {}


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    global _classifier, _segmenter, _model_metadata

    print("=" * 60)
    print("  Cardamom Leaf Disease Detection API – Starting up")
    print("=" * 60)

    model_path = os.environ.get("MODEL_PATH", "models/cardamom_model.pt")
    u2net_path = os.environ.get("U2NET_PATH")

    _segmenter = U2NetSegmenter(model_path=u2net_path)
    from .models.u2net_segmenter import _REMBG_AVAILABLE
    if _REMBG_AVAILABLE:
        print("  ✓   U2-Net background removal: enabled (rembg)")
    else:
        print("  ℹ️   U2-Net background removal: disabled (rembg not installed)")

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

    # Load or build model metadata
    meta_path = Path(model_path).with_suffix(".json")
    if meta_path.exists():
        with open(meta_path) as f:
            _model_metadata = json.load(f)
        print(f"  ✓   Model metadata loaded from {meta_path}")
    else:
        _model_metadata = {
            "model_path": model_path,
            "model_loaded": os.path.isfile(model_path),
            "note": "No model_metadata.json found alongside checkpoint.",
        }

    print("=" * 60)

    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Cardamom Leaf Disease Detection API",
    description=(
        "Classifies cardamom leaf images into Colletotrichum Blight, "
        "Phyllosticta Leaf Spot, or Healthy. Returns top-k prediction "
        "probabilities and (optionally) a Grad-CAM overlay + severity estimate."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

_cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Optional API-key authentication
# ---------------------------------------------------------------------------

_API_KEY: str | None = os.environ.get("API_KEY")


async def _check_api_key(request: Request) -> None:
    """If API_KEY env var is set, every request must supply it via
    ``Authorization: Bearer <key>`` or ``X-API-Key: <key>`` header."""
    if not _API_KEY:
        return  # auth disabled in development
    auth = request.headers.get("Authorization", "")
    x_key = request.headers.get("X-API-Key", "")
    provided = auth.removeprefix("Bearer ").strip() or x_key.strip()
    if provided != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    """Standardised error response body."""
    error_code: str
    message: str
    details: Optional[str] = None


class TopKItem(BaseModel):
    class_name: str
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability (0–1)")
    probability_pct: float = Field(..., ge=0.0, le=100.0, description="Probability (%)")


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

    # Severity fields (present only when include_severity=true)
    heatmap: Optional[str] = Field(
        None, description="Base64-encoded PNG of the Grad-CAM heatmap overlay."
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
    cam_method: str = Field(
        "none",
        description='CAM method used: "none", "gradcam", or "gradcam++".',
    )
    model_version: Optional[str] = Field(
        None, description="Model version tag from model_metadata.json, if available."
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_classes: list[str]
    device: str
    model_version: Optional[str] = None
    model_accuracy: Optional[float] = None


# ---------------------------------------------------------------------------
# Image quality guard
# ---------------------------------------------------------------------------

_MIN_IMAGE_DIM = 64          # pixels
_BLUR_THRESHOLD = float(os.environ.get("BLUR_THRESHOLD", "50.0"))
_MONO_THRESHOLD = float(os.environ.get("MONO_THRESHOLD", "10.0"))


def _check_image_quality(image: Image.Image) -> None:
    """Raise HTTPException 400 for obviously unusable images.

    Checks:
    1. Minimum dimensions  (< 64 × 64 → rejected)
    2. Near-monochromatic  (std-dev of all channels < threshold → rejected)
    3. Excessive blurring  (Laplacian variance of luminance < threshold)
    """
    w, h = image.size
    if w < _MIN_IMAGE_DIM or h < _MIN_IMAGE_DIM:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image is too small ({w}×{h} px). "
                f"Please provide an image of at least {_MIN_IMAGE_DIM}×{_MIN_IMAGE_DIM} px."
            ),
        )

    arr = np.array(image.resize((64, 64), Image.BILINEAR), dtype=np.float32)
    if arr.std() < _MONO_THRESHOLD:
        raise HTTPException(
            status_code=400,
            detail=(
                "Image appears nearly monochromatic (blank or overexposed). "
                "Please upload a clear photo of a cardamom leaf."
            ),
        )

    # Laplacian blur detection (luminance channel)
    try:
        import cv2

        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < _BLUR_THRESHOLD:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Image appears excessively blurry (sharpness score {lap_var:.1f} < {_BLUR_THRESHOLD}). "
                    "Please retake the photo in better lighting."
                ),
            )
    except ImportError:
        pass  # cv2 not available; skip blur check


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prediction_to_response(
    result: PredictionResult,
    threshold: float,
    heatmap_b64: Optional[str] = None,
    severity: Optional[SeverityResult] = None,
    cam_method: str = "none",
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
        cam_method=cam_method,
        model_version=_model_metadata.get("version"),
    )


def _find_target_conv2d(model: torch.nn.Module) -> Optional[torch.nn.Conv2d]:
    """Best-effort method to find a conv layer for Grad-CAM."""
    try:
        for module in model.features[-1].modules():  # type: ignore[attr-defined]
            if isinstance(module, torch.nn.Conv2d):
                return module
    except Exception:
        pass

    last_conv = None
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv


def _run_predict_sync(
    image: Image.Image,
    classifier: DiseaseClassifier,
    segmenter: Optional[U2NetSegmenter],
    confidence_threshold: float,
    top_k: int,
    include_severity: bool,
    severity_heatmap_threshold: Optional[float],
    use_tta: bool,
    cam_method: str,
) -> PredictResponse:
    """Blocking inference – runs in a thread-pool worker."""
    if segmenter is not None:
        image = segmenter.remove_background(image)

    classifier.confidence_threshold = float(confidence_threshold)
    classifier.top_k = min(int(top_k), 10)

    result = classifier.predict(image, use_tta=use_tta)

    if result.top_class == "Other":
        raise HTTPException(
            status_code=400,
            detail=(
                "Uncertain with the uploaded image. It may not be a cardamom leaf or is too unclear to classify. "
                "Please upload a clear photo of a cardamom leaf."
            ),
        )

    heatmap_b64: Optional[str] = None
    severity: Optional[SeverityResult] = None

    if include_severity:
        input_tensor = preprocess_image(image).to(classifier.device)

        with torch.no_grad():
            logits = classifier._model(input_tensor)  # noqa: SLF001
            probs = F.softmax(logits, dim=1)
            predicted_class = int(probs.argmax(dim=1).item())

        model = classifier._model  # noqa: SLF001
        target_layer = _find_target_conv2d(model)
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
            method=cam_method,
        )
        heatmap_b64 = overlay_and_encode(image, heatmap_np, alpha=0.4)

        ht = (
            float(severity_heatmap_threshold)
            if severity_heatmap_threshold is not None
            else get_heatmap_threshold()
        )

        severity = compute_severity_from_heatmap(
            heatmap_np,
            threshold=ht,
            thresholds=get_stage_thresholds(),
        )

    return _prediction_to_response(
        result=result,
        threshold=float(confidence_threshold),
        heatmap_b64=heatmap_b64,
        severity=severity,
        cam_method=cam_method if include_severity else "none",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    from .models.classifier import CLASS_NAMES as _cls

    loaded = _classifier is not None
    try:
        device_str = str(_classifier.device) if _classifier else "cpu"
    except Exception:
        device_str = "cpu"

    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        model_classes=_model_metadata.get("class_names", _cls),
        device=device_str,
        model_version=_model_metadata.get("version"),
        model_accuracy=_model_metadata.get("test_accuracy"),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: Request,
    file: Annotated[UploadFile, File(description="Leaf image (JPEG/PNG/WebP)")],
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
    include_severity: bool = Form(
        False,
        description=(
            "When true, compute a heuristic severity estimate from the Grad-CAM heatmap "
            "and include severity_stage, severity_percent, severity_method, and heatmap."
        ),
    ),
    severity_heatmap_threshold: Optional[float] = Form(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Override the heatmap binarisation threshold used for severity estimation (0–1). "
            f"Defaults to SEVERITY_HEATMAP_THRESHOLD (default {get_heatmap_threshold():.2f})."
        ),
    ),
    use_tta: bool = Form(
        DEFAULT_USE_TTA,
        description=(
            "When true, run Test-Time Augmentation (5 variants averaged) for more stable "
            "predictions on borderline inputs.  Slower but more robust."
        ),
    ),
    cam_method: str = Form(
        "gradcam",
        description='CAM method to use when include_severity=true. "gradcam" or "gradcam++".',
    ),
) -> PredictResponse:
    await _check_api_key(request)

    if _classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG/PNG/WebP.",
        )

    if cam_method not in ("gradcam", "gradcam++"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown cam_method '{cam_method}'. Use 'gradcam' or 'gradcam++'.",
        )

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {exc}") from exc

    _check_image_quality(image)

    t0 = time.perf_counter()

    response: PredictResponse = await asyncio.to_thread(
        _run_predict_sync,
        image,
        _classifier,
        _segmenter,
        confidence_threshold,
        top_k,
        include_severity,
        severity_heatmap_threshold,
        use_tta,
        cam_method,
    )

    latency_ms = (time.perf_counter() - t0) * 1000
    _log_prediction(
        filename=file.filename or "unknown",
        top_class=response.top_class,
        top_probability=response.top_probability,
        is_uncertain=response.is_uncertain,
        include_severity=include_severity,
        latency_ms=latency_ms,
        use_tta=use_tta,
        cam_method=cam_method,
    )

    return response


@app.post("/predict/batch", response_model=List[PredictResponse])
async def predict_batch(
    request: Request,
    files: Annotated[List[UploadFile], File(description="Up to 10 leaf images (JPEG/PNG/WebP)")],
    confidence_threshold: float = Form(DEFAULT_CONFIDENCE_THRESHOLD, ge=0.0, le=1.0),
    top_k: int = Form(DEFAULT_TOP_K, ge=1, le=10),
    include_severity: bool = Form(False),
    use_tta: bool = Form(DEFAULT_USE_TTA),
    cam_method: str = Form("gradcam"),
) -> List[PredictResponse]:
    """Run prediction on up to 10 images in a single call.

    Useful for researchers who need to evaluate a directory of images without
    writing a loop over the single-image endpoint.
    """
    await _check_api_key(request)

    if _classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Batch size limit is 10 images.")

    if cam_method not in ("gradcam", "gradcam++"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown cam_method '{cam_method}'. Use 'gradcam' or 'gradcam++'.",
        )

    results: List[PredictResponse] = []

    for upload in files:
        if upload.content_type not in ("image/jpeg", "image/png", "image/webp"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{upload.content_type}' for file '{upload.filename}'. Use JPEG/PNG/WebP.",
            )
        raw = await upload.read()
        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot read image '{upload.filename}': {exc}",
            ) from exc

        _check_image_quality(image)

        t0 = time.perf_counter()
        response: PredictResponse = await asyncio.to_thread(
            _run_predict_sync,
            image,
            _classifier,
            _segmenter,
            confidence_threshold,
            top_k,
            include_severity,
            None,  # severity_heatmap_threshold
            use_tta,
            cam_method,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        _log_prediction(
            filename=upload.filename or "unknown",
            top_class=response.top_class,
            top_probability=response.top_probability,
            is_uncertain=response.is_uncertain,
            include_severity=include_severity,
            latency_ms=latency_ms,
            use_tta=use_tta,
            cam_method=cam_method,
        )
        results.append(response)

    return results
