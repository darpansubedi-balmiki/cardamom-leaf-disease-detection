"""Pydantic models for request/response validation."""
from pydantic import BaseModel
from typing import Optional


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    model_status: Optional[str] = None


class PredictionResponse(BaseModel):
    """Prediction response model."""
    class_name: str  # Using class_name instead of class to avoid keyword conflict
    confidence: float
    heatmap: str  # Base64 encoded PNG image
    model_trained: bool = False  # Indicates if using trained or placeholder model
    warning: Optional[str] = None  # Warning message for untrained model
