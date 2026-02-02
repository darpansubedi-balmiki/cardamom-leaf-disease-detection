"""Pydantic models for request/response validation."""
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str


class PredictionResponse(BaseModel):
    """Prediction response model."""
    class_name: str  # Using class_name instead of class to avoid keyword conflict
    confidence: float
    heatmap: str  # Base64 encoded PNG image
