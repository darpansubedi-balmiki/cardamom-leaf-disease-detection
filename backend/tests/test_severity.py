"""
Unit tests for severity estimation utilities.

Tests cover:
- map_percent_to_stage: stage mapping from percentage
- compute_severity_from_heatmap: heuristic quantification with synthetic heatmaps
- /predict response schema when include_severity=true / false
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.models.classifier import (
    CLASS_NAMES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_TOP_K,
    DiseaseClassifier,
    PredictionResult,
    TopKPrediction,
)
from app.utils.severity import (
    DEFAULT_STAGE_THRESHOLDS,
    SeverityResult,
    compute_severity_from_heatmap,
    map_percent_to_stage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image_bytes(size: tuple[int, int] = (256, 256)) -> bytes:
    """Return JPEG bytes of a synthetic textured image that passes quality checks."""
    import numpy as _np
    w, h = size
    arr = _np.zeros((h, w, 3), dtype=_np.uint8)
    block = max(16, w // 8)
    for row in range(h):
        for col in range(w):
            if (row // block + col // block) % 2 == 0:
                arr[row, col] = [80, 160, 40]
            else:
                arr[row, col] = [40, 100, 20]
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_prediction_result(
    top_class: str = CLASS_NAMES[1],
    top_probability: float = 0.85,
    is_uncertain: bool = False,
) -> PredictionResult:
    top_k = [
        TopKPrediction(CLASS_NAMES[1], 0.85),
        TopKPrediction(CLASS_NAMES[0], 0.10),
        TopKPrediction(CLASS_NAMES[2], 0.05),
    ]
    return PredictionResult(
        top_class=top_class,
        top_probability=top_probability,
        is_uncertain=is_uncertain,
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def patched_classifier():
    mock_clf = MagicMock(spec=DiseaseClassifier)
    mock_clf.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
    mock_clf.top_k = DEFAULT_TOP_K
    mock_clf.predict.return_value = _make_prediction_result()

    # Mock the internal model for Grad-CAM (returns uniform logits)
    mock_model = MagicMock(spec=nn.Module)
    mock_model.return_value = torch.zeros(1, len(CLASS_NAMES))
    mock_clf._model = mock_model
    mock_clf.device = torch.device("cpu")

    with patch("app.main._classifier", mock_clf):
        yield mock_clf


# ---------------------------------------------------------------------------
# map_percent_to_stage tests
# ---------------------------------------------------------------------------


class TestMapPercentToStage:
    def test_zero_percent_is_stage_0(self):
        assert map_percent_to_stage(0.0) == 0

    def test_negative_clamped_to_stage_0(self):
        assert map_percent_to_stage(-5.0) == 0

    def test_100_percent_is_stage_4(self):
        assert map_percent_to_stage(100.0) == 4

    def test_above_100_clamped_to_stage_4(self):
        assert map_percent_to_stage(120.0) == 4

    def test_boundary_at_10_percent_is_stage_1(self):
        # 10.0 is the upper boundary for stage 1 (inclusive)
        assert map_percent_to_stage(10.0) == 1

    def test_just_above_10_percent_is_stage_2(self):
        assert map_percent_to_stage(10.01) == 2

    def test_boundary_at_25_percent_is_stage_2(self):
        assert map_percent_to_stage(25.0) == 2

    def test_just_above_25_percent_is_stage_3(self):
        assert map_percent_to_stage(25.01) == 3

    def test_boundary_at_50_percent_is_stage_3(self):
        assert map_percent_to_stage(50.0) == 3

    def test_just_above_50_percent_is_stage_4(self):
        assert map_percent_to_stage(50.01) == 4

    def test_midrange_values(self):
        assert map_percent_to_stage(5.0) == 1
        assert map_percent_to_stage(18.0) == 2
        assert map_percent_to_stage(37.0) == 3
        assert map_percent_to_stage(75.0) == 4

    def test_custom_thresholds(self):
        # Custom: 0, 20, 40, 60, 100
        thresholds = [0.0, 20.0, 40.0, 60.0, 100.0]
        assert map_percent_to_stage(0.0, thresholds) == 0
        assert map_percent_to_stage(20.0, thresholds) == 1
        assert map_percent_to_stage(21.0, thresholds) == 2
        assert map_percent_to_stage(40.0, thresholds) == 2
        assert map_percent_to_stage(61.0, thresholds) == 4

    def test_invalid_threshold_count_raises(self):
        with pytest.raises(ValueError):
            map_percent_to_stage(10.0, [0.0, 25.0, 100.0])


# ---------------------------------------------------------------------------
# compute_severity_from_heatmap tests
# ---------------------------------------------------------------------------


class TestComputeSeverityFromHeatmap:
    def test_all_zeros_heatmap_gives_stage_0(self):
        heatmap = np.zeros((8, 8))
        result = compute_severity_from_heatmap(heatmap, threshold=0.5)
        assert result.severity_percent == 0.0
        assert result.severity_stage == 0
        assert result.severity_method == "heuristic"

    def test_all_ones_heatmap_gives_100_percent(self):
        # All-ones heatmap: h_max == h_min so normalization skips,
        # every pixel equals 1.0 which is > 0.5 → 100 % affected.
        heatmap = np.ones((8, 8))
        result = compute_severity_from_heatmap(heatmap, threshold=0.5)
        assert result.severity_percent == 100.0
        assert result.severity_stage == 4

    def test_half_affected_at_threshold_05(self):
        # Left half = 1.0, right half = 0.0 → 50 % above threshold=0.5
        heatmap = np.zeros((4, 4))
        heatmap[:, :2] = 1.0
        result = compute_severity_from_heatmap(heatmap, threshold=0.5)
        assert result.severity_percent == 50.0
        assert result.severity_stage == 3

    def test_small_affected_region_gives_low_stage(self):
        # 1 out of 100 pixels above threshold
        heatmap = np.zeros((10, 10))
        heatmap[0, 0] = 1.0
        result = compute_severity_from_heatmap(heatmap, threshold=0.5)
        assert result.severity_percent == 1.0
        assert result.severity_stage == 1

    def test_returns_severity_result_instance(self):
        heatmap = np.random.rand(16, 16)
        result = compute_severity_from_heatmap(heatmap)
        assert isinstance(result, SeverityResult)

    def test_warning_is_present(self):
        heatmap = np.random.rand(8, 8)
        result = compute_severity_from_heatmap(heatmap)
        assert result.warning is not None
        assert len(result.warning) > 0

    def test_custom_threshold_changes_percent(self):
        # Heatmap: left half = 0.8, right half = 0.0.
        # After normalisation: left half → 1.0, right half → 0.0.
        # threshold=1.0: nothing strictly above 1.0 → 0 %
        # threshold=0.9: 1.0 > 0.9 = True → 50 %
        heatmap = np.zeros((4, 4))
        heatmap[:, :2] = 0.8
        result_high = compute_severity_from_heatmap(heatmap, threshold=1.0)
        result_low = compute_severity_from_heatmap(heatmap, threshold=0.9)
        assert result_high.severity_percent == 0.0
        assert result_low.severity_percent == 50.0

    def test_severity_percent_in_valid_range(self):
        heatmap = np.random.rand(32, 32)
        result = compute_severity_from_heatmap(heatmap)
        assert 0.0 <= result.severity_percent <= 100.0

    def test_stage_consistent_with_percent(self):
        heatmap = np.random.rand(32, 32)
        result = compute_severity_from_heatmap(heatmap)
        expected_stage = map_percent_to_stage(result.severity_percent)
        assert result.severity_stage == expected_stage


# ---------------------------------------------------------------------------
# Response schema validation – /predict with and without severity
# ---------------------------------------------------------------------------


class TestPredictResponseSchema:
    def test_default_request_has_no_severity_fields(self, client, patched_classifier):
        resp = client.post(
            "/predict",
            files={"file": ("leaf.jpg", _make_image_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["severity_method"] == "none"
        assert body["severity_stage"] is None
        assert body["severity_percent"] is None
        assert body["heatmap"] is None

    def test_severity_fields_absent_in_default_response(self, client, patched_classifier):
        """Core existing fields still present when severity not requested."""
        resp = client.post(
            "/predict",
            files={"file": ("leaf.jpg", _make_image_bytes(), "image/jpeg")},
        )
        body = resp.json()
        assert "top_class" in body
        assert "top_probability" in body
        assert "is_uncertain" in body
        assert "top_k" in body

    def test_include_severity_true_returns_severity_fields(self, client, patched_classifier):
        # Supply a real Conv2d as the target layer so the named_modules lookup
        # finds a valid layer before generate_gradcam_heatmap is called.
        conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        patched_classifier._model.named_modules.return_value = [("features.0.conv", conv)]

        with patch("app.main.generate_gradcam_heatmap") as mock_gc, \
             patch("app.main.overlay_and_encode") as mock_oe:
            mock_gc.return_value = np.zeros((7, 7))
            mock_oe.return_value = "base64heatmap=="

            resp = client.post(
                "/predict",
                files={"file": ("leaf.jpg", _make_image_bytes(), "image/jpeg")},
                data={"include_severity": "true"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["severity_method"] == "heuristic"
        assert body["severity_stage"] is not None
        assert body["severity_percent"] is not None
        assert body["heatmap"] == "base64heatmap=="

    def test_severity_stage_in_valid_range(self, client, patched_classifier):
        conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        patched_classifier._model.named_modules.return_value = [("features.0.conv", conv)]

        with patch("app.main.generate_gradcam_heatmap") as mock_gc, \
             patch("app.main.overlay_and_encode") as mock_oe:
            mock_gc.return_value = np.random.rand(7, 7)
            mock_oe.return_value = "base64data"

            resp = client.post(
                "/predict",
                files={"file": ("leaf.jpg", _make_image_bytes(), "image/jpeg")},
                data={"include_severity": "true"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert 0 <= body["severity_stage"] <= 4

    def test_severity_percent_in_valid_range(self, client, patched_classifier):
        conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        patched_classifier._model.named_modules.return_value = [("features.0.conv", conv)]

        with patch("app.main.generate_gradcam_heatmap") as mock_gc, \
             patch("app.main.overlay_and_encode") as mock_oe:
            mock_gc.return_value = np.random.rand(7, 7)
            mock_oe.return_value = "base64data"

            resp = client.post(
                "/predict",
                files={"file": ("leaf.jpg", _make_image_bytes(), "image/jpeg")},
                data={"include_severity": "true"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert 0.0 <= body["severity_percent"] <= 100.0
