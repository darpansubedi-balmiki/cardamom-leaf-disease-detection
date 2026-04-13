"""
Integration tests for /predict and /health endpoints.
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
import torch
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image_bytes(size: tuple[int, int] = (256, 256)) -> bytes:
    """Return JPEG bytes of a synthetic textured leaf image that passes quality checks."""
    import numpy as np
    w, h = size
    # Checkerboard pattern ensures non-zero Laplacian variance (not blurry)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
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
    """TestClient with a mocked classifier."""
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def patched_classifier():
    """Patch the global _classifier inside app.main."""
    mock_clf = MagicMock(spec=DiseaseClassifier)
    mock_clf.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
    mock_clf.top_k = DEFAULT_TOP_K
    mock_clf.predict.return_value = _make_prediction_result()

    with patch("app.main._classifier", mock_clf):
        yield mock_clf


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_returns_ok_status(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"

    def test_returns_model_loaded_field(self, client):
        body = client.get("/health").json()
        assert "model_loaded" in body

    def test_returns_device_field(self, client):
        body = client.get("/health").json()
        assert "device" in body

    def test_returns_model_classes_list(self, client):
        body = client.get("/health").json()
        assert isinstance(body.get("model_classes"), list)


# ---------------------------------------------------------------------------
# /predict – happy paths
# ---------------------------------------------------------------------------


class TestPredictSuccess:
    def test_returns_200(self, client, patched_classifier):
        resp = client.post(
            "/predict",
            files={"file": ("leaf.jpg", _make_image_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_response_contains_top_class(self, client, patched_classifier):
        resp = client.post(
            "/predict",
            files={"file": ("leaf.jpg", _make_image_bytes(), "image/jpeg")},
        )
        body = resp.json()
        assert "top_class" in body
        assert body["top_class"] == CLASS_NAMES[1]

    def test_response_contains_top_probability(self, client, patched_classifier):
        resp = client.post(
            "/predict",
            files={"file": ("leaf.jpg", _make_image_bytes(), "image/jpeg")},
        )
        body = resp.json()
        assert "top_probability" in body
        assert 0.0 <= body["top_probability"] <= 1.0

    def test_response_contains_top_probability_pct(self, client, patched_classifier):
        resp = client.post(
            "/predict",
            files={"file": ("leaf.jpg", _make_image_bytes(), "image/jpeg")},
        )
        body = resp.json()
        assert "top_probability_pct" in body
        assert abs(body["top_probability_pct"] - body["top_probability"] * 100) < 0.01

    def test_response_contains_is_uncertain(self, client, patched_classifier):
        resp = client.post(
            "/predict",
            files={"file": ("leaf.jpg", _make_image_bytes(), "image/jpeg")},
        )
        body = resp.json()
        assert "is_uncertain" in body

    def test_response_contains_top_k_list(self, client, patched_classifier):
        resp = client.post(
            "/predict",
            files={"file": ("leaf.jpg", _make_image_bytes(), "image/jpeg")},
        )
        body = resp.json()
        assert "top_k" in body
        assert isinstance(body["top_k"], list)
        assert len(body["top_k"]) == DEFAULT_TOP_K

    def test_top_k_items_have_class_name_and_probability(
        self, client, patched_classifier
    ):
        resp = client.post(
            "/predict",
            files={"file": ("leaf.jpg", _make_image_bytes(), "image/jpeg")},
        )
        for item in resp.json()["top_k"]:
            assert "class_name" in item
            assert "probability" in item
            assert "probability_pct" in item

    def test_response_contains_confidence_threshold(self, client, patched_classifier):
        resp = client.post(
            "/predict",
            files={"file": ("leaf.jpg", _make_image_bytes(), "image/jpeg")},
        )
        body = resp.json()
        assert "confidence_threshold" in body
        assert body["confidence_threshold"] == DEFAULT_CONFIDENCE_THRESHOLD

    def test_uncertain_prediction_when_low_confidence(self, client, patched_classifier):
        patched_classifier.predict.return_value = _make_prediction_result(
            top_class="Uncertain",
            top_probability=0.34,
            is_uncertain=True,
        )
        resp = client.post(
            "/predict",
            files={"file": ("leaf.jpg", _make_image_bytes(), "image/jpeg")},
        )
        body = resp.json()
        assert body["top_class"] == "Uncertain"
        assert body["is_uncertain"] is True

    def test_custom_threshold_in_form_data(self, client, patched_classifier):
        resp = client.post(
            "/predict",
            files={"file": ("leaf.jpg", _make_image_bytes(), "image/jpeg")},
            data={"confidence_threshold": "0.80"},
        )
        assert resp.status_code == 200
        # threshold should be updated on the classifier
        assert patched_classifier.confidence_threshold == 0.80

    def test_custom_top_k_in_form_data(self, client, patched_classifier):
        resp = client.post(
            "/predict",
            files={"file": ("leaf.jpg", _make_image_bytes(), "image/jpeg")},
            data={"top_k": "2"},
        )
        assert resp.status_code == 200
        assert patched_classifier.top_k == 2


# ---------------------------------------------------------------------------
# /predict – error paths
# ---------------------------------------------------------------------------


class TestPredictErrors:
    def test_unsupported_content_type_returns_400(self, client, patched_classifier):
        resp = client.post(
            "/predict",
            files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert resp.status_code == 400

    def test_missing_file_returns_422(self, client, patched_classifier):
        resp = client.post("/predict")
        assert resp.status_code == 422

    def test_tiny_image_returns_400(self, client, patched_classifier):
        """Images smaller than 64×64 px must be rejected with 400."""
        tiny = _make_image_bytes(size=(32, 32))
        resp = client.post(
            "/predict",
            files={"file": ("tiny.jpg", tiny, "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_monochromatic_image_returns_400(self, client, patched_classifier):
        """Fully blank (zero std-dev) images must be rejected with 400."""
        from PIL import Image as PILImage
        buf = io.BytesIO()
        PILImage.new("RGB", (256, 256), (255, 255, 255)).save(buf, format="JPEG")
        resp = client.post(
            "/predict",
            files={"file": ("blank.jpg", buf.getvalue(), "image/jpeg")},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /predict/batch
# ---------------------------------------------------------------------------


class TestBatchPredict:
    def test_batch_returns_list(self, client, patched_classifier):
        img = _make_image_bytes()
        resp = client.post(
            "/predict/batch",
            files=[
                ("files", ("leaf1.jpg", img, "image/jpeg")),
                ("files", ("leaf2.jpg", img, "image/jpeg")),
            ],
        )
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) == 2

    def test_batch_too_many_files_returns_400(self, client, patched_classifier):
        img = _make_image_bytes()
        files = [("files", (f"leaf{i}.jpg", img, "image/jpeg")) for i in range(11)]
        resp = client.post("/predict/batch", files=files)
        assert resp.status_code == 400

    def test_batch_each_item_has_top_class(self, client, patched_classifier):
        img = _make_image_bytes()
        resp = client.post(
            "/predict/batch",
            files=[("files", ("leaf.jpg", img, "image/jpeg"))],
        )
        assert resp.status_code == 200
        for item in resp.json():
            assert "top_class" in item

