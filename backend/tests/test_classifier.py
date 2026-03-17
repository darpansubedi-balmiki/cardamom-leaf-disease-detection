"""
Tests for DiseaseClassifier – top-k output and confidence threshold logic.
"""
from __future__ import annotations

import io
import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from unittest.mock import MagicMock, patch

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


def _make_solid_image(color: tuple[int, int, int] = (100, 150, 50)) -> Image.Image:
    """Return a small solid-color RGB image."""
    return Image.new("RGB", (256, 256), color)


def _make_classifier_with_mock_logits(
    logits: list[float],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
) -> DiseaseClassifier:
    """
    Return a DiseaseClassifier whose model always outputs the given logits,
    bypassing any real weight loading.
    """
    clf = DiseaseClassifier.__new__(DiseaseClassifier)
    clf.confidence_threshold = confidence_threshold
    clf.top_k = min(top_k, len(CLASS_NAMES))
    clf.device = torch.device("cpu")

    mock_model = MagicMock(spec=nn.Module)
    # Make the mock return a fixed logit tensor
    mock_model.return_value = torch.tensor([logits], dtype=torch.float32)
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model
    clf._model = mock_model

    return clf


# ---------------------------------------------------------------------------
# Top-k output tests
# ---------------------------------------------------------------------------


class TestTopKOutput:
    def test_returns_top_k_items(self) -> None:
        """predict() should return exactly top_k items."""
        # logits ordered: class 1 > class 2 > class 0
        clf = _make_classifier_with_mock_logits([0.1, 10.0, 5.0], top_k=3)
        result = clf.predict(_make_solid_image())

        assert isinstance(result.top_k, list)
        assert len(result.top_k) == 3

    def test_top_k_ordered_by_descending_probability(self) -> None:
        """top_k list must be sorted highest-probability first."""
        clf = _make_classifier_with_mock_logits([0.1, 10.0, 5.0], top_k=3)
        result = clf.predict(_make_solid_image())

        probs = [item.probability for item in result.top_k]
        assert probs == sorted(probs, reverse=True)

    def test_top_k_probabilities_sum_to_one(self) -> None:
        """All class probabilities together must sum to ~1."""
        clf = _make_classifier_with_mock_logits([1.0, 2.0, 3.0], top_k=3)
        result = clf.predict(_make_solid_image())

        total = sum(item.probability for item in result.top_k)
        assert abs(total - 1.0) < 1e-5

    def test_top_k_class_names_are_valid(self) -> None:
        """Each item in top_k must have a recognised class name."""
        clf = _make_classifier_with_mock_logits([1.0, 2.0, 3.0], top_k=3)
        result = clf.predict(_make_solid_image())

        for item in result.top_k:
            assert item.class_name in CLASS_NAMES

    def test_top_k_respects_k_parameter(self) -> None:
        """When top_k=1, only one item is returned."""
        clf = _make_classifier_with_mock_logits([1.0, 2.0, 3.0], top_k=1)
        result = clf.predict(_make_solid_image())

        assert len(result.top_k) == 1

    def test_top_k_probability_items_are_TopKPrediction(self) -> None:
        clf = _make_classifier_with_mock_logits([1.0, 2.0, 3.0], top_k=3)
        result = clf.predict(_make_solid_image())

        for item in result.top_k:
            assert isinstance(item, TopKPrediction)


# ---------------------------------------------------------------------------
# Confidence threshold / uncertainty tests
# ---------------------------------------------------------------------------


class TestConfidenceThreshold:
    def test_high_confidence_returns_class_name(self) -> None:
        """When top-1 prob >= threshold, top_class should be the class name."""
        # logits heavily in favour of class 1 → prob ≈ 1.0
        clf = _make_classifier_with_mock_logits(
            [-100.0, 100.0, -100.0],
            confidence_threshold=0.60,
        )
        result = clf.predict(_make_solid_image())

        assert result.top_class == CLASS_NAMES[1]
        assert result.is_uncertain is False

    def test_low_confidence_returns_uncertain(self) -> None:
        """When top-1 prob < threshold, top_class should be 'Uncertain'."""
        # Near-uniform logits → each class ≈ 33 %
        clf = _make_classifier_with_mock_logits(
            [1.0, 1.0, 1.0],
            confidence_threshold=0.60,
        )
        result = clf.predict(_make_solid_image())

        assert result.top_class == "Uncertain"
        assert result.is_uncertain is True

    def test_exactly_at_threshold_is_not_uncertain(self) -> None:
        """A probability exactly equal to the threshold is not uncertain."""
        # We need prob == 0.60 exactly.  Use logits that produce a known softmax.
        # softmax([a, b, b]) = [e^a / (e^a + 2e^b), ...]
        # Choose a so e^a / (e^a + 2) = 0.60  →  e^a = 3, a = ln(3)
        import math

        a = math.log(3.0)  # ≈ 1.099
        clf = _make_classifier_with_mock_logits(
            [a, 0.0, 0.0],
            confidence_threshold=0.60,
        )
        result = clf.predict(_make_solid_image())

        # top probability should be ≈ 0.60 (within floating-point tolerance)
        assert abs(result.top_probability - 0.60) < 1e-4
        assert result.is_uncertain is False

    def test_uncertainty_flag_matches_top_class(self) -> None:
        """is_uncertain and top_class='Uncertain' must be consistent."""
        clf_low = _make_classifier_with_mock_logits(
            [1.0, 1.0, 1.0], confidence_threshold=0.90
        )
        result_low = clf_low.predict(_make_solid_image())
        assert result_low.is_uncertain == (result_low.top_class == "Uncertain")

        clf_high = _make_classifier_with_mock_logits(
            [-100.0, 100.0, -100.0], confidence_threshold=0.10
        )
        result_high = clf_high.predict(_make_solid_image())
        assert result_high.is_uncertain == (result_high.top_class == "Uncertain")

    def test_default_threshold_is_0_60(self) -> None:
        assert DEFAULT_CONFIDENCE_THRESHOLD == 0.60

    def test_top_probability_always_present(self) -> None:
        """top_probability must be populated regardless of uncertainty."""
        clf = _make_classifier_with_mock_logits([1.0, 1.0, 1.0])
        result = clf.predict(_make_solid_image())

        assert 0.0 < result.top_probability <= 1.0

    def test_top_k_still_populated_when_uncertain(self) -> None:
        """Even for uncertain predictions, top_k must contain k items."""
        clf = _make_classifier_with_mock_logits(
            [1.0, 1.0, 1.0], confidence_threshold=0.90, top_k=3
        )
        result = clf.predict(_make_solid_image())

        assert result.is_uncertain is True
        assert len(result.top_k) == 3


# ---------------------------------------------------------------------------
# PredictionResult data class
# ---------------------------------------------------------------------------


class TestPredictionResult:
    def test_dataclass_fields(self) -> None:
        result = PredictionResult(
            top_class="Healthy",
            top_probability=0.85,
            is_uncertain=False,
            top_k=[TopKPrediction("Healthy", 0.85)],
        )
        assert result.top_class == "Healthy"
        assert result.top_probability == 0.85
        assert result.is_uncertain is False
        assert len(result.top_k) == 1
