"""
Severity estimation utilities for cardamom leaf disease detection.

Provides a heuristic method that uses the Grad-CAM heatmap to approximate
the percentage of leaf area affected, then maps that to a severity stage.

Stage mapping (default thresholds: 0, 10, 25, 50, 100):
  Stage 0 – 0 %         (healthy / no lesion)
  Stage 1 – 1–10 %      (mild)
  Stage 2 – 11–25 %     (moderate)
  Stage 3 – 26–50 %     (severe)
  Stage 4 – > 50 %      (very severe)

Environment variables
---------------------
SEVERITY_HEATMAP_THRESHOLD  float, default 0.6
    Pixels in the Grad-CAM heatmap above this normalised value are treated
    as "affected".

SEVERITY_STAGE_THRESHOLDS  comma-separated floats, default "0,10,25,50,100"
    Boundary percentages used to map severity_percent → severity_stage.
    Must be 5 values in ascending order starting with 0 and ending with 100.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Defaults / configuration
# ---------------------------------------------------------------------------

DEFAULT_HEATMAP_THRESHOLD: float = 0.6
DEFAULT_STAGE_THRESHOLDS: list[float] = [0.0, 10.0, 25.0, 50.0, 100.0]


def _parse_stage_thresholds(raw: str) -> list[float]:
    """Parse a comma-separated string into a list of 5 stage-boundary floats."""
    parts = [s.strip() for s in raw.split(",")]
    if len(parts) != 5:
        raise ValueError(
            f"SEVERITY_STAGE_THRESHOLDS must contain exactly 5 values; got {len(parts)}"
        )
    return [float(p) for p in parts]


def get_heatmap_threshold() -> float:
    """Return the configured heatmap binarisation threshold (0–1)."""
    raw = os.environ.get("SEVERITY_HEATMAP_THRESHOLD", str(DEFAULT_HEATMAP_THRESHOLD))
    return float(raw)


def get_stage_thresholds() -> list[float]:
    """Return the configured stage boundary thresholds (5 values)."""
    raw = os.environ.get(
        "SEVERITY_STAGE_THRESHOLDS",
        ",".join(str(t) for t in DEFAULT_STAGE_THRESHOLDS),
    )
    return _parse_stage_thresholds(raw)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SeverityResult:
    """Severity estimation output."""

    severity_stage: int  # 0–4
    severity_percent: float  # 0–100
    severity_method: str  # "none" | "heuristic" | "ordinal_model" | "segmentation"
    warning: str | None = None


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def map_percent_to_stage(
    percent: float,
    thresholds: list[float] | None = None,
) -> int:
    """Map a severity percentage (0–100) to a stage index (0–4).

    Args:
        percent:    Severity percentage in [0, 100].
        thresholds: Five boundary values [t0, t1, t2, t3, t4] where
                    ``t0`` must be 0 and ``t4`` must be 100.
                    Defaults to ``DEFAULT_STAGE_THRESHOLDS``.

    Returns:
        Stage integer 0–4:
          - 0 if percent == 0
          - 1 if 0  < percent <= thresholds[1]
          - 2 if thresholds[1] < percent <= thresholds[2]
          - 3 if thresholds[2] < percent <= thresholds[3]
          - 4 if percent > thresholds[3]
    """
    if thresholds is None:
        thresholds = DEFAULT_STAGE_THRESHOLDS

    if len(thresholds) != 5:
        raise ValueError(f"thresholds must have exactly 5 elements; got {len(thresholds)}")

    percent = max(0.0, min(100.0, float(percent)))

    if percent <= 0.0:
        return 0
    if percent <= thresholds[1]:
        return 1
    if percent <= thresholds[2]:
        return 2
    if percent <= thresholds[3]:
        return 3
    return 4


def compute_severity_from_heatmap(
    heatmap: np.ndarray,
    threshold: float | None = None,
    thresholds: list[float] | None = None,
) -> SeverityResult:
    """Estimate severity from a Grad-CAM heatmap using a simple threshold heuristic.

    Pixels with normalised heatmap value above *threshold* are considered
    "affected".  The ratio of affected pixels to total pixels is used as the
    severity percentage.

    Args:
        heatmap:    2-D numpy array with values in [0, 1].
        threshold:  Binarisation threshold in [0, 1].  Defaults to the
                    value from :func:`get_heatmap_threshold`.
        thresholds: Stage boundary thresholds.  Defaults to the value
                    from :func:`get_stage_thresholds`.

    Returns:
        :class:`SeverityResult` with ``severity_method="heuristic"`` and a
        warning that the estimate is approximate.
    """
    if threshold is None:
        threshold = get_heatmap_threshold()
    if thresholds is None:
        thresholds = get_stage_thresholds()

    heatmap = np.asarray(heatmap, dtype=float)

    # Normalise to [0, 1] if needed
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max > h_min:
        heatmap_norm = (heatmap - h_min) / (h_max - h_min)
    else:
        heatmap_norm = heatmap.copy()

    binary_mask = heatmap_norm > threshold
    total_pixels = binary_mask.size
    affected_pixels = int(binary_mask.sum())

    percent = (affected_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0
    stage = map_percent_to_stage(percent, thresholds)

    return SeverityResult(
        severity_stage=stage,
        severity_percent=round(percent, 2),
        severity_method="heuristic",
        warning=(
            "Severity estimated from Grad-CAM heatmap (heuristic). "
            "This is an approximation and may not reflect true lesion area."
        ),
    )
