#!/usr/bin/env python3
"""
Compute severity_percent and severity_stage from leaf + lesion mask PNGs.

Usage
-----
    python scripts/compute_severity_from_masks.py \\
        --image-dir  data/images \\
        --leaf-mask-dir  data/masks/leaf \\
        --lesion-mask-dir  data/masks/lesion \\
        --output  data/severity_labels.csv \\
        [--stage-thresholds "0,10,25,50,100"]

The script expects corresponding files in each directory to share the same
base filename (stem), e.g.::

    data/images/IMG_001.jpg
    data/masks/leaf/IMG_001.png
    data/masks/lesion/IMG_001.png

For every image that has both masks present the script computes:

    severity_percent = lesion_pixels / leaf_pixels × 100

and maps it to a stage using the provided thresholds.

The output CSV has columns:
    image_path, leaf_mask_path, lesion_mask_path, severity_percent,
    severity_stage

If ``--output`` already exists the rows are merged (updated) by
``image_path``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Stage thresholds
# ---------------------------------------------------------------------------

DEFAULT_STAGE_THRESHOLDS = [0.0, 10.0, 25.0, 50.0, 100.0]


def parse_thresholds(raw: str) -> list[float]:
    parts = [s.strip() for s in raw.split(",")]
    if len(parts) != 5:
        raise ValueError(
            f"--stage-thresholds must contain exactly 5 comma-separated values; got {len(parts)}"
        )
    values = [float(p) for p in parts]
    return values


def map_percent_to_stage(percent: float, thresholds: list[float]) -> int:
    """Map a severity percentage to a stage index (0–4)."""
    percent = max(0.0, min(100.0, percent))
    if percent <= 0.0:
        return 0
    if percent <= thresholds[1]:
        return 1
    if percent <= thresholds[2]:
        return 2
    if percent <= thresholds[3]:
        return 3
    return 4


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _mask_pixels(mask_path: Path) -> int:
    """Count foreground (> 127) pixels in a grayscale mask PNG."""
    arr = np.array(Image.open(mask_path).convert("L"))
    return int((arr > 127).sum())


def compute_severity(leaf_mask: Path, lesion_mask: Path) -> float:
    """Return severity percent (0–100) given leaf and lesion mask paths."""
    leaf_px = _mask_pixels(leaf_mask)
    lesion_px = _mask_pixels(lesion_mask)
    if leaf_px == 0:
        return 0.0
    return round(100.0 * lesion_px / leaf_px, 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _find_mask(mask_dir: Path, stem: str) -> Path | None:
    """Search for a mask file matching *stem* (any image extension)."""
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
        candidate = mask_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def run(
    image_dir: Path,
    leaf_mask_dir: Path,
    lesion_mask_dir: Path,
    output_csv: Path,
    thresholds: list[float],
) -> None:
    # Collect image files
    image_files = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
    )

    if not image_files:
        print(f"No images found in {image_dir}", file=sys.stderr)
        sys.exit(1)

    # Load existing CSV rows if present (keyed by image_path)
    existing: dict[str, dict] = {}
    fieldnames = [
        "image_path",
        "leaf_mask_path",
        "lesion_mask_path",
        "severity_percent",
        "severity_stage",
    ]
    if output_csv.exists():
        with output_csv.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                existing[row["image_path"]] = row

    processed = 0
    skipped = 0

    for img_path in image_files:
        stem = img_path.stem
        leaf_mask = _find_mask(leaf_mask_dir, stem)
        lesion_mask = _find_mask(lesion_mask_dir, stem)

        if leaf_mask is None or lesion_mask is None:
            skipped += 1
            continue

        pct = compute_severity(leaf_mask, lesion_mask)
        stage = map_percent_to_stage(pct, thresholds)

        existing[str(img_path)] = {
            "image_path": str(img_path),
            "leaf_mask_path": str(leaf_mask),
            "lesion_mask_path": str(lesion_mask),
            "severity_percent": pct,
            "severity_stage": stage,
        }
        processed += 1

    # Write out
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing.values())

    print(f"Processed: {processed}  Skipped (missing masks): {skipped}")
    print(f"Output written to: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute disease severity from leaf + lesion mask PNGs."
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        type=Path,
        help="Directory containing leaf images.",
    )
    parser.add_argument(
        "--leaf-mask-dir",
        required=True,
        type=Path,
        help="Directory containing binary leaf-area mask PNGs.",
    )
    parser.add_argument(
        "--lesion-mask-dir",
        required=True,
        type=Path,
        help="Directory containing binary lesion mask PNGs.",
    )
    parser.add_argument(
        "--output",
        default="data/severity_labels.csv",
        type=Path,
        help="Output CSV file path (default: data/severity_labels.csv).",
    )
    parser.add_argument(
        "--stage-thresholds",
        default=",".join(str(t) for t in DEFAULT_STAGE_THRESHOLDS),
        help=(
            "Comma-separated stage boundary percentages "
            f"(default: {','.join(str(t) for t in DEFAULT_STAGE_THRESHOLDS)})."
        ),
    )
    args = parser.parse_args()

    thresholds = parse_thresholds(args.stage_thresholds)
    run(
        image_dir=args.image_dir,
        leaf_mask_dir=args.leaf_mask_dir,
        lesion_mask_dir=args.lesion_mask_dir,
        output_csv=args.output,
        thresholds=thresholds,
    )


if __name__ == "__main__":
    main()
