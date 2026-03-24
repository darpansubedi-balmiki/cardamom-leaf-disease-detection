"""
Batch background removal script using rembg.

Processes all images in --input folder, removes the background,
composites onto a black RGB background, and saves as JPEG to --output.

Resume support: skips images that already exist in --output.

Usage examples
--------------
python remove_bg_batch.py --input dataset_raw/Healthy_1000 --output dataset_processed/healthy --limit 500
python remove_bg_batch.py --input dataset_raw/Phyllosticta_LS_1000 --output dataset_processed/phyllosticta_leaf_spot --limit 500
python remove_bg_batch.py --input dataset_raw/other/Apple --output dataset_processed/other --limit 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import rembg
except ImportError:
    print("Error: rembg is not installed. Run: pip install rembg", file=sys.stderr)
    sys.exit(1)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".webp", ".WEBP"}
JPEG_QUALITY = 95


def composite_on_black(rgba_image: Image.Image) -> Image.Image:
    """Composite an RGBA image onto a solid black background, returning RGB."""
    rgba = rgba_image.convert("RGBA")
    background = Image.new("RGB", rgba.size, (0, 0, 0))
    # Use the alpha channel as the paste mask
    r, g, b, a = rgba.split()
    background.paste(Image.merge("RGB", (r, g, b)), mask=a)
    return background


def remove_bg_batch(input_dir: Path, output_dir: Path, limit: int | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all image files
    all_files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix in IMAGE_EXTENSIONS
    )

    if not all_files:
        print(f"No images found in {input_dir}")
        return

    # Apply limit
    if limit is not None and limit > 0:
        all_files = all_files[:limit]

    # Filter out already-processed images (resume support)
    to_process = []
    skipped = 0
    for src in all_files:
        dest = output_dir / (src.stem + ".jpg")
        if dest.exists():
            skipped += 1
        else:
            to_process.append(src)

    if skipped:
        print(f"Skipping {skipped} already-processed image(s).")

    if not to_process:
        print("Nothing to process.")
        return

    print(f"Processing {len(to_process)} image(s) from '{input_dir}' → '{output_dir}'")

    # Initialise a single rembg session for efficiency
    session = rembg.new_session()

    for src in tqdm(to_process, unit="img"):
        try:
            with Image.open(src) as img:
                # rembg.remove() accepts a PIL Image and returns a PIL Image (RGBA)
                rgba_result = rembg.remove(img, session=session)
            rgb_result = composite_on_black(rgba_result)
            dest = output_dir / (src.stem + ".jpg")
            rgb_result.save(dest, format="JPEG", quality=JPEG_QUALITY)
        except Exception as exc:
            tqdm.write(f"WARNING: Failed to process '{src.name}': {exc}")

    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch remove backgrounds from images using rembg.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Source folder containing input images.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination folder for processed images.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to process (default: all).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.input.is_dir():
        print(f"Error: --input '{args.input}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    remove_bg_batch(args.input, args.output, args.limit)
