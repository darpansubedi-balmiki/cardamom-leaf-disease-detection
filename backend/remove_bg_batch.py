"""
Batch background removal script using rembg.

Removes backgrounds from all images in a source folder and saves them
with pure black backgrounds to an output folder.

Usage:
    python remove_bg_batch.py --input dataset_raw/Healthy_1000 --output dataset_processed/healthy
    python remove_bg_batch.py --input dataset_raw/Healthy_1000 --output dataset_processed/healthy --limit 500
"""

import argparse
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

try:
    from rembg import remove
except ImportError:
    print("Error: rembg is not installed. Run: pip install rembg")
    sys.exit(1)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
JPEG_QUALITY = 95


def remove_background_to_black(input_path: Path, output_path: Path) -> None:
    """Remove background from an image and save with a pure black background."""
    img = Image.open(input_path).convert("RGB")
    result = remove(img)  # returns RGBA
    if result.mode == "RGBA":
        black_bg = Image.new("RGB", result.size, (0, 0, 0))
        black_bg.paste(result, mask=result.split()[3])  # use alpha as mask
        result = black_bg
    result.save(output_path, "JPEG", quality=JPEG_QUALITY)


def collect_images(input_dir: Path) -> list[Path]:
    """Collect all image files from input_dir (non-recursive)."""
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(input_dir.glob(f"*{ext}"))
        images.extend(input_dir.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-remove backgrounds from images using rembg."
    )
    parser.add_argument(
        "--input", "-i", required=True, type=Path, help="Input folder containing images"
    )
    parser.add_argument(
        "--output", "-o", required=True, type=Path, help="Output folder for processed images"
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Maximum number of images to process (useful for class balancing)",
    )
    args = parser.parse_args()

    input_dir: Path = args.input
    output_dir: Path = args.output
    limit: int | None = args.limit

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    all_images = collect_images(input_dir)
    if not all_images:
        print(f"No images found in '{input_dir}'. Supported extensions: {IMAGE_EXTENSIONS}")
        sys.exit(1)

    if limit is not None:
        all_images = all_images[:limit]

    processed = 0
    skipped = 0
    failed = 0

    print(f"Input:  {input_dir}  ({len(all_images)} images to process)")
    print(f"Output: {output_dir}")

    for img_path in tqdm(all_images, desc="Removing backgrounds", unit="img"):
        output_path = output_dir / (img_path.stem + ".jpg")
        if output_path.exists():
            skipped += 1
            continue
        try:
            remove_background_to_black(img_path, output_path)
            processed += 1
        except Exception as exc:
            print(f"\nFailed to process '{img_path.name}': {exc}")
            failed += 1

    print()
    print("─" * 50)
    print(f"✅ Done!")
    print(f"   Processed : {processed}")
    print(f"   Skipped   : {skipped}  (already existed)")
    print(f"   Failed    : {failed}")
    print(f"   Total     : {processed + skipped + failed}")
    print(f"   Output    : {output_dir}")


if __name__ == "__main__":
    main()
