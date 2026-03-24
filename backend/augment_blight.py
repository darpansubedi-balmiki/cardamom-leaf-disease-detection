"""
Augmentation script for the colletotrichum_blight class.

The blight images already have black backgrounds (bg removed), so
colour-altering transforms (ColorJitter, GaussianBlur) are intentionally
excluded to avoid corrupting the solid black background.

Configuration
-------------
SOURCE_DIR   – folder that contains the original blight images.
OUTPUT_DIR   – destination folder for the augmented dataset.
TARGET_COUNT – total number of images desired in OUTPUT_DIR (originals + augmented).
SEED         – random seed for reproducibility.

Usage
-----
python augment_blight.py
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOURCE_DIR = "colletotrichum_blight"   # folder with original blight images
OUTPUT_DIR = "colletotrichum_blight_augmented"
TARGET_COUNT = 500
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
JPEG_QUALITY = 95

# ---------------------------------------------------------------------------
# Augmentation transforms (black-background safe – no colour transforms)
# ---------------------------------------------------------------------------

AUGMENT_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=45, fill=0),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15), fill=0),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5, fill=0),
])


def collect_images(source_dir: Path) -> list[Path]:
    return sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix in IMAGE_EXTENSIONS
    )


def copy_originals(source_files: list[Path], output_dir: Path) -> list[Path]:
    """Copy original images to output_dir. Returns list of copied paths."""
    copied: list[Path] = []
    for src in tqdm(source_files, desc="Copying originals", unit="img"):
        dest = output_dir / src.name
        if not dest.exists():
            shutil.copy2(src, dest)
        copied.append(dest)
    return copied


def augment_until_target(
    source_files: list[Path],
    output_dir: Path,
    current_count: int,
    target: int,
) -> None:
    """Generate augmented images until output_dir contains `target` images."""
    needed = target - current_count
    if needed <= 0:
        print(f"Already have {current_count} images — no augmentation needed.")
        return

    print(f"Generating {needed} augmented image(s) to reach TARGET_COUNT={target}…")

    idx = 0
    with tqdm(total=needed, desc="Augmenting", unit="img") as pbar:
        while idx < needed:
            src = random.choice(source_files)
            try:
                with Image.open(src) as img:
                    img_rgb = img.convert("RGB")
                augmented = AUGMENT_TRANSFORM(img_rgb)
                stem = f"aug_{idx:05d}_{src.stem}"
                dest = output_dir / f"{stem}.jpg"
                augmented.save(dest, format="JPEG", quality=JPEG_QUALITY)
                idx += 1
                pbar.update(1)
            except Exception as exc:
                tqdm.write(f"WARNING: Failed to augment '{src.name}': {exc}")


def main() -> None:
    random.seed(SEED)

    source_dir = Path(SOURCE_DIR)
    output_dir = Path(OUTPUT_DIR)

    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"SOURCE_DIR '{source_dir}' not found. "
            "Update the SOURCE_DIR variable in this script."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = collect_images(source_dir)
    if not source_files:
        raise RuntimeError(f"No images found in '{source_dir}'.")

    print(f"Found {len(source_files)} original image(s) in '{source_dir}'.")

    # Step 1 – copy originals
    copy_originals(source_files, output_dir)
    current_count = len(collect_images(output_dir))
    print(f"After copying originals: {current_count} image(s) in '{output_dir}'.")

    # Step 2 – augment to reach TARGET_COUNT
    augment_until_target(source_files, output_dir, current_count, TARGET_COUNT)

    final_count = len(collect_images(output_dir))
    print(f"\nDone. Total images in '{output_dir}': {final_count}")


if __name__ == "__main__":
    main()
