"""
Augments colletotrichum_blight images to balance with other classes.
Run ONCE after background removal:  python augment_blight.py

Assumes blight images already have BLACK backgrounds.
Does NOT use ColorJitter (would corrupt black background).
"""

import random
import shutil
from pathlib import Path

from PIL import Image
from torchvision import transforms

# ── Config ────────────────────────────────────────────────────────────────────
SOURCE_DIR = Path("dataset_processed/colletotrichum_blight")
OUTPUT_DIR = Path("dataset_processed/colletotrichum_blight_augmented")
TARGET_COUNT = 500
SEED         = 42
JPEG_QUALITY = 95
# ──────────────────────────────────────────────────────────────────────────────

random.seed(SEED)

# fill=0 keeps background pure black on all geometric transforms
augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=45, fill=0),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=0),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5, fill=0),
    # NO ColorJitter — corrupts black background
    # NO GaussianBlur — blurs sharp leaf-background edge
])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

originals = (
    list(SOURCE_DIR.glob("*.jpg"))
    + list(SOURCE_DIR.glob("*.png"))
    + list(SOURCE_DIR.glob("*.jpeg"))
)
print(f"Found {len(originals)} original Blight images")

if len(originals) == 0:
    print("❌ No images found. Make sure you ran remove_bg_batch.py first.")
    raise SystemExit(1)

# Copy originals first
for img_path in originals:
    shutil.copy(img_path, OUTPUT_DIR / img_path.name)

count = len(originals)
idx = 0
while count < TARGET_COUNT:
    src = random.choice(originals)
    img = Image.open(src).convert("RGB")
    aug = augment(img)
    out_name = f"aug_{idx:04d}_{src.stem}.jpg"
    aug.save(OUTPUT_DIR / out_name, quality=JPEG_QUALITY)
    count += 1
    idx += 1

print(f"✅ Done! {count} Blight images in {OUTPUT_DIR}")
print(f"   Original: {len(originals)}  |  Generated: {count - len(originals)}")
