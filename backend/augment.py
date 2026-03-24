"""
Augments the colletotrichum_blight class to balance with other classes.
Run this ONCE before training:  python augment_blight.py
"""
from pathlib import Path
from PIL import Image, ImageFilter
import random, shutil
from torchvision import transforms

# ── Config ────────────────────────────────────────────────────────────────────
SOURCE_DIR   = Path("dataset/new/Blight1000")   # your raw blight folder
OUTPUT_DIR   = Path("dataset/new/Blight1000_augmented")
TARGET_COUNT = 600   # how many total images you want after augmentation
SEED         = 42
# ──────────────────────────────────────────────────────────────────────────────

random.seed(SEED)

augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Copy originals first
originals = list(SOURCE_DIR.glob("*.jpg")) + list(SOURCE_DIR.glob("*.png")) + list(SOURCE_DIR.glob("*.jpeg"))
print(f"Found {len(originals)} original Blight images")

for img_path in originals:
    shutil.copy(img_path, OUTPUT_DIR / img_path.name)

# Generate augmented images until we hit target
count = len(originals)
idx   = 0
while count < TARGET_COUNT:
    src = random.choice(originals)
    img = Image.open(src).convert("RGB")
    aug = augment(img)
    out_name = f"aug_{idx:04d}_{src.stem}.jpg"
    aug.save(OUTPUT_DIR / out_name, quality=95)
    count += 1
    idx   += 1

print(f"✅ Done! {count} Blight images saved to {OUTPUT_DIR}")
print(f"   Original: {len(originals)}  |  Generated: {count - len(originals)}")