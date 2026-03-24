"""
Script to split dataset into train/val/test sets.
Reads from dataset_processed/{blight,healthy,other,spot}
Writes to   dataset/{train,val,test}/{blight,healthy,other,spot}

Typical split: 70% train, 15% val, 15% test
"""
import shutil
from pathlib import Path
import random
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
SOURCE_DIR = "dataset_processed"
CLASS_FOLDERS = ["blight", "healthy", "other", "spot"]

OUTPUT_DIR = "dataset"
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
# ──────────────────────────────────────────────────────────────────────────────


def create_directory_structure(output_dir: str, class_names: list) -> None:
    """Create train/val/test directory structure."""
    for split in ("train", "val", "test"):
        for cls in class_names:
            path = Path(output_dir) / split / cls
            path.mkdir(parents=True, exist_ok=True)
            print(f"  Created: {path}")


def split_and_copy(source_dir: str, class_name: str, output_dir: str,
                   train_ratio: float, val_ratio: float) -> None:
    """Split images from source_dir/class_name into train/val/test."""
    source_path = Path(source_dir) / class_name

    extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    all_files = [f for f in source_path.iterdir() if f.suffix in extensions]
    random.shuffle(all_files)

    total = len(all_files)
    if total == 0:
        print(f"⚠️  No images found in {source_path}")
        return

    train_end = int(total * train_ratio)
    val_end   = train_end + int(total * val_ratio)

    splits = {
        "train": all_files[:train_end],
        "val":   all_files[train_end:val_end],
        "test":  all_files[val_end:],
    }

    print(f"\n📊 {class_name}:")
    print(f"   Total: {total}")
    for split_name, files in splits.items():
        pct = len(files) / total * 100
        print(f"   {split_name.capitalize():5s}: {len(files):4d}  ({pct:.1f}%)")

    for split_name, files in splits.items():
        dest = Path(output_dir) / split_name / class_name
        for fp in tqdm(files, desc=f"  → {split_name}", leave=False):
            shutil.copy2(fp, dest / fp.name)


def main() -> None:
    print("=" * 60)
    print("Dataset Splitting Script")
    print("=" * 60)

    # Verify source folders exist
    print(f"\n🔍 Checking source folders in '{SOURCE_DIR}/'...")
    missing = []
    for cls in CLASS_FOLDERS:
        p = Path(SOURCE_DIR) / cls
        if p.exists():
            count = len([f for f in p.iterdir()
                         if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            print(f"  ✓ {p}  ({count} images)")
        else:
            print(f"  ❌ {p}  — NOT FOUND")
            missing.append(str(p))

    if missing:
        print("\n❌ Missing source folders:")
        for m in missing:
            print(f"   {m}")
        print("\nPlease create them and add images before running this script.")
        return

    # Create output structure
    print(f"\n📁 Creating output structure in '{OUTPUT_DIR}/'...")
    create_directory_structure(OUTPUT_DIR, CLASS_FOLDERS)

    # Split each class
    print("\n📂 Splitting images...")
    print("=" * 60)
    for cls in CLASS_FOLDERS:
        split_and_copy(SOURCE_DIR, cls, OUTPUT_DIR, TRAIN_RATIO, VAL_RATIO)

    # Summary
    print("\n" + "=" * 60)
    print("✅ Dataset splitting complete!")
    print("=" * 60)
    print(f"\n{OUTPUT_DIR}/")
    for split in ("train", "val", "test"):
        print(f"├── {split}/")
        for cls in CLASS_FOLDERS:
            print(f"│   ├── {cls}/")
    print("\nYou can now run:  python train.py")


if __name__ == "__main__":
    main()