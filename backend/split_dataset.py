"""
Script to split dataset into train/val/test sets
Typical split: 70% train, 15% val, 15% test
"""
import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm

# Configuration
SOURCE_FOLDERS = [
    "healthy",
    "phyllosticta_leaf_spot", 
    "colletotrichum_blight"
]

OUTPUT_DIR = "dataset"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def create_directory_structure(output_dir, class_names):
    """Create train/val/test directory structure"""
    splits = ['train', 'val', 'test']
    
    for split in splits:
        for class_name in class_names:
            path = Path(output_dir) / split / class_name
            path.mkdir(parents=True, exist_ok=True)
            print(f"Created: {path}")


def split_and_copy_files(source_folder, output_dir, train_ratio, val_ratio, test_ratio):
    """Split files from source folder into train/val/test"""
    
    source_path = Path(source_folder)
    class_name = source_path.name
    
    # Get all image files
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    all_files = []
    for ext in extensions:
        all_files.extend(list(source_path.glob(f'*{ext}')))
    
    # Shuffle files
    random.shuffle(all_files)
    
    total_files = len(all_files)
    if total_files == 0:
        print(f"⚠️  No images found in {source_folder}")
        return
    
    # Calculate split indices
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    # Split files
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]
    
    print(f"\n📊 {class_name}:")
    print(f"   Total: {total_files} images")
    print(f"   Train: {len(train_files)} ({len(train_files)/total_files*100:.1f}%)")
    print(f"   Val:   {len(val_files)} ({len(val_files)/total_files*100:.1f}%)")
    print(f"   Test:  {len(test_files)} ({len(test_files)/total_files*100:.1f}%)")
    
    # Copy files to respective directories
    def copy_files(file_list, split_name):
        dest_dir = Path(output_dir) / split_name / class_name
        for file_path in tqdm(file_list, desc=f"Copying {split_name}", leave=False):
            shutil.copy2(file_path, dest_dir / file_path.name)
    
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')


def main():
    """Main function to split dataset"""
    
    print("=" * 60)
    print("Dataset Splitting Script")
    print("=" * 60)
    
    # Verify source folders exist
    print("\n🔍 Checking source folders...")
    for folder in SOURCE_FOLDERS:
        if not Path(folder).exists():
            print(f"❌ Error: Folder '{folder}' not found!")
            print(f"   Make sure you're running this script from the correct directory.")
            return
        else:
            print(f"✓ Found: {folder}")
    
    # Create output directory structure
    print(f"\n📁 Creating directory structure in '{OUTPUT_DIR}'...")
    create_directory_structure(OUTPUT_DIR, SOURCE_FOLDERS)
    
    # Split and copy files for each class
    print("\n📂 Splitting and copying files...")
    print("=" * 60)
    
    for folder in SOURCE_FOLDERS:
        split_and_copy_files(folder, OUTPUT_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Dataset splitting complete!")
    print("=" * 60)
    print(f"\nDataset location: {OUTPUT_DIR}/")
    print("\nDirectory structure:")
    print(f"{OUTPUT_DIR}/")
    print("├── train/")
    print("│   ├── healthy/")
    print("│   ├── phyllosticta_leaf_spot/")
    print("│   └── colletotrichum_blight/")
    print("├── val/")
    print("│   ├── healthy/")
    print("│   ├── phyllosticta_leaf_spot/")
    print("│   └── colletotrichum_blight/")
    print("└── test/")
    print("    ├── healthy/")
    print("    ├── phyllosticta_leaf_spot/")
    print("    └── colletotrichum_blight/")
    print("\nYou can now run: python train.py")


if __name__ == "__main__":
    main()