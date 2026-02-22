#!/usr/bin/env python3
"""
Pre-training Environment Checker
Verifies that all requirements are met before starting training
"""

import sys
from pathlib import Path

def check_imports():
    """Check if all required packages can be imported"""
    print("=" * 60)
    print("CHECKING IMPORTS")
    print("=" * 60)
    
    required_packages = [
        ("torch", "PyTorch"),
        ("torchvision", "torchvision"),
        ("numpy", "NumPy"),
        ("tqdm", "tqdm"),
        ("matplotlib", "Matplotlib"),
        ("PIL", "Pillow")
    ]
    
    all_ok = True
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name:20s} - OK")
        except ImportError as e:
            print(f"❌ {name:20s} - MISSING")
            print(f"   Error: {e}")
            all_ok = False
    
    if not all_ok:
        print("\n⚠️  Some packages are missing!")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All imports successful!\n")
    return True


def check_torch_details():
    """Check PyTorch installation details"""
    print("=" * 60)
    print("PYTORCH DETAILS")
    print("=" * 60)
    
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"Python version: {sys.version.split()[0]}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps'):
            print(f"MPS available: {torch.backends.mps.is_available()}")
            if torch.backends.mps.is_available():
                print(f"MPS built: {torch.backends.mps.is_built()}")
        
        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        print(f"\n🖥️  Training will use: {device.upper()}")
        
        # Test device
        try:
            test_tensor = torch.randn(1, 3, 224, 224).to(device)
            print(f"✅ Device test passed - {device} is working\n")
            return True
        except Exception as e:
            print(f"❌ Device test failed: {e}")
            print(f"   Will fall back to CPU\n")
            return True  # Still OK, will use CPU
            
    except Exception as e:
        print(f"❌ Error checking PyTorch: {e}\n")
        return False


def check_dataset():
    """Check if dataset directory structure is correct"""
    print("=" * 60)
    print("CHECKING DATASET")
    print("=" * 60)
    
    dataset_path = Path("dataset")
    
    if not dataset_path.exists():
        print("❌ Dataset directory not found!")
        print(f"   Expected path: {dataset_path.absolute()}")
        print("\n📝 Dataset should have this structure:")
        print("   dataset/")
        print("   ├── train/")
        print("   │   ├── colletotrichum_blight/")
        print("   │   ├── phyllosticta_leaf_spot/")
        print("   │   └── healthy/")
        print("   ├── val/")
        print("   │   ├── colletotrichum_blight/")
        print("   │   ├── phyllosticta_leaf_spot/")
        print("   │   └── healthy/")
        print("   └── test/")
        print("       ├── colletotrichum_blight/")
        print("       ├── phyllosticta_leaf_spot/")
        print("       └── healthy/")
        print("\n💡 If you have data but haven't split it:")
        print("   Run: python split_dataset.py")
        return False
    
    # Check subdirectories
    required_splits = ["train", "val", "test"]
    required_classes = ["colletotrichum_blight", "phyllosticta_leaf_spot", "healthy"]
    
    all_ok = True
    for split in required_splits:
        split_path = dataset_path / split
        if not split_path.exists():
            print(f"❌ Missing: {split}/")
            all_ok = False
        else:
            print(f"✅ Found: {split}/")
            
            # Check classes
            for class_name in required_classes:
                class_path = split_path / class_name
                if not class_path.exists():
                    print(f"   ❌ Missing class: {split}/{class_name}/")
                    all_ok = False
                else:
                    # Count images
                    images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.jpeg")) + list(class_path.glob("*.png"))
                    print(f"   ✅ {class_name:30s} - {len(images):4d} images")
    
    if not all_ok:
        print("\n❌ Dataset structure is incomplete")
        print("   Run: python split_dataset.py")
        return False
    
    print("\n✅ Dataset structure looks good!\n")
    return True


def check_output_directory():
    """Check if models output directory exists"""
    print("=" * 60)
    print("CHECKING OUTPUT DIRECTORY")
    print("=" * 60)
    
    models_path = Path("models")
    
    if not models_path.exists():
        print(f"⚠️  Models directory not found, will create it")
        try:
            models_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {models_path.absolute()}\n")
            return True
        except Exception as e:
            print(f"❌ Could not create models directory: {e}\n")
            return False
    else:
        print(f"✅ Models directory exists: {models_path.absolute()}\n")
        return True


def main():
    """Run all checks"""
    print("\n" + "=" * 60)
    print("TRAINING ENVIRONMENT CHECK")
    print("=" * 60)
    print()
    
    checks = [
        ("Imports", check_imports),
        ("PyTorch", check_torch_details),
        ("Dataset", check_dataset),
        ("Output Directory", check_output_directory)
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20s}: {status}")
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("✅ ALL CHECKS PASSED!")
        print("\n🚀 You're ready to train!")
        print("   Run: python train.py")
    else:
        print("❌ SOME CHECKS FAILED")
        print("\n🔧 Fix the issues above before training")
        print("\n📚 Common fixes:")
        print("   - Missing packages: pip install -r requirements.txt")
        print("   - Missing dataset: python split_dataset.py")
        print("   - Check documentation: TRAINING_YOUR_MODEL.md")
    
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
