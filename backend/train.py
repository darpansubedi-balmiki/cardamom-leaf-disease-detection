"""
Training script for Cardamom Leaf Disease Classification
Uses EfficientNetV2 with transfer learning
"""
import sys
from pathlib import Path

# Check imports with helpful error messages
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
except ImportError as e:
    print(f"\n❌ Error importing PyTorch: {e}")
    print("💡 Fix: pip install torch torchvision")
    print("📚 Or run: pip install -r requirements.txt")
    sys.exit(1)

try:
    from torchvision import datasets, transforms, models
except ImportError as e:
    print(f"\n❌ Error importing torchvision: {e}")
    print("💡 Fix: pip install torchvision")
    print("📚 Or run: pip install -r requirements.txt")
    sys.exit(1)

try:
    import numpy as np
except ImportError as e:
    print(f"\n❌ Error importing numpy: {e}")
    print("💡 Fix: pip install numpy")
    print("📚 Or run: pip install -r requirements.txt")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError as e:
    print(f"\n❌ Error importing tqdm: {e}")
    print("💡 Fix: pip install tqdm")
    print("📚 Or run: pip install -r requirements.txt")
    print("\n⚠️  You may need to pull latest changes:")
    print("   git pull")
    print("   pip install -r requirements.txt")
    sys.exit(1)

import time

# Configuration
class Config:
    # Paths
    DATASET_PATH = "dataset"  # Change this to your dataset path
    MODEL_SAVE_PATH = "models/cardamom_model.pt"
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    NUM_CLASSES = 3
    IMG_SIZE = 224
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Early stopping
    PATIENCE = 10
    
    # Class names
    CLASS_NAMES = ["colletotrichum_blight", "phyllosticta_leaf_spot", "healthy"]


def get_data_transforms():
    """Define data augmentation and normalization"""
    
    train_transforms = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms


def create_model():
    """Create EfficientNetV2-S model with custom classifier"""
    
    # Load pre-trained EfficientNetV2-S
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    
    # Replace classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),           # Removed inplace=True for MPS compatibility
        nn.Linear(num_features, 512),
        nn.ReLU(),                   # Removed inplace=True for MPS compatibility
        nn.Dropout(p=0.2),           # Removed inplace=True for MPS compatibility
        nn.Linear(512, Config.NUM_CLASSES)
    )
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def train_model():
    """Main training function"""
    
    # Pre-flight checks
    print("\n" + "=" * 60)
    print("PRE-TRAINING CHECKS")
    print("=" * 60)
    
    # Check dataset directory exists
    dataset_path = Path(Config.DATASET_PATH)
    if not dataset_path.exists():
        print(f"\n❌ ERROR: Dataset directory not found!")
        print(f"   Expected: {dataset_path.absolute()}")
        print("\n💡 Solutions:")
        print("   1. If you have data but haven't organized it:")
        print("      Run: python split_dataset.py")
        print("   2. Make sure you're in the 'backend' directory")
        print("   3. Check that dataset/ folder exists with train/val/test subdirectories")
        print("\n📚 See: TRAINING_YOUR_MODEL.md for details")
        sys.exit(1)
    
    # Check dataset structure
    required_splits = ["train", "val", "test"]
    for split in required_splits:
        split_path = dataset_path / split
        if not split_path.exists():
            print(f"\n❌ ERROR: Missing dataset split: {split}/")
            print(f"   Expected: {split_path.absolute()}")
            print("\n💡 Run: python split_dataset.py")
            sys.exit(1)
    
    print("✅ Dataset directory structure OK")
    
    # Check models output directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print(f"✅ Output directory OK: {models_dir.absolute()}")
    
    # Display device info
    print(f"✅ Device: {Config.DEVICE}")
    
    # Test device
    try:
        test_tensor = torch.randn(1, 3, 224, 224).to(Config.DEVICE)
        print(f"✅ Device test passed")
        del test_tensor  # Free memory
    except Exception as e:
        print(f"\n⚠️  Warning: Device test failed: {e}")
        print("   Falling back to CPU")
        Config.DEVICE = torch.device("cpu")
    
    print("=" * 60 + "\n")
    
    print(f"Using device: {Config.DEVICE}")
    print(f"Dataset path: {dataset_path.absolute()}")
    
    # Create data transforms
    train_transforms, val_transforms = get_data_transforms()
    
    # Load datasets
    print("\nLoading datasets...")
    try:
        train_dataset = datasets.ImageFolder(
            root=f"{Config.DATASET_PATH}/train",
            transform=train_transforms
        )
        
        val_dataset = datasets.ImageFolder(
            root=f"{Config.DATASET_PATH}/val",
            transform=val_transforms
        )
    except Exception as e:
        print(f"\n❌ ERROR loading datasets: {e}")
        print("\n💡 Make sure your dataset has this structure:")
        print("   dataset/")
        print("   ├── train/")
        print("   │   ├── colletotrichum_blight/")
        print("   │   ├── phyllosticta_leaf_spot/")
        print("   │   └── healthy/")
        print("   └── val/")
        print("       ├── colletotrichum_blight/")
        print("       ├── phyllosticta_leaf_spot/")
        print("       └── healthy/")
        sys.exit(1)
    
    print(f"\nDataset loaded:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=False  # MPS doesn't support pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=False  # MPS doesn't support pin_memory
    )
    
    # Create model
    model = create_model().to(Config.DEVICE)
    print(f"\nModel created: EfficientNetV2-S")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print(f"\nStarting training for {Config.NUM_EPOCHS} epochs...")
    print("=" * 60)
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, Config.DEVICE)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc * 100:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Create models directory if it doesn't exist
            Path("models").mkdir(exist_ok=True)
            
            # Save model
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"✓ Best model saved! (Val Acc: {val_acc * 100:.2f}%)")
        else:
            patience_counter += 1
            print(f"✗ No improvement. Patience: {patience_counter}/{Config.PATIENCE}")
        
        # Early stopping
        if patience_counter >= Config.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    print("\n" + "=" * 60)
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc * 100:.2f}%")
    print(f"Model saved to: {Config.MODEL_SAVE_PATH}")
    
    return model, history


if __name__ == "__main__":
    model, history = train_model()
    
    # Optional: Plot training history
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot([x * 100 for x in history['train_acc']], label='Train Acc')
        ax2.plot([x * 100 for x in history['val_acc']], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: training_history.png")
        
    except ImportError:
        print("Matplotlib not installed. Skipping plot generation.")