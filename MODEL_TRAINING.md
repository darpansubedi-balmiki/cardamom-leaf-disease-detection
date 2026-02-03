# Model Training Guide for Cardamom Leaf Disease Detection

## Current Status: Untrained Model ⚠️

**Important**: The system currently uses randomly initialized weights (untrained model). This means:
- ❌ Predictions are essentially random
- ❌ Low confidence scores (typically 20-40%)
- ❌ Incorrect classifications
- ❌ Not suitable for production use

## Why Low Accuracy?

The model you're experiencing (35.62% confidence, incorrect prediction) is using **random weights** because no trained model file exists at `backend/models/cardamom_model.pt`.

From the code (cardamom_model.py):
```python
if model_file.exists():
    # Load trained weights
else:
    print("Using randomly initialized weights (placeholder model)")
```

## What You Need

### 1. Dataset Requirements

To train an accurate model, you need a labeled dataset with:

- **Minimum images per class**: 200-500 images
- **Recommended**: 1000+ images per class
- **Classes**:
  - Colletotrichum Blight (कोलेटोट्रिकम ब्लाइट)
  - Phyllosticta Leaf Spot (फाइलोस्टिक्टा पात दाग)
  - Healthy (स्वस्थ)

**Dataset Structure**:
```
dataset/
├── train/
│   ├── colletotrichum_blight/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── phyllosticta_leaf_spot/
│   │   ├── img001.jpg
│   │   └── ...
│   └── healthy/
│       ├── img001.jpg
│       └── ...
├── val/
│   ├── colletotrichum_blight/
│   ├── phyllosticta_leaf_spot/
│   └── healthy/
└── test/
    ├── colletotrichum_blight/
    ├── phyllosticta_leaf_spot/
    └── healthy/
```

### 2. Image Quality Guidelines

- **Resolution**: At least 224x224 pixels (higher is better)
- **Format**: JPG, PNG
- **Lighting**: Varied lighting conditions
- **Angles**: Multiple angles of leaves
- **Background**: Natural backgrounds, varied
- **Disease stages**: Include early, mid, and late stages

### 3. Data Collection Tips

- Take photos in field conditions (realistic)
- Include various leaf ages
- Capture different times of day
- Ensure clear focus on disease symptoms
- Avoid blurry or dark images
- Label carefully - accuracy depends on correct labels

## Training Process

### Step 1: Prepare Environment

```bash
cd backend
source venv/bin/activate  # Or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Additional training dependencies
pip install tensorboard scikit-learn matplotlib seaborn
```

### Step 2: Create Training Script

Create `backend/train.py`:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np
from tqdm import tqdm

from app.models.cardamom_model import CardamomClassifier

# Configuration
DATASET_PATH = "path/to/your/dataset"
MODEL_SAVE_PATH = "models/cardamom_model.pt"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
IMAGE_SIZE = 224

# Data augmentation and normalization
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(
    root=f"{DATASET_PATH}/train",
    transform=train_transform
)
val_dataset = datasets.ImageFolder(
    root=f"{DATASET_PATH}/val",
    transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CardamomClassifier(num_classes=3).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# Training loop
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * train_correct / train_total})
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    train_acc = 100. * train_correct / train_total
    val_acc = 100. * val_correct / val_total
    
    print(f"\nEpoch {epoch+1}:")
    print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        Path("models").mkdir(exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"✓ Saved best model with validation accuracy: {val_acc:.2f}%")
    
    scheduler.step(val_loss)

print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
```

### Step 3: Run Training

```bash
python train.py
```

Monitor the training:
- Training should show decreasing loss
- Validation accuracy should increase
- Target: >90% validation accuracy for production use

### Step 4: Evaluate Model

Create `backend/evaluate.py`:

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from app.models.cardamom_model import CardamomClassifier

# Load test dataset
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(
    root="path/to/dataset/test",
    transform=test_transform
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CardamomClassifier(num_classes=3).to(device)
model.load_state_dict(torch.load("models/cardamom_model.pt", map_location=device))
model.eval()

# Evaluate
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted.cpu().numpy())

# Print classification report
class_names = ['Colletotrichum Blight', 'Phyllosticta Leaf Spot', 'Healthy']
print(classification_report(all_labels, all_predictions, target_names=class_names))

# Plot confusion matrix
cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as confusion_matrix.png")
```

### Step 5: Deploy Trained Model

Once training is complete:

1. **Copy trained model** to production:
   ```bash
   cp models/cardamom_model.pt backend/models/
   ```

2. **Restart the backend server**:
   ```bash
   cd backend
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Verify model is loaded**:
   Check the server logs for:
   ```
   ✓ Loaded trained model weights from models/cardamom_model.pt
   ```

## Expected Results After Training

With a properly trained model, you should see:

- ✅ **Accuracy**: >90% on validation set
- ✅ **Confidence**: >80% for correct predictions
- ✅ **Correct classifications**: Matching the actual disease
- ✅ **Grad-CAM**: Highlighting disease-affected areas

## Data Sources

### Where to Get Training Data

1. **Field Collection**: 
   - Partner with agricultural institutions
   - Visit cardamom farms
   - Document disease cases with expert verification

2. **Public Datasets**:
   - PlantVillage (general plant diseases)
   - Agricultural research institutions
   - University agricultural departments

3. **Data Augmentation**:
   - Use rotation, flipping, color jittering
   - Synthetic data generation (carefully)
   - Transfer learning from similar crops

## Transfer Learning (Recommended)

For faster training with limited data:

```python
from torchvision import models

# Load pretrained EfficientNetV2
backbone = models.efficientnet_v2_s(weights='IMAGENET1K_V1')

# Fine-tune only the classifier layers
for param in backbone.features.parameters():
    param.requires_grad = False

# Train with smaller learning rate
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)
```

## Common Issues

### 1. Overfitting
- **Symptom**: High training accuracy, low validation accuracy
- **Solution**: 
  - Add more dropout
  - Increase data augmentation
  - Collect more diverse data

### 2. Underfitting
- **Symptom**: Both training and validation accuracy are low
- **Solution**:
  - Train for more epochs
  - Increase model capacity
  - Check data quality

### 3. Class Imbalance
- **Symptom**: Model predicts one class very frequently
- **Solution**:
  - Balance dataset (equal images per class)
  - Use weighted loss function
  - Oversample minority classes

## Monitoring Training

Use TensorBoard for visualization:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/cardamom_training')
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Accuracy/train', train_acc, epoch)
writer.add_scalar('Accuracy/val', val_acc, epoch)
```

View in browser:
```bash
tensorboard --logdir=runs
```

## Next Steps

1. **Collect Dataset**: Gather 500-1000 images per class
2. **Label Carefully**: Ensure accurate disease identification
3. **Train Model**: Follow the training script above
4. **Evaluate**: Achieve >90% accuracy
5. **Deploy**: Replace placeholder model with trained model
6. **Monitor**: Track real-world performance
7. **Iterate**: Retrain with new data as needed

## Questions?

For model training assistance:
- Review PyTorch tutorials: https://pytorch.org/tutorials/
- Check EfficientNet documentation
- Consult with agricultural domain experts for disease identification

---

**Remember**: A machine learning model is only as good as its training data. Invest time in collecting high-quality, diverse, and accurately labeled images for best results.
