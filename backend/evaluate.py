"""
Evaluate trained Cardamom Leaf Disease model on dataset/test
Prints accuracy + per-class report + confusion matrix.
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import numpy as np

# --------- Configuration (match train.py) ----------
class Config:
    DATASET_PATH = "dataset"
    MODEL_PATH = "models/cardamom_model.pt"

    BATCH_SIZE = 32
    NUM_CLASSES = 4
    IMG_SIZE = 224

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

# --------- Model definition (same as train.py) ----------
def create_model():
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, Config.NUM_CLASSES)
    )
    return model

def get_test_transforms():
    return transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def confusion_matrix_numpy(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def classification_report_from_cm(cm, class_names):
    # precision, recall, f1 per class
    report_lines = []
    report_lines.append("Per-class metrics:")
    report_lines.append("-" * 72)
    report_lines.append(f"{'class':28s}  {'precision':>9s}  {'recall':>9s}  {'f1':>9s}  {'support':>9s}")
    report_lines.append("-" * 72)

    eps = 1e-12
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        report_lines.append(f"{name:28s}  {precision:9.3f}  {recall:9.3f}  {f1:9.3f}  {support:9d}")

    report_lines.append("-" * 72)
    acc = cm.trace() / (cm.sum() + eps)
    report_lines.append(f"{'overall accuracy':28s}  {acc:9.3f}")
    return "\n".join(report_lines)

def main():
    dataset_path = Path(Config.DATASET_PATH) / "test"
    if not dataset_path.exists():
        print(f"❌ test dataset not found at: {dataset_path.absolute()}")
        sys.exit(1)

    # Load test dataset
    test_dataset = datasets.ImageFolder(
        root=str(dataset_path),
        transform=get_test_transforms()
    )
    class_names = test_dataset.classes
    print(f"✅ Found test samples: {len(test_dataset)}")
    print(f"✅ Classes: {class_names}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )

    # Load model
    model = create_model().to(Config.DEVICE)
    model_path = Path(Config.MODEL_PATH)
    if not model_path.exists():
        print(f"❌ model file not found: {model_path.absolute()}")
        print("💡 Train first: python train.py")
        sys.exit(1)

    state = torch.load(model_path, map_location=Config.DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"✅ Loaded model from: {model_path}")

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix_numpy(y_true, y_pred, num_classes=len(class_names))

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)

    print("\n" + classification_report_from_cm(cm, class_names))

if __name__ == "__main__":
    main()