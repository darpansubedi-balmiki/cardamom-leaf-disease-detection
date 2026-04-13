"""
5-Fold Cross-Validation for Cardamom Leaf Disease Classification.

Loads images directly from dataset_processed/{blight,healthy,other,spot},
trains EfficientNetV2-S for each fold, and reports mean ± std metrics.

Usage:
    cd backend
    python cross_validate.py

Outputs:
    - Per-fold accuracy/precision/recall/F1 to stdout
    - models/cv_fold_{k}.pt  (best checkpoint per fold)
    - cv_results.json        (machine-readable summary)
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SOURCE_DIR = "dataset_processed"
CLASS_FOLDERS = ["blight", "healthy", "other", "spot"]
# Display names shown in reports (alphabetical, matches ImageFolder order)
CLASS_DISPLAY_NAMES = ["blight", "healthy", "other", "spot"]

NUM_CLASSES = 4
BATCH_SIZE = 32
NUM_EPOCHS = 30          # shorter per fold; adjust upward if GPU is available
LEARNING_RATE = 0.001
PATIENCE = 7
IMG_SIZE = 224
K_FOLDS = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CardamomDataset(Dataset):
    """Simple flat image dataset that reads from class subdirectories."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    def __init__(self, source_dir: str, class_folders: list[str],
                 transform=None) -> None:
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for label, cls in enumerate(class_folders):
            folder = Path(source_dir) / cls
            if not folder.exists():
                print(f"⚠️  Folder not found: {folder} – skipping")
                continue
            for fp in folder.iterdir():
                if fp.suffix in self.EXTENSIONS:
                    self.samples.append((fp, label))

        if not self.samples:
            raise RuntimeError(
                f"No images found in {source_dir}/. "
                "Run remove_bg_batch.py / augment.py first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def create_model() -> nn.Module:
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, NUM_CLASSES),
    )
    return model

# ---------------------------------------------------------------------------
# Per-class metrics (no sklearn dependency)
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    num_classes: int) -> dict[str, Any]:
    """Return accuracy + per-class precision/recall/F1 + macro averages."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    eps = 1e-12
    precision, recall, f1 = [], [], []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p_i = tp / (tp + fp + eps)
        r_i = tp / (tp + fn + eps)
        f1_i = 2 * p_i * r_i / (p_i + r_i + eps)
        precision.append(float(p_i))
        recall.append(float(r_i))
        f1.append(float(f1_i))

    accuracy = float(cm.trace() / (cm.sum() + eps))
    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "confusion_matrix": cm.tolist(),
    }

# ---------------------------------------------------------------------------
# Single fold training
# ---------------------------------------------------------------------------

def train_fold(
    fold: int,
    train_indices: list[int],
    val_indices: list[int],
    full_dataset: CardamomDataset,
    train_tf,
    val_tf,
) -> dict[str, Any]:
    """Train one fold and return validation metrics."""
    print(f"\n{'='*60}")
    print(f"Fold {fold + 1}/{K_FOLDS}")
    print(f"  Train: {len(train_indices)} samples | Val: {len(val_indices)} samples")
    print(f"{'='*60}")

    # Apply correct transforms by wrapping with a transform-swapping dataset
    class _TransformDataset(Dataset):
        def __init__(self, base_dataset, indices, transform):
            self.base = base_dataset
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            path, label = self.base.samples[self.indices[idx]]
            image = Image.open(path).convert("RGB")
            return self.transform(image), label

    train_ds = _TransformDataset(full_dataset, train_indices, train_tf)
    val_ds = _TransformDataset(full_dataset, val_indices, val_tf)

    # Compute class weights from training split
    train_labels = np.array([full_dataset.samples[i][1] for i in train_indices])
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(float)
    class_weights = class_counts.sum() / (NUM_CLASSES * class_counts)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=False)

    model = create_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_state: dict | None = None

    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_loss = 0.0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                out = model(inputs)
                val_loss += criterion(out, labels).item() * inputs.size(0)
                total += inputs.size(0)
        val_loss /= total
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 5 == 0 or patience_counter == 0:
            print(f"  Epoch {epoch+1:3d} | val_loss={val_loss:.4f} "
                  f"| patience={patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Save best checkpoint
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    ckpt_path = models_dir / f"cv_fold_{fold+1}.pt"
    if best_state is not None:
        torch.save(best_state, ckpt_path)
        model.load_state_dict(best_state)
    print(f"  ✓ Checkpoint saved: {ckpt_path}")

    # Final evaluation on val split
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            preds = model(inputs).argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())

    metrics = compute_metrics(np.array(y_true), np.array(y_pred), NUM_CLASSES)
    print(f"\n  Fold {fold+1} results:")
    print(f"    Accuracy          : {metrics['accuracy']*100:.2f}%")
    print(f"    Macro Precision   : {metrics['macro_precision']*100:.2f}%")
    print(f"    Macro Recall      : {metrics['macro_recall']*100:.2f}%")
    print(f"    Macro F1          : {metrics['macro_f1']*100:.2f}%")
    return metrics

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("5-Fold Cross-Validation  –  Cardamom Disease Detection")
    print(f"Model      : EfficientNetV2-S")
    print(f"Device     : {DEVICE}")
    print(f"Source     : {SOURCE_DIR}/")
    print(f"Classes    : {CLASS_FOLDERS}")
    print(f"Seed       : {RANDOM_SEED}")
    print("=" * 60)

    # Verify source directory exists
    if not Path(SOURCE_DIR).exists():
        print(f"\n❌ Source directory not found: {SOURCE_DIR}/")
        print("   Run remove_bg_batch.py or augment.py first to populate it.")
        sys.exit(1)

    # Load dataset without any transform (transforms applied per-split inside fold)
    full_dataset = CardamomDataset(SOURCE_DIR, CLASS_FOLDERS, transform=None)
    n = len(full_dataset)
    print(f"\nTotal images: {n}")

    labels_array = np.array([s[1] for s in full_dataset.samples])
    for i, cls in enumerate(CLASS_FOLDERS):
        count = int((labels_array == i).sum())
        print(f"  {cls}: {count}")

    train_tf, val_tf = get_transforms()

    # Manual stratified K-Fold
    indices_per_class = [np.where(labels_array == i)[0].tolist()
                         for i in range(NUM_CLASSES)]
    for idxs in indices_per_class:
        random.shuffle(idxs)

    # Build fold index lists (stratified by class)
    folds: list[list[int]] = [[] for _ in range(K_FOLDS)]
    for class_idxs in indices_per_class:
        chunk = len(class_idxs) // K_FOLDS
        for k in range(K_FOLDS):
            start = k * chunk
            end = start + chunk if k < K_FOLDS - 1 else len(class_idxs)
            folds[k].extend(class_idxs[start:end])

    all_metrics: list[dict] = []
    for fold in range(K_FOLDS):
        val_indices = folds[fold]
        train_indices = []
        for k in range(K_FOLDS):
            if k != fold:
                train_indices.extend(folds[k])

        metrics = train_fold(fold, train_indices, val_indices,
                             full_dataset, train_tf, val_tf)
        all_metrics.append(metrics)

    # Aggregate
    print("\n" + "=" * 60)
    print("Cross-Validation Summary")
    print("=" * 60)

    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    summary: dict[str, Any] = {
        "folds": K_FOLDS,
        "random_seed": RANDOM_SEED,
        "model": "EfficientNetV2-S",
        "classes": CLASS_FOLDERS,
        "per_fold": all_metrics,
    }

    for key in keys:
        values = np.array([m[key] for m in all_metrics])
        mean, std = values.mean(), values.std()
        label = key.replace("_", " ").title()
        print(f"  {label:25s}: {mean*100:.2f}% ± {std*100:.2f}%")
        summary[f"mean_{key}"] = float(mean)
        summary[f"std_{key}"] = float(std)

    # Per-class F1 averages
    print("\n  Per-class macro F1 (mean ± std):")
    for i, cls in enumerate(CLASS_FOLDERS):
        f1_vals = np.array([m["per_class_f1"][i] for m in all_metrics])
        print(f"    {cls:12s}: {f1_vals.mean()*100:.2f}% ± {f1_vals.std()*100:.2f}%")

    # Save JSON
    out_path = Path("cv_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Results saved to: {out_path}")


if __name__ == "__main__":
    main()
