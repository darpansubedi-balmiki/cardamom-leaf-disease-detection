"""
Baseline Model Comparison for Cardamom Leaf Disease Classification.

Trains and evaluates multiple lightweight CNN architectures on the same
dataset split so that EfficientNetV2-S can be justified with concrete numbers.

Architectures compared:
  - ResNet-50
  - MobileNetV3-Small
  - EfficientNet-B0
  - EfficientNetV2-S  (our chosen model)

Usage:
    cd backend
    python baseline_comparison.py

Outputs:
    - baseline_results.json   – per-model accuracy/F1/AUC table
    - baseline_comparison.png – bar chart comparison
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_PATH = "dataset"
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 7
IMG_SIZE = 224
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def get_loaders():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(f"{DATASET_PATH}/train", transform=train_tf)
    val_ds = datasets.ImageFolder(f"{DATASET_PATH}/val", transform=val_tf)
    test_ds = datasets.ImageFolder(f"{DATASET_PATH}/test", transform=val_tf)

    num_classes = len(train_ds.classes)
    print(f"Classes ({num_classes}): {train_ds.classes}")

    targets = np.array(train_ds.targets)
    class_counts = np.bincount(targets, minlength=num_classes).astype(float)
    class_weights = torch.tensor(
        class_counts.sum() / (num_classes * class_counts), dtype=torch.float32
    ).to(DEVICE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, num_classes, class_weights


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------


def _replace_head(model: nn.Module, in_features: int, num_classes: int) -> nn.Module:
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )
    return model


MODEL_REGISTRY: dict[str, Any] = {
    "ResNet-50": lambda nc: _replace_head(
        models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        2048, nc
    ),
    "MobileNetV3-Small": lambda nc: _mobilenet_v3_small(nc),
    "EfficientNet-B0": lambda nc: _efficientnet_b0(nc),
    "EfficientNetV2-S": lambda nc: _efficientnet_v2_s(nc),
}


def _mobilenet_v3_small(num_classes: int) -> nn.Module:
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, num_classes)
    return m


def _efficientnet_b0(num_classes: int) -> nn.Module:
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )
    return m


def _efficientnet_v2_s(num_classes: int) -> nn.Module:
    m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )
    return m


# ---------------------------------------------------------------------------
# Training & evaluation helpers
# ---------------------------------------------------------------------------


def train_and_evaluate(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_weights: torch.Tensor,
) -> dict[str, Any]:
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_state: dict | None = None
    start_time = time.time()

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

        if patience_counter >= PATIENCE:
            print(f"    Early stop at epoch {epoch + 1}")
            break

    elapsed = time.time() - start_time

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on test set
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            y_prob.extend(probs.tolist())
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_prob_arr = np.array(y_prob)

    num_classes = y_prob_arr.shape[1]
    accuracy = float((y_true_arr == y_pred_arr).mean())

    # Macro F1
    eps = 1e-12
    f1_list = []
    for i in range(num_classes):
        tp = int(((y_pred_arr == i) & (y_true_arr == i)).sum())
        fp = int(((y_pred_arr == i) & (y_true_arr != i)).sum())
        fn = int(((y_pred_arr != i) & (y_true_arr == i)).sum())
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f1_list.append(2 * p * r / (p + r + eps))
    macro_f1 = float(np.mean(f1_list))

    # Macro ROC-AUC (one-vs-rest)
    try:
        from sklearn.metrics import roc_auc_score
        y_onehot = np.eye(num_classes)[y_true_arr]
        macro_auc = float(roc_auc_score(y_onehot, y_prob_arr, average="macro"))
    except Exception:
        macro_auc = float("nan")

    result = {
        "model": model_name,
        "accuracy": round(accuracy * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "macro_roc_auc": round(macro_auc, 4),
        "training_time_s": round(elapsed, 1),
    }
    print(
        f"  {model_name:20s} | Acc: {result['accuracy']:.2f}%  "
        f"F1: {result['macro_f1']:.2f}%  AUC: {result['macro_roc_auc']:.4f}  "
        f"Time: {result['training_time_s']:.0f}s"
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 70)
    print("Baseline Model Comparison  –  Cardamom Disease Detection")
    print(f"Device     : {DEVICE}")
    print(f"Dataset    : {DATASET_PATH}/")
    print(f"Epochs     : {NUM_EPOCHS}  (early stopping patience={PATIENCE})")
    print("=" * 70)

    if not Path(f"{DATASET_PATH}/train").exists():
        print(f"\n❌ Dataset not found at '{DATASET_PATH}/'. Run split_dataset.py first.")
        return

    train_loader, val_loader, test_loader, num_classes, class_weights = get_loaders()

    results = []
    for name, factory in MODEL_REGISTRY.items():
        print(f"\n▶  Training {name} ...")
        model = factory(num_classes)
        metrics = train_and_evaluate(
            name, model, train_loader, val_loader, test_loader, class_weights
        )
        results.append(metrics)

    # Save JSON
    out_json = Path("baseline_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {out_json}")

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Model':<22} {'Accuracy':>10} {'Macro F1':>10} {'ROC-AUC':>10} {'Time(s)':>10}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        print(
            f"{r['model']:<22} {r['accuracy']:>9.2f}%  "
            f"{r['macro_f1']:>9.2f}%  "
            f"{r['macro_roc_auc']:>10.4f}  "
            f"{r['training_time_s']:>8.0f}s"
        )
    print("=" * 70)

    # Bar chart
    try:
        import matplotlib.pyplot as plt

        names = [r["model"] for r in results]
        accs = [r["accuracy"] for r in results]
        f1s = [r["macro_f1"] for r in results]
        aucs = [r["macro_roc_auc"] * 100 for r in results]

        x = np.arange(len(names))
        w = 0.25

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - w, accs, w, label="Accuracy (%)", color="steelblue")
        ax.bar(x, f1s, w, label="Macro F1 (%)", color="seagreen")
        ax.bar(x + w, aucs, w, label="Macro ROC-AUC × 100", color="tomato")

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=10, ha="right")
        ax.set_ylabel("Score (%)")
        ax.set_title("Baseline Model Comparison")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 110)

        plt.tight_layout()
        chart_path = Path("baseline_comparison.png")
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        print(f"✓ Chart saved to: {chart_path}")
    except ImportError:
        print("matplotlib not installed – skipping chart")


if __name__ == "__main__":
    main()
