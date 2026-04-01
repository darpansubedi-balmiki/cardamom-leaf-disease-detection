"""
Background-Removal Ablation Study.

Evaluates the trained EfficientNetV2-S model on two conditions:
  (A) Raw test images           – no background removal
  (B) Background-removed images – processed via rembg

For each condition the script reports accuracy, precision, recall and F1
so the thesis can include a concrete ablation table.

Usage:
    cd backend
    python ablation_background_removal.py

Requirements:
    pip install rembg onnxruntime     # for condition B

Outputs:
    - Metrics for both conditions printed to stdout
    - ablation_results.json  (machine-readable summary)

Notes:
    * If rembg is not installed, only condition A is evaluated and the
      script prints a clear warning.
    * Background removal is applied in memory; no files are written.
"""
from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Configuration (must match train.py)
# ---------------------------------------------------------------------------
DATASET_PATH = "dataset"
MODEL_PATH = "models/cardamom_model.pt"
NUM_CLASSES = 4
IMG_SIZE = 224
BATCH_SIZE = 16   # smaller batch for background-removal condition (CPU heavy)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

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


def load_model() -> nn.Module:
    model = create_model().to(DEVICE)
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path.absolute()}")
        print("   Train the model first:  python train.py")
        sys.exit(1)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"✅ Loaded model from: {model_path}")
    return model

# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def _collect_test_samples(dataset_path: str) -> list[tuple[Path, int, str]]:
    """Return list of (image_path, label_index, class_name) for the test split."""
    test_dir = Path(dataset_path) / "test"
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir.absolute()}")
        print("   Run split_dataset.py first.")
        sys.exit(1)

    class_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"❌ No class subdirectories found in: {test_dir}")
        sys.exit(1)

    class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
    samples: list[tuple[Path, int, str]] = []
    for class_dir in class_dirs:
        label = class_to_idx[class_dir.name]
        for fp in class_dir.iterdir():
            if fp.suffix in EXTENSIONS:
                samples.append((fp, label, class_dir.name))
    return samples, [d.name for d in class_dirs]


class RawTestDataset(Dataset):
    """Loads test images without any background processing."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, _ = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


class BgRemovedTestDataset(Dataset):
    """Loads test images and removes the background using rembg."""

    def __init__(self, samples, transform, remove_bg_fn):
        self.samples = samples
        self.transform = transform
        self._remove_bg = remove_bg_fn

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, _ = self.samples[idx]
        with open(path, "rb") as fh:
            raw_bytes = fh.read()
        try:
            out_bytes = self._remove_bg(raw_bytes)
            image = Image.open(BytesIO(out_bytes)).convert("RGB")
        except Exception:
            # Fallback to original image on any rembg error
            image = Image.open(path).convert("RGB")
        return self.transform(image), label

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    num_classes: int) -> dict[str, Any]:
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

    return {
        "accuracy": float(cm.trace() / (cm.sum() + eps)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "confusion_matrix": cm.tolist(),
    }


def evaluate(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            preds = model(inputs).argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def print_metrics(label: str, metrics: dict, class_names: list[str]) -> None:
    print(f"\n  ── {label} ──")
    print(f"    Accuracy        : {metrics['accuracy']*100:.2f}%")
    print(f"    Macro Precision : {metrics['macro_precision']*100:.2f}%")
    print(f"    Macro Recall    : {metrics['macro_recall']*100:.2f}%")
    print(f"    Macro F1        : {metrics['macro_f1']*100:.2f}%")
    print(f"    Per-class F1    :")
    for cls, f1 in zip(class_names, metrics["per_class_f1"]):
        print(f"      {cls:15s}: {f1*100:.2f}%")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("Background-Removal Ablation Study")
    print(f"Model  : EfficientNetV2-S")
    print(f"Device : {DEVICE}")
    print(f"Data   : {DATASET_PATH}/test")
    print("=" * 60)

    model = load_model()
    samples, class_names = _collect_test_samples(DATASET_PATH)
    print(f"\nTest samples: {len(samples)}")
    for cls in class_names:
        cnt = sum(1 for _, _, c in samples if c == cls)
        print(f"  {cls}: {cnt}")

    transform = get_eval_transform()
    results: dict[str, Any] = {
        "model": "EfficientNetV2-S",
        "dataset": DATASET_PATH,
        "classes": class_names,
    }

    # ── Condition A: Raw images ──────────────────────────────────────────────
    print("\n[A] Evaluating on raw images (no background removal)…")
    raw_ds = RawTestDataset(samples, transform)
    raw_loader = DataLoader(raw_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=False)
    y_true_a, y_pred_a = evaluate(model, raw_loader)
    metrics_a = compute_metrics(y_true_a, y_pred_a, len(class_names))
    print_metrics("Condition A – Raw images", metrics_a, class_names)
    results["condition_A_raw"] = metrics_a

    # ── Condition B: Background-removed images ───────────────────────────────
    try:
        from rembg import remove as rembg_remove
    except ImportError:
        print("\n⚠️  rembg is not installed – skipping condition B.")
        print("   Install with:  pip install rembg onnxruntime")
        results["condition_B_bg_removed"] = None
    else:
        print("\n[B] Evaluating on background-removed images (rembg)…")
        print("    Note: First run downloads the ONNX model weights (~170 MB).")
        bg_ds = BgRemovedTestDataset(samples, transform, rembg_remove)
        bg_loader = DataLoader(bg_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=0,   # rembg is not fork-safe
                               pin_memory=False)
        y_true_b, y_pred_b = evaluate(model, bg_loader)
        metrics_b = compute_metrics(y_true_b, y_pred_b, len(class_names))
        print_metrics("Condition B – Background removed", metrics_b, class_names)
        results["condition_B_bg_removed"] = metrics_b

        # Delta table
        print("\n  ── Delta (B − A) ──")
        for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
            delta = (metrics_b[key] - metrics_a[key]) * 100
            sign = "+" if delta >= 0 else ""
            print(f"    {key.replace('_', ' ').title():25s}: {sign}{delta:.2f}%")

    # ── Save results ─────────────────────────────────────────────────────────
    out_path = Path("ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Ablation results saved to: {out_path}")


if __name__ == "__main__":
    main()
