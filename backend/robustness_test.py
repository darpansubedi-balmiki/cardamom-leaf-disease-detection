"""
Robustness Tests for Cardamom Leaf Disease Classification.

Evaluates the trained EfficientNetV2-S under common real-world perturbations:
  1. Gaussian blur         (simulates camera blur / motion)
  2. Brightness reduction  (low-light field conditions)
  3. Gaussian noise        (sensor noise)
  4. Random rotation       (non-standard capture angles)
  5. Center crop + resize  (partial leaf / zoom variation)

For each perturbation, accuracy and macro F1 are compared to the clean
baseline so the thesis can include a robustness table.

Usage:
    cd backend
    python robustness_test.py

Outputs:
    - Per-perturbation metrics printed to stdout
    - robustness_results.json  (machine-readable summary)

Optional: install matplotlib for a bar-chart output.
    pip install matplotlib
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Configuration (must match train.py)
# ---------------------------------------------------------------------------
DATASET_PATH = "dataset"
MODEL_PATH = "models/cardamom_model.pt"
NUM_CLASSES = 4
IMG_SIZE = 224
BATCH_SIZE = 32
_BAR_LABEL_OFFSET = 0.5   # horizontal offset (% points) for bar-chart text labels

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# ---------------------------------------------------------------------------
# Perturbation definitions
# ---------------------------------------------------------------------------

def _to_pil_transform(pil_fn):
    """Wrap a PIL → PIL function as a torchvision-compatible Transform."""
    class _T:
        def __call__(self, img):
            return pil_fn(img)
    return _T()


def _gaussian_blur(radius: float = 3.0):
    def fn(img: Image.Image) -> Image.Image:
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    return fn


def _brightness_reduce(factor: float = 0.4):
    """factor < 1 darkens the image."""
    def fn(img: Image.Image) -> Image.Image:
        return ImageEnhance.Brightness(img).enhance(factor)
    return fn


def _add_gaussian_noise(std: float = 0.15):
    """Add Gaussian noise in [0,1] float space after ToTensor."""
    class _NoiseTf:
        def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
            noise = torch.randn_like(tensor) * std
            return (tensor + noise).clamp(0.0, 1.0)
    return _NoiseTf()


def _random_rotation(degrees: float = 45.0):
    return transforms.RandomRotation(degrees=degrees)


def _center_crop(crop_ratio: float = 0.75):
    """Crop the centre crop_ratio × H × W, then resize back."""
    crop_size = int(IMG_SIZE * crop_ratio)
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ])


_NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
_RESIZE = transforms.Resize((IMG_SIZE, IMG_SIZE))
_TO_TENSOR = transforms.ToTensor()


def _build_transform(pil_augment=None, tensor_augment=None):
    """Compose an evaluation transform with optional PIL/tensor augmentations."""
    steps = [_RESIZE]
    if pil_augment is not None:
        steps.append(_to_pil_transform(pil_augment))
    steps.append(_TO_TENSOR)
    if tensor_augment is not None:
        steps.append(tensor_augment)
    steps.append(_NORMALIZE)
    return transforms.Compose(steps)


# Registry of perturbations: name → transform
PERTURBATIONS: dict[str, Any] = {
    "Baseline (clean)": _build_transform(),
    "Blur (r=3)": _build_transform(pil_augment=_gaussian_blur(3.0)),
    "Blur (r=5)": _build_transform(pil_augment=_gaussian_blur(5.0)),
    "Low brightness (×0.4)": _build_transform(pil_augment=_brightness_reduce(0.4)),
    "Low brightness (×0.2)": _build_transform(pil_augment=_brightness_reduce(0.2)),
    "Gaussian noise (σ=0.10)": _build_transform(tensor_augment=_add_gaussian_noise(0.10)),
    "Gaussian noise (σ=0.20)": _build_transform(tensor_augment=_add_gaussian_noise(0.20)),
    "Rotation (±45°)": _build_transform(pil_augment=None,
                                         tensor_augment=None),  # handled below
    "Center crop (75%)": _build_transform(),  # handled below
}

# Build rotation / crop transforms separately (they need special compose order)
PERTURBATIONS["Rotation (±45°)"] = transforms.Compose([
    _RESIZE,
    _to_pil_transform(lambda img: img),   # identity (PIL)
    transforms.RandomRotation(degrees=45),
    _TO_TENSOR,
    _NORMALIZE,
])
PERTURBATIONS["Center crop (75%)"] = transforms.Compose([
    _RESIZE,
    transforms.CenterCrop(int(IMG_SIZE * 0.75)),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    _TO_TENSOR,
    _NORMALIZE,
])

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def create_model() -> nn.Module:
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    nf = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(nf, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, NUM_CLASSES),
    )
    return model


def load_model() -> nn.Module:
    path = Path(MODEL_PATH)
    if not path.exists():
        print(f"❌ Model not found: {path.absolute()}")
        print("   Train the model first:  python train.py")
        sys.exit(1)
    model = create_model().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    print(f"✅ Loaded model: {path}")
    return model

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PerturbedDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, _ = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def collect_test_samples(dataset_path: str):
    test_dir = Path(dataset_path) / "test"
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir.absolute()}")
        print("   Run split_dataset.py first.")
        sys.exit(1)
    class_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
    samples = []
    for cd in class_dirs:
        for fp in cd.iterdir():
            if fp.suffix in EXTENSIONS:
                samples.append((fp, class_to_idx[cd.name], cd.name))
    return samples, class_names

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    num_classes: int) -> dict[str, float]:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    eps = 1e-12
    f1_list = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p_i = tp / (tp + fp + eps)
        r_i = tp / (tp + fn + eps)
        f1_list.append(2 * p_i * r_i / (p_i + r_i + eps))
    return {
        "accuracy": float(cm.trace() / (cm.sum() + eps)),
        "macro_f1": float(np.mean(f1_list)),
    }


def evaluate(model: nn.Module, loader: DataLoader):
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            preds = model(inputs).argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.array(y_pred)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("Robustness Tests  –  Cardamom Disease Detection")
    print(f"Model  : EfficientNetV2-S")
    print(f"Device : {DEVICE}")
    print(f"Data   : {DATASET_PATH}/test")
    print("=" * 60)

    model = load_model()
    samples, class_names = collect_test_samples(DATASET_PATH)
    n = len(class_names)
    print(f"\nTest samples: {len(samples)}")

    results: list[dict] = []

    print(f"\n{'Perturbation':<30s}  {'Accuracy':>9s}  {'Macro F1':>9s}  {'Δ Acc':>8s}  {'Δ F1':>8s}")
    print("-" * 75)

    baseline_acc: float | None = None
    baseline_f1: float | None = None

    for name, tf in PERTURBATIONS.items():
        ds = PerturbedDataset(samples, tf)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=False)
        y_true, y_pred = evaluate(model, loader)
        metrics = compute_metrics(y_true, y_pred, n)
        acc = metrics["accuracy"] * 100
        f1 = metrics["macro_f1"] * 100

        if baseline_acc is None:
            baseline_acc = acc
            baseline_f1 = f1
            delta_acc_str = "  —"
            delta_f1_str = "  —"
        else:
            d_acc = acc - baseline_acc
            d_f1 = f1 - baseline_f1
            delta_acc_str = f"{d_acc:+.2f}%"
            delta_f1_str = f"{d_f1:+.2f}%"

        print(f"{name:<30s}  {acc:>8.2f}%  {f1:>8.2f}%  {delta_acc_str:>8s}  {delta_f1_str:>8s}")
        results.append({
            "perturbation": name,
            "accuracy": float(acc / 100),
            "macro_f1": float(f1 / 100),
            "delta_accuracy_pp": float(acc - baseline_acc) if baseline_acc is not None else 0.0,
            "delta_f1_pp": float(f1 - baseline_f1) if baseline_f1 is not None else 0.0,
        })

    # ── Save JSON ────────────────────────────────────────────────────────────
    summary = {
        "model": "EfficientNetV2-S",
        "dataset": DATASET_PATH,
        "classes": class_names,
        "perturbations": results,
    }
    out_path = Path("robustness_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Robustness results saved to: {out_path}")

    # ── Optional: bar chart ───────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        names = [r["perturbation"] for r in results]
        accs = [r["accuracy"] * 100 for r in results]

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["steelblue" if i == 0 else "salmon" for i in range(len(names))]
        bars = ax.barh(names, accs, color=colors)
        ax.set_xlabel("Accuracy (%)")
        ax.set_title("Model Robustness Under Perturbations")
        ax.set_xlim(0, 105)
        for bar, val in zip(bars, accs):
            ax.text(val + _BAR_LABEL_OFFSET, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=9)
        ax.axvline(x=baseline_acc, color="gray", linestyle="--", linewidth=1)
        plt.tight_layout()
        chart_path = "robustness_chart.png"
        plt.savefig(chart_path, dpi=150)
        print(f"✅ Chart saved to: {chart_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
