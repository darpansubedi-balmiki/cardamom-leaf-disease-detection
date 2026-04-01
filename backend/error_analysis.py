"""
Error Analysis for Cardamom Leaf Disease Classification.

Runs inference on the test set, identifies misclassified samples, and
produces:
  - Per-class error counts and dominant confusion pairs
  - Confidence distribution for correct vs. incorrect predictions
  - CSV list of misclassified images (path, true label, predicted label,
    confidence)
  - error_analysis.json  (machine-readable summary)

Usage:
    cd backend
    python error_analysis.py

Optional: install matplotlib for confidence-histogram output.
    pip install matplotlib
"""
from __future__ import annotations

import csv
import json
import sys
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
BATCH_SIZE = 32
CONFIDENCE_BINS = 21   # number of histogram bins for confidence distribution plots

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

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

class _PathDataset(Dataset):
    """Returns (tensor, label, path_str) so we can record failing files."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, _ = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label, str(path)


def _collate_with_paths(batch):
    imgs = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    paths = [b[2] for b in batch]
    return imgs, labels, paths


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


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, loader):
    """Returns arrays: y_true, y_pred, y_conf, paths."""
    all_true, all_pred, all_conf, all_paths = [], [], [], []
    with torch.no_grad():
        for inputs, labels, paths in loader:
            inputs = inputs.to(DEVICE)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            confs = probs.max(axis=1)
            all_true.extend(labels.numpy().tolist())
            all_pred.extend(preds.tolist())
            all_conf.extend(confs.tolist())
            all_paths.extend(paths)
    return (np.array(all_true), np.array(all_pred),
            np.array(all_conf), all_paths)

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(y_true, y_pred, y_conf, paths, class_names):
    n = len(y_true)
    correct_mask = y_true == y_pred
    wrong_mask = ~correct_mask

    correct_conf = y_conf[correct_mask]
    wrong_conf = y_conf[wrong_mask]

    print(f"\nOverall accuracy : {correct_mask.sum()}/{n} = {correct_mask.mean()*100:.2f}%")
    print(f"Misclassified    : {wrong_mask.sum()} samples")

    # ── Confidence summary ───────────────────────────────────────────────────
    print("\nConfidence distribution (top-1 softmax probability):")
    print(f"  Correct predictions:")
    if len(correct_conf):
        print(f"    Mean   : {correct_conf.mean()*100:.1f}%")
        print(f"    Median : {np.median(correct_conf)*100:.1f}%")
        print(f"    Min    : {correct_conf.min()*100:.1f}%")
    print(f"  Wrong predictions:")
    if len(wrong_conf):
        print(f"    Mean   : {wrong_conf.mean()*100:.1f}%")
        print(f"    Median : {np.median(wrong_conf)*100:.1f}%")
        print(f"    Max    : {wrong_conf.max()*100:.1f}%")

    # ── Per-class breakdown ──────────────────────────────────────────────────
    print("\nPer-class error breakdown:")
    print(f"  {'Class':15s}  {'Total':>6s}  {'Wrong':>6s}  {'Error %':>8s}")
    print("  " + "-" * 42)
    for i, cls in enumerate(class_names):
        cls_mask = y_true == i
        cls_total = cls_mask.sum()
        cls_wrong = (cls_mask & wrong_mask).sum()
        pct = (cls_wrong / cls_total * 100) if cls_total > 0 else 0
        print(f"  {cls:15s}  {cls_total:>6d}  {cls_wrong:>6d}  {pct:>7.1f}%")

    # ── Dominant confusion pairs ─────────────────────────────────────────────
    nc = len(class_names)
    cm = np.zeros((nc, nc), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    print("\nTop confusion pairs (true → predicted):")
    off_diag = [(cm[i, j], class_names[i], class_names[j])
                for i in range(nc) for j in range(nc) if i != j and cm[i, j] > 0]
    off_diag.sort(reverse=True)
    for count, true_cls, pred_cls in off_diag[:10]:
        print(f"  {true_cls:15s} → {pred_cls:15s}  ({count} samples)")

    # ── Save misclassified list ───────────────────────────────────────────────
    csv_path = Path("misclassified.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_label", "predicted_label", "confidence"])
        for i in np.where(wrong_mask)[0]:
            writer.writerow([
                paths[i],
                class_names[y_true[i]],
                class_names[y_pred[i]],
                f"{y_conf[i]:.4f}",
            ])
    print(f"\n✅ Misclassified samples saved to: {csv_path}  ({wrong_mask.sum()} rows)")

    # ── JSON summary ─────────────────────────────────────────────────────────
    summary: dict[str, Any] = {
        "model": "EfficientNetV2-S",
        "dataset": DATASET_PATH,
        "classes": class_names,
        "n_total": int(n),
        "n_correct": int(correct_mask.sum()),
        "n_wrong": int(wrong_mask.sum()),
        "accuracy": float(correct_mask.mean()),
        "correct_mean_confidence": float(correct_conf.mean()) if len(correct_conf) else None,
        "wrong_mean_confidence": float(wrong_conf.mean()) if len(wrong_conf) else None,
        "confusion_matrix": cm.tolist(),
        "top_confusion_pairs": [
            {"true": tc, "predicted": pc, "count": int(cnt)}
            for cnt, tc, pc in off_diag[:10]
        ],
    }

    json_path = Path("error_analysis.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary saved to: {json_path}")

    # ── Optional: confidence histogram ───────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        bins = np.linspace(0, 1, CONFIDENCE_BINS)
        ax.hist(correct_conf, bins=bins, alpha=0.7, label="Correct", color="steelblue")
        ax.hist(wrong_conf, bins=bins, alpha=0.7, label="Wrong", color="tomato")
        ax.set_xlabel("Top-1 Confidence", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Confidence Distribution – Correct vs. Wrong Predictions")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        hist_path = "confidence_histogram.png"
        plt.savefig(hist_path, dpi=150)
        print(f"✅ Confidence histogram saved to: {hist_path}")
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("Error Analysis  –  Cardamom Disease Detection")
    print(f"Model  : EfficientNetV2-S")
    print(f"Device : {DEVICE}")
    print(f"Data   : {DATASET_PATH}/test")
    print("=" * 60)

    model = load_model()
    samples, class_names = collect_test_samples(DATASET_PATH)
    print(f"\nTest samples: {len(samples)}")
    for cls in class_names:
        cnt = sum(1 for _, _, c in samples if c == cls)
        print(f"  {cls}: {cnt}")

    transform = get_transform()
    ds = _PathDataset(samples, transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=_collate_with_paths,
                        num_workers=2, pin_memory=False)

    print("\nRunning inference…")
    y_true, y_pred, y_conf, paths = run_inference(model, loader)
    analyse(y_true, y_pred, y_conf, paths, class_names)


if __name__ == "__main__":
    main()
