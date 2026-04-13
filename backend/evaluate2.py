"""
Evaluation script for trained Cardamom Disease Detection model.
Generates detailed classification metrics, confusion matrix, ROC-AUC and PR-AUC curves.

Class names and number of outputs are derived automatically from the dataset folder
so this script stays correct regardless of how many classes the model was trained on.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm


class Config:
    """Evaluation configuration"""
    DATASET_PATH = "dataset"
    MODEL_PATH = "models/cardamom_model.pt"
    BATCH_SIZE = 32
    IMG_SIZE = 224
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def get_test_loader():
    """Create test data loader and derive class names from folder."""

    test_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(
        root=f"{Config.DATASET_PATH}/test",
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )

    return test_loader, test_dataset

def create_model(num_classes: int):
    """Create EfficientNetV2-S model with custom classifier (must match train.py)."""
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )
    return model

def load_trained_model(num_classes: int):
    """Load the trained model."""

    model_path = Path(Config.MODEL_PATH)

    if not model_path.exists():
        print(f"❌ Error: Model file not found at {Config.MODEL_PATH}")
        print(f"   Please train the model first using: python train.py")
        return None

    print(f"Loading model from: {Config.MODEL_PATH}")

    model = create_model(num_classes)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model = model.to(Config.DEVICE)
    model.eval()

    print(f"✓ Model loaded successfully ({num_classes} classes)")
    return model


def evaluate_model(model, test_loader):
    """Evaluate model on test set; return predictions, labels, and probabilities."""

    print(f"\n{'='*60}")
    print("Running Evaluation on Test Set")
    print(f"{'='*60}\n")

    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(Config.DEVICE)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)


def print_classification_report(y_true, y_pred, class_names):
    """Print detailed classification metrics."""

    print(f"\n{'='*60}")
    print("Classification Report")
    print(f"{'='*60}\n")

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    print(report)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{'='*60}")
    print(f"Overall Test Accuracy: {accuracy * 100:.2f}%")
    print(f"{'='*60}\n")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png"):
    """Create and save confusion matrix visualization."""

    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'},
                ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)

    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Percentage (%)'},
                vmin=0, vmax=100,
                ax=ax2)
    ax2.set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {save_path}")

    return cm


def plot_per_class_metrics(y_true, y_pred, class_names, save_path="per_class_metrics.png"):
    """Plot per-class performance metrics."""

    from sklearn.metrics import precision_recall_fscore_support

    labels_idx = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx
    )

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, precision * 100, width, label='Precision', color='skyblue')
    bars2 = ax.bar(x, recall * 100, width, label='Recall', color='lightgreen')
    bars3 = ax.bar(x + width, f1 * 100, width, label='F1-Score', color='salmon')

    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])

    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Per-class metrics saved to: {save_path}")


def _class_colors(num_classes: int) -> np.ndarray:
    """Return an array of ``num_classes`` distinct RGBA colors."""
    return plt.cm.Set1(np.linspace(0, 1, num_classes))  # type: ignore[attr-defined]


def plot_roc_curves(y_true, y_prob, class_names, save_path="roc_curves.png"):
    """Plot one-vs-rest ROC curves and compute macro ROC-AUC."""

    num_classes = len(class_names)
    # One-hot encode labels
    y_onehot = np.eye(num_classes)[y_true]

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = _class_colors(num_classes)
    aucs = []
    for i, (cls, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_onehot[:, i], y_prob[:, i])
        auc = roc_auc_score(y_onehot[:, i], y_prob[:, i])
        aucs.append(auc)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
    macro_auc = float(np.mean(aucs))
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curves (Macro AUC = {macro_auc:.3f})', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curves saved to: {save_path}")
    print(f"  Macro ROC-AUC: {macro_auc:.4f}")
    for cls, auc in zip(class_names, aucs):
        print(f"    {cls}: {auc:.4f}")

    return macro_auc


def plot_pr_curves(y_true, y_prob, class_names, save_path="pr_curves.png"):
    """Plot one-vs-rest Precision-Recall curves and compute macro AP."""

    num_classes = len(class_names)
    y_onehot = np.eye(num_classes)[y_true]

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = _class_colors(num_classes)
    aps = []
    for i, (cls, color) in enumerate(zip(class_names, colors)):
        precision, recall, _ = precision_recall_curve(y_onehot[:, i], y_prob[:, i])
        ap = average_precision_score(y_onehot[:, i], y_prob[:, i])
        aps.append(ap)
        ax.plot(recall, precision, color=color, lw=2, label=f"{cls} (AP = {ap:.3f})")

    macro_ap = float(np.mean(aps))
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curves (Macro AP = {macro_ap:.3f})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ PR curves saved to: {save_path}")
    print(f"  Macro PR-AUC (mAP): {macro_ap:.4f}")
    for cls, ap in zip(class_names, aps):
        print(f"    {cls}: {ap:.4f}")

    return macro_ap


def analyze_confidence(probabilities, labels, predictions):
    """Analyze prediction confidence."""

    print(f"\n{'='*60}")
    print("Confidence Analysis")
    print(f"{'='*60}\n")

    confidences = np.max(probabilities, axis=1)

    correct_mask = predictions == labels
    correct_confidences = confidences[correct_mask]
    incorrect_confidences = confidences[~correct_mask]

    print(f"Correct Predictions:")
    print(f"  Count: {len(correct_confidences)}/{len(predictions)} ({len(correct_confidences)/len(predictions)*100:.1f}%)")
    print(f"  Mean confidence: {np.mean(correct_confidences)*100:.1f}%")
    print(f"  Median confidence: {np.median(correct_confidences)*100:.1f}%")
    print(f"  Min confidence: {np.min(correct_confidences)*100:.1f}%")

    if len(incorrect_confidences) > 0:
        print(f"\nIncorrect Predictions:")
        print(f"  Count: {len(incorrect_confidences)}/{len(predictions)} ({len(incorrect_confidences)/len(predictions)*100:.1f}%)")
        print(f"  Mean confidence: {np.mean(incorrect_confidences)*100:.1f}%")
        print(f"  Median confidence: {np.median(incorrect_confidences)*100:.1f}%")
        print(f"  Max confidence: {np.max(incorrect_confidences)*100:.1f}%")
    else:
        print(f"\n🎉 Perfect predictions! No incorrect classifications!")

    print(f"\nPredictions by Confidence Level:")
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
    for threshold in thresholds:
        count = np.sum(confidences >= threshold)
        accuracy = np.sum((confidences >= threshold) & correct_mask) / count * 100 if count > 0 else 0
        print(f"  ≥{threshold*100:.0f}%: {count} predictions ({count/len(predictions)*100:.1f}%) - Accuracy: {accuracy:.1f}%")


def main():
    """Main evaluation function."""

    print(f"\n{'='*60}")
    print("Cardamom Disease Detection Model Evaluation")
    print(f"{'='*60}\n")
    print(f"Device: {Config.DEVICE}")
    print(f"Dataset: {Config.DATASET_PATH}/test")

    # Load test data (class names derived from folder names)
    print(f"\nLoading test dataset...")
    test_loader, test_dataset = get_test_loader()
    class_names = test_dataset.classes  # derived dynamically from folder names
    num_classes = len(class_names)
    print(f"✓ Test samples: {len(test_dataset)}")
    print(f"  Classes ({num_classes}): {class_names}")

    class_counts = {}
    for _, label in test_dataset.samples:
        class_name = test_dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    print(f"\n  Class distribution:")
    for class_name, count in class_counts.items():
        print(f"    - {class_name}: {count} images")

    # Load model
    model = load_trained_model(num_classes)
    if model is None:
        return

    # Run evaluation
    predictions, labels, probabilities = evaluate_model(model, test_loader)

    # Print classification report
    print_classification_report(labels, predictions, class_names)

    # Analyze confidence
    analyze_confidence(probabilities, labels, predictions)

    # Create visualizations
    print(f"\n{'='*60}")
    print("Generating Visualizations")
    print(f"{'='*60}\n")

    plot_confusion_matrix(labels, predictions, class_names, "confusion_matrix.png")
    plot_per_class_metrics(labels, predictions, class_names, "per_class_metrics.png")
    plot_roc_curves(labels, probabilities, class_names, "roc_curves.png")
    plot_pr_curves(labels, probabilities, class_names, "pr_curves.png")

    print(f"\n{'='*60}")
    print("✅ Evaluation Complete!")
    print(f"{'='*60}\n")
    print("Generated files:")
    print("  - confusion_matrix.png")
    print("  - per_class_metrics.png")
    print("  - roc_curves.png")
    print("  - pr_curves.png")
    print("\nNext steps:")
    print("  1. Review the metrics above")
    print("  2. Check the confusion matrix for misclassifications")
    print("  3. If accuracy is ≥85%, deploy to production")
    print("  4. If accuracy is low, consider:")
    print("     - Collecting more training data")
    print("     - Balancing the dataset")
    print("     - Training for more epochs")
    print("     - Adjusting hyperparameters")


if __name__ == "__main__":
    main()

