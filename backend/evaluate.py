"""
Evaluation script for trained Cardamom Disease Detection model
Generates detailed classification metrics and confusion matrix
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

from app.models.cardamom_model import CardamomClassifier


class Config:
    """Evaluation configuration"""
    DATASET_PATH = "dataset"
    MODEL_PATH = "models/cardamom_model.pt"
    BATCH_SIZE = 32
    IMG_SIZE = 224
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Class names matching training
    CLASS_NAMES = ["Colletotrichum Blight", "Phyllosticta Leaf Spot", "Healthy"]
    CLASS_NAMES_SHORT = ["colletotrichum_blight", "phyllosticta_leaf_spot", "healthy"]


def get_test_loader():
    """Create test data loader"""
    
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


def load_trained_model():
    """Load the trained model"""
    
    model_path = Path(Config.MODEL_PATH)
    
    if not model_path.exists():
        print(f"❌ Error: Model file not found at {Config.MODEL_PATH}")
        print(f"   Please train the model first using: python train.py")
        return None
    
    print(f"Loading model from: {Config.MODEL_PATH}")
    
    model = CardamomClassifier(num_classes=3)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model = model.to(Config.DEVICE)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    return model


def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
    
    print(f"\n{'='*60}")
    print("Running Evaluation on Test Set")
    print(f"{'='*60}\n")
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(Config.DEVICE)
            
            # Get predictions
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)


def print_classification_report(y_true, y_pred):
    """Print detailed classification metrics"""
    
    print(f"\n{'='*60}")
    print("Classification Report")
    print(f"{'='*60}\n")
    
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=Config.CLASS_NAMES,
        digits=4
    )
    print(report)
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{'='*60}")
    print(f"Overall Test Accuracy: {accuracy * 100:.2f}%")
    print(f"{'='*60}\n")


def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    """Create and save confusion matrix visualization"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=Config.CLASS_NAMES,
                yticklabels=Config.CLASS_NAMES,
                cbar_kws={'label': 'Count'},
                ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    
    # Plot 2: Percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=Config.CLASS_NAMES,
                yticklabels=Config.CLASS_NAMES,
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


def plot_per_class_metrics(y_true, y_pred, save_path="per_class_metrics.png"):
    """Plot per-class performance metrics"""
    
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2]
    )
    
    # Create bar plot
    x = np.arange(len(Config.CLASS_NAMES))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, precision * 100, width, label='Precision', color='skyblue')
    bars2 = ax.bar(x, recall * 100, width, label='Recall', color='lightgreen')
    bars3 = ax.bar(x + width, f1 * 100, width, label='F1-Score', color='salmon')
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(Config.CLASS_NAMES, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Add value labels on bars
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


def analyze_confidence(probabilities, labels, predictions):
    """Analyze prediction confidence"""
    
    print(f"\n{'='*60}")
    print("Confidence Analysis")
    print(f"{'='*60}\n")
    
    # Get confidence for predicted class
    confidences = np.max(probabilities, axis=1)
    
    # Correct vs incorrect predictions
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
    
    # Confidence thresholds
    print(f"\nPredictions by Confidence Level:")
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
    for threshold in thresholds:
        count = np.sum(confidences >= threshold)
        accuracy = np.sum((confidences >= threshold) & correct_mask) / count * 100 if count > 0 else 0
        print(f"  ≥{threshold*100:.0f}%: {count} predictions ({count/len(predictions)*100:.1f}%) - Accuracy: {accuracy:.1f}%")


def main():
    """Main evaluation function"""
    
    print(f"\n{'='*60}")
    print("Cardamom Disease Detection Model Evaluation")
    print(f"{'='*60}\n")
    print(f"Device: {Config.DEVICE}")
    print(f"Dataset: {Config.DATASET_PATH}/test")
    
    # Load test data
    print(f"\nLoading test dataset...")
    test_loader, test_dataset = get_test_loader()
    print(f"✓ Test samples: {len(test_dataset)}")
    print(f"  Classes: {test_dataset.classes}")
    
    # Count samples per class
    class_counts = {}
    for _, label in test_dataset.samples:
        class_name = test_dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"\n  Class distribution:")
    for class_name, count in class_counts.items():
        print(f"    - {class_name}: {count} images")
    
    # Load model
    model = load_trained_model()
    if model is None:
        return
    
    # Run evaluation
    predictions, labels, probabilities = evaluate_model(model, test_loader)
    
    # Print classification report
    print_classification_report(labels, predictions)
    
    # Analyze confidence
    analyze_confidence(probabilities, labels, predictions)
    
    # Create visualizations
    print(f"\n{'='*60}")
    print("Generating Visualizations")
    print(f"{'='*60}\n")
    
    plot_confusion_matrix(labels, predictions, "confusion_matrix.png")
    plot_per_class_metrics(labels, predictions, "per_class_metrics.png")
    
    print(f"\n{'='*60}")
    print("✅ Evaluation Complete!")
    print(f"{'='*60}\n")
    print("Generated files:")
    print("  - confusion_matrix.png")
    print("  - per_class_metrics.png")
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
