# Reproducible Model Pipeline

This document describes the end-to-end research pipeline for the
**Cardamom Leaf Disease Detection** system.  Every configuration choice that
affects reproducibility is listed here so that experiments can be exactly
replicated.

---

## 1. Dataset

| Property | Value |
|---|---|
| Total images | 1,724 |
| Classes | Colletotrichum Blight · Phyllosticta Leaf Spot · Healthy · Other |
| Split ratios | 70 % train / 15 % val / 15 % test |
| Random seed | 42 (set in `split_dataset.py`) |
| Source folder | `backend/dataset_processed/{blight,healthy,other,spot}` |
| Output folder | `backend/dataset/{train,val,test}/{blight,healthy,other,spot}` |

Run `backend/split_dataset.py` to reproduce the split.

---

## 2. Preprocessing

All images pass through the following deterministic preprocessing pipeline
before being fed to the model.

### 2a. Background Removal (optional, ablation condition B)

- Tool: [`rembg`](https://github.com/danielgatis/rembg) (U²-Net ONNX backend)
- Removes background before training/evaluation to isolate leaf features
- Evaluated in `backend/ablation_background_removal.py`
- Images are converted to RGB after background removal (alpha channel dropped)

### 2b. Resize

All images are resized to **224 × 224** pixels using bilinear interpolation
(torchvision `transforms.Resize`).

### 2c. Augmentation (training only)

Applied only to the training split to reduce over-fitting:

| Augmentation | Parameters |
|---|---|
| Random horizontal flip | p = 0.5 |
| Random vertical flip | p = 0.5 |
| Random rotation | ± 30° |
| Color jitter | brightness, contrast, saturation Δ = 0.2 |
| Random affine translate | ± 10 % of image size |

### 2d. Normalization

ImageNet statistics applied to all splits:

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

## 3. Class Weighting

To handle class imbalance, inverse-frequency weights are computed from
training-split class counts and passed to `nn.CrossEntropyLoss`:

```python
class_counts = np.bincount(train_targets)
class_weights = class_counts.sum() / (len(class_counts) * class_counts)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
```

This calculation is performed at runtime inside `backend/train.py` so that
weights automatically adapt to the actual class distribution.

---

## 4. Model Architecture

| Property | Value |
|---|---|
| Backbone | EfficientNetV2-S (torchvision, ImageNet pre-trained) |
| Classifier head | Dropout(0.3) → Linear(1280, 512) → ReLU → Dropout(0.2) → Linear(512, 4) |
| Output classes | 4 |
| Input size | 224 × 224 × 3 |

### Training hyperparameters

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Batch size | 32 |
| Max epochs | 50 |
| Early stopping patience | 10 (monitored on val loss) |
| Loss function | CrossEntropyLoss with class weights |

---

## 5. Explainability — Grad-CAM

Grad-CAM is computed against the **last convolutional layer** of the
EfficientNetV2-S backbone.

```python
# Target layer in torchvision EfficientNetV2-S
target_layer = model.features[-1]
```

Implementation: `backend/app/utils/grad_cam.py`

The API endpoint `POST /predict` returns a base64-encoded heatmap overlaid on
the original image when Grad-CAM is enabled (default).  The mobile and web
apps display this heatmap alongside the predicted class.

---

## 6. Evaluation Protocol

### 6a. Standard evaluation (single split)

```bash
cd backend
python evaluate.py     # confusion matrix + per-class metrics
```

### 6b. 5-Fold Cross-Validation

```bash
cd backend
python cross_validate.py
```

Produces `cv_results.json` with mean ± std for accuracy, precision, recall
and F1 across 5 stratified folds.

### 6c. Background-Removal Ablation

```bash
cd backend
python ablation_background_removal.py
```

Compares condition A (raw images) vs condition B (rembg-processed images).
Produces `ablation_results.json`.

### 6d. Error Analysis

```bash
cd backend
python error_analysis.py
```

Produces `misclassified.csv` and `error_analysis.json` with dominant
confusion pairs and per-class error rates.

### 6e. Robustness Tests

```bash
cd backend
python robustness_test.py
```

Evaluates accuracy and macro F1 under blur, low-brightness, Gaussian noise,
rotation, and center-crop perturbations.  Produces `robustness_results.json`.

---

## 7. Deployment

The trained model checkpoint (`backend/models/cardamom_model.pt`) is served
via the FastAPI backend (`backend/app/main.py`).  The inference path is:

```
POST /predict
  → image preprocessing (resize + normalize)
  → EfficientNetV2-S forward pass
  → softmax probabilities
  → uncertainty check (confidence_threshold = 0.60)
  → Grad-CAM heatmap generation
  → JSON response {top_class, top_probability, top_k, heatmap, …}
```

The web frontend (React/TypeScript) and mobile app (React Native/Expo) both
consume this endpoint and display results bilingually (English + Nepali).

---

## 8. Reproducibility Checklist

- [x] Random seed fixed (`RANDOM_SEED = 42`) in `split_dataset.py` and `cross_validate.py`
- [x] Augmentation parameters documented above
- [x] Class weights computed deterministically from training labels
- [x] Confidence threshold documented (`DEFAULT_CONFIDENCE_THRESHOLD = 0.60`)
- [x] Model checkpoint path documented (`models/cardamom_model.pt`)
- [x] All evaluation scripts are self-contained and produce JSON outputs
- [x] `requirements.txt` pins all Python dependencies
