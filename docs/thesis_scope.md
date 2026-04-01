# Thesis Scope and Research Contributions

## Project Title

**An Explainable Bilingual Cardamom Leaf Disease Detection System Using
EfficientNetV2**

---

## Research Baseline

This project is inspired by and extends the methodology described in:

> Sunil C. K., Jaidhar C. D., and Nagamma Patil,
> "Cardamom Plant Disease Detection Approach Using EfficientNetV2,"
> *IEEE Access*, 2022.

The baseline paper contributes:
- A field-collected dataset of 1,724 cardamom leaf images (3 disease classes)
- A pipeline using U²-Net for background removal followed by EfficientNetV2
  for classification
- Reported best accuracy of **98.26 %** with EfficientNetV2-L on the
  cardamom dataset

---

## What This Thesis Extends

| Dimension | Baseline Paper | This Thesis |
|---|---|---|
| Classes | 3 (Blight, Spot, Healthy) | 4 (adds **Other** for rejection) |
| Best model | EfficientNetV2-L | EfficientNetV2-S (deployable size) |
| Background removal | U²-Net | rembg (U²-Net ONNX); ablation study included |
| Class imbalance | Not reported | Inverse-frequency class weighting |
| Augmentation policy | Not detailed | Documented (flip, rotation, jitter, affine) |
| Evaluation | Single train/test split | 5-fold cross-validation + single split |
| Ablation study | None | Background-removal ablation (A vs B) |
| Error analysis | Confusion matrix | Misclassification CSV + dominant pairs |
| Robustness tests | None | Blur, noise, brightness, rotation, crop |
| Explainability | Not included | Grad-CAM heatmaps (integrated into API + apps) |
| Deployment | Research paper only | FastAPI backend + React web app + React Native mobile app |
| Language support | English | English + Nepali (नेपाली) |
| Severity estimation | None | Heatmap-based severity stage (mild / moderate / severe) |
| Uncertainty flagging | None | Confidence threshold (0.60); "Uncertain" label |

---

## Thesis Contributions

### 1. Reproducible Research Pipeline
A fully documented training, validation, and evaluation pipeline with:
- Fixed random seeds for reproducibility
- Documented augmentation policy
- Inverse-frequency class weighting
- All scripts included in the repository

### 2. Expanded Class Taxonomy
Addition of an **Other** class to handle images that are not cardamom leaves,
making the system more practical for field deployment where non-cardamom
images may be submitted.

### 3. Evaluation Depth
- **5-fold cross-validation** (mean ± std accuracy, precision, recall, F1)
- **Background-removal ablation** showing the quantitative impact of
  U²-Net/rembg preprocessing on classification metrics
- **Error analysis** identifying dominant misclassification patterns and
  providing per-sample CSV output
- **Robustness tests** under blur, noise, brightness reduction, rotation,
  and center crop perturbations

### 4. Explainability via Grad-CAM
Grad-CAM heatmaps are generated for every prediction and exposed to end-users
through the web and mobile interfaces, making the system interpretable and
building user trust.

### 5. Full-Stack Bilingual Deployment
A complete production-style system comprising:
- **FastAPI REST backend** with automatic OpenAPI documentation
- **React / TypeScript web application** (Vite)
- **React Native / Expo mobile application** for Android and iOS
- **English + Nepali (नेपाली)** bilingual interface, improving accessibility
  for farmers in Nepal

### 6. Severity Estimation
A post-classification severity stage is estimated from the Grad-CAM heatmap,
providing actionable guidance (mild / moderate / severe) beyond a bare
disease label.

---

## What Is Not Claimed as Novel

- The EfficientNetV2 architecture (Tan & Le, 2021)
- The U²-Net background-removal architecture (Qin et al., 2020)
- The original cardamom disease dataset (Sunil C. K. et al., 2022)
- The Grad-CAM method (Selvaraju et al., 2017)

---

## Scope Boundaries

| In scope | Out of scope |
|---|---|
| Cardamom leaf disease detection (Blight, Spot, Healthy, Other) | Other crop species |
| EfficientNetV2-S as the deployment model | EfficientNetV2-M / L (size constraints) |
| rembg-based background removal | Full U²-Net fine-tuning |
| Single-image prediction via REST API | Real-time video stream processing |
| Severity estimation from Grad-CAM | Agronomic treatment recommendation |
| Nepali + English UI | Other languages |

---

## Citation

```bibtex
@article{sunil2022cardamom,
  author  = {Sunil, C. K. and Jaidhar, C. D. and Patil, Nagamma},
  title   = {Cardamom Plant Disease Detection Approach Using {EfficientNetV2}},
  journal = {IEEE Access},
  year    = {2022},
  doi     = {10.1109/ACCESS.2022.3175013}
}
```
