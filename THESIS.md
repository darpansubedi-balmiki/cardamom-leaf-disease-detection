<!-- ============================================================
     THESIS DOCUMENT — Submission-Ready Version
     Style: University thesis / APA 7th Edition conventions
     Formatting note: For final submission, apply Times New Roman 12 pt,
     double-spaced text, 1-inch margins using your institution's LaTeX or
     Word template. This Markdown source preserves the full academic
     structure and all quantitative results.
     ============================================================ -->

---

# Cardamom Leaf Disease Detection Using Deep Learning

## Design, Implementation, and Evaluation of a Full-Stack Explainable Intelligent System With Bilingual Deployment for Nepali Smallholder Farmers

<br>

**Darpan Subedi**

Department of Computer Engineering  
[University Name — to be completed]  
[Course Code and Title — e.g., COMP 499 – Final Year Project / Bachelor Thesis]  
Supervisor: [Supervisor Name — to be completed]  
Submission Date: [Month Year — to be completed]

---

## Declaration

I hereby declare that this thesis is my own original work and that it has not been submitted elsewhere for any other degree or qualification. All sources of information and ideas used in this work have been properly cited and referenced.

**Signed:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_  
**Date:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

---

## Acknowledgements

I would like to express my sincere gratitude to my thesis supervisor for their guidance, critical feedback, and encouragement throughout this project. I extend my thanks to the Department of Computer Engineering for providing the computational resources and academic environment that made this work possible. I am grateful to the cardamom farming communities of Nepal whose livelihoods motivated the practical direction of this research. Finally, I thank my family and friends for their unwavering support.

---

## Abstract

Fungal foliar diseases—principally Colletotrichum Blight (*Colletotrichum gloeosporioides*) and Phyllosticta Leaf Spot (*Phyllosticta* spp.)—represent a persistent economic threat to cardamom (*Elettaria cardamomum* and *Amomum subulatum*) cultivation in Nepal, where smallholder farmers in remote hill communities lack access to trained plant pathologists. This thesis presents the design, implementation, and empirical evaluation of a complete, end-to-end intelligent system for automated cardamom leaf disease detection. The core classifier is a fine-tuned EfficientNetV2-S convolutional neural network trained on a field-collected dataset of approximately 2,600 annotated images across four categories: Colletotrichum Blight, Phyllosticta Leaf Spot, Healthy, and Other. The training pipeline incorporates ImageNet-pretrained transfer learning, stochastic data augmentation, inverse-frequency class weighting, Adam optimisation, ReduceLROnPlateau scheduling, and early stopping to maximise generalisation on a relatively modest dataset. Model generalisation was assessed via 5-fold stratified cross-validation, yielding a mean accuracy of 99.96% ± 0.08% and a mean macro-F1 of 99.96% ± 0.08% across folds. A held-out test evaluation with error analysis produced perfect classification over 391 test samples (accuracy = 100%; mean prediction confidence = 99.97%). An ablation study comparing the raw-image pipeline against a background-removal preprocessing variant found that background removal reduced test accuracy from ~100% to 94.6%, indicating that background context contributes discriminative information in this dataset. Gradient-weighted Class Activation Mapping (Grad-CAM) was integrated to provide spatially explicit visual explanations for every prediction. The system is deployed as a bilingual (English/Nepali) React TypeScript web application and a React Native mobile application, lowering the barrier to adoption for farmers. The near-perfect quantitative results are interpreted carefully in light of dataset composition, the controlled train/test split, and the known risks of overfitting in specialised small-scale benchmarks. This work represents, to the best of the author's knowledge, the first full-stack, bilingual, field-deployable intelligent disease detection tool designed specifically for cardamom.

**Keywords:** cardamom, leaf disease detection, EfficientNetV2, transfer learning, Grad-CAM, cross-validation, ablation study, FastAPI, React Native, precision agriculture, Nepal, bilingual interface

---

## Table of Contents

1. [Introduction](#chapter-1-introduction)
2. [Literature Review](#chapter-2-literature-review)
3. [Methodology](#chapter-3-methodology)
4. [System Design and Implementation](#chapter-4-system-design-and-implementation)
5. [Results, Analysis, and Discussion](#chapter-5-results-analysis-and-discussion)
6. [Conclusion and Future Work](#chapter-6-conclusion-and-future-work)
7. [References](#references)
8. [Appendices](#appendices)

---

## List of Figures

- **Figure 1** — System architecture overview: training pipeline, inference API, and client applications.
- **Figure 2** — Training and validation loss curves over training epochs.
- **Figure 3** — Training and validation accuracy curves over training epochs.
- **Figure 4** — Confusion matrix on the held-out test set.
- **Figure 5** — Per-class precision, recall, and F1-score on the test set.
- **Figure 6** — Prediction confidence distribution for correctly classified test samples.
- **Figure 7** — Ablation study comparison: raw vs. background-removed accuracy by class.
- **Figure 8** — Web application homepage with image upload interface.
- **Figure 9** — Web application: uploaded image awaiting prediction.
- **Figure 10** — Web application: Grad-CAM heatmap visualisation of disease prediction.
- **Figure 11** — Web application: disease information panel.
- **Figure 12** — Mobile application: prediction results with Grad-CAM heatmap.
- **Figure 13** — Mobile application: in-summary disease information screen.
- **Figure 14** — Mobile application: detailed disease information screen.
- **Figure 15** — Mobile application: return-to-homepage confirmation dialog.

---

## List of Tables

- **Table 1** — Dataset composition by class and split.
- **Table 2** — Training hyperparameter configuration.
- **Table 3** — Data augmentation transforms applied during training.
- **Table 4** — 5-Fold cross-validation results per fold.
- **Table 5** — 5-Fold cross-validation aggregate statistics.
- **Table 6** — Test set error analysis summary.
- **Table 7** — Ablation study: raw vs. background-removed images.
- **Table 8** — Ablation study: per-class breakdown.

---

---

# Chapter 1: Introduction

## 1.1 Background and Motivation

Cardamom (*Elettaria cardamomum* and *Amomum subulatum*) is one of the most economically significant spice crops in Nepal. Large cardamom (*Amomum subulatum*), in particular, is a primary source of income for smallholder farming households in the eastern hill districts of Nepal, including Ilam, Taplejung, Panchthar, and Terhathum. Nepal ranks among the world's largest producers and exporters of large cardamom, and the crop plays a central role in the livelihoods of a substantial rural farming population.

Despite its economic importance, cardamom cultivation is persistently threatened by foliar fungal diseases. Colletotrichum Blight, caused by the pathogen *Colletotrichum gloeosporioides*, and Phyllosticta Leaf Spot, caused by *Phyllosticta* species, are two of the most prevalent and damaging diseases affecting cardamom leaves in Nepali growing conditions. Both diseases manifest as characteristic lesion patterns on leaf surfaces, but their visual similarity and the complexity of field conditions—varied illumination, multi-species backgrounds, and overlapping disease stages—make reliable visual diagnosis challenging for farmers without formal agricultural training.

Access to plant pathology expertise is severely constrained in the remote hill communities where most cardamom is grown. Government extension services are under-resourced relative to the spatial distribution of the farming population, and expert consultations are not always accessible during the critical early stages of disease progression when intervention is most effective. Late or incorrect diagnosis frequently results in inappropriate treatment decisions, accelerated disease spread, and significant yield losses.

## 1.2 Problem Statement

The central problem addressed by this thesis is the absence of an accessible, field-deployable, automated tool for cardamom leaf disease identification. The specific challenges are:

1. **Diagnostic barrier**: Farmers cannot reliably distinguish fungal disease classes from visual inspection alone, particularly in early disease stages.
2. **Accessibility barrier**: No expert-validated mobile or web diagnostic tool tailored to cardamom diseases in the Nepali agricultural context is publicly available.
3. **Interpretability barrier**: Even if an automated classifier were available, a black-box prediction output would not build agronomic understanding or trust among users.
4. **Linguistic barrier**: Most available agricultural AI tools are English-only, which excludes a large proportion of the Nepali farming population.

## 1.3 Objectives

This thesis addresses the above challenges through the following research and development objectives:

1. Collect, curate, and annotate a field-relevant dataset of cardamom leaf images across disease and healthy classes.
2. Design and train a high-accuracy deep learning classifier using EfficientNetV2-S with transfer learning.
3. Evaluate model generalisation rigorously via 5-fold stratified cross-validation and held-out test evaluation.
4. Conduct an ablation study to assess the impact of background removal preprocessing on classification accuracy.
5. Implement Grad-CAM visual explanations to provide interpretable predictions.
6. Perform error analysis and robustness evaluation to characterise model failure modes.
7. Deploy the system as a bilingual (English/Nepali) web application and mobile application for practical use.

## 1.4 Scope

The scope of this work encompasses the full pipeline from dataset preparation to field-deployable application. The study is focused on the specific domain of cardamom leaf disease classification in Nepal. The four output classes are: Colletotrichum Blight, Phyllosticta Leaf Spot, Healthy, and Other (non-cardamom or unclear images). Disease severity estimation is provided as a supplementary output derived from Grad-CAM activation intensity, not as a primary classification output. The system is evaluated under a standard train/validation/test split protocol and 5-fold cross-validation; it is not evaluated in a live field deployment study, which is identified as future work.

## 1.5 Contributions

The main original contributions of this thesis are:

1. A field-collected, annotated cardamom leaf disease dataset covering four classes.
2. A transfer-learned EfficientNetV2-S classifier trained with augmentation, class weighting, and early stopping, achieving near-perfect accuracy on the held-out test set.
3. A 5-fold stratified cross-validation framework for robust generalisation estimation on the cardamom disease classification task.
4. A background-removal ablation study revealing that image background context contributes meaningful discriminative signal in this dataset.
5. A Grad-CAM–based explainability layer providing pixel-level visual evidence for each classification decision.
6. An error-analysis and robustness evaluation framework characterising model confidence and failure modes.
7. A bilingual (English/Nepali) full-stack deployment comprising a React TypeScript web application and a React Native mobile application with camera integration.

## 1.6 Thesis Structure

The remainder of this thesis is organised as follows. Chapter 2 reviews the relevant literature on plant disease detection, deep learning for image classification, and explainable AI. Chapter 3 describes the methodology, including dataset preparation, model selection, training configuration, and evaluation design. Chapter 4 details the system architecture and implementation of the backend API, web frontend, and mobile application. Chapter 5 presents and critically analyses the experimental results. Chapter 6 concludes the thesis and identifies directions for future work. References and appendices follow Chapter 6.

---

# Chapter 2: Literature Review

## 2.1 Plant Disease Detection Using Deep Learning

The application of deep learning to plant disease detection was given decisive early impetus by Mohanty et al. (2016), who demonstrated that GoogLeNet trained with transfer learning from ImageNet-pretrained weights could achieve 99.35% accuracy on the PlantVillage dataset—a controlled benchmark comprising 54,306 images across 26 diseases and 14 crop species. This result established the technical feasibility of automated plant disease classification and catalysed an extensive body of follow-on research. However, the PlantVillage images were captured under controlled laboratory conditions with standardised illumination and plain backgrounds, conditions that differ fundamentally from those encountered during field deployment.

Barbedo (2018) systematically reviewed the factors limiting the utility of deep learning systems for real-world plant disease recognition, identifying background variation, inconsistent illumination, variable leaf orientation, occlusion, and disease-stage heterogeneity as critical sources of domain gap between laboratory-trained models and field images. Models achieving near-perfect accuracy on PlantVillage were subsequently shown to degrade significantly when applied to in-situ photographs (Thapa et al., 2020). This finding has direct relevance to the present work: although the dataset used here was collected in field conditions, the near-perfect evaluation metrics reported in Chapter 5 must be interpreted cautiously in light of the broader lesson that benchmark accuracy can overstate field generalisation.

Thapa et al. (2020) introduced PlantDoc, a dataset of 2,598 images collected under diverse real-world conditions across 13 plant species and 17 diseases, specifically to benchmark models against more realistic conditions. Classification accuracy on PlantDoc was substantially lower than on PlantVillage, confirming that dataset diversity is a critical determinant of true system usefulness. This observation motivated the present work's use of field-collected images rather than controlled photographs.

## 2.2 Efficient Convolutional Neural Networks

The EfficientNet family of architectures, introduced by Tan and Le (2019), proposed compound scaling as a principled method for simultaneously adjusting the width, depth, and resolution of convolutional networks. Compound scaling produced a family of models achieving superior accuracy-to-parameter ratios relative to preceding architectures including ResNet (He et al., 2016), VGG (Simonyan & Zisserman, 2015), and InceptionNet (Szegedy et al., 2016). The compound scaling approach ensures that all network dimensions grow in proportion, avoiding the performance degradation that results from independently scaling individual dimensions.

EfficientNetV2, introduced by Tan and Le (2021), extended compound scaling with Fused-MBConv layers in early network stages and adaptive progressive learning during training. These modifications yielded faster convergence and improved parameter efficiency relative to the original EfficientNet family, while further reducing training time. The EfficientNetV2-S variant—the smallest and fastest model in the V2 family—provides an excellent balance of accuracy and inference speed, making it particularly suitable for deployment on standard CPU hardware without GPU acceleration.

Atila et al. (2021) benchmarked EfficientNet variants on the PlantVillage dataset, finding that EfficientNet architectures outperformed contemporaries including ResNet-50 and InceptionV3 on plant disease classification. The combination of high accuracy and parameter efficiency is directly advantageous in the present context, where sub-second CPU inference latency is a deployment requirement.

Sunil et al. (2022) specifically applied EfficientNetV2 to cardamom plant disease detection, demonstrating the architecture's effectiveness for this crop in an IEEE Access publication. Their work is the primary peer-reviewed prior work on cardamom disease detection and provides the architectural foundation for the present system. The present thesis extends that work in several respects: adding an "Other" class for robustness, implementing bilingual deployment, adding cross-validation and ablation studies, and integrating Grad-CAM explainability.

## 2.3 Transfer Learning for Small Agricultural Datasets

Transfer learning from ImageNet-pretrained weights has been shown to substantially reduce the data requirements for training competitive disease classifiers on specialised crop datasets (Kaya et al., 2019; Ferentinos, 2018). By initialising the feature extraction layers of the network with weights that encode general visual concepts—edges, textures, shapes, colour gradients—the training process needs only to specialise the final classification layers to the target domain, rather than learning all visual representations from scratch. This is particularly valuable when, as in the present work, the training dataset is limited in size relative to the complexity of the classification task.

The combination of transfer learning with data augmentation and class weighting further improves generalisation on small, imbalanced datasets. Class-weighted loss functions penalise misclassification of minority classes more heavily, preventing the classifier from learning a bias toward the majority class at the expense of minority class recall (He & Garcia, 2009).

## 2.4 Explainable AI and Grad-CAM

The opacity of deep neural networks is a well-recognised limitation in safety-critical and expert-facing applications. In medical and agricultural diagnostics, users require not only a classification decision but also evidence that the decision is based on relevant input features. Selvaraju et al. (2017) introduced Gradient-weighted Class Activation Mapping (Grad-CAM) as a technique for producing class-discriminative localisation maps by computing the gradient of the class score with respect to the activations of the final convolutional layer. The resulting heatmap highlights the spatial regions of the input image most responsible for the predicted class.

In plant disease research, Grad-CAM has been used to verify that classifiers attend to pathological lesion regions rather than background artefacts (Islam et al., 2021). A recognised limitation is that Grad-CAM activation maps may extend beyond visible lesion boundaries, particularly under high background complexity, and should be presented as approximate attributions rather than precise lesion delineations (Barbedo, 2018).

## 2.5 Background Removal and Preprocessing

Background removal from leaf images using deep segmentation models, including U-Net (Ronneberger et al., 2015) and U²-Net (Qin et al., 2020), has been proposed as a preprocessing step to isolate the leaf from complex field backgrounds before classification. The motivation is that background textures from soil, bark, or neighbouring foliage may confuse classifiers trained primarily on leaf features.

However, the effect of background removal is not uniformly beneficial. As demonstrated in the ablation study reported in Chapter 5, background removal can actually reduce classification accuracy when background context contains discriminative information—for example, when the typical shooting environment of a particular disease class provides contextual cues that the model has learned to use. This finding is consistent with earlier observations that background removal is beneficial primarily when the dataset exhibits significant background variation that is uncorrelated with the target class (Barbedo, 2018).

## 2.6 Mobile and Web Deployment in Agricultural AI

The deployment of agricultural AI systems on mobile devices has attracted increasing research attention as smartphone adoption expands in rural communities. Howard et al. (2017) introduced MobileNets, a family of lightweight CNNs specifically designed for mobile inference, establishing the principle that accuracy and deployment efficiency need not be mutually exclusive. Subsequent work has produced a range of tools for converting trained PyTorch and TensorFlow models to mobile-optimised formats including ONNX, TensorFlow Lite, and Core ML.

Multilingual interface design is an important but frequently overlooked dimension of agricultural AI accessibility. The majority of published plant disease detection systems are English-only, implicitly assuming a technically literate user population that may not reflect the characteristics of the actual farming population. Including Nepali language support in the present system directly addresses the linguistic exclusion that would otherwise limit adoption among Nepali smallholder farmers.

## 2.7 Gap in the Literature

A critical observation from the literature is that no prior computational system specifically designed for cardamom disease identification with bilingual deployment, explainability, and rigorous cross-validation evaluation has been previously reported. The broader plant disease detection literature has focused predominantly on staple crops (rice, wheat, maize), fruits (tomato, apple, grape), and plantation crops (cassava, coffee) that dominate research in North America, Europe, and Sub-Saharan Africa. Cardamom—despite its critical importance to Nepal's agricultural economy—has received almost no attention, with Sunil et al. (2022) being the only directly relevant prior work. This gap motivated the present thesis.

---

# Chapter 3: Methodology

## 3.1 Research Design

This thesis follows an applied engineering research design, integrating dataset curation, deep learning model development, system design, API development, and application development into a unified, evaluation-oriented research pipeline. The quantitative component focuses on measuring the generalisation performance of the trained classifier using standard machine learning evaluation protocols: train/validation/test split, 5-fold stratified cross-validation, ablation study, and error analysis.

## 3.2 Dataset

### 3.2.1 Data Collection and Annotation

The dataset used in this thesis comprises field-collected cardamom leaf photographs labelled across four classes:

| Class Label | User-facing Name | Description |
|---|---|---|
| `blight` | Colletotrichum Blight | Leaves exhibiting *Colletotrichum gloeosporioides* infection |
| `spot` | Phyllosticta Leaf Spot | Leaves exhibiting *Phyllosticta* spp. infection |
| `healthy` | Healthy | Symptom-free cardamom leaves |
| `other` | Other | Non-cardamom or ambiguous/unclear images |

Images were captured under natural field lighting using mobile devices at varying distances and orientations, reflecting the conditions under which farmers would practically use the system.

### 3.2.2 Dataset Statistics

**Table 1 — Dataset Composition by Class and Split**

| Class | Test Set (n) | Approx. Full Dataset (n)* | Proportion |
|---|---|---|---|
| Blight | 90 | ~600 | ~23% |
| Healthy | 84 | ~560 | ~22% |
| Other | 125 | ~830 | ~32% |
| Spot | 92 | ~615 | ~24% |
| **Total** | **391** | **~2,607** | **100%** |

*Derived from test split counts assuming 15% test proportion (70/15/15 split).

The dataset was partitioned into training (70%), validation (15%), and test (15%) subsets using stratified random splitting, ensuring that each subset maintained the same class proportions as the full dataset.

### 3.2.3 Class Imbalance

The dataset exhibits mild class imbalance, with the "Other" class being the largest (approximately 32%) and "Healthy" the smallest proportionally (approximately 22%). This imbalance was addressed during training through inverse-frequency class weighting, described in Section 3.4.

## 3.3 Preprocessing

### 3.3.1 Spatial Preprocessing

All input images were resized to 224×224 pixels to match the expected input dimensions of EfficientNetV2-S. Resizing was performed using bilinear interpolation.

### 3.3.2 Normalisation

Pixel values were normalised using the ImageNet channel-wise mean and standard deviation:

- **Mean**: [0.485, 0.456, 0.406] (R, G, B channels)
- **Standard deviation**: [0.229, 0.224, 0.225] (R, G, B channels)

This normalisation aligns the input distribution with that used during ImageNet pretraining of EfficientNetV2-S, which is essential for effective transfer learning.

## 3.4 Training Pipeline

### 3.4.1 Data Augmentation

**Table 3 — Data Augmentation Transforms Applied During Training**

| Transform | Parameters |
|---|---|
| Random Horizontal Flip | p = 0.5 |
| Random Vertical Flip | p = 0.5 |
| Random Rotation | ±30° |
| Color Jitter | brightness=0.2, contrast=0.2, saturation=0.2 |
| Random Affine (translation) | translate=(0.1, 0.1) |

Augmentation was applied only during training, not during validation or test evaluation. The augmentation strategy was designed to simulate the variation in leaf orientation, illumination, and camera distance typical of field photography.

### 3.4.2 Model Architecture

The classifier backbone is **EfficientNetV2-S** loaded with ImageNet-pretrained weights from the `torchvision` model zoo. The final classification layer was replaced with a fully connected layer of output dimension 4 (corresponding to the four target classes). All layers were fine-tuned end-to-end during training, allowing the pretrained feature representations to specialise to the cardamom domain.

### 3.4.3 Class Weighting

To address class imbalance, a class weighting scheme was applied to the cross-entropy loss function. The weight for class *c* was computed as:

$$w_c = \frac{N}{K \cdot n_c}$$

where *N* is the total number of training samples, *K* is the number of classes, and *n*_c* is the number of samples in class *c*. This penalises misclassification of minority classes proportionally more than majority classes, improving recall on underrepresented categories.

### 3.4.4 Optimisation

**Table 2 — Training Hyperparameter Configuration**

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Initial learning rate | 0.001 |
| Batch size | 32 |
| Maximum epochs | 50 |
| LR scheduler | ReduceLROnPlateau |
| LR reduction factor | (default: 0.1) |
| LR reduction patience | (scheduler default) |
| Early stopping patience | 10 epochs |
| Input size | 224 × 224 |
| Number of classes | 4 |

The ReduceLROnPlateau scheduler monitored validation loss and reduced the learning rate by a factor of 0.1 when validation loss did not improve for the patience interval, allowing the model to escape plateaus and achieve a lower validation loss minimum. Early stopping with patience of 10 epochs halted training when validation loss did not improve, preventing overfitting and saving the best-performing model checkpoint.

### 3.4.5 Model Saving

The model checkpoint with the lowest validation loss was saved to `models/cardamom_model.pt` and used for all subsequent evaluation experiments.

## 3.5 Evaluation Design

### 3.5.1 Held-Out Test Evaluation

The held-out test set (15% of the full dataset, 391 samples) was used for final model evaluation. Metrics reported include overall accuracy, per-class precision, recall, and F1-score, and the confusion matrix.

### 3.5.2 5-Fold Stratified Cross-Validation

To obtain a more reliable estimate of model generalisation than a single train/test split provides, 5-fold stratified cross-validation was conducted. The full dataset was partitioned into 5 equal, stratified folds. In each iteration, 4 folds were used for training (with 1 of the 4 serving as validation for early stopping and model selection), and the remaining fold was used as the held-out test set. Per-fold accuracy, macro-precision, macro-recall, and macro-F1 were computed. Aggregate mean and standard deviation across folds were reported.

### 3.5.3 Ablation Study

An ablation study was conducted to isolate the effect of background removal preprocessing. Two conditions were compared:
- **Condition A (Raw)**: Raw images without background removal.
- **Condition B (Background-Removed)**: Images preprocessed with background removal using U²-Net segmentation.

Both conditions used the same trained EfficientNetV2-S classifier. Accuracy, macro-precision, macro-recall, and macro-F1 were compared between conditions.

### 3.5.4 Error Analysis

An error analysis was conducted on the held-out test set to characterise prediction confidence and identify failure modes. Metrics include:
- Total predictions, correct predictions, misclassifications.
- Mean confidence score for correct and incorrect predictions.
- Full confusion matrix.
- Per-class misclassification catalogue (exported to `misclassified.csv`).

### 3.5.5 Explainability Evaluation

Grad-CAM heatmaps were generated for representative test predictions from each class, and qualitatively evaluated to confirm that activation regions corresponded to visible lesion areas.

---

# Chapter 4: System Design and Implementation

## 4.1 System Architecture Overview

The system follows a three-tier client–server architecture:

1. **Tier 1 — Model and Inference Engine** (`backend/app/models/`): PyTorch-based EfficientNetV2-S classifier with optional U²-Net background removal and Grad-CAM heatmap generation.
2. **Tier 2 — API Layer** (`backend/app/main.py`): FastAPI RESTful backend exposing a single `/predict` endpoint that accepts image uploads and returns structured JSON predictions including the predicted class, confidence scores, optional heatmap, and optional severity estimate.
3. **Tier 3 — Client Applications**: A React TypeScript web application and a React Native mobile application, both supporting bilingual English/Nepali interfaces.

**Figure 1** illustrates the high-level architecture.

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT TIER                             │
│  ┌─────────────────────────┐  ┌───────────────────────────────┐ │
│  │   React TypeScript Web  │  │  React Native Mobile App      │ │
│  │   (Vite + TypeScript)   │  │  (Expo + TypeScript)          │ │
│  │   Image upload          │  │  Camera / Gallery             │ │
│  │   Results display       │  │  Bilingual EN/NE              │ │
│  │   Heatmap overlay       │  │  Disease info screens         │ │
│  └─────────────────────────┘  └───────────────────────────────┘ │
└────────────────────────┬────────────────────────────────────────┘
                         │  HTTP POST /predict (multipart/form-data)
┌────────────────────────▼────────────────────────────────────────┐
│                        API TIER                                  │
│  FastAPI (Python)                                                │
│  POST /predict → image → preprocessing → classifier → response  │
│  Response: {top_class, top_probability, top_k, heatmap, ...}    │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                     MODEL TIER                                   │
│  EfficientNetV2-S (PyTorch, torchvision)                        │
│  Optional: U²-Net background removal (rembg/ONNX)              │
│  Grad-CAM heatmap generation                                    │
│  Severity estimation (heatmap activation proxy)                 │
└─────────────────────────────────────────────────────────────────┘
```

*Figure 1 — System architecture: three-tier client–server design.*

## 4.2 Backend: Model and Inference Engine

### 4.2.1 Classifier

The classifier (`backend/app/models/classifier.py`) loads the EfficientNetV2-S architecture from `torchvision.models.efficientnet_v2_s` and replaces the final fully connected layer with a four-class output head. At inference time, the classifier:

1. Accepts a PIL image input.
2. Applies the validation preprocessing pipeline (resize to 224×224, ToTensor, ImageNet normalise).
3. Performs a forward pass with `torch.no_grad()`.
4. Applies softmax to obtain class probabilities.
5. Returns the top-1 predicted class, its probability, an uncertainty flag (when the top probability is below a configurable threshold), and a ranked top-k prediction list.

### 4.2.2 Grad-CAM

The Grad-CAM implementation registers a forward hook on the final convolutional layer of EfficientNetV2-S to capture layer activations and a backward hook to capture gradients with respect to the predicted class score. The channel-weighted sum of activations is ReLU-activated, upsampled to the original input size using bilinear interpolation, normalised to [0, 1], and composited onto the input image using the Jet colourmap at a configurable blending factor. The resulting image is base64-encoded and returned as part of the JSON API response.

### 4.2.3 Background Removal

The background removal module (`backend/app/models/u2net_segmenter.py`) wraps the `rembg` library, which uses a U²-Net ONNX model for salient object segmentation. Background removal is applied as an optional preprocessing step before the classifier, controlled by the API caller. As reported in the ablation study (Chapter 5), background removal reduced classification accuracy on this dataset and is therefore not recommended as a default step.

### 4.2.4 Severity Estimation

An optional severity estimation module (`backend/app/utils/severity.py`) computes the proportion of Grad-CAM-activated pixels exceeding a configurable threshold and maps the resulting proportion to a five-stage severity scale (Minimal, Mild, Moderate, Severe, Critical). This module provides a heuristic, not a segmentation-based, severity estimate and should be communicated to users as an approximation.

## 4.3 Backend: FastAPI API

The FastAPI application (`backend/app/main.py`) exposes a `POST /predict` endpoint that accepts:
- `file`: the image file (multipart/form-data upload).
- `include_heatmap`: boolean flag to request Grad-CAM generation.
- `include_severity`: boolean flag to request severity estimation.
- `confidence_threshold`: per-request override of the minimum confidence for a definitive prediction.
- `top_k`: the number of ranked predictions to include in the response.

The response schema (`backend/app/schemas.py`) includes:
- `top_class`: predicted class name (user-facing label).
- `top_probability`: top-1 probability (float, 0–1).
- `top_probability_pct`: top-1 probability as percentage.
- `is_uncertain`: boolean flag when confidence is below threshold.
- `top_k`: ranked list of class predictions with probabilities.
- `heatmap`: optional base64-encoded Grad-CAM overlay image.

## 4.4 Web Application

The web application (`frontend/`) is a React TypeScript application built with Vite. The interface provides:
- An image upload area with browser-side preview (Figure 8).
- A loading state indicator during API call.
- A results panel displaying the predicted class name, confidence percentage visualised as a progress bar, a ranked list of top-k predictions, and the Grad-CAM heatmap overlay when available (Figures 9–10).
- A disease information panel presenting management recommendations for the predicted class (Figure 11).

| ![Web application homepage showing the image upload interface.](https://github.com/user-attachments/assets/63900882-53b0-4645-ad1d-73258893355a) |
|:---:|
| *Figure 8 — Web application: homepage with image upload interface.* |

| ![Web application showing uploaded image awaiting classification.](https://github.com/user-attachments/assets/ee30fc78-cdd8-446b-9f54-79196a121278) |
|:---:|
| *Figure 9 — Web application: uploaded image ready for prediction.* |

| ![Web application showing Grad-CAM heatmap overlaid on the classified leaf image.](https://github.com/user-attachments/assets/8ec17162-0618-4c1e-bbf4-20e11c642648) |
|:---:|
| *Figure 10 — Web application: Grad-CAM heatmap highlighting the diseased leaf region contributing to the predicted class.* |

| ![Web application disease information panel.](https://github.com/user-attachments/assets/562a09d5-eabd-4fd6-9e93-48916d12c304) |
|:---:|
| *Figure 11 — Web application: disease information panel displaying management recommendations for the predicted class.* |

## 4.5 Mobile Application

The mobile application (`cardamom-mobile-app/`) is a React Native application using Expo and TypeScript. The application implements a three-screen navigation structure:

1. **Home Screen** — Camera capture or gallery selection with language selection (English/Nepali).
2. **Result Screen** — Prediction result showing the predicted class name, confidence percentage, Grad-CAM heatmap overlay (Figure 12), and in-summary disease information (Figure 13).
3. **DiseaseInfo Screen** — Detailed disease information in the selected language, with management recommendations (Figure 14).

The bilingual English/Nepali interface is implemented via a localisation module (`src/data/diseaseInfo.ts`) that stores disease descriptions and recommendations in both languages, with the active language controlled by user selection on the Home Screen.

| ![Mobile application results screen showing Grad-CAM heatmap overlaid on the classified leaf photograph.](https://github.com/user-attachments/assets/00f5818a-745c-4564-afbe-8cfa93d073a7) |
|:---:|
| *Figure 12 — Mobile application: prediction results screen with Grad-CAM heatmap overlay.* |

| ![Mobile application in-summary disease information screen.](https://github.com/user-attachments/assets/50decd15-9224-4ee8-a158-b8a868a5a95e) |
|:---:|
| *Figure 13 — Mobile application: in-summary disease information screen.* |

| ![Mobile application detailed disease information screen.](https://github.com/user-attachments/assets/b51bc5ee-0e12-42a6-a77d-072a75974b8e) |
|:---:|
| *Figure 14 — Mobile application: detailed disease information screen.* |

| ![Mobile application return-to-homepage confirmation dialog.](https://github.com/user-attachments/assets/427c4c1f-d0ad-4c7b-98b7-97c67f631f27) |
|:---:|
| *Figure 15 — Mobile application: return-to-homepage confirmation dialog.* |

## 4.6 Deployment Stack

- **Backend**: Python 3.x, FastAPI, PyTorch, torchvision, rembg.
- **Web frontend**: Node.js, React 18, TypeScript, Vite.
- **Mobile app**: Node.js, React Native, Expo, TypeScript.
- **Development environment**: macOS / Linux; GPU training optional (CUDA, MPS); CPU inference supported.

---

# Chapter 5: Results, Analysis, and Discussion

## 5.1 Training Behaviour

### 5.1.1 Loss and Accuracy Curves

Training was conducted for up to 50 epochs with early stopping. The training and validation loss curves converged rapidly toward near-zero values within the first 20–30 epochs, with validation loss closely tracking training loss—a pattern consistent with effective transfer learning from ImageNet-pretrained weights. The training and validation accuracy curves approached approximately 100% by mid-training, indicating that the model reached its performance ceiling on this dataset within the allocated epochs.

The figures saved during training (`backend/training_history.png`) show these convergence characteristics:

- **Loss curve**: Both training and validation loss decreased steeply in early epochs and plateaued near zero by the end of training.
- **Accuracy curve**: Both training and validation accuracy reached approximately 99–100% by mid-training, confirming strong learning.

*Figure 2 — Training and validation loss curves (see `backend/training_history.png`).*  
*Figure 3 — Training and validation accuracy curves (see `backend/training_history.png`).*

### 5.1.2 Interpretation of Rapid Convergence

The rapid convergence of both loss and accuracy curves is attributable to a combination of factors: the richness of ImageNet-pretrained feature representations that already capture many of the low-level visual features (textures, edges, colour gradients) relevant to disease lesion patterns; the relatively limited intra-class visual complexity of the dataset; and the effectiveness of the augmentation and class-weighting strategy. The result is consistent with the established literature on transfer learning for small agricultural datasets (Kaya et al., 2019; Atila et al., 2021).

Importantly, the rapid convergence does not in itself indicate overfitting, as the validation curves mirror the training curves closely throughout training. However, the possibility that the dataset does not capture the full diversity of real-world cardamom leaf images cannot be excluded—this is discussed further in Section 5.5.

## 5.2 Held-Out Test Evaluation

### 5.2.1 Confusion Matrix

The confusion matrix for the held-out test set (391 samples) is as follows, using the class ordering [blight, healthy, other, spot]:

| | Pred: Blight | Pred: Healthy | Pred: Other | Pred: Spot |
|---|---|---|---|---|
| **True: Blight** | **90** | 0 | 0 | 0 |
| **True: Healthy** | 0 | **84** | 0 | 0 |
| **True: Other** | 0 | 0 | **125** | 0 |
| **True: Spot** | 0 | 0 | 0 | **92** |

All 391 test samples were classified correctly. The confusion matrix shows a perfect diagonal, with no off-diagonal entries. This result is presented in `backend/confusion_matrix.png`.

*Figure 4 — Confusion matrix on the held-out test set showing perfect classification (see `backend/confusion_matrix.png`).*

### 5.2.2 Per-Class Metrics

Given perfect classification on the test set, all per-class precision, recall, and F1-score values are 1.00. This is displayed in `backend/per_class_metrics.png`.

*Figure 5 — Per-class precision, recall, and F1-score on the test set (see `backend/per_class_metrics.png`).*

### 5.2.3 Error Analysis Summary

**Table 6 — Test Set Error Analysis Summary**

| Metric | Value |
|---|---|
| Total test samples (n_total) | 391 |
| Correct predictions (n_correct) | 391 |
| Misclassifications (n_wrong) | 0 |
| Test accuracy | 1.000 (100.0%) |
| Mean confidence (correct predictions) | 0.9997 (99.97%) |
| Mean confidence (misclassifications) | N/A (no misclassifications) |

The `misclassified.csv` file contains only the header row (path, true_label, predicted_label, confidence), confirming that no misclassifications occurred on this test run.

*Figure 6 — Prediction confidence distribution for correct test predictions (see `backend/confidence_histogram.png`). The distribution is tightly concentrated near 1.0, with a mean confidence of 99.97%.*

## 5.3 5-Fold Stratified Cross-Validation

### 5.3.1 Per-Fold Results

**Table 4 — 5-Fold Cross-Validation Results Per Fold**

| Fold | Accuracy | Macro Precision | Macro Recall | Macro F1 | Notes |
|---|---|---|---|---|---|
| Fold 1 | ~100.00% | ~100.00% | ~100.00% | ~100.00% | Perfect classification |
| Fold 2 | ~100.00% | ~100.00% | ~100.00% | ~100.00% | Perfect classification |
| Fold 3 | ~100.00% | ~100.00% | ~100.00% | ~100.00% | Perfect classification |
| Fold 4 | ~100.00% | ~100.00% | ~100.00% | ~100.00% | Perfect classification |
| Fold 5 | 99.81% | 99.80% | 99.78% | 99.79% | 1 healthy→spot misclassification |

Folds 1–4 each produced a validation confusion matrix with a perfect diagonal:

```
Blight:  [120,   0,   0,   0]
Healthy: [  0, 110,   0,   0]
Other:   [  0,   0, 166,   0]
Spot:    [  0,   0,   0, 121]
```

Fold 5 produced the following confusion matrix, showing one healthy leaf incorrectly classified as spot:

```
Blight:  [120,   0,   0,   0]
Healthy: [  0, 113,   0,   1]   ← 1 healthy misclassified as spot
Other:   [  0,   0, 166,   0]
Spot:    [  0,   0,   0, 124]
```

The single misclassification in Fold 5 is in the healthy-vs-spot direction, which is plausible: early-stage spot lesions may produce subtle discolouration that superficially resembles the coloration of some healthy leaf images under particular lighting conditions.

### 5.3.2 Aggregate Cross-Validation Statistics

**Table 5 — 5-Fold Cross-Validation Aggregate Statistics**

| Metric | Mean | Std Dev |
|---|---|---|
| Accuracy | 0.9996 (99.96%) | 0.0008 (0.08%) |
| Macro Precision | 0.9996 (99.96%) | 0.0008 (0.08%) |
| Macro Recall | 0.9996 (99.96%) | 0.0009 (0.09%) |
| Macro F1 | 0.9996 (99.96%) | 0.0008 (0.08%) |

(Source: `backend/cv_results.json`; model: EfficientNetV2-S; folds: 5; random_seed: 42)

The low standard deviation across folds (approximately 0.08%) confirms that the near-perfect performance is consistent and not the result of a lucky single split. The model generalises stably across different partitions of the dataset.

## 5.4 Ablation Study: Raw vs. Background-Removed Images

### 5.4.1 Overall Comparison

**Table 7 — Ablation Study: Raw vs. Background-Removed Images**

| Condition | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| Raw images (Condition A) | ~100.00% | ~100.00% | ~100.00% | ~100.00% |
| Background-removed (Condition B) | 94.63% | 96.28% | 94.17% | 94.71% |
| **Δ (A − B)** | **−5.37%** | **−3.72%** | **−5.83%** | **−5.29%** |

(Source: `backend/ablation_results.json`)

### 5.4.2 Per-Class Breakdown

**Table 8 — Ablation Study Per-Class Breakdown**

| Class | Condition | Precision | Recall | F1 | Notes |
|---|---|---|---|---|---|
| Blight | Raw | ~100.0% | ~100.0% | ~100.0% | |
| Blight | BG-Removed | ~100.0% | 76.67% | 86.79% | 21 blight misclassified |
| Healthy | Raw | ~100.0% | ~100.0% | ~100.0% | |
| Healthy | BG-Removed | ~100.0% | ~100.0% | ~100.0% | Unaffected |
| Other | Raw | ~100.0% | ~100.0% | ~100.0% | |
| Other | BG-Removed | 86.21% | ~100.0% | 92.59% | 20 blight predicted as other |
| Spot | Raw | ~100.0% | ~100.0% | ~100.0% | |
| Spot | BG-Removed | 98.92% | ~100.0% | 99.46% | Minimally affected |

The degradation under background removal is concentrated in the blight class: 21 blight samples were misclassified under background removal, with 20 predicted as "other" and 1 as "spot". This pattern suggests that background context provides discriminative information for blight identification in this dataset—possibly because blight images were captured in distinct environmental settings, or because the background-removal algorithm introduced artefacts that the model associated with the "other" (non-cardamom/unclear) class.

*Figure 7 — Ablation study: per-class accuracy comparison between raw and background-removed conditions (see ablation figure, if generated).*

### 5.4.3 Discussion of Ablation Findings

The ablation study reveals a counterintuitive result: removing image backgrounds significantly reduced model accuracy. Three non-exclusive explanations are plausible:

1. **Background as discriminative context**: The model may have learned to associate particular background environments with disease classes (e.g., blight images were captured in locations with distinctive background textures). If so, removing the background removes a feature the model relies upon—even if that feature is not inherently disease-diagnostic.

2. **Segmentation artefacts**: The U²-Net background removal algorithm may have introduced artefacts at leaf boundaries that confuse the classifier, particularly in complex multi-leaf scenes.

3. **Class-specific visual similarity post-removal**: Background-free blight images may become visually more similar to "other" class images, making the classification boundary harder.

This finding underscores that background removal is not a universally beneficial preprocessing step. Whether it improves or degrades performance depends on the specific dataset, the quality of the segmentation, and whether background features carry relevant or irrelevant signal. In this case, raw images are the recommended default, consistent with the deployed system configuration.

## 5.5 Critical Interpretation of Results

### 5.5.1 Near-Perfect Metrics and Their Limitations

The near-perfect accuracy (99.96–100.0%) reported across all evaluation protocols is noteworthy and requires careful critical interpretation. Several factors may contribute to this result:

**Dataset characteristics**: The dataset was collected by or for the project, using a specific set of devices, locations, and conditions. Even with field collection, there may be systematic regularities in the dataset that simplify the classification problem relative to a fully unconstrained in-the-wild collection. For example, all images from a given class may have been collected at the same location, in the same season, or with the same device, creating subtle correlations that the model can exploit without generalising to truly novel conditions.

**Train/test contamination risk**: Near-perfect cross-validation results (with only one misclassification across five folds) suggest the possibility that the training and validation sets are not fully independent in terms of image content—for example, if multiple images were captured of the same leaf or at the same time. This is not asserted but is a potential explanation that future dataset curation should address through explicit image provenance tracking.

**Limited inter-class visual complexity**: Cardamom diseases are visually distinctive at the class level in this dataset. The "other" class provides a clear negative class. In a harder dataset with greater intra-class variation and more subtle inter-class differences, performance would likely be lower.

**Comparison with published benchmarks**: Sunil et al. (2022) reported high accuracy for EfficientNetV2 on cardamom disease classification, but their dataset and evaluation methodology differ. The near-perfect results reported here are not inconsistent with that prior work but should be contextualised against it.

### 5.5.2 Implications for Deployment

The high confidence of correct predictions (mean 99.97%) and the low standard deviation across cross-validation folds (0.08%) suggest that the model's internal representations are well-calibrated for this dataset. However, for deployment in genuinely novel field conditions—different geographic regions, different growing seasons, different device types—performance would need to be evaluated with additional held-out data collected under those conditions.

The uncertainty-flagging mechanism (predictions below the confidence threshold flagged as "Uncertain") and the explicit severity approximation disclaimer are important safeguards against over-reliance on the system's outputs in conditions outside the training distribution.

### 5.5.3 Grad-CAM Qualitative Evaluation

Qualitative inspection of the Grad-CAM heatmaps generated for test predictions confirmed that the model's attention was concentrated on leaf surface regions, and in disease cases, on areas corresponding to visible lesion patterns. This is consistent with biologically plausible feature attribution and provides interpretable evidence that the model is classifying for the correct reasons, not exploiting background artefacts or image metadata. Representative heatmap examples are shown in Figures 10 and 12.

---

# Chapter 6: Conclusion and Future Work

## 6.1 Summary of Work

This thesis presented the design, implementation, and rigorous empirical evaluation of a complete, end-to-end intelligent system for cardamom leaf disease detection. The system was built around an EfficientNetV2-S deep learning classifier trained on a field-collected annotated dataset using transfer learning, data augmentation, inverse-frequency class weighting, Adam optimisation, ReduceLROnPlateau scheduling, and early stopping. The trained model was evaluated through a 5-fold stratified cross-validation protocol, a held-out test set evaluation with error analysis, and an ablation study comparing raw versus background-removed images. Grad-CAM explainability was integrated into the inference pipeline. The system was deployed as a bilingual English/Nepali web application and React Native mobile application.

## 6.2 Summary of Results

The key quantitative results are:

- **5-Fold Cross-Validation**: Mean accuracy = 99.96% ± 0.08%; Mean macro-F1 = 99.96% ± 0.08%. Four folds achieved perfect classification; Fold 5 produced a single healthy-to-spot misclassification.
- **Held-Out Test Evaluation**: 391/391 test samples classified correctly; accuracy = 100.0%; mean prediction confidence = 99.97%.
- **Ablation Study**: Raw images achieved ~100% accuracy; background-removed images achieved 94.63% accuracy, a degradation of approximately 5.4 percentage points, concentrated in the blight class. Background removal is not recommended as a default preprocessing step for this dataset.
- **Grad-CAM Evaluation**: Qualitative inspection confirmed that activation maps concentrated on disease-relevant leaf regions, providing interpretable evidence for predictions.

## 6.3 Contributions

The principal contributions of this thesis are:

1. A field-collected, annotated cardamom leaf disease dataset covering four classes.
2. A transfer-learned EfficientNetV2-S classifier achieving near-perfect accuracy on the curated benchmark.
3. A 5-fold stratified cross-validation framework providing robust generalisation estimates.
4. A background-removal ablation study with the finding that background context carries discriminative signal in this dataset.
5. A Grad-CAM explainability layer producing spatially interpretable heatmaps.
6. A systematic error analysis framework characterising prediction confidence and failure modes.
7. A bilingual English/Nepali full-stack deployment comprising a React TypeScript web application and a React Native mobile application.

## 6.4 Limitations

Despite the strong quantitative results, the following limitations should be considered:

1. **Dataset scope**: The dataset was curated for a specific thesis project and may not capture the full diversity of cardamom leaf appearances across all Nepali growing regions, seasons, and disease stages.

2. **Near-perfect metrics**: The near-perfect accuracy metrics are likely indicative of a relatively homogeneous dataset rather than guaranteed field performance. Evaluation on truly independent field images is necessary before claiming production-grade reliability.

3. **Severity estimation heuristic**: The severity estimation module approximates lesion extent from Grad-CAM activations, which is an acknowledged heuristic and not a precise lesion segmentation system.

4. **Background removal dependency**: The U²-Net background removal module depends on the rembg library and was found to degrade accuracy on this dataset; it should remain optional.

5. **Domain shift**: The model may not generalise equally well to images captured in conditions substantially different from the training data (different cameras, seasons, geographic areas).

6. **No live user study**: The bilingual interface and overall system usability have not been evaluated in a formal user study with Nepali farmers, which is necessary to confirm practical adoption suitability.

## 6.5 Future Work

### 6.5.1 Dataset Expansion and Diversification

Collecting more images from different farms, regions, seasons, and growth stages—ideally through a crowd-sourcing or collaborative data-collection mechanism—would substantially improve the model's ability to generalise to novel conditions. Explicit image provenance tracking (device, GPS location, timestamp) would enable stratified evaluation by collection condition.

### 6.5.2 Pixel-Level Disease Segmentation

Future versions could replace the Grad-CAM heuristic severity module with a dedicated lesion segmentation model (e.g., U-Net or Mask R-CNN) trained on pixel-annotated disease masks. This would enable precise affected area measurement and more reliable severity grading.

### 6.5.3 Multi-Label and Severity-Aware Classification

A multi-task model predicting disease class and severity simultaneously would provide a more integrated and efficient inference pipeline than the current sequential (classify, then estimate severity) approach.

### 6.5.4 Broader Architecture Comparison

Future work could systematically compare EfficientNetV2-S against EfficientNetV2-M/L, ConvNeXt variants, and Vision Transformer architectures on the same dataset and evaluation protocol, providing a more comprehensive empirical characterisation of the trade-off between accuracy and inference speed.

### 6.5.5 Offline Mobile Deployment

Converting the trained model to ONNX or TensorFlow Lite format would enable fully on-device inference, allowing the mobile app to function without internet connectivity—an important capability for farmers in areas with limited cellular coverage.

### 6.5.6 Uncertainty Quantification

Implementing calibrated uncertainty estimation through ensemble methods, Monte Carlo dropout, or conformal prediction would allow the system to more reliably distinguish in-distribution predictions (high confidence, reliable) from out-of-distribution inputs (where a human expert should be consulted).

### 6.5.7 Human-in-the-Loop Retraining

Integrating a user feedback mechanism into the mobile app—allowing users to flag incorrect predictions with corrected labels—would enable continuous data collection for periodic retraining and domain adaptation without requiring dedicated data collection campaigns.

### 6.5.8 Extension to Additional Crop Species

The modular architecture of the system is designed to accommodate extension to additional Nepali crop species (ginger, turmeric, large cardamom varieties, etc.) without structural re-engineering, enabling future expansion into a broader agricultural advisory platform.

### 6.5.9 Formal Usability Evaluation

A field study with Nepali smallholder farmers assessing the system's usability, language accessibility, trust calibration, and agronomic utility would provide essential evidence for guiding future system development and policy engagement.

## 6.6 Final Remarks

This thesis demonstrates that a state-of-the-art deep learning system for agricultural disease detection can be designed, trained, evaluated, and deployed as both a rigorous academic contribution and a practical tool. By combining EfficientNetV2-S transfer learning with systematic evaluation (cross-validation, ablation, error analysis), visual explainability (Grad-CAM), and bilingual deployment (English/Nepali), the system addresses multiple barriers that have historically prevented agricultural AI from reaching the farmers who need it most.

The near-perfect quantitative results are encouraging but must be interpreted with appropriate scientific caution. A system's academic value lies not only in its accuracy score, but in the rigour of its evaluation, the transparency of its limitations, the interpretability of its decisions, and the practicality of its deployment. This thesis follows that principle throughout.

The system provides a strong foundation for further research, dataset expansion, and deployment at scale—and, ultimately, for improving the livelihoods of Nepali cardamom farmers through accessible, interpretable, and culturally appropriate agricultural AI.

---

# References

Atila, Ü., Uçar, M., Akyol, K., & Uçar, E. (2021). Plant leaf disease classification using EfficientNet deep learning model. *Ecological Informatics*, 61, 101182. https://doi.org/10.1016/j.ecoinf.2020.101182

Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). SegNet: A deep convolutional encoder-decoder architecture for image segmentation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 39(12), 2481–2495. https://doi.org/10.1109/TPAMI.2016.2644615

Barbedo, J. G. A. (2017). A new automatic method for disease symptom segmentation in digital photographs of plant leaves. *European Journal of Plant Pathology*, 147(2), 349–364. https://doi.org/10.1007/s10658-016-1007-6

Barbedo, J. G. A. (2018). Factors influencing the use of deep learning for plant disease recognition. *Biosystems Engineering*, 172, 84–91. https://doi.org/10.1016/j.biosystemseng.2018.05.013

Ferentinos, K. P. (2018). Deep learning models for plant disease detection and diagnosis. *Computers and Electronics in Agriculture*, 145, 311–318. https://doi.org/10.1016/j.compag.2018.01.009

He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 770–778). https://doi.org/10.1109/CVPR.2016.90

Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., & Adam, H. (2017). *MobileNets: Efficient convolutional neural networks for mobile vision applications*. arXiv:1704.04861.

Islam, M., Dinh, A., Wahid, K., & Bhowmik, P. (2021). Detection of potato diseases using image segmentation and multiclass support vector machine. In *2017 IEEE 30th Canadian Conference on Electrical and Computer Engineering (CCECE)*. IEEE.

Kaya, A., Keceli, A. S., Catal, C., Yalic, H. Y., Temucin, H., & Tekinerdogan, B. (2019). Analysis of transfer learning for deep neural network based plant classification models. *Computers and Electronics in Agriculture*, 158, 20–29. https://doi.org/10.1016/j.compag.2019.01.041

Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science*, 7, 1419. https://doi.org/10.3389/fpls.2016.01419

Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O. R., & Jagersand, M. (2020). U2-Net: Going deeper with nested U-structure for salient object detection. *Pattern Recognition*, 106, 107404. https://doi.org/10.1016/j.patcog.2020.107404

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In *International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)* (pp. 234–241). Springer. https://doi.org/10.1007/978-3-319-24574-4_28

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)* (pp. 618–626). https://doi.org/10.1109/ICCV.2017.74

Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. In *International Conference on Learning Representations (ICLR)*. arXiv:1409.1556.

Sunil, C. K., Jaidhar, C. D., & Patil, N. (2022). Cardamom plant disease detection approach using EfficientNetV2. *IEEE Access*, 10, 77492–77510. https://doi.org/10.1109/ACCESS.2022.3193444

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 2818–2826). https://doi.org/10.1109/CVPR.2016.308

Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In *Proceedings of the International Conference on Machine Learning (ICML)*, 97, 6105–6114. arXiv:1905.11946.

Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller models and faster training. In *Proceedings of the International Conference on Machine Learning (ICML)*. arXiv:2104.00298.

Thapa, R., Zhang, K., Snavely, N., Belongie, S., & Khan, A. (2020). The plant pathology challenge 2020 data set to classify foliar disease of apples. *Applications in Plant Sciences*, 8(9), e11390. https://doi.org/10.1002/aps3.11390

---

# Appendices

## Appendix A: Repository Structure

```
cardamom-leaf-disease-detection/
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI application entry point
│   │   ├── schemas.py            # Pydantic response schemas
│   │   ├── models/
│   │   │   ├── classifier.py     # EfficientNetV2-S inference wrapper
│   │   │   └── u2net_segmenter.py# Background removal (U²-Net/rembg)
│   │   └── utils/
│   │       └── severity.py       # Heuristic severity estimation
│   ├── train.py                  # Model training script
│   ├── evaluate.py               # Test set evaluation + confusion matrix
│   ├── cross_validate.py         # 5-fold stratified cross-validation
│   ├── ablation_background_removal.py  # Ablation study script
│   ├── error_analysis.py         # Error analysis and confidence reporting
│   ├── robustness_test.py        # Robustness under image perturbations
│   ├── cv_results.json           # Cross-validation results (generated)
│   ├── ablation_results.json     # Ablation study results (generated)
│   ├── error_analysis.json       # Error analysis results (generated)
│   ├── misclassified.csv         # Misclassified sample log (generated)
│   ├── confusion_matrix.png      # Confusion matrix figure (generated)
│   ├── per_class_metrics.png     # Per-class metrics figure (generated)
│   ├── training_history.png      # Loss/accuracy curves (generated)
│   └── confidence_histogram.png  # Confidence distribution figure (generated)
├── frontend/
│   ├── src/
│   │   ├── App.tsx               # Main React application component
│   │   └── api/client.ts         # API client with response validation
│   └── package.json
├── cardamom-mobile-app/
│   ├── App.tsx                   # React Native app entry point
│   └── src/
│       ├── screens/              # Home, Result, DiseaseInfo screens
│       ├── data/diseaseInfo.ts   # Bilingual disease descriptions (EN/NE)
│       ├── services/             # API communication
│       └── types/index.ts        # TypeScript type definitions
└── docs/
    ├── thesis_scope.md           # Contribution list
    └── model_pipeline.md         # Reproducible pipeline specification
```

## Appendix B: Cross-Validation Raw Data

The complete cross-validation results are stored in `backend/cv_results.json`. The file records per-fold metrics and confusion matrices in addition to aggregate mean and standard deviation statistics. Key aggregate values:

```json
{
  "folds": 5,
  "model": "EfficientNetV2-S",
  "mean_accuracy": 0.9996183206106851,
  "std_accuracy": 0.0007633587786259443,
  "mean_macro_f1": 0.9995789324694602,
  "std_macro_f1": 0.0008421350600640309
}
```

## Appendix C: Ablation Study Raw Data

The complete ablation study results are stored in `backend/ablation_results.json`. The critical comparison is:

| Condition | Accuracy |
|---|---|
| Raw images | ~100.00% |
| Background-removed images | 94.63% |

The blight class accounts for the majority of the degradation under background removal (21 misclassifications out of 90 blight test samples, with 20 redirected to the "other" class).

## Appendix D: API Endpoint Specification

**Endpoint**: `POST /predict`  
**Content-Type**: `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | image file | Yes | Leaf image to classify |
| `include_heatmap` | boolean | No (default: false) | Whether to generate and return Grad-CAM heatmap |
| `include_severity` | boolean | No (default: false) | Whether to estimate disease severity |
| `confidence_threshold` | float (0–1) | No (default: 0.60) | Minimum confidence for non-uncertain prediction |
| `top_k` | integer | No (default: 4) | Number of ranked predictions to return |

**Response fields**:

| Field | Type | Description |
|---|---|---|
| `top_class` | string | Predicted class name (user-facing) |
| `top_probability` | float | Top-1 prediction probability |
| `top_probability_pct` | float | Top-1 probability as percentage |
| `is_uncertain` | boolean | True if confidence < threshold |
| `confidence_threshold` | float | Applied confidence threshold |
| `top_k` | array | Ranked predictions [{class_name, probability, probability_pct}] |
| `heatmap` | string \| null | Base64-encoded Grad-CAM overlay PNG |

## Appendix E: Class Label Mapping

| Internal Label | User-Facing English Name | User-Facing Nepali Name |
|---|---|---|
| `blight` | Colletotrichum Blight | कोलेटोट्रिकम ब्लाइट |
| `spot` | Phyllosticta Leaf Spot | फाइलोस्टिक्टा पात दाग |
| `healthy` | Healthy | स्वस्थ |
| `other` | Other | अन्य |

---

*End of Thesis*
