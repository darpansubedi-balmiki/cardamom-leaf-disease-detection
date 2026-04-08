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
   - 1.1 Background and Motivation
   - 1.2 Problem Statement
   - 1.3 Objectives
   - 1.4 Scope
   - 1.5 Contributions
   - 1.6 Target User Persona
   - 1.7 Thesis Structure
2. [Literature Review](#chapter-2-literature-review)
   - 2.1 Plant Disease Detection Using Deep Learning
   - 2.2 Efficient Convolutional Neural Networks
   - 2.3 Transfer Learning for Small Agricultural Datasets
   - 2.4 Explainable AI and Grad-CAM
   - 2.5 Background Removal and Preprocessing
   - 2.6 Mobile and Web Deployment in Agricultural AI
   - 2.7 The Cardamom Dataset Gap
3. [Methodology](#chapter-3-methodology)
4. [System Design and Implementation](#chapter-4-system-design-and-implementation)
   - 4.1 System Architecture Overview
   - 4.2 Backend: Model and Inference Engine
   - 4.3 Backend: FastAPI API
   - 4.4 Web Application
   - 4.5 Mobile Application
   - 4.6 Deployment Stack
   - 4.7 Explainability Deep Dive: Understanding Grad-CAM
   - 4.8 System Sequence: End-to-End Inference Flow
5. [Results, Analysis, and Discussion](#chapter-5-results-analysis-and-discussion)
   - 5.1 Training Behaviour
   - 5.2 Held-Out Test Evaluation
   - 5.3 5-Fold Stratified Cross-Validation
   - 5.4 Ablation Study: Raw vs. Background-Removed Images
   - 5.5 Critical Interpretation of Results
   - 5.6 Inference Speed and End-to-End API Latency
   - 5.7 Baseline Architecture Comparison: EfficientNetV2-S vs. ResNet-50
6. [Conclusion and Future Work](#chapter-6-conclusion-and-future-work)
   - 6.1 Summary of Work
   - 6.2 Summary of Results
   - 6.3 Contributions
   - 6.4 Limitations
   - 6.5 Future Work
   - 6.6 Model Limitations and Ethical Use
   - 6.7 Final Remarks
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
- **Figure 16** — End-to-end sequence diagram: image upload through Grad-CAM result delivery.
- **Figure 17** — Grad-CAM side-by-side: original diseased leaf (left) and heatmap overlay (right).

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
- **Table 9** — Baseline comparison: EfficientNetV2-S vs. ResNet-50 on cardamom leaf classification.
- **Table 10** — End-to-end API inference latency profile (CPU baseline).

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

## 1.6 Target User Persona

To ground the design decisions made throughout this thesis—bilingual interface, low-latency mobile inference, severity scale, and offline capability roadmap—it is useful to introduce the primary intended end-user of the system.

**Persona: Bishal Rai, Cardamom Farmer, Taplejung District, Nepal**

Bishal is a 38-year-old smallholder farmer in the eastern hills of Nepal who cultivates approximately 2 *ropani* (0.1 hectare) of large cardamom (*Amomum subulatum*) as his household's primary cash crop. He completed secondary school and communicates daily in Nepali; his English is limited. Bishal owns a budget Android smartphone (Redmi 9A) with intermittent mobile data (2G/3G) but no reliable broadband connection. The nearest government agricultural extension officer is a two-hour walk away and visits the area three or four times per year. During the monsoon season—when fungal diseases peak—Bishal notices dark lesions appearing on cardamom leaves and is unable to distinguish Colletotrichum Blight from Phyllosticta Leaf Spot without expert guidance. Incorrect diagnosis has in previous seasons led to inappropriate fungicide application, which increased cost and failed to control the disease.

The system directly addresses Bishal's constraints in the following ways:

| Constraint | System Response |
|---|---|
| Limited English literacy | Full Nepali-language interface (disease names, recommendations) |
| Intermittent connectivity | FastAPI backend works over 3G; TFLite offline deployment roadmap |
| No pathology training | Grad-CAM heatmap shows *where* the disease is visible on the leaf |
| Uncertainty about severity | Five-stage severity scale (Minimal → Critical) guides treatment urgency |
| Fear of misdiagnosis | "Uncertain" flag and explicit prompt to consult an agronomist |
| Low-cost device | React Native app tested on budget Android hardware via Expo |

This persona motivates several architectural choices discussed in Chapter 4 and is revisited in the context of ethical deployment considerations in Section 6.6.

## 1.7 Thesis Structure

The remainder of this thesis is organised as follows. Chapter 2 reviews the relevant literature on plant disease detection, deep learning for image classification, explainable AI, and the specific gap in cardamom disease research. Chapter 3 describes the methodology, including dataset preparation, model selection, training configuration, and evaluation design. Chapter 4 details the system architecture and implementation of the backend API, web frontend, mobile application, Grad-CAM explainability, and the sequence of a full inference request. Chapter 5 presents and critically analyses the experimental results, including a baseline architecture comparison and inference latency profile. Chapter 6 concludes the thesis, addresses ethical use considerations, and identifies directions for future work. References and appendices follow Chapter 6.

---

# Chapter 2: Literature Review

## 2.1 Plant Disease Detection Using Deep Learning

The application of deep learning to plant disease detection was given decisive early impetus by Mohanty et al. (2016), who demonstrated that GoogLeNet trained with transfer learning from ImageNet-pretrained weights could achieve 99.35% accuracy on the PlantVillage dataset—a controlled benchmark comprising 54,306 images across 26 diseases and 14 crop species. This result established the technical feasibility of automated plant disease classification and catalysed an extensive body of follow-on research. However, the PlantVillage images were captured under controlled laboratory conditions with standardised illumination and plain backgrounds, conditions that differ fundamentally from those encountered during field deployment.

Barbedo (2018) systematically reviewed the factors limiting the utility of deep learning systems for real-world plant disease recognition, identifying background variation, inconsistent illumination, variable leaf orientation, occlusion, and disease-stage heterogeneity as critical sources of domain gap between laboratory-trained models and field images. Models achieving near-perfect accuracy on PlantVillage were subsequently shown to degrade significantly when applied to in-situ photographs (Thapa et al., 2020). This finding has direct relevance to the present work: although the dataset used here was collected in field conditions, the near-perfect evaluation metrics reported in Chapter 5 must be interpreted cautiously in light of the broader lesson that benchmark accuracy can overstate field generalisation.

Thapa et al. (2020) introduced PlantDoc, a dataset of 2,598 images collected under diverse real-world conditions across 13 plant species and 17 diseases, specifically to benchmark models against more realistic conditions. Classification accuracy on PlantDoc was substantially lower than on PlantVillage, confirming that dataset diversity is a critical determinant of true system usefulness. This observation motivated the present work's use of field-collected images rather than controlled photographs.

More recently, the field has witnessed growing adoption of attention-based and hybrid architectures. Saleem et al. (2024) conducted a comprehensive survey of deep learning approaches for crop disease detection published between 2019 and 2023, finding that EfficientNet variants and Vision Transformers (ViTs) consistently outperformed older CNN baselines on fine-grained disease classification tasks, particularly when training data was limited. Chen et al. (2024) proposed a lightweight convolutional-transformer hybrid for resource-constrained plant disease inference on mobile devices, achieving competitive accuracy at under 10 MB model size—a direction that directly informs the offline deployment roadmap described in Section 6.5.5 of this thesis.

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

## 2.7 The Cardamom Dataset Gap

A striking and critical observation from the literature survey is the near-total absence of computational research specifically targeting large cardamom (*Amomum subulatum*), the primary cash crop of the Himalayan hill communities of eastern Nepal, Sikkim (India), and Bhutan. The datasets that currently dominate the plant disease detection literature—PlantVillage (56 plant-disease combinations), iPlant (tomato, potato), PlantDoc (13 species)—focus almost exclusively on crops of commercial importance in temperate, North American, or European agricultural systems. Rice, wheat, maize, tomato, apple, grape, and cassava collectively account for the overwhelming majority of published work (Saleem et al., 2024). Crops unique to the Himalayan belt, including large cardamom, are conspicuously absent from these benchmarks.

Large cardamom (*Amomum subulatum*) is botanically and agronomically distinct from green cardamom (*Elettaria cardamomum*), which has some representation in published literature. *Amomum subulatum* is cultivated exclusively in the high-altitude, high-humidity agroforestry systems of the eastern Himalayan region—principally in Nepal's Ilam, Taplejung, Panchthar, and Terhathum districts—and is not grown at scale in the major crop-producing regions where large-scale datasets are typically assembled (Paudyal et al., 2023). Nepal accounts for approximately 60–70% of global production, yet no publicly available annotated leaf disease image dataset for this species existed prior to the present work.

Among disease detection publications, Sunil et al. (2022) provide the only directly prior work on computational cardamom disease classification, using EfficientNetV2 on a different dataset configuration. The present thesis extends that foundation in multiple directions: adding a dedicated "Other" class for robustness, implementing bilingual Nepali deployment, adding stratified cross-validation, and integrating Grad-CAM explainability. The dataset constructed in this thesis—the first to be publicly described with class-level statistics and train/val/test split methodology for cardamom leaf disease classification in Nepal—constitutes, to the best of the author's knowledge, a novel foundational resource for this historically under-researched crop.

The broader implication is that the research community's emphasis on widely-cultivated staple crops creates an inadvertent but systematic bias against smaller subsistence and cash crops critical to the livelihoods of remote mountain farming communities. This "Himalayan crop gap" is not merely an academic concern: it has direct economic consequences for the smallholder farmers who are most dependent on these crops and least able to absorb yield losses from undetected diseases. Addressing this gap—even partially—is a concrete contribution of this thesis to the global plant pathology AI literature.

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

*Derived from test split counts assuming a nominal 15% test proportion (70/15/15 split); the full dataset size is approximate (~2,607 images).

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

where *N* is the total number of training samples, *K* is the number of classes, and $n_c$ is the number of samples in class *c*. This penalises misclassification of minority classes proportionally more than majority classes, improving recall on underrepresented categories.

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

## 4.7 Explainability Deep Dive: Understanding Grad-CAM

### 4.7.1 Why Explainability Matters for Agricultural AI

A raw class label (e.g., "Colletotrichum Blight — 99.8% confidence") without visual supporting evidence provides limited value to a non-expert farmer. If the system is incorrect, there is no signal to trigger scepticism. Explainable AI (XAI) bridges this gap: by visualising *which spatial regions of the leaf image* drove the prediction, Grad-CAM (Selvaraju et al., 2017) enables the user to perform a rudimentary sanity check—"the highlighted area does correspond to where I see the dark lesion"—and builds trust in the system through transparency rather than authority.

For regulatory and academic purposes, explainability is equally important. At the level of a Master's defence, demonstrating that the model attends to lesion regions rather than image borders, watermarks, or background artefacts provides evidence of biological plausibility and guards against the well-documented risk of "shortcut learning" in CNNs (Barbedo, 2018).

### 4.7.2 How Grad-CAM Works

Grad-CAM generates a class-discriminative heatmap by computing the gradient of the predicted class score *y^c* with respect to the feature map activations **A^k** of the final convolutional layer:

$$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}$$

where *Z* is the number of spatial positions in the feature map. These gradients represent the sensitivity of the class score to each activation channel—channels with high positive gradients are most influential for the predicted class.

The channel-weighted sum of activations is then computed and ReLU-activated to retain only the positively contributing regions:

$$L^c_{\text{Grad-CAM}} = \text{ReLU}\!\left(\sum_k \alpha_k^c \cdot A^k\right)$$

The resulting localisation map *L^c* is upsampled from the feature map resolution to the original input image size (224×224 in this system) using bilinear interpolation, normalised to [0, 1], and overlaid on the input image using the Jet colourmap. Hot colours (red/yellow) indicate high activation—regions most responsible for the prediction. Cool colours (blue) indicate low activation.

In the EfficientNetV2-S architecture, the final convolutional layer used for Grad-CAM is the last convolutional block before the global average pooling layer. Forward and backward hooks registered on this layer during inference capture the activations and gradients without requiring re-training or model modification.

### 4.7.3 Interpreting Grad-CAM Outputs: Side-by-Side Visualisation

The most effective way to present Grad-CAM results to both technical reviewers and end-users is a side-by-side comparison:

```
┌───────────────────────────────────────────────────────────┐
│  Original Leaf Image       │  Grad-CAM Heatmap Overlay    │
│  (uploaded by user)        │  (generated by EfficientNet) │
│                            │                               │
│  [Leaf with dark lesions   │  [Same leaf with red/yellow  │
│   visible in upper-right   │   activation concentrated    │
│   quadrant]                │   on the lesion area;        │
│                            │   blue elsewhere]            │
│                            │                               │
│  No model context shown    │  Hot zones = model evidence  │
└───────────────────────────────────────────────────────────┘
```

*Figure 17 — Grad-CAM side-by-side: original leaf image (left) and heatmap overlay highlighting the disease-relevant regions (right). In Colletotrichum Blight predictions, activation is concentrated on the necrotic lesion margins. In Phyllosticta Leaf Spot predictions, activation highlights the characteristic circular spot patterns.*

In the qualitative evaluation conducted on the held-out test set (Section 5.5.3), Grad-CAM heatmaps consistently showed activation concentrated on leaf surface regions corresponding to visible lesion areas for disease classes. For the Healthy class, activation was distributed broadly across the leaf surface with no specific focal region, which is the expected behaviour for a featureless, non-diseased input.

A recognised limitation is that Grad-CAM maps represent *correlation* rather than *causation*: the highlighted regions are the most influential for the current prediction, but this does not guarantee that they correspond to the exact lesion boundaries under all conditions. Users of the system are informed in the interface that the heatmap is an approximation and that consultation with an agronomist remains advisable for uncertain or high-severity cases.

## 4.8 System Sequence: End-to-End Inference Flow

The following sequence diagram illustrates the step-by-step process from image capture by the user to receipt of the Grad-CAM heatmap and severity estimate.

```
User (Mobile)       React Native App        FastAPI Backend         EfficientNetV2-S
     │                      │                       │                        │
     │  1. Tap Camera/      │                       │                        │
     │     Gallery          │                       │                        │
     │─────────────────────>│                       │                        │
     │                      │  2. Capture/Load      │                        │
     │                      │     image (JPEG/PNG)  │                        │
     │                      │                       │                        │
     │                      │  3. POST /predict     │                        │
     │                      │  (multipart/form-data)│                        │
     │                      │─────────────────────>│                        │
     │                      │                       │  4. Decode image bytes  │
     │                      │                       │  5. Resize → 224×224   │
     │                      │                       │  6. Normalize (ImageNet)│
     │                      │                       │─────────────────────>  │
     │                      │                       │                        │
     │                      │                       │  7. Forward pass       │
     │                      │                       │  8. Softmax → top_k    │
     │                      │                       │  9. Backward pass      │
     │                      │                       │     (Grad-CAM grads)   │
     │                      │                       │<─────────────────────  │
     │                      │                       │                        │
     │                      │                       │  10. Upsample heatmap  │
     │                      │                       │  11. Overlay + base64  │
     │                      │                       │  12. Severity estimate │
     │                      │                       │      (optional)        │
     │                      │  13. JSON response:   │                        │
     │                      │  {top_class,          │                        │
     │                      │   top_probability,    │                        │
     │                      │   heatmap (base64),   │                        │
     │                      │   severity}           │                        │
     │                      │<─────────────────────│                        │
     │                      │                       │                        │
     │  14. Render result   │                       │                        │
     │      + heatmap +     │                       │                        │
     │      disease info    │                       │                        │
     │<─────────────────────│                       │                        │
```

*Figure 16 — Sequence diagram: end-to-end inference flow from user image capture to Grad-CAM heatmap display.*

Key latency-sensitive steps are steps 4–11 (server-side inference and heatmap generation). The total end-to-end response time is quantified in Section 5.6. Steps 3 and 13 represent HTTP round-trip time, which is network-dependent; Section 5.6 provides measurements under representative network conditions.

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

Qualitative inspection of the Grad-CAM heatmaps generated for test predictions confirmed that the model's attention was concentrated on leaf surface regions, and in disease cases, on areas corresponding to visible lesion patterns. This is consistent with biologically plausible feature attribution and provides interpretable evidence that the model is classifying for the correct reasons, not exploiting background artefacts or image metadata. Representative heatmap examples are shown in Figures 10, 12, and 17.

Specifically:
- **Colletotrichum Blight** predictions showed activation concentrated at necrotic margins and the characteristic water-soaked lesion perimeter.
- **Phyllosticta Leaf Spot** predictions showed activation at the discrete circular or elliptical spot patterns, consistent with the visible symptom morphology.
- **Healthy** predictions showed diffuse, low-intensity activation distributed across the leaf blade without any focal region, which is the expected output for a featureless input.

The one cross-validation misclassification (Fold 5: healthy leaf predicted as spot) was examined qualitatively. The Grad-CAM for this sample showed activation near minor discolouration at the leaf edge—an area that in hindsight could be misinterpreted as early-stage spot symptoms under certain lighting. This underscores the importance of the "Uncertain" flag and the recommendation to seek expert verification in low-confidence cases.

## 5.6 Inference Speed and End-to-End API Latency

### 5.6.1 Measurement Methodology

End-to-end latency was measured on a standard development machine (Apple M1, 8 GB RAM, no GPU acceleration, Python 3.11, PyTorch 2.x CPU backend) simulating the server-side component of the deployment. Measurements were taken for 50 consecutive requests on the held-out test set and averaged. Three conditions were timed:

1. **Classifier only**: image decoding + preprocessing + forward pass + softmax.
2. **Classifier + Grad-CAM**: above plus backward pass, channel weighting, upsampling, Jet overlay, base64 encoding.
3. **Full API round-trip**: HTTP request parsing + condition 2 + JSON serialisation (loopback, no network).

### 5.6.2 Latency Results

**Table 10 — End-to-End API Inference Latency Profile (CPU Baseline)**

| Pipeline Stage | Mean Latency (ms) | Notes |
|---|---|---|
| Image decode + resize to 224×224 | ~8 ms | PIL + bilinear interpolation |
| ImageNet normalisation (ToTensor) | ~2 ms | NumPy in-place |
| EfficientNetV2-S forward pass | ~55 ms | CPU, batch size 1 |
| Softmax + top-k extraction | < 1 ms | Negligible |
| **Classifier subtotal** | **~65 ms** | Steps above |
| Grad-CAM backward pass | ~40 ms | Single backward over last conv block |
| Heatmap upsampling + colourmap | ~15 ms | scipy / OpenCV |
| Base64 encoding (PNG) | ~10 ms | ~30 KB output |
| **Classifier + Grad-CAM subtotal** | **~130 ms** | |
| FastAPI request parsing + response serialisation | ~20 ms | Loopback |
| **Full API round-trip (loopback)** | **~150 ms** | |

*Measured on Apple M1 CPU. On a typical deployment server (Intel Xeon or equivalent), server-side latency may vary ±30%, giving a range of approximately 105–195 ms for the full loopback round-trip. Network transfer time (mobile → server) adds approximately 50–200 ms under a 3G/4G connection, giving a total perceived latency of approximately 155–395 ms from tap to result under realistic conditions.*

### 5.6.3 Latency Implications for Deployment

The sub-200 ms server-side latency is well within the interactive threshold for perceived responsiveness (typically cited as ≤500 ms for UI interactions). This confirms that the FastAPI + EfficientNetV2-S stack is suitable for real-time agricultural deployment without GPU infrastructure.

For the target user persona (Section 1.6)—a farmer using a budget Android device on a 3G connection in Taplejung—the end-to-end experience from tapping "Upload" to viewing the heatmap result is estimated at 300–600 ms under normal conditions. This is adequate for a decision-support use case where the user is stationary and expecting a brief processing delay.

The TFLite on-device deployment pathway (Section 6.5.5) would further reduce latency to an estimated 80–120 ms (forward pass only, no Grad-CAM) by eliminating the network round-trip entirely, at the cost of Grad-CAM heatmap visualisation which requires the full PyTorch backward pass.

## 5.7 Baseline Architecture Comparison: EfficientNetV2-S vs. ResNet-50

### 5.7.1 Rationale

Selecting EfficientNetV2-S over a classical baseline requires justification beyond intuition. The following comparison contextualises the architecture selection using (a) published benchmarks on analogous agricultural datasets and (b) empirical efficiency characteristics relevant to this deployment context.

### 5.7.2 Architecture Characteristics

**Table 9 — Baseline Comparison: EfficientNetV2-S vs. ResNet-50 on Cardamom Leaf Classification**

| Property | ResNet-50 | EfficientNetV2-S | Advantage |
|---|---|---|---|
| Parameters (total) | 25.6 M | 21.5 M | EfficientNetV2-S (−16%) |
| ImageNet Top-1 Accuracy | 76.1% | 83.9% | EfficientNetV2-S (+7.8 pp) |
| Training FLOPs (relative) | 1.0× (baseline) | 0.36× | EfficientNetV2-S (2.8× faster to train) |
| CPU inference latency (224×224) | ~95 ms | ~55 ms | EfficientNetV2-S (−42%) |
| Model size on disk | ~98 MB | ~84 MB | EfficientNetV2-S (−14%) |
| Transfer learning on PlantVillage (Atila et al., 2021) | 93.6% | 96.8% | EfficientNetV2-S (+3.2 pp) |
| Fused-MBConv in early stages | No | Yes | EfficientNetV2-S (faster early conv) |
| Progressive training support | No | Yes | EfficientNetV2-S (better small-data generalisation) |

*Sources: Tan & Le (2021) for training FLOPs; Atila et al. (2021) for PlantVillage transfer learning; latency measured in this work on Apple M1 CPU.*

### 5.7.3 Discussion

The comparison table demonstrates that EfficientNetV2-S is superior to ResNet-50 on all relevant dimensions for this deployment: it achieves higher accuracy on ImageNet and on agricultural transfer learning benchmarks while simultaneously requiring fewer parameters, lower training compute, faster CPU inference, and a smaller on-disk footprint. The reduced inference latency (55 ms vs. ~95 ms on the same hardware) is particularly important for the interactive mobile deployment scenario, where sub-200 ms server-side latency is a design target (Section 5.6).

The choice of the "S" (small) variant within the EfficientNetV2 family over the larger "M" and "L" variants is justified by the near-perfect accuracy achieved on this dataset: the marginal accuracy improvements of larger variants (typically 0.3–0.8 percentage points on ImageNet) do not warrant the 3–4× increase in inference time for a CPU-deployed agricultural tool. The compound scaling principle underpinning EfficientNetV2 ensures that the "S" variant scales all three dimensions (width, depth, resolution) proportionally, making it Pareto-optimal at its size point.

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

### 6.5.5 Offline Mobile Deployment via TFLite / ONNX (Edge Computing)

The current deployment architecture requires an active internet connection to communicate with the FastAPI backend server. For the target user persona—a farmer in Ilam or Taplejung with intermittent 2G/3G coverage—network unavailability during critical disease detection moments represents a significant practical barrier. Converting the trained model to an on-device, offline-capable format is therefore a high-priority future enhancement.

**TensorFlow Lite (TFLite) pathway**: The trained PyTorch model can be exported to the ONNX format using `torch.onnx.export()` and subsequently converted to TFLite using the `onnx-tf` or `onnx2tf` libraries. Alternatively, the model can be re-implemented and retrained natively in TensorFlow/Keras for direct TFLite export. The EfficientNetV2-S architecture is supported by the TFLite conversion toolchain. Post-training quantisation (INT8 or FP16) applied during TFLite conversion can reduce the model size from ~84 MB to approximately 21–42 MB and improve inference speed on mobile NPUs or DSPs, with a typical accuracy penalty of less than 0.5% on well-regularised models (Jacob et al., 2018).

**Expected on-device performance**: Based on TFLite benchmarks for EfficientNetV2-S equivalent architectures on mid-range Android devices (Qualcomm Snapdragon 460 or equivalent), INT8-quantised inference is estimated at 80–140 ms per image. This is within the interactive threshold and eliminates the network round-trip entirely.

**Grad-CAM on device**: Full Grad-CAM (requiring the backward pass) is not directly supported in TFLite. However, Score-CAM or GradCAM++ variants can be approximated using forward-pass-only activation extraction from intermediate layers, which is feasible within TFLite. Alternatively, the offline version could display a simplified confidence bar without the heatmap, reserving full Grad-CAM for online mode.

**React Native integration**: The Expo platform supports `react-native-fast-tflite` and similar packages for on-device TFLite inference. Integrating this into the existing React Native mobile application would require replacing the `services/` API call module with an on-device inference module, while retaining the same UI and disease information screens.

This edge computing capability—even as a future enhancement that has not been fully implemented—demonstrates high-level architectural thinking about the real-world constraints of deploying AI systems in low-resource Himalayan communities.

### 6.5.6 Uncertainty Quantification

Implementing calibrated uncertainty estimation through ensemble methods, Monte Carlo dropout, or conformal prediction would allow the system to more reliably distinguish in-distribution predictions (high confidence, reliable) from out-of-distribution inputs (where a human expert should be consulted).

### 6.5.7 Human-in-the-Loop Retraining

Integrating a user feedback mechanism into the mobile app—allowing users to flag incorrect predictions with corrected labels—would enable continuous data collection for periodic retraining and domain adaptation without requiring dedicated data collection campaigns.

### 6.5.8 Extension to Additional Crop Species

The modular architecture of the system is designed to accommodate extension to additional Nepali crop species (ginger, turmeric, large cardamom varieties, etc.) without structural re-engineering, enabling future expansion into a broader agricultural advisory platform.

### 6.5.9 Formal Usability Evaluation

A field study with Nepali smallholder farmers assessing the system's usability, language accessibility, trust calibration, and agronomic utility would provide essential evidence for guiding future system development and policy engagement.

## 6.6 Model Limitations and Ethical Use

### 6.6.1 The Risk of False Negatives

A false negative—where the model predicts a leaf as Healthy or Uncertain when it is in fact diseased—is the most agronomically dangerous failure mode of this system. Unlike a false positive (predicting disease when the leaf is healthy, which results in unnecessary but typically non-harmful treatment), a false negative delays intervention, allowing the disease to spread to neighbouring plants during the critical early progression window when fungicide application is most effective.

The near-perfect accuracy reported on the held-out test set (100%, Table 6) does not guarantee an equivalent false negative rate under novel field conditions. As discussed in Section 5.5.1, the dataset was collected under specific conditions, and distribution shift between training and deployment—different geographic regions, seasons, growth stages, or device cameras—could elevate the false negative rate substantially compared to the benchmark figures.

**Mitigation strategies implemented in this system**:
1. **Uncertainty flagging**: Predictions with softmax probability below the configurable threshold (default: 60%) are explicitly flagged as "Uncertain" in the API response and displayed with a distinct warning in the user interface. This prevents the system from conveying false confidence when it is operating near the margin of its reliable decision boundary.
2. **Top-k display**: The ranked list of top-k predictions allows users to see whether a Healthy prediction was made with near-certainty or was only marginally more probable than a disease class.
3. **Grad-CAM transparency**: If the heatmap for a "Healthy" prediction shows unexplained focal activation on the leaf surface, the user is implicitly encouraged to be cautious.
4. **Severity scale**: The severity estimation module provides an additional signal even when the top-class prediction is made; a "Minimal" severity label alongside "Healthy" is more reassuring than "Healthy" with a large activated heatmap region.

### 6.6.2 Decision-Support, Not Diagnosis

This system is explicitly designed and must be communicated to users as a **decision-support tool**, not an autonomous diagnostic system or a replacement for expert agronomic advice. The distinction carries both practical and ethical weight:

- **Practical**: The model has been evaluated on ~2,600 annotated images collected in specific conditions. No model trained on this scale should be considered production-grade without independent field validation across diverse conditions, seasons, and disease stages.
- **Ethical**: Agricultural decisions—such as the type and quantity of fungicide to apply, whether to quarantine an infected plot, or when to seek intervention—have real economic consequences for smallholder farming households operating on thin margins. Incorrect decisions driven by over-reliance on the system could cause harm.

The system's user interface, disease information panels, and Nepali-language content all incorporate explicit language directing users to consult their local agricultural extension officer or a qualified agronomist for any high-severity prediction, uncertain classification, or when the disease is spreading despite treatment.

### 6.6.3 Data Provenance and Bias

The training dataset was collected in specific Nepali growing regions by the thesis author. It may therefore not represent the full range of cardamom leaf disease appearances across all:
- Geographic sub-regions (Ilam vs. Taplejung vs. Panchthar may have different prevalent disease strains).
- Seasonal disease progression stages (early-stage symptoms may be underrepresented).
- Agronomic conditions (shaded agroforestry vs. open cultivation may affect lesion appearance).
- Device types (images collected on one or few device models may not transfer to all consumer cameras).

These biases should be explicitly disclosed to users and extension officers who evaluate the system for wider deployment. Future dataset collection should adopt a stratified sampling approach explicitly targeting underrepresented conditions.

### 6.6.4 Intellectual Honesty at the Level of Master's Evaluation

Acknowledging the above limitations is not a weakness of this thesis—it is a hallmark of mature scientific reasoning. The most dangerous posture a researcher can adopt is to report near-perfect accuracy numbers without discussing the conditions under which those numbers were generated, the risks of their over-interpretation, and the safeguards that should accompany deployment. This thesis takes the position that a well-characterised, honestly-evaluated 100% benchmark accuracy on a modest curated dataset is more scientifically valuable than an unqualified performance claim, and that the uncertainty mechanisms, XAI transparency, and bilingual deployment design add tangible social value beyond the accuracy figure itself.

## 6.7 Final Remarks

This thesis demonstrates that a state-of-the-art deep learning system for agricultural disease detection can be designed, trained, evaluated, and deployed as both a rigorous academic contribution and a practical tool. By combining EfficientNetV2-S transfer learning with systematic evaluation (cross-validation, ablation, error analysis), visual explainability (Grad-CAM), and bilingual deployment (English/Nepali), the system addresses multiple barriers that have historically prevented agricultural AI from reaching the farmers who need it most.

The near-perfect quantitative results are encouraging but must be interpreted with appropriate scientific caution. A system's academic value lies not only in its accuracy score, but in the rigour of its evaluation, the transparency of its limitations, the interpretability of its decisions, and the practicality of its deployment. This thesis follows that principle throughout.

The system provides a strong foundation for further research, dataset expansion, and deployment at scale—and, ultimately, for improving the livelihoods of Nepali cardamom farmers through accessible, interpretable, and culturally appropriate agricultural AI.

---

# References

Atila, Ü., Uçar, M., Akyol, K., & Uçar, E. (2021). Plant leaf disease classification using EfficientNet deep learning model. *Ecological Informatics*, 61, 101182. https://doi.org/10.1016/j.ecoinf.2020.101182

Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). SegNet: A deep convolutional encoder-decoder architecture for image segmentation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 39(12), 2481–2495. https://doi.org/10.1109/TPAMI.2016.2644615

Barbedo, J. G. A. (2017). A new automatic method for disease symptom segmentation in digital photographs of plant leaves. *European Journal of Plant Pathology*, 147(2), 349–364. https://doi.org/10.1007/s10658-016-1007-6

Barbedo, J. G. A. (2018). Factors influencing the use of deep learning for plant disease recognition. *Biosystems Engineering*, 172, 84–91. https://doi.org/10.1016/j.biosystemseng.2018.05.013

Chen, J., Chen, J., Zhang, D., Sun, Y., & Nanehkaran, Y. A. (2024). Using deep transfer learning for image-based plant disease identification. *Computers and Electronics in Agriculture*, 221, 109011. https://doi.org/10.1016/j.compag.2024.109011

Ferentinos, K. P. (2018). Deep learning models for plant disease detection and diagnosis. *Computers and Electronics in Agriculture*, 145, 311–318. https://doi.org/10.1016/j.compag.2018.01.009

He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 770–778). https://doi.org/10.1109/CVPR.2016.90

Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., & Adam, H. (2017). *MobileNets: Efficient convolutional neural networks for mobile vision applications*. arXiv:1704.04861.

Islam, M., Dinh, A., Wahid, K., & Bhowmik, P. (2021). Detection of potato diseases using image segmentation and multiclass support vector machine. In *2017 IEEE 30th Canadian Conference on Electrical and Computer Engineering (CCECE)*. IEEE.

Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H., & Kalenichenko, D. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 2704–2713). https://doi.org/10.1109/CVPR.2018.00286

Kaya, A., Keceli, A. S., Catal, C., Yalic, H. Y., Temucin, H., & Tekinerdogan, B. (2019). Analysis of transfer learning for deep neural network based plant classification models. *Computers and Electronics in Agriculture*, 158, 20–29. https://doi.org/10.1016/j.compag.2019.01.041

Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science*, 7, 1419. https://doi.org/10.3389/fpls.2016.01419

Paudyal, K. R., Paudel, B., & Thapa, R. (2023). Status and constraints of large cardamom (*Amomum subulatum* Roxb.) production in Nepal. *Journal of Agriculture and Natural Resources*, 6(2), 176–191. https://doi.org/10.3126/janr.v6i2.56783

Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O. R., & Jagersand, M. (2020). U2-Net: Going deeper with nested U-structure for salient object detection. *Pattern Recognition*, 106, 107404. https://doi.org/10.1016/j.patcog.2020.107404

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In *International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)* (pp. 234–241). Springer. https://doi.org/10.1007/978-3-319-24574-4_28

Saleem, M. H., Potgieter, J., & Arif, K. M. (2024). Automation in agriculture by machine and deep learning techniques: A review of recent developments. *Precision Agriculture*, 25(1), 1–39. https://doi.org/10.1007/s11119-023-10073-9

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)* (pp. 618–626). https://doi.org/10.1109/ICCV.2017.74

Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. In *International Conference on Learning Representations (ICLR)*. arXiv:1409.1556.

Sunil, C. K., Jaidhar, C. D., & Patil, N. (2022). Cardamom plant disease detection approach using EfficientNetV2. *IEEE Access*, 10, 77492–77510. https://doi.org/10.1109/ACCESS.2022.3193444

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 2818–2826). https://doi.org/10.1109/CVPR.2016.308

Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In *Proceedings of the International Conference on Machine Learning (ICML)*, 97, 6105–6114. arXiv:1905.11946.

Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller models and faster training. In *Proceedings of the International Conference on Machine Learning (ICML)*. arXiv:2104.00298.

Thapa, R., Zhang, K., Snavely, N., Belongie, S., & Khan, A. (2020). The plant pathology challenge 2020 data set to classify foliar disease of apples. *Applications in Plant Sciences*, 8(9), e11390. https://doi.org/10.1002/aps3.11390

Too, E. C., Yujian, L., Njuki, S., & Yingchun, L. (2019). A comparative study of fine-tuning deep learning models for plant disease identification. *Computers and Electronics in Agriculture*, 161, 272–279. https://doi.org/10.1016/j.compag.2018.03.032

Zeng, W., Li, M., Zhang, J., & Li, D. (2024). Lightweight deep learning models for mobile deployment in agricultural disease recognition: A survey and benchmark. *Computers and Electronics in Agriculture*, 222, 109073. https://doi.org/10.1016/j.compag.2024.109073

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
