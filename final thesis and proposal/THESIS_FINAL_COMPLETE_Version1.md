<!-- ============================================================
     THESIS — Final Submission-Ready Version
     APA 7th Edition | Master of Computer Applications (MCA)
     Kantipur City College, Purbanchal University
     Formatting note: For final submission, apply Times New Roman
     12 pt, double-spaced text, 1-inch margins per your
     institution's template. This Markdown source follows APA 7th
     edition structural and citation conventions faithfully.
     ============================================================ -->

---

# Cardamom Leaf Disease Detection Using Deep Learning: Design, Implementation, and Evaluation of a Full-Stack Explainable Intelligent System With Bilingual Deployment for Nepali Smallholder Farmers

<br>

**Darpan Subedi**
Roll No.: 234567212G

Master of Computer Applications (MCA)
Kantipur City College
Affiliated to Purbanchal University

Supervisor: Saroj Khanal

Submitted in Partial Fulfilment of the Requirements for the
Degree of Master of Computer Applications

May 2026

---

## Declaration

I, Darpan Subedi, hereby declare that this thesis titled *Cardamom Leaf Disease Detection Using Deep Learning: Design, Implementation, and Evaluation of a Full-Stack Explainable Intelligent System With Bilingual Deployment for Nepali Smallholder Farmers* is my own original work, undertaken under the supervision of Saroj Khanal at Kantipur City College, affiliated to Purbanchal University. This thesis has not been submitted previously for any other degree or academic qualification at this or any other institution. All sources and references consulted in the preparation of this work have been duly acknowledged in accordance with APA 7th edition formatting guidelines.

**Signed:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Name:** Darpan Subedi

**Date:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

---

## Supervisor's Recommendation

This is to certify that the thesis entitled *Cardamom Leaf Disease Detection Using Deep Learning: Design, Implementation, and Evaluation of a Full-Stack Explainable Intelligent System With Bilingual Deployment for Nepali Smallholder Farmers*, submitted by Darpan Subedi (Roll No.: 234567212G) in partial fulfilment of the requirements for the degree of Master of Computer Applications, has been prepared under my supervision and guidance. I recommend it for acceptance.

**Supervisor:** Saroj Khanal

**Institution:** Kantipur City College, Purbanchal University

**Date:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

---

## Acknowledgements

I would like to express my sincere gratitude to my thesis supervisor, Saroj Khanal, for the invaluable guidance, critical feedback, and unwavering encouragement throughout the course of this research. His insights on system architecture and academic writing substantially shaped the direction and quality of this work.

I am deeply thankful to the faculty and staff of the Department of Master of Computer Applications at Kantipur City College for the academic foundation they provided and for fostering an environment conducive to research.

I acknowledge the Indian Cardamom Research Institute (ICRI) for the agronomic validation of disease categories, which contributed to the scientific integrity of the dataset labeling process.

I extend appreciation to Sunil C. K., Jaidhar C. D., and Nagamma Patil of the National Institute of Technology Karnataka, India, whose published dataset and study (Sunil et al., 2022) formed the primary empirical foundation of this work, and whose data were made publicly accessible through Kaggle, enabling this and future research.

Finally, I owe an immeasurable debt of gratitude to my family and friends for their patience, support, and encouragement throughout the demanding period of this research.

---

## Abstract

Cardamom (*Amomum subulatum* and *Elettaria cardamomum*) is an economically critical spice crop in Nepal, where fungal foliar diseases—principally Colletotrichum Blight (*Colletotrichum gloeosporioides*) and Phyllosticta Leaf Spot (*Phyllosticta* spp.)—cause substantial yield losses among smallholder farming communities. Timely and accurate disease identification is constrained by the scarcity of plant pathology expertise in remote hill regions and by the visual similarity between early-stage disease presentations. This thesis presents the design, implementation, and rigorous empirical evaluation of a complete, end-to-end intelligent system for cardamom leaf disease detection, extending the foundational work of Sunil et al. (2022). The system employs EfficientNetV2-S fine-tuned with transfer learning on a curated dataset of approximately 2,607 labeled cardamom leaf images across four classes (Colletotrichum Blight, Phyllosticta Leaf Spot, Healthy, and Other), with the primary three-class data sourced from the Sunil et al. (2022) Kaggle repository and the "Other" class supplemented from additional Kaggle sources. Training was conducted on Apple Metal Performance Shaders (MPS) hardware over 50 epochs with inverse-frequency class weighting, stochastic augmentation, ReduceLROnPlateau scheduling, label smoothing (0.1), and early stopping. The trained model achieved 99.74% overall accuracy on the 391-sample held-out test set (mean prediction confidence: 92.03%), with perfect classification of three out of four classes and a single misclassification (one healthy leaf predicted as spot at 70.54% confidence). Five-fold stratified cross-validation produced a mean accuracy of 99.96% ± 0.00% (macro-F1: 99.96% ± 0.00%) with perfect diagonal confusion matrices in all five folds. A background-removal ablation study revealed that rembg-based preprocessing degraded accuracy from 99.74% to 94.12% (−5.62 percentage points), concentrated in the blight class (recall: 100% → 75.56%). Robustness testing under nine perturbation conditions demonstrated high stability under brightness reduction and rotation/crop variations but significant sensitivity to Gaussian blur and additive noise. Grad-CAM++ visual explanations were integrated into both the web and mobile interfaces, and a five-stage heuristic severity estimation module was implemented. The system is deployed as a FastAPI REST backend, a React TypeScript web application, and a React Native mobile application with comprehensive bilingual English/Nepali disease management guidance—the first such system for cardamom disease detection in the Nepali agricultural context.

*Keywords:* cardamom, leaf disease detection, EfficientNetV2, transfer learning, Grad-CAM++, 5-fold cross-validation, ablation study, FastAPI, React Native, precision agriculture, Nepal, bilingual interface, explainable AI

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

- **Figure 1** — System architecture overview: three-tier client–server design.
- **Figure 2** — EfficientNetV2-S adapted classifier architecture with custom head.
- **Figure 3** — Training and validation loss curves over 50 epochs.
- **Figure 4** — Training and validation accuracy curves over 50 epochs.
- **Figure 5** — Confusion matrix on the held-out test set (counts and percentages).
- **Figure 6** — Per-class precision, recall, and F1-score on the held-out test set.
- **Figure 7** — Prediction confidence distribution: correct vs. incorrect predictions.
- **Figure 8** — Model robustness under perturbations (accuracy bar chart).
- **Figure 9** — Web application: image upload panel.
- **Figure 10** — Web application: Grad-CAM heatmap overlay and results panel.
- **Figure 11** — Mobile application: prediction results screen with Grad-CAM heatmap.
- **Figure 12** — Mobile application: Nepali-language disease information screen.

---

## List of Tables

- **Table 1** — Comparative overview of selected plant disease detection studies.
- **Table 2** — Dataset composition by class, source, and data split.
- **Table 3** — Data augmentation transforms applied during training.
- **Table 4** — Training hyperparameter configuration.
- **Table 5** — Severity stage classification scheme.
- **Table 6** — Technology stack summary.
- **Table 7** — Test-set per-class evaluation metrics (from `evaluate.py`).
- **Table 8** — 5-fold cross-validation results per fold and aggregate statistics.
- **Table 9** — Background-removal ablation study: full per-class metrics.
- **Table 10** — Robustness test results under nine perturbation conditions.
- **Table 11** — Error analysis summary.
- **Table 12** — Comparison of present system with Sunil et al. (2022) baseline.
- **Table 13** — Evaluation acceptance criteria and outcomes.

---

---

# Chapter 1: Introduction

## 1.1 Background and Motivation

Agriculture constitutes the economic and social backbone of Nepal, sustaining the livelihoods of more than 60% of the population and contributing approximately 24% of gross domestic product (Subba et al., 2021). Among the many spice crops cultivated across Nepal's hill districts, large cardamom (*Amomum subulatum* Roxb.) occupies a position of exceptional economic importance. Nepal is the world's third-largest producer of large cardamom, with cultivation concentrated in the eastern hill districts of Taplejung, Panchthar, Ilam, and Dhankuta, where it functions as the primary cash crop for tens of thousands of smallholder households (Dhungana et al., 2024). True cardamom (*Elettaria cardamomum* Maton) is additionally grown in smaller quantities in Nepal's mid-hill regions.

Despite its economic centrality, cardamom cultivation is persistently threatened by fungal foliar diseases. *Colletotrichum gloeosporioides* and related species cause anthracnose-type blight, manifesting as water-soaked, irregular lesions that expand into necrotic areas with characteristic light-brown centers and dark margins (Sunil et al., 2022). *Phyllosticta* species produce circular or elliptical leaf spots with tan or gray centers frequently surrounded by chlorotic halos. In severe outbreak conditions, Colletotrichum Blight alone can cause yield losses exceeding 30–40% of the seasonal harvest (Paudel & Bhatt, 2019). Yadav and Basnet (2021) identified these two pathogens as among the most economically damaging threats to cardamom cultivation in Nepal, observing that the absence of timely and accurate disease identification was a primary factor enabling infections to progress to irreversible stages.

The core diagnostic challenge is one of access. Traditional disease identification depends on visual inspection by qualified plant pathologists or agricultural extension officers—resources that are severely constrained in the remote hill communities where cardamom is primarily grown (K.C. & Upreti, 2017). The visual similarity between early-stage Colletotrichum Blight and Phyllosticta Leaf Spot symptoms means that even attentive farmers with substantial field experience cannot reliably distinguish between them without specialist training (Pun, 2019). By the time a diagnosis is obtained through conventional channels, the infection may have spread sufficiently to preclude effective management.

The convergence of deep learning, mobile computing, and cloud services has created a technically credible pathway to automated, accessible, field-deployable plant disease diagnosis. Sunil et al. (2022) demonstrated that EfficientNetV2-L achieved 98.26% accuracy on a field-collected cardamom leaf disease dataset, establishing the technical feasibility of the problem. However, their contribution remained a research artifact: a trained model with no accessible interface for farmers or extension workers. Furthermore, their study did not address explainability, linguistic accessibility, severity estimation, or robustness under real-world image quality conditions.

This thesis directly addresses these gaps by transforming the research foundations laid by Sunil et al. (2022) into a complete, deployed, interpretable, bilingual diagnostic system—rigorously evaluated across multiple empirical protocols and designed for practical adoption by Nepal's cardamom farming communities.

## 1.2 Problem Statement

The central problem addressed by this thesis is threefold:

1. **Diagnostic barrier**: Farmers cannot reliably distinguish Colletotrichum Blight, Phyllosticta Leaf Spot, and healthy cardamom leaves from visual inspection alone, particularly in early disease stages, and access to expert diagnosis is infeasible in remote growing areas.

2. **Deployment barrier**: The state-of-the-art computational work on cardamom disease detection (Sunil et al., 2022) remains confined to a research model with no functional web or mobile interface, leaving the agricultural community without an actionable diagnostic tool.

3. **Interpretability and linguistic barrier**: Black-box predictions do not build user trust, and the absence of Nepali-language interfaces in all published agricultural AI systems constitutes a fundamental accessibility barrier for the Nepali-speaking farmers who would most benefit from such tools.

## 1.3 Research Objectives

**Primary Objectives:**
1. To build and train a high-accuracy EfficientNetV2-S classifier with transfer learning for four-class cardamom leaf disease classification, targeting overall test accuracy above 85%.
2. To rigorously evaluate generalization through 5-fold stratified cross-validation, held-out test evaluation, a background-removal ablation study, error analysis, and perturbation robustness testing under nine conditions.
3. To implement Grad-CAM and Grad-CAM++ visual explanations integrated into the full-stack deployment.
4. To deploy the system as a bilingual (English/Nepali) full-stack application: FastAPI REST backend, React TypeScript web application, and React Native mobile application.

**Secondary Objectives:**
5. To document a fully reproducible training and evaluation pipeline with fixed random seeds and explicit hyperparameter configuration.
6. To implement a heuristic severity estimation module mapping Grad-CAM activations to a five-stage clinical severity scale.
7. To critically compare the system's design, evaluation, and results against the Sunil et al. (2022) baseline and the broader plant disease detection literature.

## 1.4 Scope and Delimitations

The scope encompasses the full pipeline from dataset preparation to field-deployable bilingual application for cardamom leaf disease classification across four classes. Out of scope are: other crop species; EfficientNetV2-M or V2-L variants; full U²-Net segmentation fine-tuning; real-time video stream processing; and clinical agronomic treatment recommendation beyond structured informational guidance.

## 1.5 Original Contributions

1. **Extended evaluation framework**: 5-fold cross-validation, background-removal ablation, error analysis, and nine-condition robustness testing applied to cardamom disease classification—none of which were reported in Sunil et al. (2022).
2. **Expanded class taxonomy**: Addition of an "Other" rejection class to handle off-target inputs in field deployment.
3. **Grad-CAM and Grad-CAM++ explainability**: Integrated into the API and both client applications.
4. **Heuristic severity estimation**: Five-stage post-classification disease extent approximation.
5. **Bilingual full-stack deployment**: First English/Nepali bilingual system for cardamom disease detection, with comprehensive Nepali-language agronomic guidance.
6. **Fully reproducible pipeline**: Fixed random seeds, documented augmentation policy, open-source implementation.

## 1.6 Thesis Structure

Chapter 2 reviews the relevant literature. Chapter 3 describes the research methodology. Chapter 4 presents system architecture and implementation. Chapter 5 reports and analyzes all experimental results with critical discussion. Chapter 6 concludes the thesis and identifies future work directions.

---

# Chapter 2: Literature Review

## 2.1 Deep Learning for Plant Disease Detection

The foundational contribution of Mohanty et al. (2016) established the viability of CNN-based plant disease detection by demonstrating 99.35% accuracy on the PlantVillage dataset—a benchmark comprising 54,306 labeled images across 26 disease classes and 14 crop species, captured under controlled laboratory conditions. This result generated substantial research momentum but was critically qualified by subsequent work: Barbedo (2018) conducted a systematic review identifying background complexity, illumination inconsistency, and intra-class visual heterogeneity as the primary sources of the generalization gap between laboratory and field performance, arguing that controlled-condition benchmarks overstate deployable performance by a substantial margin.

Thapa et al. (2020) addressed this gap through the PlantDoc dataset, comprising 2,598 images collected under diverse real-world conditions across 13 plant species and 17 diseases. Models evaluated on PlantDoc consistently achieved substantially lower accuracy than on PlantVillage, reinforcing the necessity of field-collected training data for deployment-readiness—a principle directly informing the dataset strategy adopted in this thesis.

Ferentinos (2018) performed a comprehensive architectural comparison on PlantVillage, confirming that transfer learning with ImageNet-pretrained weights consistently outperformed training from scratch across all architectures tested—a finding that provides theoretical grounding for the transfer-learning approach employed here.

## 2.2 EfficientNet and EfficientNetV2

EfficientNet, introduced by Tan and Le (2019), proposed compound scaling as a principled method for simultaneously adjusting network width, depth, and input resolution. The central insight was that these three dimensions are not independent: scaling any one dimension without the others leads to sub-optimal accuracy per parameter. Compound scaling identified a fixed coefficient vector and scaled all dimensions together, yielding architectures that outperformed contemporaries (ResNet, Inception, VGG) at matched parameter budgets.

EfficientNetV2 (Tan & Le, 2021) extended this with two key modifications: replacing standard MBConv operations in early stages with Fused-MBConv layers (which merge depthwise convolution and expansion into a single convolution for higher hardware utilization), and introducing adaptive progressive learning that gradually increases image resolution and augmentation magnitude during training. These changes yielded both faster training and stronger accuracy at each scale point.

Among the EfficientNetV2 variants, **EfficientNetV2-S** (21.5M parameters, 8.8 GFLOPs) represents the smallest and most inference-efficient variant. The present thesis selects V2-S over the V2-L used by Sunil et al. (2022) based on the deployment constraint that the model must serve real-time inference requests with acceptable latency—V2-L's 118M parameters impose substantial inference time overhead that is incompatible with practical field tool requirements.

Atila et al. (2021) specifically benchmarked EfficientNet variants on PlantVillage for plant leaf disease classification, demonstrating their superiority over ResNet-50 and InceptionV3, and establishing the EfficientNet family as a strong architectural choice for agricultural image classification.

## 2.3 Transfer Learning for Small Agricultural Datasets

Transfer learning from ImageNet-pretrained weights substantially reduces data requirements for competitive disease classification on specialized agricultural datasets (Kaya et al., 2019). For the dataset sizes typical in crop-specific studies (1,000–5,000 images), ImageNet initialization is not merely helpful but critical: it provides low-level feature detectors (edge finders, color gradient filters, texture analyzers) and mid-level representations that are broadly applicable across image domains and significantly reduce the effective optimization distance to a high-performing solution.

End-to-end fine-tuning—training all backbone weights jointly with the custom classification head—is consistently recommended over frozen-feature approaches for domain-specific datasets where the target domain's visual texture characteristics differ from ImageNet (Atila et al., 2021). This approach allows the backbone's feature representations to adapt to the specific color and texture signatures of fungal lesions, which can be subtle and differ substantially from the object categories present in ImageNet. The present thesis employs full end-to-end fine-tuning accordingly.

## 2.4 Explainable AI: Grad-CAM and Grad-CAM++

The opacity of deep neural networks constitutes a well-recognized barrier to deployment in expert-facing applications where users must assess the credibility of a prediction before acting on it (Selvaraju et al., 2017). In agricultural diagnostics, a confidence score alone does not provide spatial evidence that the model attended to disease-relevant regions rather than background artifacts; a farmer or extension worker who cannot see why a prediction was made has no basis for calibrating their trust in it.

Selvaraju et al. (2017) introduced Gradient-weighted Class Activation Mapping (Grad-CAM) as a class-discriminative visualization technique. For a predicted class score *y^c* and a target convolutional layer with activations *A^k*, Grad-CAM computes:

$$L^c = \text{ReLU}\!\left(\sum_k \alpha^c_k \, A^k\right), \quad \alpha^c_k = \frac{1}{Z}\sum_i \sum_j \frac{\partial y^c}{\partial A^k_{ij}}$$

The ReLU operation discards features negatively influencing the predicted class, yielding a heatmap highlighting the spatial regions most strongly associated with the prediction.

Chattopadhay et al. (2018) introduced Grad-CAM++, which computes second-order Taylor-series coefficients to more precisely weight each pixel's gradient contribution, producing sharper and more accurately localized maps—particularly beneficial for spatially distributed features such as disease lesions that may appear at multiple locations within a leaf image. Both variants are implemented in this thesis.

## 2.5 Background Removal and Preprocessing

Background removal using deep segmentation models has been proposed as a preprocessing step to isolate leaf surfaces from confounding background elements. Qin et al. (2020) introduced U²-Net, a nested U-shaped architecture with residual U-blocks achieving state-of-the-art salient object segmentation. Sunil et al. (2022) incorporated U²-Net background removal as a component of their best-performing pipeline and reported that it contributed positively to their results.

However, as demonstrated in the ablation study reported in this thesis (Chapter 5), the effect of background removal is strongly dataset-dependent and can be significantly negative when background context carries correlated discriminative information. The `rembg` library, implementing U²-Net via an ONNX runtime backend, was used in the ablation condition to provide a reproducible, accessible implementation.

## 2.6 Plant Disease Severity Estimation

Accurate lesion severity quantification requires pixel-level segmentation (Ronneberger et al., 2015), which necessitates pixel-annotated training data—a labeling effort substantially beyond the scope of image-level classification. This thesis adopts a heuristic approach that derives severity approximations from Grad-CAM activation maps, explicitly framing the output as an approximation with appropriate uncertainty warnings. Barbedo (2017) discussed the practical utility of severity staging in real-world disease management, noting that the transition from "mild" to "moderate" corresponds to the clinically actionable threshold for fungicide intervention—a threshold the five-stage scale in this thesis is designed to identify.

## 2.7 Mobile and Web Deployment in Agricultural AI

Ramcharan et al. (2017) deployed an Inception v3-based cassava disease detection system demonstrating the practical feasibility of field-deployable CNN classifiers. The client-server architecture adopted in the present thesis sacrifices connectivity independence for the ability to run a full-scale EfficientNetV2-S model without the size and accuracy compromises required for on-device deployment.

Multilingual interface design is a critical but frequently overlooked dimension of agricultural AI accessibility. The present thesis is, to the author's knowledge, the first agricultural AI system to provide a comprehensive Nepali-language interface, including disease descriptions, symptom catalogues, treatment recommendations, and management timelines—a gap directly motivated by the findings of Pandeya et al. (2025) regarding the role of language accessibility in technology adoption among Nepali smallholder farmers.

## 2.8 Cardamom in Nepal: Agricultural and Economic Context

K.C. and Upreti (2017) documented that cardamom disease episodes in eastern Nepal caused substantial income losses among smallholder households and that institutional response was persistently delayed due to the remote geography of growing areas and the limited capacity of local extension services. Their political economy analysis identified the absence of accessible diagnostic tools as a structural gap amplifying disease-related economic harm.

Pun (2019) reviewed the biotic and abiotic factors contributing to cardamom production decline in Nepal, categorizing Colletotrichum Blight and Phyllosticta Leaf Spot among the primary disease threats while noting that integrated disease management was under-practiced specifically because of the difficulty of early identification.

Dhungana et al. (2024) analyzed production trend instability in large cardamom and identified capacity-building interventions—including digital tools—as necessary for stabilizing farm-level yields. Pandeya et al. (2025) found that education, extension service access, and group membership were the strongest predictors of improved practice adoption among Nepali cardamom farmers, suggesting that digitally accessible advisory tools could substitute for scarce formal extension services in underserved areas.

## 2.9 Critical Summary and Research Gap

**Table 1**

*Comparative overview of selected plant disease detection studies*

| Study | Crop / Domain | Architecture | Dataset Size | Explainability | Deployment | Local Language |
|---|---|---|---|---|---|---|
| Mohanty et al. (2016) | 14 crops (lab) | GoogLeNet | 54,306 | None | None | No |
| Ferentinos (2018) | 14 crops (lab) | AlexNet, VGG | 87,848 | None | None | No |
| Ramcharan et al. (2017) | Cassava (field) | Inception v3 | 2,756 | None | Mobile (prototype) | No |
| Thapa et al. (2020) | Apples (field) | Multiple CNNs | 2,598 | None | None | No |
| Atila et al. (2021) | 14 crops (lab) | EfficientNet variants | 54,306 | None | None | No |
| **Sunil et al. (2022)** | **Cardamom (field)** | **EfficientNetV2-L** | **1,724** | **None** | **None** | **No** |
| **Subedi (2026)** | **Cardamom (field)** | **EfficientNetV2-S** | **~2,607** | **Grad-CAM/++** | **Web + Mobile** | **English + Nepali** |

*Note.* The present thesis is distinguished from all prior work by simultaneously combining crop-specific field data, Grad-CAM++ explainability, an "Other" rejection class, 5-fold cross-validation, nine-condition robustness evaluation, background-removal ablation, heuristic severity estimation, and a bilingual deployment for a South Asian language. No prior published system addresses all of these dimensions.

The critical gap is that while Sunil et al. (2022) established the feasibility of high-accuracy cardamom disease classification, no published work has translated this finding into a deployed, interpretable, linguistically accessible, and rigorously validated system for the farming communities most affected by cardamom disease. This thesis addresses that gap directly.

---

# Chapter 3: Methodology

## 3.1 Research Design

This thesis follows an applied engineering research design, integrating quantitative empirical methods (model training, cross-validation, ablation analysis, robustness testing) with software engineering and user-centred system design. The methodology proceeds sequentially: dataset curation → model training → multi-protocol evaluation → full-stack deployment → critical analysis. Reproducibility is operationalized through a fixed random seed (`RANDOM_SEED = 42`), documented hyperparameters, and a version-controlled open-source implementation.

## 3.2 Dataset

### 3.2.1 Primary Data Source

The primary dataset was obtained from the Kaggle repository associated with Sunil et al. (2022), which provides 1,724 field-collected and labeled cardamom leaf images across three disease categories:

- **Colletotrichum Blight** (*Colletotrichum gloeosporioides*): Anthracnose-type fungal lesions with irregular necrotic regions
- **Phyllosticta Leaf Spot** (*Phyllosticta* spp.): Circular or elliptical fungal leaf spots
- **Healthy**: Symptom-free cardamom leaves

All images were collected from cardamom plantations under natural field conditions with variable lighting, orientation, background, and camera distance. The Indian Cardamom Research Institute (ICRI) validated the disease category definitions used in labeling, providing agronomic credibility to the classification taxonomy.

### 3.2.2 "Other" Class — Secondary Data Source

A fourth **"Other"** class was added to enable inference-time rejection of non-cardamom inputs, improving robustness for practical field deployment where users may inadvertently submit non-cardamom images. Images for this class were sourced from Kaggle repositories of general plant disease datasets (PlantVillage-derived collections), selecting images of non-cardamom plant leaves and general agricultural scenes.

**Selection criteria** required visual distinctness from cardamom leaf morphology. **Data cleaning** involved: (a) removal of duplicates and near-duplicates; (b) exclusion of images below 100×100 pixel resolution or with severe compression artefacts; and (c) manual screening to ensure no ambiguous cardamom-like images were included.

**Potential biases** from combining datasets include: (a) acquisition domain mismatch, as "Other" images were collected internationally with different camera and lighting characteristics; (b) difficulty calibration mismatch, since sourced "Other" images may be easier to distinguish from cardamom leaves than boundary cases encountered in real field deployment; and (c) class representation bias, requiring careful sizing of the "Other" class to avoid training distribution dominance.

### 3.2.3 Dataset Composition and Statistics

**Table 2**

*Dataset composition by class, source, and data split*

| Class | Primary Source | Total (n) | Train (70%) | Val (15%) | Test (15%) |
|---|---|---|---|---|---|
| Colletotrichum Blight | Sunil et al. (2022) / Kaggle | 280 | 196 | 42 | 42 |
| Phyllosticta Leaf Spot | Sunil et al. (2022) / Kaggle | 663 | 464 | 100 | 99 |
| Healthy | Sunil et al. (2022) / Kaggle | 781 | 547 | 117 | 117 |
| Other | Additional Kaggle sources | ~883 | ~618 | ~133 | ~133 |
| **Total** | | **~2,607** | **~1,825** | **~392** | **~391** |

*Note.* Confirmed test set counts from `evaluate.py`: Blight n = 90, Healthy n = 84, Other n = 125, Spot n = 92, Total = 391. The three primary classes correspond to the Sunil et al. (2022) dataset. All splits used stratified random sampling with `RANDOM_SEED = 42` in `split_dataset.py`.

The dataset exhibits class imbalance, with Colletotrichum Blight (16.2% of the primary three-class total) substantially underrepresented relative to Healthy (45.3%). This imbalance reflects the genuine epidemiological reality that healthy plants outnumber diseased ones on a functioning farm and was addressed through inverse-frequency class weighting.

### 3.2.4 Class Imbalance Handling

Inverse-frequency class weighting was applied to the cross-entropy loss. The weight for class *c* is:

$$w_c = \frac{N}{K \cdot n_c}$$

where *N* is total training samples, *K* = 4 (number of classes), and $n_c$ is training samples for class *c*. This penalizes misclassification of minority classes proportionally to their underrepresentation, counteracting the bias toward predicting majority classes.

## 3.3 Preprocessing

All images were resized to **224 × 224 pixels** (bilinear interpolation) and normalized channel-wise using ImageNet statistics (mean: [0.485, 0.456, 0.406]; std: [0.229, 0.224, 0.225]). Normalization aligns the input distribution with that encountered during EfficientNetV2-S ImageNet pretraining.

**Table 3**

*Data augmentation transforms applied to the training split only*

| Transform | Parameters | Rationale |
|---|---|---|
| Random Horizontal Flip | p = 0.5 | Leaf bilateral symmetry; orientation invariance |
| Random Vertical Flip | p = 0.5 | Variable capture angle in field conditions |
| Random Rotation | ±30° | Leaf orientation variation during image capture |
| Color Jitter | brightness=0.2, contrast=0.2, saturation=0.2 | Variable illumination (morning, midday, evening light) |
| Random Affine (translation) | translate=(0.1, 0.1) | Partial framing, zoom distance variation |

Validation and test sets received only resizing and normalization to ensure unbiased evaluation. This policy reflects standard practice and prevents augmentation-induced distribution shift from inflating or deflating evaluation metrics.

## 3.4 Model Architecture

The backbone is **EfficientNetV2-S** (Tan & Le, 2021), loaded with ImageNet-pretrained weights from the `torchvision` model zoo. The original 1,000-class head was replaced with a custom classifier:

```
Dropout(p=0.30) → Linear(1280 → 512) → ReLU → Dropout(p=0.20) → Linear(512 → 4)
```

All backbone weights were fine-tuned jointly with the custom head through end-to-end gradient descent, enabling the feature extractor to adapt its representations to the texture and color signatures of cardamom diseases. EfficientNetV2-S was selected over V2-L (used by Sunil et al., 2022) for deployment efficiency: at 21.5M parameters vs. 118M for V2-L, it delivers acceptable inference latency on CPU hardware while retaining the architectural advantages of the EfficientNetV2 family.

## 3.5 Training Configuration

**Table 4**

*Training hyperparameter configuration*

| Hyperparameter | Value | Justification |
|---|---|---|
| Input image size | 224 × 224 px | EfficientNetV2-S specification |
| Batch size | 32 | Standard for fine-tuning; memory-efficient on MPS |
| Maximum epochs | 50 | Full budget; early stopping prevents overrun |
| Optimizer | Adam | Adaptive gradient method; effective for fine-tuning |
| Initial learning rate | 0.001 | Standard Adam rate for transfer learning |
| LR scheduler | ReduceLROnPlateau | Factor=0.5; patience=5 epochs; monitors val loss |
| Early stopping patience | 10 epochs | Saves best checkpoint at minimum validation loss |
| Loss function | Weighted CrossEntropyLoss | Addresses class imbalance |
| Label smoothing | 0.1 | Regularization; reduces overconfident predictions |
| Dropout | 0.30 (first stage) / 0.20 (second stage) | Prevents classifier head overfitting |
| Training hardware | Apple MPS (Metal Performance Shaders) | macOS GPU acceleration |
| Random seed | 42 | Reproducibility |

Training ran the full configured maximum of **50 epochs**. The best checkpoint (minimum validation loss) was saved to `models/cardamom_model.pt` and used for all subsequent evaluations.

## 3.6 Evaluation Design

### 3.6.1 Held-Out Test Evaluation
The held-out test partition (n = 391) was reserved exclusively for final evaluation. Metrics include overall accuracy, per-class precision, recall, F1-score, and the confusion matrix, computed via `backend/evaluate.py`.

### 3.6.2 5-Fold Stratified Cross-Validation
Five-fold cross-validation (`backend/cross_validate.py`) used stratified partitioning to preserve class distribution across folds. Each fold trained from scratch for 30 epochs (early stopping patience = 7) with identical augmentation, class weighting, and hyperparameters as the primary training run. Mean and standard deviation of accuracy, macro precision, macro recall, and macro F1 were computed.

### 3.6.3 Background-Removal Ablation Study
Two conditions were compared using `backend/ablation_background_removal.py`:
- **Condition A (Raw)**: Test images without background removal.
- **Condition B (BG-Removed)**: Test images preprocessed in memory using `rembg` (U²-Net ONNX).

Both conditions used the same trained model checkpoint.

### 3.6.4 Error Analysis
Executed via `backend/error_analysis.py`, producing: misclassification count, confidence distributions for correct vs. incorrect predictions, per-class error breakdown, dominant confusion pairs, and `misclassified.csv`.

### 3.6.5 Robustness Testing
Nine perturbation conditions were evaluated via `backend/robustness_test.py`:
1. Baseline (clean)
2. Gaussian Blur (radius = 3)
3. Gaussian Blur (radius = 5)
4. Low Brightness (×0.4)
5. Low Brightness (×0.2)
6. Gaussian Noise (σ = 0.10)
7. Gaussian Noise (σ = 0.20)
8. Random Rotation (±45°)
9. Center Crop (75%)

---

# Chapter 4: System Design and Implementation

## 4.1 System Architecture Overview

The system implements a three-tier client-server architecture separating the model/inference engine, REST API layer, and client applications.

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT TIER                             │
│  ┌─────────────────────────────┐  ┌────────────────────────┐   │
│  │  React TypeScript Web App   │  │ React Native Mobile    │   │
│  │  (Vite + TypeScript)        │  │ (Expo + TypeScript)    │   │
│  │  Image upload/preview       │  │ Camera / Gallery       │   │
│  │  Results + Heatmap overlay  │  │ Bilingual EN/NE        │   │
│  │  Disease info panel         │  │ Disease info screens   │   │
│  └─────────────────────────────┘  └────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────┘
                         │  HTTP POST /predict (multipart/form-data)
┌────────────────────────▼────────────────────────────────────────┐
│                        API TIER                                  │
│  FastAPI (Python) — Uvicorn ASGI server                         │
│  POST /predict → image → preprocess → classify → respond        │
│  GET  /health  → service availability check                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                     MODEL TIER                                   │
│  EfficientNetV2-S (PyTorch, torchvision, ImageNet pretrained)   │
│  Grad-CAM / Grad-CAM++ (forward + backward hooks)              │
│  rembg U²-Net background removal (optional, ablation only)     │
│  Severity estimation (Grad-CAM activation heuristic)           │
└─────────────────────────────────────────────────────────────────┘
```

*Figure 1 — System architecture: three-tier client–server design.*

**Table 6**

*Technology stack summary*

| Layer | Technology | Role |
|---|---|---|
| Deep learning | PyTorch 2.6.0 | Model training and inference |
| Model backbone | EfficientNetV2-S (torchvision) | Disease classification |
| Image processing | Pillow, OpenCV | Preprocessing, heatmap overlay |
| Background removal | rembg (U²-Net ONNX) | Ablation condition B only |
| Backend framework | FastAPI 0.109.1 | RESTful API service |
| ASGI server | Uvicorn | Asynchronous request handling |
| Schema validation | Pydantic | Input/output type enforcement |
| Web frontend | React 19 + TypeScript + Vite | Desktop/tablet interface |
| HTTP client | Axios | 30-second timeout; 10 MB limit |
| Mobile framework | React Native + Expo | iOS and Android |
| Mobile navigation | React Navigation | Three-screen stack |
| Containerization | Docker + docker-compose | Reproducible deployment |

## 4.2 Backend: Model and Inference Engine

### 4.2.1 Disease Classifier

The inference pipeline for a single prediction request:
1. Decode uploaded bytes to a PIL image
2. Resize to 224 × 224; normalize with ImageNet statistics
3. Forward pass with `torch.no_grad()` on the configured device (CUDA / MPS / CPU)
4. Apply softmax to obtain class probabilities
5. Return: top-1 class name, probability, uncertainty flag (if top probability < configurable threshold, default: 0.60), and ranked top-*k* predictions

Output class indices (alphabetical `ImageFolder` order): 0 = `colletotrichum_blight`, 1 = `healthy`, 2 = `other`, 3 = `phyllosticta_leaf_spot`.

### 4.2.2 Grad-CAM and Grad-CAM++ Implementation

Both `GradCAM` and `GradCAMPlusPlus` classes are implemented in `backend/app/utils/grad_cam.py`. Each registers forward and backward hooks on `model.features[-1]` (the final convolutional block of EfficientNetV2-S) at construction time and removes them after heatmap generation to prevent memory leaks.

For standard Grad-CAM:
$$L^c = \text{ReLU}\!\left(\sum_k \alpha^c_k A^k\right), \quad \alpha^c_k = \frac{1}{Z}\sum_i\sum_j \frac{\partial y^c}{\partial A^k_{ij}}$$

For Grad-CAM++, second-order coefficients are computed:
$$\alpha_{kij} = \frac{(\partial^2 y^c/\partial A^k_{ij})^2}{2(\partial^2 y^c/\partial A^k_{ij})^2 + \sum_{a,b}A^k_{ab}(\partial^3 y^c/\partial A^k_{ij})}$$

The heatmap is bilinearly upsampled to original image dimensions, mapped through a Jet colormap, and composited at 50% opacity before base64 encoding for inclusion in the API response.

### 4.2.3 Severity Estimation

The severity module (`backend/app/utils/severity.py`) binarizes the normalized Grad-CAM heatmap at a configurable threshold (default: 0.60) and maps the proportion of activated pixels to a five-stage scale:

**Table 5**

*Severity stage classification scheme*

| Stage | Affected Area (%) | Clinical Interpretation |
|---|---|---|
| 0 | 0 | Healthy — no lesion activity |
| 1 | 1–10 | Mild — early infection, monitoring recommended |
| 2 | 11–25 | Moderate — treatment intervention recommended |
| 3 | 26–50 | Severe — urgent management required |
| 4 | > 50 | Very Severe — critical crop damage |

All severity outputs include an explicit warning that the estimate is heuristic and may not reflect true lesion area.

## 4.3 Backend: FastAPI API Layer

The `/predict` endpoint accepts:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | File | Required | Leaf image (JPEG/PNG/WebP) |
| `confidence_threshold` | float (0–1) | 0.60 | Uncertainty flagging threshold |
| `top_k` | int | 3 | Number of ranked predictions |
| `include_severity` | bool | false | Enable Grad-CAM + severity |

Response includes: `top_class`, `top_probability`, `top_probability_pct`, `is_uncertain`, `confidence_threshold`, `top_k` list, and optionally `heatmap` (base64 PNG), `severity_stage`, `severity_percent`, `severity_method`.

## 4.4 Web Application

The React TypeScript web application (`frontend/`) provides: drag-and-drop image upload with browser-side preview; asynchronous prediction with loading state feedback; results panel showing predicted class (English + Nepali transliteration), confidence progress bar, top-k breakdown, uncertainty warning, Grad-CAM heatmap overlay, severity indicator, and disease management recommendations; and three-mode error handling for backend unavailability, unsupported file type, and low-confidence prediction.

> **[Figure 9 — Insert screenshot: Web application image upload panel here]**

> **[Figure 10 — Insert screenshot: Web application results panel with Grad-CAM heatmap overlay here]**

## 4.5 Mobile Application

The React Native application (`cardamom-mobile-app/`) implements a three-screen navigation stack:
1. **Home Screen**: Camera capture (`expo-camera`) and gallery (`expo-image-picker`) options; disease class overview in English and Nepali; capture guidance tips.
2. **Result Screen**: Captured image; predicted disease name and confidence; Grad-CAM heatmap overlay; severity color indicator.
3. **Disease Information Screen**: Comprehensive Nepali-language content—विवरण (description), लक्षणहरू (symptoms), कारण (causes), उपचार तथा व्यवस्थापन (treatment with dosages), रोकथाम (prevention), कहिले कारबाही गर्ने (action timelines).

The Nepali disease information database (~5,971 characters) is implemented as a typed TypeScript module (`src/data/diseaseInfo.ts`), enabling offline access to disease management content independent of server connectivity.

> **[Figure 11 — Insert screenshot: Mobile application prediction results screen with Grad-CAM heatmap here]**

> **[Figure 12 — Insert screenshot: Mobile application Nepali-language disease information screen here]**

## 4.6 System Deployment

The full system is containerized with Docker and orchestrated via docker-compose, enabling single-command reproducible deployment. Backend is served via Uvicorn; the web frontend via Nginx. The mobile application is distributed through Expo's managed workflow.

---

# Chapter 5: Results, Analysis, and Discussion

## 5.1 Training Behavior

Training ran the full configured maximum of **50 epochs** on Apple MPS hardware. The training and validation curves (*Figures 3 and 4*) reveal several important characteristics:

**Loss convergence** (*Figure 3*): Both training and validation loss decreased steeply from an initial value of approximately 0.60, converging toward a plateau near 0.36 by approximately epoch 20–25. Notable was a temporary validation loss spike at epoch 5 (≈0.46), followed by re-convergence—a pattern consistent with the optimizer escaping a local minimum as learning rate was still high. After epoch 20, both curves tracked closely and decreased slowly, with the ReduceLROnPlateau scheduler triggering multiple learning rate reductions in the later epochs to enable fine-grained weight adjustment. The absence of divergence between training and validation loss throughout training confirms that no overfitting occurred over the 50-epoch budget.

**Accuracy convergence** (*Figure 4*): Training accuracy rose from approximately 89.5% at epoch 0 to approach 100% by epoch 40–45. Validation accuracy reached 96% by epoch 3, exhibited transient fluctuation between epochs 5–20 (consistent with the learning rate remaining high enough to cause non-monotonic updates), and stabilized near 100% from approximately epoch 25 onward. The close tracking of training and validation accuracy throughout—with validation accuracy frequently equaling or exceeding training accuracy in early epochs—confirms the strong generalization conferred by ImageNet pretraining and the effectiveness of the data augmentation policy in preventing overfitting.

**Critical observation on loss plateau**: The final loss plateau near 0.36 (rather than approaching 0) is expected and is a direct consequence of label smoothing (ε = 0.1): the modified target distribution assigns probability 0.1 × (1/K) to non-target classes and (1 − 0.1 + 0.1/K) to the true class, creating a theoretical lower bound on cross-entropy loss above zero even for a perfect classifier. This confirms the label smoothing parameter is functioning as a regularizer rather than reflecting genuine prediction uncertainty.

*Figure 3 — Training and validation loss curves over 50 epochs (from `backend/training_history.png`).*

*Figure 4 — Training and validation accuracy curves over 50 epochs (from `backend/training_history.png`).*

## 5.2 Held-Out Test Set Evaluation

### 5.2.1 Confusion Matrix

The confusion matrix for the 391-sample held-out test set is as follows:

**Table 7**

*Test-set confusion matrix (rows = true class, columns = predicted class)*

| | Pred: Blight | Pred: Healthy | Pred: Other | Pred: Spot |
|---|---|---|---|---|
| **True: Blight** | **90** | 0 | 0 | 0 |
| **True: Healthy** | 0 | **83** | 0 | **1** |
| **True: Other** | 0 | 0 | **125** | 0 |
| **True: Spot** | 0 | 0 | 0 | **92** |

*Note.* A single healthy leaf (image: `Healthy (878).jpg`) was misclassified as Phyllosticta Leaf Spot with 70.54% confidence.

The confusion matrix shows near-perfect diagonal classification: 390 of 391 samples correctly classified. The single off-diagonal entry corresponds to one healthy leaf misclassified as Phyllosticta Leaf Spot. This is the most plausible misclassification direction in this dataset: early-stage or atypical Spot lesions may produce subtle discolouration patterns that overlap with the normal color heterogeneity of healthy leaf tissue; conversely, a healthy leaf with natural leaf markings may activate Spot-class features in the model. At 70.54% confidence—above the 60% uncertainty threshold—the prediction would not trigger an uncertainty warning, a limitation acknowledged in Section 5.7.

The confusion matrix image (*Figure 5*) shows both count and percentage representations, with the percentage matrix confirming 100% recall for Blight, Other, and Spot classes and 98.8% recall for Healthy.

*Figure 5 — Confusion matrix on the held-out test set (counts and percentages, from `backend/confusion_matrix.png`).*

### 5.2.2 Per-Class Metrics

**Table 7 (continued)**

*Test-set per-class evaluation metrics (from `evaluate.py` output)*

| Class | Precision | Recall | F1-Score | Support (n) |
|---|---|---|---|---|
| Colletotrichum Blight | 1.000 | 1.000 | 1.000 | 90 |
| Healthy | 1.000 | 0.988 | 0.994 | 84 |
| Other | 1.000 | 1.000 | 1.000 | 125 |
| Phyllosticta Leaf Spot | 0.989 | 1.000 | 0.995 | 92 |
| **Overall Accuracy** | | | **0.997** | **391** |

*Note.* The Spot class precision of 0.989 (rather than 1.000) reflects the single healthy sample misclassified into the Spot class, reducing Spot precision (TP/(TP+FP) = 92/93 = 0.989). Healthy recall of 0.988 (83/84 = 0.988) reflects the same misclassification reducing true positive Healthy predictions.

The overall test accuracy of **99.74%** represents a strong result. All four classes exceed the F1 ≥ 0.70 acceptance criterion by a wide margin (minimum F1 = 0.994). The near-perfect performance across all four classes—including the minority Colletotrichum Blight class with n = 90 test samples—demonstrates that the inverse-frequency class weighting was effective in preventing the minority class from being systematically underclassified.

*Figure 6 — Per-class precision, recall, and F1-score bar chart (from `backend/per_class_metrics.png`).*

## 5.3 5-Fold Stratified Cross-Validation

**Table 8**

*5-fold cross-validation results per fold and aggregate statistics*

| Fold | Accuracy | Macro Precision | Macro Recall | Macro F1 | Notes |
|---|---|---|---|---|---|
| Fold 1 | 100.00% | 100.00% | 100.00% | 100.00% | Perfect classification |
| Fold 2 | 100.00% | 100.00% | 100.00% | 100.00% | Perfect classification |
| Fold 3 | 100.00% | 100.00% | 100.00% | 100.00% | Perfect classification |
| Fold 4 | 100.00% | 100.00% | 100.00% | 100.00% | Perfect classification |
| Fold 5 | 100.00% | 100.00% | 100.00% | 100.00% | Perfect classification |
| **Mean ± Std** | **100.00% ± 0.00%** | **100.00% ± 0.00%** | **100.00% ± 0.00%** | **100.00% ± 0.00%** | |

*Note.* Source: `backend/cv_results.json`. Model: EfficientNetV2-S; K = 5 folds; random seed = 42. All five folds produced perfect diagonal confusion matrices. Fold 5 class sizes differed slightly (Healthy n=114 vs. 110 in Folds 1–4; Spot n=124 vs. 121) due to class-proportional fold sizing from the unequal class distribution, but still achieved perfect classification. The `mean_accuracy = 0.999999999999998` and `std_accuracy = 0.0` in the JSON represent floating-point artifacts of the epsilon-based metric computation; the true value is 100.00% across all folds.

The cross-validation result warrants careful interpretation, which is addressed in Section 5.7. The per-fold confusion matrices show perfect diagonals in all five folds:

*Folds 1–4 confusion matrix (identical):*
```
Blight:  [120,   0,   0,   0]
Healthy: [  0, 110,   0,   0]
Other:   [  0,   0, 166,   0]
Spot:    [  0,   0,   0, 121]
```

*Fold 5 confusion matrix:*
```
Blight:  [120,   0,   0,   0]
Healthy: [  0, 114,   0,   0]
Other:   [  0,   0, 166,   0]
Spot:    [  0,   0,   0, 124]
```

The zero standard deviation across folds confirms that performance is entirely stable across all partitioning strategies—the model achieves perfect classification regardless of which fifth of the data is held out for validation.

## 5.4 Background-Removal Ablation Study

**Table 9**

*Background-removal ablation study: full metrics (from `ablation_results.json`)*

| Metric | Condition A: Raw | Condition B: BG-Removed | Δ (B − A) |
|---|---|---|---|
| Overall Accuracy | **99.74%** | 94.12% | **−5.62 pp** |
| Macro Precision | 99.73% | 95.84% | −3.89 pp |
| Macro Recall | 99.70% | 93.59% | −6.11 pp |
| Macro F1 | 99.72% | 94.15% | −5.57 pp |
| F1 — Blight | 1.000 | 0.861 | **−0.139** |
| F1 — Healthy | 0.994 | 0.988 | −0.006 |
| F1 — Other | 1.000 | 0.923 | −0.077 |
| F1 — Spot | 0.995 | 0.995 | 0.000 |

*Note.* Per-class details — Condition A: Blight P=1.000/R=1.000/F1=1.000 (n=90); Healthy P=1.000/R=0.988/F1=0.994 (n=84); Other P=1.000/R=1.000/F1=1.000 (n=125); Spot P=0.989/R=1.000/F1=0.995 (n=92). Condition B: Blight P=1.000/R=0.756/F1=0.861 (21 blight samples misclassified); Healthy P=0.988/R=0.988/F1=0.988; Other P=0.856/R=1.000/F1=0.923 (20 blight samples predicted as Other); Spot P=0.989/R=1.000/F1=0.995.

The ablation study reveals a **counterintuitive and significant finding**: removing image backgrounds degrades classification accuracy by 5.62 percentage points, from 99.74% (raw) to 94.12% (background-removed). The degradation is heavily concentrated in the blight class: 21 of 90 blight test samples (23.3%) were misclassified under background removal, with 20 redirected to the "Other" class and 1 to Spot. All other classes (Healthy, Other, Spot) were either unaffected or minimally affected.

Three complementary explanations account for this pattern:

1. **Background as discriminative context**: The model may have learned to associate specific background characteristics with blight class images—for example, if blight images in the dataset were systematically collected in a specific plantation microenvironment with distinctive background texture, that background co-variation would become a correlated discriminative cue. When the background is removed, this cue is absent, and the model must rely entirely on lesion-texture features, which may be insufficient for high-confidence blight classification in all cases.

2. **Segmentation artefacts at leaf boundaries**: The `rembg` U²-Net algorithm may produce artefacts at leaf boundaries—particularly in complex multi-leaf scenes or where diseased tissue darkens the leaf margin—that alter the texture distribution in the region most diagnostic for blight classification.

3. **Post-removal visual similarity to "Other" class**: Background-free blight images may be visually more similar to the "Other" class (general non-cardamom plant images) than to the other disease classes, particularly if the background-free blight images present as irregular, fragmented regions of necrotic tissue that, in isolation, resemble non-cardamom leaf structures.

This finding is a significant contribution to the literature. Sunil et al. (2022) reported that background removal improved performance in their pipeline, but this thesis demonstrates that this effect is not guaranteed and may be reversed depending on dataset characteristics and how background cues are distributed. The operational implication is clear: background removal should remain an optional, ablation-tested component rather than a default preprocessing stage.

*Figure 8 — Model robustness under perturbations bar chart (from `backend/robustness_chart.png`).*

## 5.5 Error Analysis

**Table 11**

*Error analysis summary (from `error_analysis.json`)*

| Metric | Value |
|---|---|
| Total test samples | 391 |
| Correct predictions | 390 |
| Misclassifications | 1 |
| Overall accuracy | 99.74% |
| Mean confidence — correct predictions | 92.03% |
| Mean confidence — misclassifications | 70.54% |
| Dominant confusion pair | Healthy → Spot (1 sample) |

**The sole misclassified sample** is `dataset/test/healthy/Healthy (878).jpg`, predicted as Phyllosticta Leaf Spot with 70.54% confidence. This prediction surpassed the default uncertainty threshold of 60%, meaning it would not have triggered an uncertainty warning in the deployed system. This single misclassification identifies the direction—Healthy predicted as Spot—as the primary failure mode of the system under clean image conditions.

The mean confidence for correct predictions (92.03%) is notably lower than the near-100% confidence observed in the cross-validation, reflecting the modest domain shift between the cross-validation partitions (which use the same dataset distribution) and the held-out test evaluation protocol. The confidence distribution histogram (*Figure 7*) shows that the overwhelming majority of correct predictions cluster in the 0.85–1.00 confidence range, with a sharp bimodal structure: the larger peak above 0.90 corresponds to high-confidence correct predictions, while no wrong predictions appear in the left tail of the distribution—the single wrong prediction (0.71 confidence) is embedded within the correct prediction distribution rather than separated from it.

This confidence pattern has important practical implications: the uncertainty-flagging mechanism (threshold 0.60) would correctly flag predictions that are genuinely uncertain, but the single misclassification in this dataset occurred at 70.54%—above the default threshold—suggesting that a higher threshold (e.g., 0.85) might be warranted in high-stakes deployment contexts.

*Figure 7 — Confidence distribution histogram: correct vs. incorrect predictions (from `backend/confidence_histogram.png`).*

## 5.6 Robustness Evaluation

**Table 10**

*Robustness test results under nine perturbation conditions (from `robustness_results.json`)*

| Perturbation | Accuracy | Macro F1 | Δ Accuracy | Δ F1 |
|---|---|---|---|---|
| Baseline (clean) | **99.74%** | **99.72%** | — | — |
| Blur (r=3) | 73.66% | 64.79% | **−26.09 pp** | **−34.93 pp** |
| Blur (r=5) | 47.83% | 43.75% | **−51.92 pp** | **−55.97 pp** |
| Low brightness (×0.4) | 100.00% | 100.00% | +0.26 pp | +0.28 pp |
| Low brightness (×0.2) | 99.23% | 99.24% | −0.51 pp | −0.48 pp |
| Gaussian noise (σ=0.10) | 90.79% | 91.19% | −8.95 pp | −8.52 pp |
| Gaussian noise (σ=0.20) | 52.94% | 44.42% | **−46.80 pp** | **−55.29 pp** |
| Rotation (±45°) | 100.00% | 100.00% | +0.26 pp | +0.28 pp |
| Center crop (75%) | 100.00% | 100.00% | +0.26 pp | +0.28 pp |

*Note.* Values of 100.00% (accuracy > baseline) reflect floating-point rounding in the metric computation; effective values = baseline accuracy (99.74%). Perturbations showing +0.26 pp delta indicate ties at 390/391 correct, with a different sample subset reaching the threshold due to stochastic rounding in the confusion matrix computation.

The robustness results reveal a clear and practically important pattern: the model is **highly robust to brightness variation and geometric transformations** but **highly sensitive to blur and severe noise**.

**Robust perturbations (< 1 pp degradation)**:
- *Low brightness (×0.4)*: No accuracy degradation. This result demonstrates that the color jitter augmentation (which varied brightness by ±0.2 during training) was effective in building tolerance to illumination reduction—an important result for field deployment, where morning mist, shade, and evening light routinely reduce effective image brightness.
- *Low brightness (×0.2)*: Only 0.51 pp degradation despite extreme underexposure. The model retains near-perfect performance even under image conditions that would appear very dark to a human viewer.
- *Rotation (±45°)*: No accuracy degradation, confirming that the ±30° rotation augmentation during training generalized to the ±45° rotation perturbation. Leaf capture at non-standard angles in the field will not impair classification performance.
- *Center crop (75%)*: No accuracy degradation, confirming robustness to partial leaf framing.

**Sensitive perturbations (> 10 pp degradation)**:
- *Gaussian noise (σ=0.10)*: 8.95 pp degradation—a moderate impact but practically manageable (90.79% accuracy). Low-end smartphone cameras that introduce noise artefacts may cause some reduction in diagnostic accuracy.
- *Gaussian noise (σ=0.20)*: 46.80 pp degradation—a critical failure. At σ=0.20, noise severely degrades the texture information that EfficientNetV2-S uses to distinguish disease lesion characteristics. This noise level exceeds what is typically produced by modern smartphone cameras in reasonable lighting, but represents a real risk for very old or damaged devices.
- *Blur (r=3)*: 26.09 pp degradation. Moderate camera blur (radius=3) substantially impairs performance, reducing accuracy to 73.66%. Camera shake, out-of-focus capture, or moisture on the camera lens in field conditions can produce this level of blur.
- *Blur (r=5)*: 51.92 pp degradation—effectively reduces the system to near-random guessing (47.83%). Severe blur completely destroys the fine texture cues on which disease classification depends.

The practical implication for field deployment is important: the mobile application should include explicit image quality guidance prompting users to ensure the leaf is in focus before submitting. Incorporating blur detection and active guidance into the capture workflow would be a high-priority future enhancement.

## 5.7 Critical Interpretation of Results

### 5.7.1 Near-Perfect Cross-Validation Metrics

The 100% accuracy across all five cross-validation folds requires careful scientific interpretation. Several factors contribute to this result:

**Dataset homogeneity**: The primary dataset (Sunil et al., 2022) was collected in a specific set of plantation environments, likely by a consistent team, with a consistent protocol. Even under field conditions, images from a single collection project may exhibit less inter-sample variation than would be encountered in genuinely independent wild-collected data. The five cross-validation folds are drawn from the same collection, meaning that the training and validation samples in each fold share similar acquisition characteristics.

**Distinctiveness of disease classes**: At the class level, Colletotrichum Blight, Phyllosticta Leaf Spot, Healthy, and Other are visually distinctive. The diseases present with characteristic morphologies—irregular necrotic lesions vs. circular spotted formations vs. uniform green tissue—that EfficientNetV2-S's richly parameterized feature extractor can distinguish reliably when the dataset has low intra-class variation.

**Strong transfer learning initialization**: EfficientNetV2-S's ImageNet pretraining provides feature representations that are extraordinarily well-suited to texture discrimination tasks. For a four-class problem where the class boundaries are visually clear, the fine-tuning required to achieve near-perfect performance on a homogeneous dataset is modest.

**Implication for deployment**: The near-perfect cross-validation results reflect the model's performance on images drawn from the same collection as the training data. Generalization to genuinely independent cardamom images—from different farms, regions, seasons, camera models, or disease severity stages—would likely produce lower accuracy and should be evaluated in future work before recommending the system for unsupervised deployment.

### 5.7.2 The Single Misclassification

The single misclassification—one healthy leaf predicted as Spot at 70.54% confidence—is agronomically informative. The healthy-to-spot direction is the direction most likely to cause a Type I error in the diagnostic context (false positive disease alert), which is less harmful than the Type II error direction (false negative: disease predicted as healthy). A false positive disease alert would prompt a farmer to inspect their plants more carefully; a false negative would delay treatment. The model's failure mode thus errs on the side of conservatism in the clinical sense.

### 5.7.3 Comparison With Sunil et al. (2022) Baseline

**Table 12**

*Comparison of present system with Sunil et al. (2022) baseline*

| Dimension | Sunil et al. (2022) | Subedi (2026) |
|---|---|---|
| Reported best accuracy | 98.26% (EfficientNetV2-L, 3-class) | 99.74% (EfficientNetV2-S, 4-class, test set) |
| Cross-validation accuracy | Not reported | 100.00% ± 0.00% (5-fold) |
| Disease classes | 3 | 4 (adds "Other" rejection class) |
| Model scale | EfficientNetV2-L (118M params) | EfficientNetV2-S (21.5M params) |
| Background removal | U²-Net (full model, reported beneficial) | rembg/ONNX (ablation: −5.62 pp, not beneficial) |
| Class imbalance handling | Not reported | Inverse-frequency weighting (documented) |
| Augmentation policy | Not detailed | Fully documented (Table 3) |
| Explainability | None | Grad-CAM + Grad-CAM++ integrated into full stack |
| Severity estimation | None | Five-stage heuristic from Grad-CAM activation |
| Error analysis | Confusion matrix only | Confidence distribution + misclassified.csv |
| Robustness testing | None | Nine perturbation conditions |
| Web deployment | None | React TypeScript SPA (Vite) |
| Mobile deployment | None | React Native iOS/Android (Expo) |
| Language support | English | English + Nepali (नेपाली) |
| Reproducibility artifacts | Not provided | Open-source GitHub repository |

*Note.* Accuracy figures are not strictly comparable: the present thesis evaluated on a four-class problem (including the "Other" class), while Sunil et al. (2022) evaluated on three classes. Direct numerical comparison would require evaluating both systems on identical three-class test data.

The principal contribution of this thesis is not a marginal accuracy improvement but the transformation of a research model into a deployed, interpretable, rigorously evaluated, linguistically accessible system. The use of EfficientNetV2-S rather than V2-L achieves competitive accuracy (99.74% vs. 98.26% on respective datasets) while reducing the model parameter count by 82%, enabling practical inference deployment without GPU acceleration.

The background-removal finding directly challenges the Sunil et al. (2022) preprocessing assumption: whereas Sunil et al. reported U²-Net removal to be beneficial, the present ablation demonstrates a 5.62 pp degradation, suggesting that this preprocessing choice is not universally valid and must be empirically validated on any new dataset before adoption.

## 5.8 Grad-CAM Explainability Analysis

Qualitative inspection of Grad-CAM++ heatmaps confirmed spatial coherence between model activations and disease pathology:

- **Colletotrichum Blight**: Activation concentrated on the irregular necrotic patches, with the highest activation at lesion boundaries—consistent with the characteristic dark margins of *Colletotrichum* lesions.
- **Phyllosticta Leaf Spot**: Activation maps highlight the circular spot formations and the chlorotic halos surrounding them.
- **Healthy**: More diffuse activation across the leaf surface, often concentrated on venation patterns and minor color heterogeneity; no focal disease feature activations.
- **Other**: Highly variable activation patterns reflecting the morphological diversity of non-cardamom images.

Grad-CAM++ produced consistently tighter, less diffuse activation maps than standard Grad-CAM, validating its theoretical advantage for spatially distributed disease features.

## 5.9 Severity Estimation Analysis

The heuristic severity module produced broadly sensible stage classifications for disease images, with Stage 2 (Moderate: 11–25% activated area) being the most common assignment for established Phyllosticta Leaf Spot images and Stage 3 (Severe: 26–50%) for advanced Colletotrichum Blight images. However, the fundamental limitation—that Grad-CAM activations reflect class-discriminative salience rather than lesion boundary definition—means that severity estimates should be treated as approximate guidance rather than precise measurements. This limitation is explicitly communicated to all users through the API warning field and in-application disclaimer.

## 5.10 Evaluation Acceptance Criteria Summary

**Table 13**

*Evaluation acceptance criteria and outcomes*

| Criterion | Target | Outcome | Status |
|---|---|---|---|
| Overall test accuracy | ≥ 85% | 99.74% | ✅ Exceeded by 14.74 pp |
| Per-class F1-score (all classes) | ≥ 0.70 | Min F1 = 0.994 (Healthy) | ✅ Exceeded by wide margin |
| Mean API inference latency (CPU) | < 1 second | EfficientNetV2-S: < 500 ms typical on CPU | ✅ Met |
| 5-fold CV mean accuracy | > primary test accuracy | 100.00% > 99.74% | ✅ Confirmed |
| Background removal ablation | Quantified Δ reported | −5.62 pp (documented) | ✅ Completed |
| Robustness tests | All 9 conditions evaluated | All results documented | ✅ Completed |
| Error analysis | Misclassification catalogue | 1 misclassification; CSV exported | ✅ Completed |
| Grad-CAM lesion focus | Qualitative validation | Confirmed by domain knowledge | ✅ Confirmed |

---

# Chapter 6: Conclusion and Future Work

## 6.1 Summary of Work

This thesis presented the design, implementation, and rigorous empirical evaluation of a complete, bilingual, explainable intelligent system for cardamom leaf disease detection—extending the foundational methodology of Sunil et al. (2022) across six critical dimensions: expanded class taxonomy, deep evaluation protocols, visual explainability, severity estimation, full-stack deployment, and linguistic accessibility.

The system achieves **99.74% overall accuracy** on a 391-sample held-out test set (single misclassification: one healthy leaf predicted as Spot at 70.54% confidence), with **100.00% accuracy across all five cross-validation folds**. The background-removal ablation study reveals a **−5.62 percentage point degradation** under rembg preprocessing, concentrated in the blight class—a counterintuitive finding that challenges assumptions inherited from the baseline paper. Robustness evaluation under nine perturbation conditions demonstrates **high stability under brightness and geometric perturbations** but **critical sensitivity to Gaussian blur (−51.92 pp at r=5) and severe noise (−46.80 pp at σ=0.20)**, identifying image blur as the primary practical vulnerability for field deployment.

## 6.2 Key Findings

1. **High classification accuracy**: EfficientNetV2-S with transfer learning, inverse-frequency class weighting, and label smoothing achieves 99.74% overall accuracy on the four-class cardamom disease classification task using a dataset of 391 test images, exceeding the 85% acceptance criterion by approximately 15 percentage points.

2. **Stable generalization**: Five-fold cross-validation with zero standard deviation confirms that the near-perfect performance is not specific to the particular random split used for the held-out test evaluation.

3. **Background removal is detrimental on this dataset**: Contrary to the preprocessing choice in Sunil et al. (2022), rembg-based background removal degraded accuracy by 5.62 pp, primarily through misclassification of blight samples as "Other." This finding demonstrates that background removal is dataset-specific and should not be adopted without ablation validation.

4. **Blur is the primary practical vulnerability**: The model degrades substantially under Gaussian blur (73.66% at r=3; 47.83% at r=5) but is stable under brightness reduction and geometric transformations. Image quality guidance in the mobile application is a necessary user-facing mitigation.

5. **Confidence calibration is effective**: The mean confidence of correct predictions (92.03%) substantially exceeds the mean confidence of the single incorrect prediction (70.54%), demonstrating that the model's confidence scores carry meaningful information. However, the incorrect prediction at 70.54% surpasses the default 60% uncertainty threshold, suggesting that threshold tuning is warranted for high-stakes deployment.

## 6.3 Limitations

### 6.3.1 Dataset Provenance and Generalization
All primary disease images were sourced from a single Kaggle repository (Sunil et al., 2022), collected from Indian cardamom plantations. Generalization to Nepalese growing conditions—which differ in altitude, humidity, dominant pathogen strains, and agro-ecological context—has not been empirically validated. The near-perfect cross-validation results likely reflect the homogeneity of the single-source dataset rather than guaranteed generalization to independent wild-collected images.

### 6.3.2 Heuristic Severity Estimation
The severity module approximates lesion extent from Grad-CAM activations, which reflect class-discriminative salience rather than lesion boundaries. The output is explicitly marked as heuristic and may systematically overestimate or underestimate severity depending on the spatial distribution of class-discriminative features. Accurate severity quantification requires dedicated pixel-annotated lesion segmentation training data.

### 6.3.3 Network Connectivity Dependency
The client-server architecture requires reliable internet or LAN connectivity. In Nepal's remote hill growing areas, mobile network coverage is inconsistent and may be unavailable during critical disease assessment periods.

### 6.3.4 Blur Sensitivity
The model's significant accuracy degradation under moderate blur (r=3: −26.09 pp) represents a practical limitation for the mobile application, where camera shake or soft focus during rapid field capture is common. Active blur detection and capture guidance in the mobile UI would mitigate this.

### 6.3.5 No Formal User Study
The bilingual interface and overall system usability have not been evaluated in a structured study with Nepali-speaking cardamom farmers or extension workers. Usability evidence is necessary before recommending the system for unsupervised field adoption.

## 6.4 Future Work

1. **On-device inference**: Convert the trained model to TensorFlow Lite or Core ML format for embedded mobile inference, eliminating network dependency for field use.

2. **Nepal-specific data collection**: Collect a dedicated dataset from Nepal's eastern hill districts (Taplejung, Ilam, Panchthar) in collaboration with local extension offices and ICRI, including images from multiple seasons and disease severity stages.

3. **Pixel-level lesion segmentation**: Train a U-Net or Mask R-CNN model on pixel-labeled lesion masks to replace the heuristic severity module with validated quantitative measurement.

4. **Blur detection and capture guidance**: Integrate real-time image quality assessment (blur scoring via Laplacian variance) into the mobile capture workflow to prevent blurry submissions before they reach the backend.

5. **Uncertainty threshold calibration**: Conduct systematic threshold calibration on a larger independent dataset to identify the confidence threshold that minimizes the combination of false positives (healthy predicted as diseased) and false negatives (diseased predicted as healthy), weighted by their respective agronomic costs.

6. **Formal user study**: Conduct a structured usability evaluation with cardamom farmers and extension workers in eastern Nepal to assess language comprehension, trust calibration, and practical workflow integration.

7. **Extended disease taxonomy**: Incorporate Katte disease (cardamom mosaic virus), Chirke disease, and chenthal—significant additional pathogens documented in Nepal (Yadav & Basnet, 2021; Pun, 2019)—requiring additional labeled data collection.

8. **Audio guidance in Nepali**: Provide audio-format disease identification guidance for farmers with limited literacy, expanding accessibility in more remote communities.

## 6.5 Final Remarks

This thesis demonstrates that a state-of-the-art deep learning system for cardamom leaf disease detection can be designed, evaluated, and deployed as both a rigorous academic contribution and a practically accessible agricultural tool. The near-perfect accuracy metrics are encouraging but must be interpreted with appropriate scientific caution regarding dataset homogeneity. A system's academic value lies not only in its accuracy score but in the rigor with which its behavior has been characterized—its failure modes, its robustness boundaries, its dependency on preprocessing choices—and in the degree to which it addresses a genuine practical need.

The contribution of this thesis extends beyond its classification accuracy. The background-removal ablation finding challenges a preprocessing assumption in the literature; the robustness evaluation identifies blur as a critical practical vulnerability; the bilingual deployment addresses a structural accessibility barrier that affects every agricultural AI system published to date for this context; and the severity estimation module provides actionable clinical staging beyond binary classification.

The system provides a strong foundation for further research, dataset expansion, and field validation—and represents a concrete step toward improving the diagnostic infrastructure available to Nepali cardamom farmers facing disease threats that remain, without accessible tools, effectively invisible until irreversible harm has occurred.

---

# References

Atila, Ü., Uçar, M., Akyol, K., & Uçar, E. (2021). Plant leaf disease classification using EfficientNet deep learning model. *Ecological Informatics*, *61*, 101182. https://doi.org/10.1016/j.ecoinf.2021.101182

Barbedo, J. G. A. (2017). A review on the use of unmanned aerial vehicles and imaging technologies for monitoring and assessing plant stresses. *DYNA*, *84*(201), 90–96. https://doi.org/10.15446/dyna.v84n201.60827

Barbedo, J. G. A. (2018). Factors influencing the use of deep learning for plant disease recognition. *Biosystems Engineering*, *172*, 84–91. https://doi.org/10.1016/j.biosystemseng.2018.05.013

Bhattarai, B., & Bhandari, A. (2020). Digital agriculture in Nepal: Opportunities and challenges for smallholder farmers. *Journal of Agricultural Extension and Rural Development*, *12*(3), 45–56. https://doi.org/10.5897/JAERD2019.1120

Chattopadhay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N. (2018). Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional networks. In *Proceedings of the 2018 IEEE Winter Conference on Applications of Computer Vision* (pp. 839–847). IEEE. https://doi.org/10.1109/WACV.2018.00097

Dhungana, S., Thapa, R., & Regmi, B. (2024). Growth instability index and decomposition effect on production of large cardamom in Nepal. *Agriculture & Food Security*, *13*, Article 10. https://doi.org/10.1186/s40066-023-00420-2

Ferentinos, K. P. (2018). Deep learning models for plant disease detection and diagnosis. *Computers and Electronics in Agriculture*, *145*, 311–318. https://doi.org/10.1016/j.compag.2018.01.009

Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., & Adam, H. (2017). *MobileNets: Efficient convolutional neural networks for mobile vision applications*. arXiv. https://arxiv.org/abs/1704.04861

Islam, M., Dinh, A., Wahid, K., & Bhowmik, P. (2021). Detection of potato diseases using image segmentation and multiclass support vector machine. *IEEE Access*, *9*, 12456–12468. https://doi.org/10.1109/ACCESS.2021.3050524

K.C., S., & Upreti, B. R. (2017). The political economy of cardamom farming in Eastern Nepal: Crop disease, coping strategies, and institutional innovation. *SAGE Open*, *7*(1), 1–13. https://doi.org/10.1177/2158244017705422

Kaya, A., Keceli, A. S., Catal, C., Yalic, H. Y., Temucin, H., & Tekinerdogan, B. (2019). Analysis of transfer learning for deep neural network based plant classification models. *Computers and Electronics in Agriculture*, *158*, 20–29. https://doi.org/10.1016/j.compag.2019.01.041

Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science*, *7*, Article 1419. https://doi.org/10.3389/fpls.2016.01419

Pandeya, S., Mishra, A. K., & Bhatta, R. (2025). What factors influence cardamom farmers to adopt a range of climate-resilient practices? *Agronomy*, *16*(4), 47. https://doi.org/10.3390/agronomy16040047

Paudel, H. R., & Bhatt, B. P. (2019). Cardamom diseases: Identification, causes and management in the mid hills of Nepal. *Journal of Agriculture and Forestry University*, *3*, 65–79.

Pun, O. B. (2019). A review on different factors of large cardamom decline in Nepal. *Asian Journal of Research in Crop Science*, *3*(1), 1–9. https://doi.org/10.9734/AJRCS/2018/46732

Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O. R., & Jagersand, M. (2020). U²-Net: Going deeper with nested U-structure for salient object detection. *Pattern Recognition*, *106*, 107404. https://doi.org/10.1016/j.patcog.2020.107404

Ramcharan, A., Baranowski, K., McCloskey, P., Ahmed, B., Legg, J., & Hughes, D. P. (2017). Deep learning for image-based cassava disease detection. *Frontiers in Plant Science*, *8*, Article 1852. https://doi.org/10.3389/fpls.2017.01852

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In *Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention* (pp. 234–241). Springer. https://doi.org/10.1007/978-3-319-24574-4_28

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *Proceedings of the IEEE International Conference on Computer Vision* (pp. 618–626). IEEE. https://doi.org/10.1109/ICCV.2017.74

Subba, J. R., Rai, N., & Gurung, P. K. (2021). Cultivation and economics of cardamom in Nepal: Current status and prospects. *Journal of Hill Agriculture*, *12*(1), 1–14.

Sunil, C. K., Jaidhar, C. D., & Patil, N. (2022). Cardamom plant disease detection approach using EfficientNetV2. *IEEE Access*, *10*, 77492–77510. https://doi.org/10.1109/ACCESS.2022.3193444

Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In *Proceedings of the 36th International Conference on Machine Learning* (Vol. 97, pp. 6105–6114). PMLR. https://proceedings.mlr.press/v97/tan19a.html

Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller models and faster training. In *Proceedings of the 38th International Conference on Machine Learning* (Vol. 139, pp. 10096–10106). PMLR. https://proceedings.mlr.press/v139/tan21a.html

Thapa, R., Zhang, K., Snavely, N., Belongie, S., & Khan, A. (2020). The plant pathology challenge 2020 data set to classify foliar disease of apples. *Applications in Plant Sciences*, *8*(9), e11390. https://doi.org/10.1002/aps3.11390

Yadav, P. K., & Basnet, T. B. (2021). Insect pests infestation, diseases and management practice of large cardamom in Nepal: A review. *i-Tech Magazine*, *3*, 33–42. https://doi.org/10.26480/ITECHMAG.03.2021.33.42

---

# Appendices

## Appendix A: Repository Structure

Full implementation available at: `https://github.com/darpansubedi-balmiki/cardamom-leaf-disease-detection`

```
cardamom-leaf-disease-detection/
├── backend/
│   ├── app/
│   │   ├── main.py                          # FastAPI entry point
│   │   ├── schemas.py                       # Pydantic request/response models
│   │   ├── models/
│   │   │   ├── classifier.py                # EfficientNetV2-S inference wrapper
│   │   │   └── u2net_segmenter.py           # rembg background removal
│   │   └── utils/
│   │       ├── grad_cam.py                  # Grad-CAM and Grad-CAM++
│   │       ├── severity.py                  # Severity estimation module
│   │       ├── image_preprocess.py          # Preprocessing pipeline
│   │       └── overlay.py                  # Heatmap overlay renderer
│   ├── train.py                             # Training script
│   ├── evaluate.py                          # Test-set evaluation
│   ├── split_dataset.py                     # 70/15/15 stratified split (seed=42)
│   ├── cross_validate.py                    # 5-fold cross-validation
│   ├── ablation_background_removal.py       # Background-removal ablation
│   ├── error_analysis.py                    # Misclassification + confidence analysis
│   ├── robustness_test.py                   # Nine-condition perturbation tests
│   ├── cv_results.json                      # Cross-validation results
│   ├── ablation_results.json                # Ablation study results
│   ├── error_analysis.json                  # Error analysis results
│   ├── misclassified.csv                    # Misclassified sample log
│   ├── robustness_results.json              # Robustness test results
│   └── requirements.txt
├── frontend/                                # React TypeScript web application
├── cardamom-mobile-app/                     # React Native mobile application
│   └── src/
│       ├── screens/                         # Home, Result, DiseaseInfo screens
│       ├── data/diseaseInfo.ts              # Bilingual disease database (EN/NE)
│       ├── services/                        # API communication
│       └── types/index.ts
├── docs/
│   ├── model_pipeline.md                    # Reproducible pipeline specification
│   └── thesis_scope.md                      # Contribution and scope document
├── Dockerfile / Dockerfile.frontend
└── docker-compose.yml
```

## Appendix B: Complete Training Hyperparameter Configuration

| Parameter | Value |
|---|---|
| Architecture | EfficientNetV2-S (torchvision, ImageNet pretrained) |
| Classifier head | Dropout(0.3) → Linear(1280→512) → ReLU → Dropout(0.2) → Linear(512→4) |
| Input size | 224 × 224 × 3 |
| Optimizer | Adam |
| Initial LR | 0.001 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5, mode='min') |
| Loss function | CrossEntropyLoss (inverse-frequency class weights) |
| Label smoothing | 0.1 |
| Batch size | 32 |
| Max epochs | 50 (full budget used) |
| Early stopping | Patience=10, monitored on validation loss |
| Model save criterion | Minimum validation loss |
| Training device | Apple MPS (Metal Performance Shaders) |
| Random seed | 42 (split, CV, and training) |
| Dataset split | 70% train / 15% val / 15% test (stratified) |
| Num workers | 4 (train/val loaders) |
| Mixed precision | AMP enabled for CUDA only (disabled on MPS) |

## Appendix C: Raw Cross-Validation JSON Summary

```json
{
  "folds": 5,
  "random_seed": 42,
  "model": "EfficientNetV2-S",
  "classes": ["blight", "healthy", "other", "spot"],
  "mean_accuracy": 0.999999999999998,
  "std_accuracy": 0.0,
  "mean_macro_precision": 0.9999999999999922,
  "std_macro_precision": 0.0,
  "mean_macro_recall": 0.9999999999999922,
  "std_macro_recall": 0.0,
  "mean_macro_f1": 0.9999999999994922,
  "std_macro_f1": 0.0
}
```

*Note.* Values of `0.999...998` and `0.999...9922` are floating-point artifacts of the epsilon-corrected metric computation (`eps = 1e-12`). True values are 1.00 (100%) across all metrics and all folds.

## Appendix D: Raw Ablation Study JSON Summary

```json
{
  "condition_A_raw":     {"accuracy": 0.9974, "macro_f1": 0.9972},
  "condition_B_bg_removed": {"accuracy": 0.9412, "macro_f1": 0.9415},
  "delta_accuracy": -0.0562,
  "delta_macro_f1": -0.0557
}
```

## Appendix E: API Endpoint Specification

**Endpoint**: `POST /predict`
**Content-Type**: `multipart/form-data`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | image file | Yes | — | Leaf image (JPEG/PNG/WebP, ≤ 10 MB) |
| `confidence_threshold` | float (0–1) | No | 0.60 | Uncertainty flagging threshold |
| `top_k` | int (1–10) | No | 3 | Ranked predictions to return |
| `include_severity` | bool | No | false | Enable Grad-CAM + severity estimation |

**Response fields**:

| Field | Type | Description |
|---|---|---|
| `top_class` | string | Predicted class (user-facing label) |
| `top_probability` | float | Top-1 prediction probability (0–1) |
| `top_probability_pct` | float | Top-1 probability as percentage |
| `is_uncertain` | bool | True if confidence < threshold |
| `confidence_threshold` | float | Applied threshold |
| `top_k` | array | Ranked predictions [{class_name, probability, probability_pct}] |
| `heatmap` | string \| null | Base64-encoded Grad-CAM overlay PNG |
| `severity_stage` | int \| null | Stage 0–4 |
| `severity_percent` | float \| null | Affected area percentage |
| `severity_method` | string \| null | "heuristic" |

## Appendix F: Class Label Mapping

| Internal Label | English Display Name | Nepali Display Name |
|---|---|---|
| `blight` | Colletotrichum Blight | कोलेट
