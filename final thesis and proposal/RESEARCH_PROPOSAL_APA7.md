<!-- ============================================================
     RESEARCH PROPOSAL — APA 7th Edition
     Master of Computer Applications (MCA)
     Kantipur City College, Purbanchal University
     Formatting note: For final submission, apply Times New Roman
     12 pt, double-spaced text, 1-inch margins, and running head
     flush left with page numbers flush right per your
     institution's template.
     ============================================================ -->

---

**Running Head:** CARDAMOM LEAF DISEASE DETECTION USING DEEP LEARNING

---

# Cardamom Leaf Disease Detection Using Deep Learning: Proposed Design, Implementation, and Evaluation of a Full-Stack Explainable Intelligent System With Bilingual Deployment for Nepali Smallholder Farmers

<br>

**Darpan Subedi**
Roll No.: 234567212G

Master of Computer Applications (MCA)
Department of Computer Applications
Kantipur City College
Affiliated to Purbanchal University

Proposed Supervisor: Saroj Khanal

Submitted in Partial Fulfilment of the Requirements for the
Approval of Research Proposal for the Degree of
Master of Computer Applications

May 2026

---

## Declaration

I, Darpan Subedi, hereby declare that this research proposal titled *Cardamom Leaf Disease Detection Using Deep Learning: Proposed Design, Implementation, and Evaluation of a Full-Stack Explainable Intelligent System With Bilingual Deployment for Nepali Smallholder Farmers* is an original document prepared by me under the guidance of my proposed supervisor, Saroj Khanal, at Kantipur City College, affiliated to Purbanchal University. All secondary sources consulted in the preparation of this proposal have been properly acknowledged following APA 7th edition citation guidelines.

**Signed:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Name:** Darpan Subedi

**Date:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

---

## Supervisor's Endorsement

This is to certify that the research proposal entitled *Cardamom Leaf Disease Detection Using Deep Learning: Proposed Design, Implementation, and Evaluation of a Full-Stack Explainable Intelligent System With Bilingual Deployment for Nepali Smallholder Farmers*, submitted by Darpan Subedi (Roll No.: 234567212G), has been reviewed and is endorsed for consideration by the Research Committee.

**Supervisor:** Saroj Khanal

**Institution:** Kantipur City College, Purbanchal University

**Date:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Problem Statement](#3-problem-statement)
4. [Research Objectives](#4-research-objectives)
5. [Research Questions](#5-research-questions)
6. [Literature Review](#6-literature-review)
7. [Methodology](#7-methodology)
8. [Expected Outcomes](#8-expected-outcomes)
9. [Research Timeline](#9-research-timeline)
10. [Ethical Considerations](#10-ethical-considerations)
11. [Budget Estimate](#11-budget-estimate)
12. [References](#12-references)

---

---

## 1. Abstract

Nepal is among the world's foremost producers of large cardamom (*Amomum subulatum* Roxb.), a crop of critical economic importance to tens of thousands of smallholder farming households concentrated in the eastern hill districts of Taplejung, Panchthar, Ilam, and Dhankuta. Two fungal foliar diseases—Colletotrichum Blight (*Colletotrichum gloeosporioides*) and Phyllosticta Leaf Spot (*Phyllosticta* spp.)—constitute the most economically damaging biotic threats to cardamom production, capable of causing yield losses exceeding 30–40% of the seasonal harvest under severe outbreak conditions (Paudel & Bhatt, 2019). Early and accurate diagnosis is the cornerstone of effective disease management; however, the remote geography of Nepal's cardamom-growing communities and the severe scarcity of agricultural extension expertise mean that timely expert diagnosis is inaccessible to the majority of affected farmers.

This research proposal presents a plan to design, implement, and rigorously evaluate a complete, end-to-end deep learning-based intelligent system for automated cardamom leaf disease detection. The proposed system will extend the foundational work of Sunil et al. (2022)—who demonstrated 98.26% classification accuracy using EfficientNetV2-L on a field-collected cardamom disease dataset—by introducing six dimensions absent from that baseline: (a) an "Other" class to enable inference-time rejection of non-cardamom inputs; (b) a comprehensive multi-protocol evaluation framework including 5-fold stratified cross-validation, background-removal ablation, error analysis, and robustness testing under multiple perturbation conditions; (c) Gradient-weighted Class Activation Mapping (Grad-CAM) and Grad-CAM++ visual explanations; (d) a heuristic five-stage disease severity estimation module; (e) full-stack deployment as a FastAPI REST backend, a React TypeScript web application, and a React Native mobile application; and (f) a comprehensive bilingual English/Nepali interface—the first for any crop disease detection system in Nepal. The system will be trained using EfficientNetV2-S with transfer learning, inverse-frequency class weighting, and stochastic data augmentation on a curated dataset of approximately 2,607 labeled images sourced primarily from the Sunil et al. (2022) Kaggle repository. Overall test accuracy above 85% is targeted, with all per-class F1-scores above 0.70.

*Keywords:* cardamom, leaf disease detection, EfficientNetV2, transfer learning, Grad-CAM++, explainable AI, FastAPI, React Native, precision agriculture, Nepal, bilingual interface, ablation study

---

## 2. Introduction

### 2.1 Background and Motivation

Agriculture is the economic and social foundation of Nepal, sustaining the livelihoods of more than 60% of the population and contributing approximately 24% of gross domestic product (Subba et al., 2021). Within this agricultural economy, spice crops play an outsize role: Nepal is the world's third-largest producer of large cardamom (*Amomum subulatum* Roxb.), with cultivation concentrated in the eastern hill districts where the crop functions as the primary source of household cash income for tens of thousands of smallholder families (Dhungana et al., 2024). True cardamom (*Elettaria cardamomum* Maton) is additionally grown in smaller quantities in Nepal's mid-hill regions.

The economic foundations of cardamom cultivation are persistently undermined by two fungal pathogens. *Colletotrichum gloeosporioides* produces anthracnose-type Blight, characterized by water-soaked irregular lesions that expand into necrotic areas with light-brown centers and dark margins. *Phyllosticta* species produce circular or elliptical Leaf Spot formations with tan or gray centers surrounded by chlorotic halos (Sunil et al., 2022). Paudel and Bhatt (2019) documented yield losses exceeding 30–40% in severe blight episodes, while Yadav and Basnet (2021) identified both diseases as primary threats to the long-term viability of cardamom cultivation in Nepal, emphasizing that delayed identification was the key factor enabling progression to economically irreversible disease stages.

The challenge of early and accurate disease identification is fundamentally one of access. Qualified plant pathologists and agricultural extension officers are severely scarce in Nepal's remote hill communities; an accurate expert diagnosis may require a journey of several hours from a remote farm (K.C. & Upreti, 2017). Furthermore, the visual similarity between early-stage Colletotrichum Blight and Phyllosticta Leaf Spot—both presenting as small, discolored lesions before distinctive morphological differentiation—means that even experienced farmers frequently cannot distinguish between them without specialist training (Pun, 2019). The consequence is delayed intervention, disease spread, and avoidable economic loss.

The convergence of deep learning for image classification, cloud-based inference services, and smartphone proliferation has created a technically credible pathway to automated, field-deployable plant disease diagnosis. Sunil et al. (2022) demonstrated that EfficientNetV2-L—a state-of-the-art efficient convolutional neural network—could classify cardamom leaf diseases at 98.26% accuracy from a field-collected image dataset, establishing the technical feasibility of the problem. However, their work remained a research artifact: no deployed tool, no visual explanation for predictions, no provision for non-cardamom inputs, no language accessibility for Nepali-speaking farmers, and no evaluation beyond a single train-test split.

This research proposes to bridge that gap by transforming the Sunil et al. (2022) research model into a complete, deployed, interpretable, rigorously evaluated, and linguistically accessible diagnostic system. The work is situated within the precision agriculture movement and the growing literature on responsible AI deployment in resource-constrained agricultural contexts (Pandeya et al., 2025; Bhattarai & Bhandari, 2020).

### 2.2 Significance of the Study

The significance of this research extends across four dimensions:

**Scientific significance**: This research will produce the first rigorously multi-protocol evaluated deep learning system for cardamom disease detection, introducing 5-fold cross-validation, background-removal ablation, robustness testing, and error analysis to a domain that has lacked systematic evaluation depth. The background-removal ablation, in particular, will empirically test an assumption made but not validated in the primary reference study (Sunil et al., 2022).

**Technical significance**: The integration of Grad-CAM++ explainability into a deployed production system—not merely as a research visualization tool—represents an engineering contribution that advances the state of practice for agricultural AI deployment. The five-stage heuristic severity module operationalizes visual attribution maps as actionable management guidance.

**Social significance**: Nepal's cardamom smallholder farming communities are among the most geographically isolated agricultural populations in South Asia. The development of a bilingual English/Nepali diagnostic tool specifically designed for their context represents a direct response to the structural accessibility barriers that have historically excluded Nepali farmers from the benefits of agricultural AI.

**Methodological significance**: The comparison between raw and background-removed image conditions, and between different perturbation types in robustness testing, will generate generalizable evidence about the conditions under which preprocessing choices help or harm classification performance—evidence applicable beyond cardamom to other crop disease detection projects operating with field-collected datasets.

---

## 3. Problem Statement

Despite Nepal's status as a leading producer of large cardamom, the farming communities responsible for this production lack access to accurate, timely, and affordable leaf disease diagnostic tools. The two principal fungal diseases—Colletotrichum Blight and Phyllosticta Leaf Spot—are visually similar in early stages, geographically remote from expert consultation, and collectively capable of destroying more than a third of the seasonal harvest if not identified and treated promptly.

The research literature has established the technical feasibility of automated cardamom leaf disease detection using convolutional neural networks (Sunil et al., 2022), but has not translated this feasibility into practical utility. Specifically, the following problems remain unaddressed:

1. **The deployment gap**: No functional web or mobile interface exists for the Sunil et al. (2022) model or any comparable cardamom disease detection system; the agricultural community has no actionable tool.

2. **The explainability gap**: Existing models produce classification outputs without visual explanations, preventing users from verifying that predictions are based on disease-relevant leaf features rather than background artifacts—a prerequisite for trust in expert-support applications.

3. **The evaluation depth gap**: The Sunil et al. (2022) study reported performance on a single train-test split without cross-validation, ablation analysis, or robustness testing, leaving uncertainty about the stability and limitations of the reported accuracy.

4. **The linguistic accessibility gap**: All published agricultural AI systems in this domain provide exclusively English interfaces, making them inaccessible to the large proportion of Nepali-speaking farmers who are the primary intended beneficiaries of these tools.

5. **The severity estimation gap**: Classification into disease categories is insufficient for practical management decisions; farmers and extension workers need guidance on disease severity (early, moderate, severe) to calibrate the urgency and intensity of their response.

This research directly addresses all five gaps through a unified system design, implementation, and evaluation program.

---

## 4. Research Objectives

### 4.1 Primary Objectives

**Objective 1**: To design and train a high-accuracy deep learning classifier using EfficientNetV2-S with ImageNet-pretrained transfer learning for four-class cardamom leaf disease classification (Colletotrichum Blight, Phyllosticta Leaf Spot, Healthy, and Other), achieving an overall test accuracy target of ≥ 85%.

**Objective 2**: To conduct a rigorous, multi-protocol empirical evaluation of the trained classifier, comprising: (a) held-out test set evaluation; (b) 5-fold stratified cross-validation; (c) a background-removal ablation study comparing rembg-preprocessed versus raw image conditions; (d) comprehensive error analysis with confidence distribution examination; and (e) robustness testing under nine image perturbation conditions simulating real-world field degradation.

**Objective 3**: To implement Gradient-weighted Class Activation Mapping (Grad-CAM) and its second-order extension Grad-CAM++ as integrated components of the production system, providing spatially coherent visual explanations that highlight disease-relevant leaf regions for each prediction.

**Objective 4**: To design, implement, and deploy a complete three-tier full-stack system: (a) a FastAPI REST backend providing AI inference as a service; (b) a React TypeScript web application for extension workers and researchers; and (c) a React Native mobile application with comprehensive bilingual English/Nepali disease management guidance for smallholder farmers.

### 4.2 Secondary Objectives

**Objective 5**: To implement a heuristic five-stage disease severity estimation module that derives approximate lesion extent from Grad-CAM activation maps, providing clinically interpretable staging (mild, moderate, severe, very severe) to guide management urgency.

**Objective 6**: To document a fully reproducible training and evaluation pipeline with fixed random seeds, explicit hyperparameter configuration, and open-source implementation hosted in a version-controlled repository.

**Objective 7**: To critically compare the system's design choices, evaluation results, and limitations against the Sunil et al. (2022) baseline and the broader plant disease detection literature, identifying the specific contributions and the conditions under which the findings generalize.

---

## 5. Research Questions

The following research questions guide the investigation:

**RQ1**: Can EfficientNetV2-S, fine-tuned with transfer learning on a field-collected cardamom leaf disease dataset with inverse-frequency class weighting and stochastic augmentation, achieve overall test accuracy exceeding 85% and per-class F1-scores exceeding 0.70 on a four-class classification task including an "Other" rejection class?

**RQ2**: Does the model's performance generalize consistently across different data partitioning strategies, as measured by mean accuracy and standard deviation in 5-fold stratified cross-validation?

**RQ3**: Does background removal using rembg (U²-Net ONNX implementation) improve or degrade classification accuracy relative to the raw image baseline, and which disease classes are most affected by this preprocessing decision?

**RQ4**: Which types of image perturbation—blur, brightness reduction, additive noise, rotation, and partial cropping—cause the greatest degradation in model performance, and which are effectively mitigated by the data augmentation policy?

**RQ5**: Do Grad-CAM++ activation maps for disease-classified images concentrate on visually identifiable disease lesion regions rather than on background artifacts or non-diagnostic leaf areas?

**RQ6**: Can a bilingual English/Nepali full-stack system—including disease diagnosis, visual explanation, severity staging, and agronomic management guidance—be designed, implemented, and containerized for reproducible deployment?

---

## 6. Literature Review

### 6.1 Deep Learning for Plant Disease Detection

The systematic application of deep learning to plant disease detection was catalyzed by Mohanty et al. (2016), who demonstrated that a GoogLeNet CNN fine-tuned with transfer learning achieved 99.35% accuracy on the PlantVillage benchmark—a controlled laboratory dataset of 54,306 labeled images across 26 disease classes and 14 crop species. While this result established technical feasibility, subsequent research exposed a critical limitation: the benchmark's controlled capture conditions (uniform backgrounds, standardized lighting) meant that models trained on PlantVillage exhibited substantially degraded performance when evaluated on field-collected images (Barbedo, 2018).

Barbedo (2018) systematically catalogued the factors responsible for this laboratory-to-field performance gap: background heterogeneity (soil, stems, multiple leaves), illumination variation (morning mist, direct sunlight, shadow), image quality variation (smartphone cameras of varying quality), and intra-class visual variation (disease appearing at different developmental stages). These factors are directly relevant to cardamom disease detection in Nepal's hill plantation environments, where capture conditions are far from controlled.

Thapa et al. (2020) directly measured the laboratory-to-field gap by constructing PlantDoc, a 2,598-image dataset collected in real field conditions across 13 species and 17 diseases. Models consistently achieved substantially lower accuracy on PlantDoc than on PlantVillage, empirically confirming Barbedo's predictions. The lesson for the present research is that field-collected training data—as provided by Sunil et al. (2022)—is essential for building a system with realistic field performance.

Ferentinos (2018) performed a systematic architectural comparison on PlantVillage, confirming that transfer learning with ImageNet pretraining outperformed training from scratch across all architectures benchmarked. This result provides theoretical grounding for the transfer-learning approach that this research will employ.

### 6.2 EfficientNet and EfficientNetV2

EfficientNet (Tan & Le, 2019) introduced compound scaling, a principled methodology for simultaneously adjusting network width, depth, and input resolution through a fixed scaling coefficient. The key theoretical contribution was the observation that these three dimensions are not independent: scaling one without the others produces sub-optimal accuracy per parameter. Compound scaling yields architectures that consistently outperform contemporaries (ResNet, Inception, VGG, DenseNet) at matched computational budgets.

EfficientNetV2 (Tan & Le, 2021) extended this with Fused-MBConv operations in early network stages—merging the depthwise convolution and pointwise expansion into a single regular convolution for higher hardware throughput—and adaptive progressive training, which gradually increases input resolution and augmentation strength during training. These changes yielded faster training and stronger accuracy. EfficientNetV2-S, the smallest variant (21.5M parameters, 8.8 GFLOPs), provides the best inference efficiency for deployment in a CPU-served API context and is therefore selected as the classification backbone for this research.

Sunil et al. (2022) applied EfficientNetV2-L to cardamom disease classification on a 1,724-image field-collected dataset across three disease classes, reporting 98.26% accuracy with U²-Net background removal as a preprocessing stage. This study is the direct baseline for the present research. The selection of EfficientNetV2-S over EfficientNetV2-L is motivated by the deployment context: V2-L's 118M parameters impose inference overhead incompatible with CPU-based API serving at reasonable latency, whereas V2-S's 21.5M parameters provide acceptable latency without GPU acceleration.

Atila et al. (2021) systematically benchmarked EfficientNet variants on PlantVillage, demonstrating consistent superiority over ResNet-50 and InceptionV3, establishing the EfficientNet family as the state of the art for plant disease image classification tasks.

### 6.3 Transfer Learning for Agricultural Image Classification

The theoretical and empirical case for ImageNet-pretrained transfer learning in agricultural image classification is well-established (Kaya et al., 2019). For the small to medium dataset sizes encountered in crop-specific studies (1,000–5,000 images), the optimization landscape is sufficiently complex that training from random initialization is prone to convergence to poor local minima. ImageNet-pretrained weights provide feature detectors that capture low-level image primitives (edges, colors, gradients) and mid-level texture representations that transfer broadly across vision domains. These representations substantially reduce the effective optimization distance to a high-performing solution for the target domain.

Full end-to-end fine-tuning—updating all backbone weights jointly with the custom classification head—has been shown to outperform frozen-feature approaches for domain-specific applications where the target domain's texture characteristics differ from ImageNet (Atila et al., 2021). Disease lesion textures are visually quite different from ImageNet object categories, making full fine-tuning the appropriate strategy. This research will employ full end-to-end fine-tuning throughout.

### 6.4 Explainable AI: Grad-CAM and Grad-CAM++

Selvaraju et al. (2017) introduced Grad-CAM, a technique that produces class-discriminative localization maps by computing the gradient of a predicted class score with respect to the activations of a target convolutional layer and forming a weighted sum of those activation maps:

$$L^c = \text{ReLU}\!\left(\sum_k \alpha^c_k A^k\right), \quad \alpha^c_k = \frac{1}{Z}\sum_i \sum_j \frac{\partial y^c}{\partial A^k_{ij}}$$

The ReLU operation discards features negatively correlated with the predicted class, producing a heatmap that highlights the spatial regions most associated with the prediction. In plant disease research, Grad-CAM has been used to verify that classifiers attend to lesion regions rather than background artifacts, lending biological credibility to automated diagnoses (Islam et al., 2021).

Chattopadhay et al. (2018) proposed Grad-CAM++, which uses second-order Taylor-series coefficients to weight each pixel's gradient contribution more precisely, producing sharper activation maps particularly valuable for spatially distributed features. This is especially relevant for disease lesion visualization, where a single leaf may contain multiple discrete lesion sites. Both Grad-CAM and Grad-CAM++ will be implemented in this research.

The critical importance of explainability for agricultural AI deployment goes beyond technical curiosity: a farmer or extension worker who cannot see why a prediction was made has no basis for calibrating trust in it. A system that achieves 99% accuracy but provides no explanation may be less useful in practice than a 95%-accurate system that highlights the lesion supporting its prediction, because the former cannot be verified and therefore may not be adopted.

### 6.5 Background Removal

Background removal via deep segmentation has been proposed as a preprocessing step to isolate leaf surfaces from confounding background elements in plant disease classification. Qin et al. (2020) introduced U²-Net, a nested U-shaped architecture that achieves state-of-the-art salient object detection by stacking multiple U-Net-like blocks within a larger U-Net structure, enabling multi-scale feature extraction without the memory overhead of feature concatenation across scales.

Sunil et al. (2022) incorporated U²-Net background removal in their best-performing pipeline and reported that it contributed positively to classification accuracy. The `rembg` library provides an accessible ONNX runtime implementation of U²-Net suitable for batch preprocessing. However, the effect of background removal is theoretically ambiguous: if background texture co-varies with disease class in the training data, removing the background removes a legitimate discriminative signal. An ablation study is therefore proposed as a core evaluation component of this research, to empirically test whether background removal helps or harms classification on this specific dataset.

### 6.6 Disease Severity Estimation

Clinical disease management depends not only on accurate categorical diagnosis but on quantification of disease extent. The decision to apply fungicide treatment, and at what concentration, depends on whether an infection is in an early, moderate, or severe stage (Barbedo, 2017). Accurate pixel-level severity measurement requires semantic segmentation of lesion regions—a task requiring pixel-annotated training data that substantially exceeds the data collection burden of image-level classification.

A heuristic approach that approximates severity from Grad-CAM activation maps—binarizing the normalized heatmap at a threshold and computing the proportion of activated pixels—can provide a practically useful approximation without requiring pixel annotations. The five-stage severity scale proposed in this research (healthy, mild, moderate, severe, very severe) corresponds to clinically meaningful intervention thresholds derived from the agronomic literature on fungal foliar disease management (Barbedo, 2017; Paudel & Bhatt, 2019). Severity estimates will be explicitly labeled as heuristic approximations.

### 6.7 Mobile and Web Deployment in Agricultural AI

Ramcharan et al. (2017) demonstrated the practical feasibility of mobile-deployed CNN classifiers for field crop diagnostics through their Inception v3-based cassava disease detection system. The client-server architecture proposed in this research—where inference is performed by a centralized FastAPI backend rather than on-device—trades connectivity independence for the ability to deploy a full-scale EfficientNetV2-S model without the size and accuracy compromises imposed by mobile compression.

Howard et al. (2017) introduced MobileNets, architectures specifically designed for on-device inference, which could be considered for a future on-device deployment iteration. For this research, the priority is to establish a functional, accurate, deployed system; on-device conversion is identified as a future work direction.

Multilingual interface design is a critical and systematically neglected dimension of agricultural AI accessibility. All reviewed plant disease detection systems—including Sunil et al. (2022)—provide exclusively English interfaces. Given that the primary intended users of this system are Nepali-speaking smallholder farmers, a comprehensive Nepali-language interface—including disease names, symptom descriptions, treatment recommendations, prevention measures, and action timelines—is proposed as a core deliverable rather than an optional enhancement.

### 6.8 Cardamom Cultivation and Disease in Nepal

Nepal's cardamom industry, concentrated in the eastern hill districts, faces compound threats from climate change, disease pressure, and market volatility (K.C. & Upreti, 2017). K.C. and Upreti's (2017) political economy analysis documented that disease episodes caused measurable income losses among smallholder households and that institutional response was slow due to remote geography and limited extension capacity—identifying the absence of accessible diagnostic tools as a structural gap amplifying harm.

Pun (2019) reviewed the biotic and abiotic factors contributing to cardamom production decline in Nepal, identifying Colletotrichum Blight and Phyllosticta Leaf Spot among the primary threats and noting that integrated disease management was under-practiced specifically because of the difficulty of early identification. Dhungana et al. (2024) analyzed production trend instability in Nepalese large cardamom, identifying capacity-building interventions—including digital tools—as necessary for stabilizing farm-level yields. Pandeya et al. (2025) found that education, extension service access, and farmer group membership were the strongest predictors of improved practice adoption among Nepali cardamom farmers, suggesting that accessible digital advisory tools could functionally substitute for scarce formal extension services.

### 6.9 Research Gap

The reviewed literature establishes the following composite gap:

1. Deep learning, and specifically EfficientNetV2-based transfer learning, achieves strong performance on cardamom leaf disease classification from field-collected images (Sunil et al., 2022).
2. No deployed, accessible diagnostic tool exists for cardamom farmers or extension workers.
3. The existing evaluation evidence is limited to a single train-test split; cross-validation, ablation, and robustness evidence are absent.
4. No cardamom (or broader crop) disease detection system provides a Nepali-language interface.
5. The role of background removal as a preprocessing step has not been empirically validated against a raw-image baseline on this dataset.

This research directly and comprehensively addresses all five gaps.

---

## 7. Methodology

### 7.1 Research Design

This research adopts an applied engineering research design that integrates quantitative empirical methods (model training, cross-validation, ablation analysis, robustness testing) with system design, software engineering, and user-centred design principles. The methodology proceeds through five sequential phases, described below. Reproducibility is operationalized through a fixed random seed (`RANDOM_SEED = 42`), fully documented hyperparameter configuration, and a version-controlled open-source implementation.

### 7.2 Dataset

#### 7.2.1 Primary Dataset

The primary dataset will be obtained from the Kaggle repository associated with Sunil et al. (2022), which provides 1,724 field-collected and labeled cardamom leaf images across three disease categories:

- **Colletotrichum Blight** (*Colletotrichum gloeosporioides*): Anthracnose-type fungal lesions with irregular necrotic regions, light-brown centers, and dark margins.
- **Phyllosticta Leaf Spot** (*Phyllosticta* spp.): Circular or elliptical fungal leaf spots with tan or gray centers, frequently surrounded by chlorotic halos.
- **Healthy**: Symptom-free cardamom leaves.

Images were collected from cardamom plantations under natural field conditions, with variable illumination, orientation, background, and camera distance. The Indian Cardamom Research Institute (ICRI) provided agronomic validation of the disease category definitions used in labeling.

#### 7.2.2 "Other" Class

A fourth **"Other"** class will be constructed by sourcing images of non-cardamom plant leaves and general agricultural scenes from publicly available Kaggle repositories (principally PlantVillage-derived collections). Selection criteria will require visual distinctness from cardamom leaf morphology. Data cleaning will involve: (a) removal of duplicates and near-duplicates; (b) exclusion of images below 100 × 100 pixel resolution or with severe compression artefacts; and (c) manual screening to eliminate ambiguous images that could be confused with cardamom leaves.

**Rationale**: Without a rejection class, a deployed system would classify any submitted image as one of the three disease categories regardless of whether the input is actually a cardamom leaf, producing unreliable predictions for off-target inputs. The "Other" class enables principled handling of non-cardamom submissions.

**Limitations of dataset combination**: Combining the Sunil et al. (2022) images (collected in India under field conditions) with Kaggle-sourced "Other" class images introduces three potential biases: (a) acquisition domain mismatch due to different camera characteristics and collection protocols; (b) difficulty calibration mismatch if "Other" images are easier to distinguish from cardamom leaves than genuine field boundary cases; and (c) class size imbalance if "Other" images dominate the training distribution. These limitations will be acknowledged explicitly in the thesis and addressed through careful class sizing and inverse-frequency weighting.

#### 7.2.3 Planned Dataset Statistics

| Class | Source | Approx. Total | Train (70%) | Val (15%) | Test (15%) |
|---|---|---|---|---|---|
| Colletotrichum Blight | Sunil et al. (2022) / Kaggle | 280 | 196 | 42 | 42 |
| Phyllosticta Leaf Spot | Sunil et al. (2022) / Kaggle | 663 | 464 | 100 | 99 |
| Healthy | Sunil et al. (2022) / Kaggle | 781 | 547 | 117 | 117 |
| Other | Additional Kaggle sources | ~883 | ~618 | ~133 | ~132 |
| **Total** | | **~2,607** | **~1,825** | **~392** | **~390** |

*Note.* All splits will use stratified random sampling with `RANDOM_SEED = 42` to preserve class distribution and ensure reproducibility.

#### 7.2.4 Class Imbalance Handling

The dataset exhibits class imbalance: Colletotrichum Blight represents approximately 16.2% of the primary three-class total, substantially underrepresented relative to Healthy (45.3%). Inverse-frequency class weighting will be applied to the cross-entropy loss:

$$w_c = \frac{N}{K \cdot n_c}$$

where *N* is total training samples, *K* = 4 (classes), and $n_c$ is training samples for class *c*. This formulation assigns higher loss weight to underrepresented classes, penalizing minority-class misclassifications proportionally to underrepresentation.

### 7.3 Preprocessing and Data Augmentation

All images will be resized to **224 × 224 pixels** (bilinear interpolation) and normalized channel-wise using ImageNet statistics (mean: [0.485, 0.456, 0.406]; std: [0.229, 0.224, 0.225]).

The following stochastic augmentations will be applied **to training images only**:

| Transform | Parameters | Rationale |
|---|---|---|
| Random Horizontal Flip | p = 0.5 | Leaf bilateral symmetry; orientation invariance |
| Random Vertical Flip | p = 0.5 | Variable capture angle in field conditions |
| Random Rotation | ±30° | Leaf orientation variation during image capture |
| Color Jitter | brightness=0.2, contrast=0.2, saturation=0.2 | Variable illumination (morning, midday, evening) |
| Random Affine (translation) | translate=(0.1, 0.1) | Partial framing, zoom distance variation |

Validation and test sets will receive only resizing and normalization, ensuring unbiased performance estimation.

### 7.4 Model Architecture

The classification backbone will be **EfficientNetV2-S** (Tan & Le, 2021), loaded with ImageNet-pretrained weights from the `torchvision` model zoo. The original 1,000-class ImageNet classification head will be replaced with a custom two-layer fully connected head:

```
Dropout(p=0.30) → Linear(1280 → 512) → ReLU → Dropout(p=0.20) → Linear(512 → 4)
```

All backbone weights will be fine-tuned jointly with the custom head through end-to-end gradient descent, enabling the feature extractor to adapt its representations to the texture and color signatures of cardamom diseases. The two-stage Dropout regularization (p=0.30, p=0.20) will prevent overfitting in the classification head.

**Justification for EfficientNetV2-S**: The deployment context requires a model that can serve inference requests at acceptable latency on CPU hardware (< 1 second per image). EfficientNetV2-S's 21.5M parameters and 8.8 GFLOPs are consistent with this requirement; EfficientNetV2-L's 118M parameters impose prohibitive CPU inference overhead. The accuracy trade-off between V2-S and V2-L is modest at this scale of fine-tuning task, and is justified by the deployment requirement.

### 7.5 Training Configuration

| Hyperparameter | Planned Value | Justification |
|---|---|---|
| Input image size | 224 × 224 px | EfficientNetV2-S specification |
| Batch size | 32 | Standard for fine-tuning; memory-efficient |
| Maximum epochs | 50 | Sufficient with early stopping |
| Optimizer | Adam | Adaptive gradient; effective for fine-tuning |
| Initial learning rate | 0.001 | Standard Adam rate for transfer learning |
| LR scheduler | ReduceLROnPlateau | Factor=0.5; patience=5; monitors val loss |
| Early stopping patience | 10 epochs | Saves best checkpoint at min val loss |
| Loss function | Weighted CrossEntropyLoss | Addresses class imbalance |
| Label smoothing | 0.1 | Regularization; prevents overconfidence |
| Dropout | 0.30 / 0.20 (two-stage) | Classifier head regularization |
| Random seed | 42 | Full reproducibility |

### 7.6 Evaluation Protocols

#### 7.6.1 Held-Out Test Evaluation

The held-out test partition (approximately 15% of the dataset) will be reserved exclusively for final evaluation. Metrics: overall accuracy, per-class precision, recall, F1-score, and confusion matrix.

#### 7.6.2 5-Fold Stratified Cross-Validation

Five-fold cross-validation with stratified partitioning will provide a statistically more reliable estimate of generalization than a single split. Each fold will train from scratch using identical configuration for 30 epochs (early stopping patience = 7). Mean ± standard deviation will be reported for accuracy, macro precision, macro recall, and macro F1.

#### 7.6.3 Background-Removal Ablation Study

Two conditions will be evaluated on the same held-out test set using the same trained model checkpoint:
- **Condition A (Raw)**: Test images evaluated without preprocessing beyond resizing and normalization.
- **Condition B (BG-Removed)**: Test images preprocessed in-memory using `rembg` (U²-Net ONNX) before evaluation.

Full per-class metrics and their differences (Δ) will be reported.

#### 7.6.4 Error Analysis

Executed post-evaluation, this will produce: total and per-class misclassification counts; mean and median confidence scores for correct and incorrect predictions; dominant confusion pairs; and a CSV catalogue of all misclassified images with their predicted labels and confidence scores.

#### 7.6.5 Robustness Testing

Nine perturbation conditions will simulate common image degradation in field conditions:

| Condition | Parameters | Simulated Real-World Cause |
|---|---|---|
| Baseline (clean) | — | Reference |
| Gaussian Blur | radius = 3 | Camera shake, soft focus |
| Gaussian Blur | radius = 5 | Severe camera shake, moisture on lens |
| Low Brightness | ×0.4 | Evening light, shade |
| Low Brightness | ×0.2 | Extreme underexposure |
| Gaussian Noise | σ = 0.10 | Low-end smartphone sensor noise |
| Gaussian Noise | σ = 0.20 | Severely degraded camera sensor |
| Rotation | ±45° | Extreme capture angle |
| Center Crop | 75% | Partial leaf framing |

Accuracy and macro F1 under each condition, and their deltas from the clean baseline, will be reported.

### 7.7 Grad-CAM and Grad-CAM++ Implementation

Both standard Grad-CAM (Selvaraju et al., 2017) and Grad-CAM++ (Chattopadhay et al., 2018) will be implemented using PyTorch forward and backward hooks registered on the final convolutional block of EfficientNetV2-S (`model.features[-1]`). Hooks will be removed after each heatmap generation to prevent memory leaks.

Standard Grad-CAM will compute global average-pooled gradients $\alpha^c_k = \frac{1}{Z}\sum_{i,j}\frac{\partial y^c}{\partial A^k_{ij}}$ and form the heatmap as $L^c = \text{ReLU}(\sum_k \alpha^c_k A^k)$.

Grad-CAM++ will apply second-order pixel-level weighting to improve localization precision. Heatmaps will be bilinearly upsampled to the original image dimensions, mapped through a Jet colormap, and composited with the input image at 50% opacity for display.

### 7.8 Severity Estimation Module

A heuristic severity module will binarize the normalized Grad-CAM heatmap at a configurable threshold (default: 0.60) and compute the proportion of activated pixels as the severity percentage. This will be mapped to a five-stage clinical scale:

| Stage | Affected Area (%) | Clinical Interpretation |
|---|---|---|
| 0 | 0 | Healthy — no lesion activity |
| 1 | 1–10 | Mild — early infection, monitoring recommended |
| 2 | 11–25 | Moderate — treatment intervention recommended |
| 3 | 26–50 | Severe — urgent management required |
| 4 | > 50 | Very Severe — critical crop damage |

All severity outputs will include an explicit warning that the estimate is heuristic.

### 7.9 System Architecture and Deployment

**Three-tier architecture**:

1. **Model Tier**: EfficientNetV2-S PyTorch model; Grad-CAM/++ engine; rembg background removal (optional); severity estimator.
2. **API Tier**: FastAPI REST backend; Uvicorn ASGI server; Pydantic input/output validation.
3. **Client Tier**: React TypeScript web application (Vite build toolchain); React Native mobile application (Expo managed workflow).

**Technology stack**:

| Component | Technology |
|---|---|
| Deep learning | PyTorch 2.x |
| Model backbone | EfficientNetV2-S (torchvision) |
| Backend framework | FastAPI |
| API validation | Pydantic |
| Web frontend | React 19 + TypeScript + Vite |
| Mobile | React Native + Expo |
| Mobile navigation | React Navigation |
| Containerization | Docker + docker-compose |

**Bilingual content**: All user-facing text, disease names, symptom descriptions, treatment recommendations, prevention guidelines, and action timelines will be provided in both English and Nepali (नेपाली) across both the web and mobile applications. A comprehensive Nepali-language disease information database will be implemented as a TypeScript module enabling offline access to management content.

### 7.10 Tools and Technologies Summary

| Purpose | Tool/Library |
|---|---|
| Model training and inference | Python 3.11, PyTorch 2.x, torchvision |
| Image processing | Pillow, OpenCV |
| Background removal | rembg (U²-Net ONNX) |
| Data manipulation | NumPy, scikit-learn |
| REST API | FastAPI, Uvicorn, Pydantic |
| Web frontend | React 19, TypeScript, Vite, Axios |
| Mobile | React Native, Expo, React Navigation |
| Containerization | Docker, docker-compose |
| Experiment tracking | JSON result files; version-controlled repository |
| Version control | Git, GitHub |

---

## 8. Expected Outcomes

### 8.1 Primary Deliverables

**Deliverable 1 — Trained EfficientNetV2-S model**: A serialized model checkpoint (`cardamom_model.pt`) achieving ≥ 85% overall test accuracy and per-class F1 ≥ 0.70 across all four classes, with documented training history and evaluation metrics.

**Deliverable 2 — Evaluation evidence corpus**: A complete set of evaluation outputs including:
- Held-out test confusion matrix and per-class metrics
- 5-fold cross-validation results (`cv_results.json`) with mean ± standard deviation statistics
- Background-removal ablation results (`ablation_results.json`) with per-class Δ metrics
- Robustness test results (`robustness_results.json`) under nine perturbation conditions
- Error analysis outputs (`error_analysis.json`, `misclassified.csv`) including confidence distribution

**Deliverable 3 — FastAPI backend**: A production-style REST API serving disease classification, Grad-CAM/++ visual explanation, and five-stage severity estimation, containerized for reproducible deployment.

**Deliverable 4 — React TypeScript web application**: A functional, accessible web interface for disease detection, heatmap visualization, and disease management guidance, suitable for extension workers and researchers.

**Deliverable 5 — React Native bilingual mobile application**: A camera-first, bilingual English/Nepali mobile application for iOS and Android, including disease prediction, Grad-CAM overlay, severity indicator, and comprehensive Nepali-language management guidance content.

**Deliverable 6 — Research thesis**: A complete, APA 7th edition-formatted thesis documenting the research rationale, methodology, results, analysis, and conclusions, submitted to Kantipur City College in fulfilment of the MCA degree requirements.

### 8.2 Expected Research Findings

Based on the established literature and the characteristics of the proposed methodology, the following outcomes are expected:

- **Classification accuracy**: EfficientNetV2-S with ImageNet-pretrained transfer learning is expected to achieve overall test accuracy substantially exceeding the 85% acceptance criterion, consistent with the 98.26% reported by Sunil et al. (2022) using the larger V2-L variant on the same primary dataset.

- **Cross-validation stability**: Low standard deviation across folds is expected, reflecting the discriminative clarity of the four proposed classes and the stability of the transfer-learned representations.

- **Background-removal effect**: The ablation study is expected to produce a non-trivial Δ in either direction, providing empirical evidence about whether background context carries correlated discriminative information in this dataset.

- **Robustness profile**: Strong robustness to brightness and geometric perturbations (mitigated by the augmentation policy) is expected, with greater sensitivity to blur and additive noise (not fully mitigated by the augmentation policy).

- **Grad-CAM coherence**: Activation maps for disease predictions are expected to concentrate on lesion regions, as evidenced by qualitative alignment with disease morphology documented in the agronomic literature.

### 8.3 Academic Contributions

This research will contribute to the academic literature in the following specific ways:

1. **First rigorous multi-protocol evaluation of cardamom disease classification**: Establishing a methodological template for evaluation depth in crop-specific disease detection research.

2. **Empirical evidence on background removal effects**: The ablation study will contribute generalizable evidence about the conditions under which background removal helps or harms classification performance in field-collected datasets.

3. **First bilingual English/Nepali deployment for agricultural AI**: Setting a precedent for language-accessible design in agricultural AI systems for South Asian contexts.

4. **Practical robustness characterization**: The nine-condition robustness test will generate actionable evidence about which image quality factors most threaten the practical reliability of this class of system.

---

## 9. Research Timeline

The research is planned across **12 months**, organized into five phases:

| Phase | Activities | Duration | Months |
|---|---|---|---|
| **Phase 1: Preparation and Review** | Comprehensive literature review; gap analysis; dataset acquisition (Sunil et al., 2022 Kaggle repository + "Other" class from Kaggle); ICRI agronomic consultation; dataset cleaning, labeling verification, and 70/15/15 stratified split implementation | 2 months | 1–2 |
| **Phase 2: Model Development** | EfficientNetV2-S architecture configuration and custom head implementation; augmentation pipeline development; inverse-frequency class weighting; training loop implementation with ReduceLROnPlateau and early stopping; 50-epoch training on MPS/CUDA hardware; checkpoint management | 2 months | 3–4 |
| **Phase 3: Evaluation** | Held-out test evaluation; 5-fold stratified cross-validation (5 × 30-epoch training runs); background-removal ablation study; error analysis and confidence distribution examination; nine-condition robustness testing; Grad-CAM++ coherence evaluation | 2 months | 5–6 |
| **Phase 4: System Development and Deployment** | FastAPI backend implementation and containerization; React TypeScript web application development; React Native mobile application development; Nepali-language disease information database compilation; bilingual UI implementation; end-to-end integration testing; docker-compose deployment validation | 4 months | 7–10 |
| **Phase 5: Writing and Submission** | Thesis writing (all chapters); supervisor review and revision cycles; final formatting per APA 7th edition; institutional submission procedures | 2 months | 11–12 |

**Milestone summary**:

| Milestone | Target Month |
|---|---|
| Dataset preparation and splits complete | End of Month 2 |
| EfficientNetV2-S training complete; model checkpoint saved | End of Month 4 |
| All evaluation protocols complete; results exported | End of Month 6 |
| FastAPI backend deployed and tested | End of Month 7 |
| Web application feature-complete | End of Month 8 |
| Mobile application feature-complete (bilingual) | End of Month 10 |
| Thesis draft submitted to supervisor | End of Month 11 |
| Final thesis submitted to institution | End of Month 12 |

---

## 10. Ethical Considerations

### 10.1 Data Usage and Attribution

All training data will be sourced from publicly accessible Kaggle repositories under their stated licenses. The primary dataset (Sunil et al., 2022) will be used in strict accordance with the terms of its Kaggle publication. All data sources will be fully cited in the thesis following APA 7th edition guidelines. No private, proprietary, or personally identifiable data will be collected or used.

### 10.2 Agronomic Responsibility

The disease management recommendations integrated into the mobile application, particularly those involving fungicide dosages and application timing, carry an agronomic responsibility extending beyond technical performance metrics. All management content will be (a) validated against published agronomic guidance from ICRI and the Nepalese Department of Agriculture; (b) explicitly presented as informational guidance rather than clinical prescription; and (c) accompanied by a clear recommendation to consult a qualified agricultural extension officer before undertaking chemical treatment.

The severity estimation module will be clearly labeled as a heuristic approximation, with explicit warnings that the estimate may not accurately reflect true lesion area and should not be used as the sole basis for treatment decisions.

### 10.3 Transparency and Reproducibility

The full implementation—training scripts, evaluation scripts, model architecture, hyperparameter configuration, and deployment code—will be made publicly available as an open-source GitHub repository. All random seeds and data partitioning procedures will be documented to enable full reproducibility of the reported results.

### 10.4 Responsible Deployment

The system is intended as a decision-support tool, not a diagnostic replacement for qualified agronomic expertise. All user interfaces will include clear statements to this effect. The uncertainty-flagging mechanism (flagging predictions below a configurable confidence threshold) will be implemented as a safeguard against acting on low-confidence predictions.

---

## 11. Budget Estimate

| Item | Description | Estimated Cost (NPR) |
|---|---|---|
| Compute resources | Cloud GPU time for 5-fold CV training runs (if MPS hardware insufficient) | 5,000–10,000 |
| Kaggle dataset access | Free (public Kaggle datasets) | 0 |
| Software licenses | All software open-source (Python, PyTorch, FastAPI, React, Expo) | 0 |
| Internet and connectivity | Stable broadband for development, cloud access, dataset download | 3,000–5,000 |
| Mobile testing device | Android test device (if not already available) | 10,000–15,000 |
| Printing and binding | Thesis printing and institutional submission | 2,000–3,000 |
| Miscellaneous | Stationery, cloud storage, backup | 2,000–3,000 |
| **Total** | | **~22,000–36,000** |

*Note.* The primary compute requirement (EfficientNetV2-S training on Apple MPS hardware) incurs no additional cost. The cross-validation training runs (5 × 30 epochs) and ablation/robustness evaluations are computationally modest and can be completed on the same local hardware. Cloud GPU time is budgeted as a contingency.

---

## 12. References

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

*This research proposal was prepared in accordance with APA 7th edition formatting and citation guidelines (American Psychological Association, 2020). Estimated word count (excluding tables and references): approximately 5,800 words.*

*End of Research Proposal*