# Predictive Modeling for Multi-Hospital Cardiac Diagnostics

This repository contains the complete machine learning decision-support architecture and data engineering pipeline developed for the *Applied Machine Learning (Basic)* course. The architecture transitions an integrated multi-centric clinical database into a stable diagnostic portfolio designed to safely flag coronary artery disease while accounting for real-world administrative recording artifacts and geographic clinical imbalances.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
![Area](https://img.shields.io/badge/Field-Bioinformatics-red.svg)
---

## 👥 Course Metadata

**Student Name**: Mirza Muhammad Hasan Ali 

**Course**: Applied Machine Learning (Basic)

**Instructors**: Prof. Daniele Bonacorsi & Dr. Luca Clissa 

---

## 🚀 1. Getting Started & Reproducibility

This section outlines the environment configurations, setup protocols, and dataset staging pipelines mandatory to ensure exact, end-to-end mathematical replication of the clinical audits.

### 📋 Environment Workspace Setup

Execute repository cloning and initialize local path positioning via terminal execution:

```bash
git clone https://github.com/hasanmirza72/Heart-Disease-Prediction-AML-Basic.git
cd Heart-Disease-Prediction-AML-Basic

```

### 📦 Dependency Manifest Installation

Deploy the certified external libraries using the unified package manager to prevent local environmental runtime mismatches:

```bash
pip install -r requirements.txt

```

### 📊 Dataset Staging Constraints

To preserve a lightweight codebase repository structure, raw medical text records are decoupled from core source control parameters:
* **Primary Source**

Download the raw source archive from the official [UCI Heart Disease Dataset Repository](https://archive.ics.uci.edu/dataset/45/heart+disease).
* **Local Workspace Allocation**

Extract the target compressed folder to locate the individual multi-hospital files (such as processed.cleveland.data, processed.hungarian.data, processed.switzerland.data, and processed.va.data). Compile these target sub-cohort files into a unified dataset matrix named heart_disease_raw.csv and save it directly inside a directory folder named `Data/` configured within your root repository pathing layout.

---

## 🏗️ 2. Project Layout & Structure

The repository architecture is strictly modularized to cleanly separate raw data streams, clinical preprocessing engines, grid-search hyperparameter tuning loops, reporting visual assets, and formal scholarly records:

```text
Predictive-Modeling-Multi-Hospital-Cardiac-Diagnostics/
├── requirements.txt            # Unified library constraints ensuring project reproducibility
├── README.md                   # Core user onboarding documentation and metric scorecards
│
├── Data/                       # Raw and Engineered Dataset Tracking Repository
│   ├── processed.cleveland.data   # Raw Cleveland sub-cohort matrix containing '?' placeholders
│   ├── processed.hungarian.data   # Raw Hungarian sub-cohort matrix containing '?' placeholders
│   ├── processed.switzerland.data # Raw Zurich sub-cohort matrix containing '?' placeholders
│   ├── processed.va.data          # Raw Long Beach sub-cohort matrix containing '?' placeholders
│   ├── heart_disease_raw.csv      # Combined multi-hospital source validation master dataset
│   └── heart_disease_cleaned.csv  # Final engineered feature space post KNN imputation and normalization
│
├── Scripts/                    # Modular Execution & Model Evaluation Pipelines
│   ├── Phase-1 Dataset Construction.py # Data integration script combining records across the 4 clinical sites
│   ├── Phase-2 Pre processing Dataset.py # Implements lambda label binning and biomedical KNN multivariate imputation
│   ├── Phase-3 Exploratory Data Analysis.py # Generates continuous KDE charts and geographic prevalence metrics
│   ├── Phase-4 Model Training & Evaluation.py # Evaluates unoptimized baseline frameworks over an 80/20 data split
│   ├── Phase-5 Global Optimization & Robustness Validation.py #Executes parallel grid search optimization across regularization strengths, distance geometries, and tree constraints via 5-fold cross-validation
│   └── Phase-6 Presentation & Clinical Audit.py # Compiles production matrices and final model scorecards.
|
├── Visuals/                    # Production-Grade Performance Charts & Clinical Graphics
│   ├── figure1_class_prevalence.png  # Target label bar chart validating global baseline balance (411 vs 509)
│   ├── figure2_hospital_comparison.png # Multi-centric composition profile tracking regional intake skews
│   ├── figure3_lipid_anomalies.png     # Continuous KDE curves isolating the non-physiological 0 mg/dl artifact
│   ├── figure4_correlation_matrix.png  # 14x14 product-moment correlation matrix mapping feature dependencies
│   ├── figure5_baseline_discovery.png  # Metric trends for unoptimized models evaluated over an 80/20 data split
│   ├── figure6_robust_audit.png        # Performance bar chart displaying optimized 5-fold cross-validation trends
│   ├── figure7_confusion_matrices.png  # 5-Model multi-grid matrix gallery tracking true/false error quadrants
│   └── figure8_feature_importance.png  # Horizontal bar plot charting global Random Forest Gini impurity cuts
│
└── Report/                     # Formal Scholarly Documentation
    └── Applied_Machine_Learning_Heart_Disease_Triage_Analysis_Report.docx # Complete project report

```

## 📝 3. Abstract & Problem Statement

Cardiovascular diseases remain the leading etiology of global human mortality, creating a profound diagnostic challenge for international triage infrastructure. While conventional clinical interpretation relies heavily on subjective practitioner experience, applied machine learning provides an objective secondary validation pathway by isolating complex, non-linear feature interactions hidden within multi-variate physiological records.

This research project evaluates the predictive utility of five distinct classification frameworks initialized over an integrated clinical dataset spanning four international healthcare facilities. Moving beyond naive, accuracy-centric evaluation models which generate massive blind spots when handling asymmetric data densities, this study establishes a safety-critical evaluation framework driven by the Matthews Correlation Coefficient ($MCC$) and true positive sensitivity ($Recall$). By executing regularized 5-Fold Cross-Validation alongside exhaustive hyperparameter optimization via `GridSearchCV`, this study builds a reliable diagnostic portfolio, identifying an optimized Random Forest ensemble as our most robust diagnostic auditor and an RBF-kernel Support Vector Machine as our premier high-sensitivity screening tool.

---

## 📊 4. Dataset & Exploratory Data Analysis (EDA) Summary

### Integrated Framework Core Elements

* **Integrated Clinical Sites**

Cleveland Clinic Foundation (USA), Hungarian Institute of Cardiology (Hungary), University Hospital (Zurich, Switzerland), and VA Medical Center (Long Beach, USA).
* **Cohort Scale**

Encompasses 920 total patient matrices.
* **Attribute Dimension Space**

Comprises 13 predictive feature fields mapped across demographic, physiological, metabolic, stress test, anatomical, and electrical diagnostic categories. The linear correlation heatmap matrix expands to a $14 \times 14$ grid because the binary target variable is integrated directly alongside the 13 predictive inputs to map feature-to-target dependencies.

### Core EDA Anomaly Discoveries

* **Global Class Symmetries**

The combined binary target label displays a balanced distribution consisting of 411 healthy profiles (44.7%) and 509 active disease profiles (55.3%), protecting downstream optimization from majority class collapsing shortcuts.

![Figure 1 - Global Class Prevalence Profile](Visuals/figure1_class_prevalence.png)

* **Geographic Bifurcation**

The Cleveland and Hungary sub-cohorts represent balanced outpatient screening environments. Conversely, the Switzerland (8 normal vs. 115 diseased) and VA Long Beach (50 normal vs. 148 diseased) registries function as high-acuity interventional tertiary units dominated by advanced pathology.

![Figure 2 - Center-Specific Bias and Geographic Variations](Visuals/figure2_hospital_comparison.png)

* **Administrative Recording Artifact**

A heavy, non-physiological spike resting precisely at 0 mg/dl was discovered in the continuous cholesterol distribution. Cross-referencing proved this trace belonged exclusively to the Switzerland and VA Long Beach registries, where lipid panel tests were routinely skipped during emergency intake triage.

![Figure 3 - Feature Scale Discrepancies and Cholesterol Artifact](Visuals/figure3_lipid_anomalies.png)

* **Linear Associations**

The variables with the highest positive linear correlation to heart disease are chest pain type (`cp`, $r = 0.47$) and exercise-induced angina (`exang`, $r = 0.45$), while maximum heart rate achieved (`thalach`, $r = -0.39$) exhibits a robust inverse relationship.

![Figure 4 - Clinical Feature Correlation Heatmap](Visuals/figure4_correlation_matrix.png)

---

## ⚙️ 5. Data Preprocessing & Engineering Pipeline

To resolve the architectural scale variances, geographic skews, and zero-masked tracking artifacts uncovered during the exploratory data analysis phase, raw records pass through an automated mathematical engineering sequence:

### 💡 Target Re-Formulation (Label Binning)

The multi-stage coronary artery disease severity index (Stages 0–4) is transformed into a safety-critical binary target framework using a specialized lambda mapping script:

$$Y = 0 \quad \text{if raw stage} = 0 \ (\text{healthy})$$

$$Y = 1 \quad \text{if raw stage} \in \{1, 2, 3, 4\} \ (\text{diseased})$$

### 💡 Algorithmic Missing Value Reconstruction

To prevent a catastrophic loss of 66% of overall rows caused by listwise deletion, a K-Nearest Neighbors (`KNNImputer`) setup is deployed with $k=5$. Proximity across the remaining numeric variables is determined using a pairwise, NaN-adjusted Euclidean distance formula:

$$d(x,y) = \sqrt{\frac{m}{p} \sum_{i \in \text{valid}} (x_i - y_i)^2}$$

where $p$ represents the count of valid features present in both vectors, and $m$ represents the total feature dimension space of 13 attributes. The missing or zero-masked elements are then reconstructed via local unweighted neighbor averaging:

$$x_{\text{imputed}} = \frac{1}{k} \sum_{j=1}^{k} N_j$$

### 💡 Z-Score Standardization

To neutralize severe scale variations across continuous measurements (e.g., cholesterol expanding up to 603 mg/dl swacking a tight fractional ST waveform reading), features are normalized to prevent spatial distance distortion inside distance-sensitive frameworks:

$$X_{\text{scaled}} = \frac{X - \mu}{\sigma}$$

### 💡 Data Leakage Safeguards

To guarantee clean experimental boundaries, scaling parameters are isolated using strict pipeline ordering structures across both project phases:

* **Baseline Eighty-Twenty Split**

The data is split into a static training cohort (80%, 736 records) and a testing cohort (20%, 184 records). The `StandardScaler` computes its structural mean and standard deviation parameters strictly from the training partition alone before transforming the unseen test set.
* **Final Five-Fold Cross-Validation**

The split scales into an iterative five-fold stratified partitioning framework. Scaling transformations are embedded directly within a pipeline loop, forcing the scaling parameters to recalculate from scratch strictly within the active training fold (80% training data vs. 20% validation data per fold).

---

## 🧠 6. Model Portfolio & Hyperparameter Configurations

Five frameworks were evaluated using a `GridSearchCV` routine optimized specifically for the macro $F_1$-score to enforce balanced decision boundaries across both target categories:

* **Baseline Logistic Regression**

Regularized with an inverse weight of $C = 0.1$ paired with the `lbfgs` optimization solver and restricted to 1000 maximum iterations to apply a strict $L_2$ penalty that suppresses center-specific tracking noise.
* **Instance-Based K-Nearest Neighbors (KNN)**

Parameterized using a neighborhood pool of 19 consensus nodes, uniform voting weights, and the Manhattan distance ($L_1$ norm) metric to apply a low-pass smoothing filter that prevents wide-magnitude metrics from swamping electrical wave indicators.
* **Rule-Based Decision Tree**

Restricted to a maximum depth of 5 layers driven by entropy splitting metrics with a minimum split threshold of 10 samples to terminate branches early and enforce structural regularization.
* **Random Forest Ensemble**

Compiled using 100 independent decision trees initialized with unconstrained depths and the Gini variance impurity criterion to average out uncorrelated classification variances across bootstrap aggregated bags.
* **Support Vector Machine (SVM)**

Locked to a soft margin strength of $C = 0.1$, the scale gamma coefficient, and a non-linear Radial Basis Function (RBF) transformation kernel to tolerate minor training anomalies and maximize true positive clinical detection boundaries.

---

## 📈 7. Clinical Performance Audit & Scorecard

The optimized portfolio displays the following out-of-sample performance trends derived via robust 5-Fold Cross-Validation on the independent testing partition:

| Classification Architecture | Accuracy | Recall (Safety Margin) | $F_1$-Score | MCC (Robustness Floor) |
| --- | --- | --- | --- | --- |
| **Random Forest Ensemble** | 86.4% | 84.4% | 0.880 | **0.727** |
| **K-Nearest Neighbors** | 85.8% | 83.4% | 0.875 | 0.717 |
| **Logistic Regression Baseline** | 82.6% | 80.7% | 0.846 | 0.651 |
| **Support Vector Machine (SVM)** | 80.9% | **88.0%** | 0.829 | 0.622 |
| **Decision Tree (Control Node)** | 78.8% | 75.2% | 0.808 | 0.573 |

### 🩺 Deep-Dive Confusion Matrix Interpretations

Cross-referencing the compiled validation portfolio gallery isolates two clear operational pathways for deployment:

![Figure 7 - Diagnostic Validation Portfolio Gallery](Visuals/figure7_confusion_matrices.png)

* **The Balanced Auditor (Random Forest)**

Establishes our highest diagnostic robustness floor ($MCC = 0.727$, Accuracy = 86.4%). It successfully isolates 67 true negatives and 92 true positives while limiting false alarms to just 8 false positives, making it ideal for routine hospital resource verification loops.
* **The High-Sensitivity Screening Line (SVM)**

Delivers an elite clinical safety ceiling by achieving a Recall score of 88.0%. It successfully flags 96 out of 109 true diseased patients and drops critical missed diagnoses down to an absolute low of 13 false negatives, accepting an elevated trade-off of 15 false positives to protect high-stakes preliminary intake lines.

---

## 🩺 8. Explainable Medical Logic Validation

Extracting the global feature importance rankings based on the mean decrease in node Gini impurity across the 100 independent trees of the champion Random Forest ensemble verifies that the mathematical gradients converged on organic human biology rather than administrative data shortcuts:

![Figure 8 - Clinical Feature Importance](Visuals/figure8_feature_importance.png)

1. **Chest Pain Type (`cp`, Importance: 0.144)**

The primary predictive driver, capturing immediate clinical red flags for active coronary artery obstructions and myocardial ischemia.

2. **Maximum Heart Rate Achieved (`thalach`, Importance: 0.113)**

Captures the restricted physiological ceiling and loss of cardiovascular reserve capacity typical of ischemic heart tissue under strain.

3. **Number of Major Vessels (`ca`, Importance: 0.106)** 

Tracks anatomical calcification evidence discovered via fluoroscopy imaging.

4. **Secondary Risk Layers** 

Incorporates serum cholesterol (`chol`, Importance: 0.097) to track continuous baseline metabolic risk profiles, and ST depression (`oldpeak`, Importance: 0.093) to parse immediate electrical waveform changes recorded during physical stress testing.

### 🧪 Methodological Note on Alternative Architecture Logic

* **Standalone Decision Tree Volatility**

While a single tree computes feature splits using similar mechanisms, it lacks bootstrap aggregation safeguards. A single tree is highly sensitive to training variance, risking ranking variables based on localized data noise or site-specific anomalies within an individual hospital registry. The Random Forest averages split importances across 100 uncorrelated paths to provide a stable, generalizable biological ranking.
* **Support Vector Machine Non-Linear Processing**

Because our optimized support vector machine utilizes a non-linear Radial Basis Function (RBF) kernel, it projects the 13 input fields into a higher-dimensional space to maximize the geometric margin of a separating hyperplane. In this space, variables are mathematically transformed and combined non-linearly. Consequently, individual linear feature weights do not exist for an RBF architecture; the model evaluates complex, compounding physiological interactions rather than tracking isolated, standalone metrics.

---

## 📊 9. Mathematical Audit Specifications

To maintain absolute scientific transparency, all downstream evaluation metrics are derived directly from the four primary quadrants of the validation confusion matrix:

$$\text{Overall Classification Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Recall / Sensitivity (Clinical Safety)} = \frac{TP}{TP + FN}$$

$$F_1\text{-Score (Harmonic Performance Mean)} = \frac{2 \times TP}{2 \times TP + FP + FN}$$

$$\text{Matthews Correlation Coefficient (MCC)} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$

### 📋 Clinical Metric Key

* **True Positive (TP)**

Sick patients correctly identified as having active coronary heart disease.
* **True Negative (TN)**

Healthy individuals correctly identified as normal.
* **False Positive (FP)**

Healthy individuals incorrectly identified as sick, generating an unneeded false diagnostic alarm.
* **False Negative (FN)**

Sick patients incorrectly identified as healthy, representing the single most critical failure rate to minimize in clinical diagnostics.

---

## 🏆 10. Conclusion & Operational Directives

This research project successfully demonstrates the transition from a messy, multi-centric clinical database into a highly stable, machine learning decision-support architecture. By utilizing an advanced data engineering pipeline driven by biomedical K-Nearest Neighbors multivariate imputation and strict stratified standardization, we successfully resolved intense multi-hospital collection skews and administrative recording artifacts without corrupting the underlying physiological dataset logic.

Our final performance evaluation outlines a powerful diagnostic portfolio. For routine verification and hospital resource auditing tasks, the Random Forest ensemble provides the most reliable balance across all metrics ($MCC = 0.727$). For high-stakes preliminary screening lines where minimizing missed diagnoses is the single most critical task, the Support Vector Machine provides an elite clinical safety ceiling (88.0% Recall), ensuring high diagnostic protection across diverse patient cohorts.

Ultimately, this study proves that machine learning models can extract universal cardiovascular risk structures from diverse data sources, provided that data-cleaning decisions are actively guided by exploratory domain auditing. To expand this pipeline further, the next logical engineering phase must involve testing these architectures against an entirely external, unintegrated hospital cohort to actively evaluate cross-border domain adaptation and algorithmic generalization boundaries under varying institutional testing constraints.
