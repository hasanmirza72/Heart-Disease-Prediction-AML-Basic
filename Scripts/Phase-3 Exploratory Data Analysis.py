"""
Project: Heart Disease Prediction (Multi-Hospital Integration)
Phase: 3 - Exploratory Data Analysis (EDA) & Clinical Artifact Auditing
Author: Mirza Muhammad Hasan Ali
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================================
# 1. HARMONIZED COHORT MATRIX INGESTION
# =========================================================================
# Loading the preprocessed and imputed dataset checkfile from Phase 2.
# Enforcing a unified global style theme ensures that all exported plots
# maintain a consistent grid layout and crisp canvas dimensions for auditing.
# =========================================================================
input_file = 'heart_disease_cleaned.csv'
df = pd.read_csv(input_file)

# Establishes clean visual framing parameters across all export vectors
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print(f"--- Initializing Phase 3 Pipeline: Auditing {len(df)} Patient Profiles ---")

# =========================================================================
# 2. TARGET PREVALENCE STRATIFICATION AUDIT
# =========================================================================
# We plot the final binary class balance to verify gradient optimization
# stability. Isolating the ratio between healthy (0) and diseased (1) vectors
# proves that the cohort lacks severe skews, confirming that downstream
# classifiers cannot cheat via majority class collapse frequency shortcuts.
# =========================================================================
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='target', data=df, hue='target', palette='viridis', legend=False)
plt.title('Distribution of Heart Disease Presence', fontsize=15)
plt.xlabel('Diagnosis (0.0 = Healthy, 1.0 = Disease Present)', fontsize=12)
plt.ylabel('Patient Count', fontsize=12)

# Dynamically rendering absolute frequencies over bar geometric vectors
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.savefig('eda_class_distribution.png')
print("Saved: eda_class_distribution.png")

# =========================================================================
# 3. GLOBAL LINEAR CORRELATION DIMENSIONS (PEARSON MATRIX)
# =========================================================================
# Computing a global Pearson product-moment matrix to identify directional
# dependencies between predictive attributes and target distress states.
# The categorical origin tracking labels ('hospital') are explicitly dropped
# to isolate purely continuous and binned clinical features.
# =========================================================================
plt.figure(figsize=(14, 10))

# Decoupling categorical strings from numerical array dimensions
correlation_matrix = df.drop(columns=['hospital']).corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Clinical Feature Correlation Heatmap', fontsize=16)
plt.savefig('eda_correlation_heatmap.png')
print("Saved: eda_correlation_heatmap.png")

# =========================================================================
# 4. PATIENT RISK KERNEL DENSITY ESTIMATION & ARTIFACT LOCATING
# =========================================================================
# Using continuous Kernel Density Estimation (KDE) to analyze feature spreads.
# The cholesterol plot is critical: it visually exposes the non-physiological
# spike resting at 0 mg/dl, proving the presence of an administrative logging
# artifact in emergency registries where lipid panels were omitted.
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Isolating patient age distribution trends across diagnostic categories
sns.kdeplot(data=df, x='age', hue='target', fill=True, ax=axes[0], palette='magma')
axes[0].set_title('Age Distribution by Diagnosis', fontsize=14)

# Isolating metabolic cholesterol levels to track zero-masked artifacts
sns.kdeplot(data=df, x='chol', hue='target', fill=True, ax=axes[1], palette='magma')
axes[1].set_title('Cholesterol (mg/dl) Distribution by Diagnosis', fontsize=14)

plt.savefig('eda_clinical_distributions.png')
print("Saved: eda_clinical_distributions.png")

# =========================================================================
# 5. GEOGRAPHIC PREVALENCE CROSS-EXAMINATION (SITE-SPECIFIC BIAS)
# =========================================================================
# Cross-tabulating target disease classifications across individual hospital
# origin strings. This plot uncovers severe reporting skews—such as Switzerland's
# clinical focus on active pathology—which alerts us that center-tracking
# strings must be excluded from active modeling attributes to prevent memorization.
# =========================================================================
plt.figure(figsize=(10, 6))
sns.countplot(x='hospital', hue='target', data=df, palette='Set2')
plt.title('Disease Prevalence Across International Hospitals', fontsize=15)
plt.xlabel('Hospital Location', fontsize=12)
plt.ylabel('Patient Count', fontsize=12)
plt.legend(title='Target', labels=['Healthy', 'Diseased'])

plt.savefig('eda_hospital_comparison.png')
print("Saved: eda_hospital_comparison.png")

print("\n" + 40 * "=")
print("EXPLORATORY DATA ANALYSIS PHASE COMPLETE")
print(40 * "=")
print("All high-resolution diagnostic matrices successfully exported to root workspace.")