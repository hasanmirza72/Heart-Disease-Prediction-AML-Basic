"""
Project: Heart Disease Prediction (Multi-Hospital Integration)
Phase: 3 - Exploratory Data Analysis (EDA)
Objective: Generate clinical visualizations to understand feature correlations,
           class balance, and hospital-specific data distributions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------------
# SECTION 1: DATA LOADING
# -------------------------------------------------------------------------
# Load the cleaned dataset produced in Phase 2.
# -------------------------------------------------------------------------
input_file = 'heart_disease_cleaned.csv'
df = pd.read_csv(input_file)

# Set the visual style for all plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print(f"--- Phase 3 Started: Analyzing {len(df)} records ---")

# -------------------------------------------------------------------------
# SECTION 2: CLASS DISTRIBUTION (HEALTHY vs. DISEASED)
# -------------------------------------------------------------------------
# A balanced dataset is crucial for model performance. Visualize the
# ratio of healthy patients (0) to diseased patients (1).
# -------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='target', data=df, hue='target', palette='viridis', legend=False)
plt.title('Distribution of Heart Disease Presence', fontsize=15)
plt.xlabel('Diagnosis (0 = Healthy, 1 = Disease Present)', fontsize=12)
plt.ylabel('Patient Count', fontsize=12)

# Adding count labels on top of bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.savefig('eda_class_distribution.png')
print("Saved: eda_class_distribution.png")

# -------------------------------------------------------------------------
# SECTION 3: CLINICAL FEATURE CORRELATIONS
# -------------------------------------------------------------------------
# Use a Correlation Matrix to see which clinical markers have the
# strongest relationship with the target diagnosis.
# The correlation coefficient (r) ranges from -1 to +1.
# -------------------------------------------------------------------------
plt.figure(figsize=(14, 10))

# Exclude 'hospital' as it is a categorical string
correlation_matrix = df.drop(columns=['hospital']).corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Clinical Feature Correlation Heatmap', fontsize=16)
plt.savefig('eda_correlation_heatmap.png')
print("Saved: eda_correlation_heatmap.png")

# -------------------------------------------------------------------------
# SECTION 4: AGE AND CHOLESTEROL DISTRIBUTIONS
# -------------------------------------------------------------------------
# Visualize how Age and Cholesterol differ across patient groups.
# Overlapping distributions indicate more challenging classification tasks.
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Age Distribution
sns.kdeplot(data=df, x='age', hue='target', fill=True, ax=axes[0], palette='magma')
axes[0].set_title('Age Distribution by Diagnosis', fontsize=14)

# Cholesterol Distribution
sns.kdeplot(data=df, x='chol', hue='target', fill=True, ax=axes[1], palette='magma')
axes[1].set_title('Cholesterol (mg/dl) Distribution by Diagnosis', fontsize=14)

plt.savefig('eda_clinical_distributions.png')
print("Saved: eda_clinical_distributions.png")

# -------------------------------------------------------------------------
# SECTION 5: SITE-SPECIFIC ANALYSIS (HOSPITAL BIAS)
# -------------------------------------------------------------------------
# Visualize the disease prevalence across different hospital locations.
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.countplot(x='hospital', hue='target', data=df, palette='Set2')
plt.title('Disease Prevalence Across International Hospitals', fontsize=15)
plt.xlabel('Hospital Location', fontsize=12)
plt.ylabel('Patient Count', fontsize=12)
plt.legend(title='Target', labels=['Healthy', 'Diseased'])

plt.savefig('eda_hospital_comparison.png')
print("Saved: eda_hospital_comparison.png")

print("\n" + 30 * "=")
print("EDA PHASE COMPLETE")
print(30 * "=")
print("Review the .png files in your project folder to see the results.")