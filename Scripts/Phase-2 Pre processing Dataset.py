"""
Project: Heart Disease Prediction (Multi-Hospital Integration)
Phase: 2 - Clinical Target Binning & Multivariate Imputation Pipeline
Author: Mirza Muhammad Hasan Ali
"""

import pandas as pd
from sklearn.impute import KNNImputer

# =========================================================================
# 1. CONSOLIDATED DATA MATRIX INGESTION
# =========================================================================
# Loading the intermediate raw master file produced during Phase 1.
# Operating on this standardized baseline ensures that downstream mathematical
# transformations remain consistent across all four global sub-cohorts.
# =========================================================================
input_file = 'heart_disease_raw.csv'
df = pd.read_csv(input_file)

print(f"--- Initializing Phase 2 Pipeline: Processing {len(df)} Patient Vectors ---")

# =========================================================================
# 2. CLINICAL LOGIC TARGET RE-FORMULATION (LABEL BINNING)
# =========================================================================
# The raw dataset indexes coronary disease across an incremental 0-4 severity
# classification. In a primary screening framework, separating normal individuals
# from those with any grade of arterial occlusion is critical for triage safety.
# We apply a lambda function to collapse the label space into a binary state
# (0 = Healthy, 1 = Cardiac Distress Present), maximizing learning stability.
# =========================================================================
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

print("Step 1 Complete: Categorical target variable converted to binary (0/1).")

# =========================================================================
# 3. MULTIVARIATE MATRIX RECONSTRUCTION (KNN IMPUTATION, k=5)
# =========================================================================
# Discarding rows with missing values would decimate 66% of the multi-center rows.
# To reconstruct missing blocks without introducing human bias, we leverage
# an instance-based K-Nearest Neighbors mathematical imputer.
#
# Process Sequence:
# 1. Metadata Isolation: Set aside the textual 'hospital' tracker so string
#    characters do not distort downstream spatial distance calculations.
# 2. Distance Mapping: Compute pairwise NaN-Euclidean proximity values across
#    continuous features to locate the 5 closest clinical matching patients.
# 3. Vector Imputation: Replace missing markers in fields like 'ca' and 'thal'
#    by taking the unweighted mean value of those 5 neighbors, preserving
#    underlying biological attribute correlations.
# =========================================================================
# Isolating non-numeric metadata vectors from active numerical coordinates
hospital_metadata = df['hospital']
numeric_data = df.drop(columns=['hospital'])

# Initializing distance imputer. k=5 smooths out localized recording biases
imputer = KNNImputer(n_neighbors=5)

# Executing spatial neighbor searches and vector reconstruction
imputed_array = imputer.fit_transform(numeric_data)

# Re-constructing dataframe architecture over the imputed matrix array
df_cleaned = pd.DataFrame(imputed_array, columns=numeric_data.columns)

# Re-attaching the source hospital origin tags for downstream EDA tracking
df_cleaned['hospital'] = hospital_metadata.values

print("Step 2 Complete: Missing data matrices reconstructed via KNN Imputation.")

# =========================================================================
# 4. PRE-FLIGHT DATA INTEGRITY & CLASS BALANCE AUDIT
# =========================================================================
# Before passing data to the Exploratory Data Analysis visualization engine,
# we execute a strict pre-flight check to verify complete missing data resolution
# and verify the exact numerical ratios of our target labels.
# =========================================================================
print("\n" + 40 * "=")
print("HARMONIZED PREPROCESSED DATASET SUMMARY")
print(40 * "=")
print(f"Target Label Stratification Distribution:\n{df_cleaned['target'].value_counts()}")
print(f"Residual Unresolved Missing Values in Matrix: {df_cleaned.isnull().sum().sum()}")
print(40 * "=")

# =========================================================================
# 5. PIPELINE CHECKPOINT EXPORT (SERIALIZED RECONSTRUCTION)
# =========================================================================
# Serializing the preprocessed and imputed dataset as 'heart_disease_cleaned.csv'.
# This establishes a high-integrity checkpoint file, ensuring that the Phase 3
# EDA curves reflect true clinical values without raw noise artifacts.
# =========================================================================
output_file = 'heart_disease_cleaned.csv'
df_cleaned.to_csv(output_file, index=False)

print(f"\nPhase 2 Complete. Preprocessed data checkpoint saved as: {output_file}")