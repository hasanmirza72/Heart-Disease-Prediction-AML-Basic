"""
Project: Heart Disease Prediction (Multi-Hospital Integration)
Phase: 2 - Data Preprocessing & Clinical Imputation
Objective: Prepare the raw dataset for machine learning by standardizing
           the target variable and handling missing clinical values.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# -------------------------------------------------------------------------
# SECTION 1: DATA INGESTION
# -------------------------------------------------------------------------
# We load the raw merged dataset created in Phase 1. This modular step
# ensures we are working on a stable clinical baseline.
# -------------------------------------------------------------------------
input_file = 'heart_disease_raw.csv'
df = pd.read_csv(input_file)

print(f"--- Phase 2 Started: Processing {len(df)} records ---")

# -------------------------------------------------------------------------
# SECTION 2: BINARY TARGET CONVERSION (LABEL BINNIG)
# -------------------------------------------------------------------------
# Clinically, the original data ranks heart disease from 0 (Healthy) to
# 4 (Severe). For a binary classification task, we simplify this:
# 0 = No Disease (Healthy)
# 1, 2, 3, 4 = Disease Present (Mapped to 1)
# This improves model reliability given the sample size constraints.
# -------------------------------------------------------------------------
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

print("Step 1 Complete: Target variable converted to binary (0/1).")

# -------------------------------------------------------------------------
# SECTION 3: HANDLING MISSING VALUES VIA KNN IMPUTATION
# -------------------------------------------------------------------------
# Clinical data is often incomplete. Instead of deleting rows (which causes
# bias), we use K-Nearest Neighbors (KNN) to 'estimate' missing values.
# Logic: KNN finds 5 'similar' patients based on other clinical markers
# and uses their values to fill the gaps in features like 'ca' and 'thal'.
# -------------------------------------------------------------------------

# The 'hospital' column is text and must be temporarily removed for KNN math
hospital_metadata = df['hospital']
numeric_data = df.drop(columns=['hospital'])

# Initialize the Imputer. n_neighbors=5 is a standard clinical default.
imputer = KNNImputer(n_neighbors=5)

# fit_transform calculates the distances between patients and fills NaNs
imputed_array = imputer.fit_transform(numeric_data)

# Re-create the dataframe with original clinical headers
df_cleaned = pd.DataFrame(imputed_array, columns=numeric_data.columns)

# Re-attach the hospital metadata we set aside earlier
df_cleaned['hospital'] = hospital_metadata.values

print("Step 2 Complete: Missing values filled using clinical KNN Imputation.")

# -------------------------------------------------------------------------
# SECTION 4: FINAL INTEGRITY CHECK & SUMMARY
# -------------------------------------------------------------------------
# Before moving to EDA (Phase 3), we verify that all missing values are
# resolved and check the final balance of our classes.
# -------------------------------------------------------------------------
print("\n" + 30 * "=")
print("PREPROCESSED DATASET SUMMARY")
print(30 * "=")
print(f"Target Distribution:\n{df_cleaned['target'].value_counts()}")
print(f"Remaining Missing Values: {df_cleaned.isnull().sum().sum()}")
print(30 * "=")

# -------------------------------------------------------------------------
# SECTION 5: EXPORT CLEANED DATASET
# -------------------------------------------------------------------------
# We save this version as 'heart_disease_cleaned.csv'. This file will
# serve as the primary source for our Phase 3 Visualizations.
# -------------------------------------------------------------------------
output_file = 'heart_disease_cleaned.csv'
df_cleaned.to_csv(output_file, index=False)

print(f"\nPhase 2 Complete. Cleaned data saved as: {output_file}")