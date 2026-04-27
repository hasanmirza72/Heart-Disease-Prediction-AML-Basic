"""
Project: Heart Disease Prediction (Multi-Hospital Integration)
Phase: 1 - Dataset Construction & Standardizing
Objective: Consolidate four international datasets into a single clinical master file.
"""

import pandas as pd
import numpy as np

# -------------------------------------------------------------------------
# SECTION 1: CLINICAL ATTRIBUTE DEFINITION
# -------------------------------------------------------------------------
# The raw UCI files do not contain header rows. We manually define the
# 14 standard attributes based on the UCI clinical documentation.
# Terms like 'trestbps' (Resting Blood Pressure) and 'ca' (Number of
# Major Vessels) are standardized for cross-hospital analysis.
# -------------------------------------------------------------------------
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# -------------------------------------------------------------------------
# SECTION 2: DATA SOURCE DEFINITION (INTERNATIONAL SCOPE)
# -------------------------------------------------------------------------
# Track four distinct geographical locations to account for site-specific
# biases in medical data collection.
# -------------------------------------------------------------------------
files = {
    'Cleveland': 'processed.cleveland.data',
    'Hungary': 'processed.hungarian.data',
    'Switzerland': 'processed.switzerland.data',
    'VA_Long_Beach': 'processed.va.data'
}

# List to store individual dataframes before concatenation
dataframes = []

# -------------------------------------------------------------------------
# SECTION 3: SMART DATA INGESTION LOOP
# -------------------------------------------------------------------------
# Iterate through each file to perform three critical tasks:
# 1. Labeling: Applying the 14 clinical column names.
# 2. Missing Value Handling: Mapping '?' to NaN for mathematical processing.
# 3. Metadata Tagging: Adding a 'hospital' column to preserve the data origin.
# -------------------------------------------------------------------------
print("--- Loading Clinical Data ---")

for hospital, file_path in files.items():
    # 'na_values' is used to identify that '?' represents missing clinical data
    df = pd.read_csv(file_path, names=columns, na_values='?')

    # Tagging the source ensures we can analyze hospital-specific distributions later
    df['hospital'] = hospital

    dataframes.append(df)
    print(f"Successfully loaded: {hospital} ({len(df)} records)")

# -------------------------------------------------------------------------
# SECTION 4: DATASET CONCATENATION & INTEGRITY CHECK
# -------------------------------------------------------------------------
# Merging individual hospital files into a unified master dataset.
# Reset the index to provide a continuous row count for the master dataframe.
# -------------------------------------------------------------------------
heart_df = pd.concat(dataframes, ignore_index=True)

print("\n" + 30 * "=")
print("MASTER DATASET SUMMARY")
print(30 * "=")
print(f"Total Combined Records: {heart_df.shape[0]}")
print(f"Missing Values Identified per Category:\n{heart_df.isnull().sum()}")
print(30 * "=")

# -------------------------------------------------------------------------
# SECTION 5: DATA PERSISTENCE (MODULAR SAVING)
# -------------------------------------------------------------------------
# Save the raw merged data to a CSV. This modular approach allows for
# a clean transition to Phase 2 (Preprocessing) without re-loading raw files.
# -------------------------------------------------------------------------
output_filename = 'heart_disease_raw.csv'
heart_df.to_csv(output_filename, index=False)

print(f"\nPhase 1 Complete. Merged data saved as: {output_filename}")
