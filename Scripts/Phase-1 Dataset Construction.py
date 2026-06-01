"""
Project: Heart Disease Prediction (Multi-Hospital Integration)
Phase: 1 - Multi-Centric Cohort Consolidation & Data Harmonization
Author: Mirza Muhammad Hasan Ali
"""

import pandas as pd

# =========================================================================
# 1. CLINICAL SCHEMATIC MATRIX DEFINITION
# =========================================================================
# The raw source files from the UCI repository are headerless matrices.
# We define the 14 standard attributes based on the original clinical
# documentation to establish a uniform feature space across all four hospitals.
# Mapping shorthand codes like 'trestbps' and 'ca' allows us to align the
# features for integrated multivariate parsing.
# =========================================================================
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# =========================================================================
# 2. SEPARATED GEOGRAPHIC DATA SOURCES
# =========================================================================
# Defining explicit entry paths for each file to handle differences in
# clinical logging setups across different data collection centers.
# =========================================================================
files = {
    'Cleveland': 'processed.cleveland.data',
    'Hungary': 'processed.hungarian.data',
    'Switzerland': 'processed.switzerland.data',
    'VA_Long_Beach': 'processed.va.data'
}

# Buffer array to stack individual frames before unified structural merging
dataframes = []

# =========================================================================
# 3. INTERNATIONALLY ISOLATED COHORT INGESTION
# =========================================================================
# We iterate through the hospital registries to execute three essential tasks:
# 1. Structural Header Mapping: Applies our standardized attribute columns.
# 2. Character-to-Null Normalization: Converts the string placeholder '?'
#    to native missing values to ensure compatibility with downstream math operations.
# 3. Origin Tracking injection: Appends a 'hospital' category label to preserve
#    source metadata, which allows us to audit collection biases later.
# =========================================================================
print("--- Initializing Multi-Hospital Data Harmonization ---")

for hospital, file_path in files.items():
    # Explicitly parsing '?' into native missing values prevents mathematical column skew
    df = pd.read_csv(file_path, names=columns, na_values='?')

    # Tracking source identity is necessary to evaluate clinical reporting variances later
    df['hospital'] = hospital

    dataframes.append(df)
    print(f"Successfully integrated: {hospital} ({len(df)} patient profiles rows loaded)")

# =========================================================================
# 4. DATASET CONCATENATION & INITIAL INTEGRITY AUDIT
# =========================================================================
# Stacking separate sheets into a single master index. Resetting the index
# yields a continuous coordinate space across all 920 records, preparing
# the rows for cross-validation splitting without positional mixing.
# =========================================================================
heart_df = pd.concat(dataframes, ignore_index=True)

print("\n" + 40 * "=")
print("CONSOLIDATED DATASET SUMMARY")
print(40 * "=")
print(f"Total Combined Multi-Hospital Cohort Size: {heart_df.shape[0]} patient rows")
print(f"Missing Values Identified per Predictive Attribute Field:\n{heart_df.isnull().sum()}")
print(40 * "=")

# =========================================================================
# 5. STREAMLINED STORAGE PASS (MODULAR PIPELINE TRANSITION)
# =========================================================================
# Exporting the raw, unified database sheet to an intermediate file.
# This establishes a clean checkpoint, allowing us to move smoothly to
# Preprocessing and EDA without needing to reload or remap the individual text fields.
# =========================================================================
output_filename = 'heart_disease_raw.csv'
heart_df.to_csv(output_filename, index=False)

print(f"\nPhase 1 Complete. Consolidated master file saved as: {output_filename}")