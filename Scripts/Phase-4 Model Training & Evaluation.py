"""
Project: Heart Disease Prediction (Multi-Hospital Integration)
Phase: 4 - Advanced Model Training & Clinical Evaluation
Objective: Train and audit five classification algorithms using a comprehensive
           suite of metrics including Accuracy, Recall, F1-Score, and MCC.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, matthews_corrcoef

# -------------------------------------------------------------------------
# SECTION 1: DATA PREPARATION & SCALING
# -------------------------------------------------------------------------
# Load the cleaned dataset. For clinical models, we split the data
# before scaling to prevent 'Data Leakage' (information from the test set
# influencing the training set).
# -------------------------------------------------------------------------
df = pd.read_csv('heart_disease_cleaned.csv')

# Features (X) exclude the target and the hospital metadata
X = df.drop(columns=['target', 'hospital'])
y = df['target']

# 80/20 split is the standard for datasets of this size (~920 records)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# StandardScaler is mandatory for distance-sensitive models (SVM, KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"--- Phase 4 Started: Auditing 5 Models on {len(X_train)} training samples ---")

# -------------------------------------------------------------------------
# SECTION 2: MODEL INITIALIZATION
# -------------------------------------------------------------------------
# Selected a diverse range of algorithms:
# 1. Linear (Logistic Regression)
# 2. Instance-based (KNN)
# 3. Tree-based (Decision Tree, Random Forest)
# 4. Kernel-based (SVM)
# -------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

results = []

# -------------------------------------------------------------------------
# SECTION 3: CLINICAL EVALUATION LOOP
# -------------------------------------------------------------------------
# For a medical screening tool, we prioritize:
# - Recall: Minimizing False Negatives (Missing a sick patient).
# - MCC: A robust correlation coefficient ($+1$ to $-1$) that considers
#   True Negatives—essential for identifying healthy individuals accurately.
# -------------------------------------------------------------------------
for name, model in models.items():
    # Model Training
    model.fit(X_train_scaled, y_train)

    # Model Prediction
    y_pred = model.predict(X_test_scaled)

    # Metric Calculation
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Recall": rec,
        "F1-Score": f1,
        "MCC": mcc
    })
    print(f"Audit Complete: {name}")

# Convert results to a DataFrame for analysis
performance_df = pd.DataFrame(results)

# -------------------------------------------------------------------------
# SECTION 4: MULTI-METRIC VISUALIZATION
# -------------------------------------------------------------------------
# Create a grouped bar chart. A high-performing clinical model should
# show high scores across all four metrics, particularly in Recall and MCC.
# -------------------------------------------------------------------------
performance_melted = performance_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(14, 8))
sns.barplot(data=performance_melted, x="Model", y="Score", hue="Metric", palette="viridis")

plt.title("Heart Disease Model Comparison: Accuracy vs. Recall vs. F1 vs. MCC", fontsize=16)
plt.ylim(0.6, 1.0)  # Magnify the differences between top models
plt.ylabel("Performance Score")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig('clinical_model_comparison.png')

# -------------------------------------------------------------------------
# SECTION 5: FINAL PERFORMANCE SUMMARY
# -------------------------------------------------------------------------
print("\n" + 40 * "=")
print("FINAL CLINICAL PERFORMANCE AUDIT")
print(40 * "=")
print(performance_df.sort_values(by="MCC", ascending=False).to_string(index=False))
print(40 * "=")
print("Visualization saved as: clinical_model_comparison.png")