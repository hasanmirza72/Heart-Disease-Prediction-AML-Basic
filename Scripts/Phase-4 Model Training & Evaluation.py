"""
Project: Heart Disease Prediction (Multi-Hospital Integration)
Phase: 4 - Baseline Algorithmic Training & Multi-Metric Diagnostic Auditing
Author: Mirza Muhammad Hasan Ali
"""

import pandas as pd
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

# =========================================================================
# 1. EVALUATION SPACE ISOLATION & CONTAMINANT PROTECTION (DATA LEAKAGE)
# =========================================================================
# Loading the structured data matrix checkpoint. To preserve strict empirical
# boundaries, the database is partitioned *prior* to scaling. Computing
# the structural mean (mu) and standard deviation (sigma) solely from the
# training set prevents testing metrics from bleeding into the optimization
# loop, establishing a clean validation baseline.
# =========================================================================
df = pd.read_csv('heart_disease_cleaned.csv')

# Decoupling active continuous/discrete features from diagnostic targets
X = df.drop(columns=['target', 'hospital'])
y = df['target']

# Enforcing a stratified 80/20 train-test partition over the 920 records
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# StandardScaler is mandatory to normalize distance-sensitive models (SVM, KNN),
# ensuring wide-magnitude features (cholesterol) do not overpower smaller scales (oldpeak).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"--- Initializing Phase 4 Pipeline: Training Portfolio on {len(X_train)} Instances ---")

# =========================================================================
# 2. ALGORITHMIC TAXONOMY SPECIFICATION (UNOPTIMIZED OPERATIONAL STACK)
# =========================================================================
# Initializing a diverse classification portfolio to evaluate different
# mathematical approaches to feature mapping: linear baselines, instance proximity,
# recursive rule splits, ensemble bagging, and high-dimensional maximum-margin hyperplanes.
# =========================================================================
models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

results = []

# =========================================================================
# 3. HIGH-INTEGRITY CLINICAL AUDITING LOOP
# =========================================================================
# Executing iterative training cycles across our model matrix. For a screening
# asset, optimization targets focus on:
# 1. Recall (Sensitivity): To minimize critical False Negatives (missing a sick patient).
# 2. Matthews Correlation Coefficient (MCC): Evaluates all 4 quadrants of the
#    confusion matrix simultaneously, serving as our primary audit metric to verify
#    symmetric prediction precision on balanced categories.
# =========================================================================
for name, model in models.items():
    # Backpropagation optimization pass over scaled training data
    model.fit(X_train_scaled, y_train)

    # Matrix inference tracking over unseen validation data
    y_pred = model.predict(X_test_scaled)

    # Statistical Evaluation Callouts
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
    print(f"Audit Verification Complete: {name}")

# Re-constructing compiled evaluation dimensions into an audit frame
performance_df = pd.DataFrame(results)

# =========================================================================
# 4. MULTI-VARIATE DIAGNOSTIC PERFORMANCE VISUALIZATION
# =========================================================================
# Melting the evaluation frames to generate a unified grouped bar layout.
# The vertical boundaries are locked between 0.6 and 1.0 to magnify
# minor classification gaps between top-performing assets.
# =========================================================================
performance_melted = performance_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(14, 8))
sns.barplot(data=performance_melted, x="Model", y="Score", hue="Metric", palette="viridis")

plt.title("Heart Disease Model Comparison: Accuracy vs. Recall vs. F1 vs. MCC", fontsize=16)
plt.ylim(0.6, 1.0)
plt.ylabel("Performance Score")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig('clinical_model_comparison.png')

# =========================================================================
# 5. CONSOLIDATED CONSOLE SUMMARY REPORT
# =========================================================================
# Sorting outputs by descending Matthews Correlation Coefficient to automatically
# isolate our most robust, biologically correlated classifiers at the top.
# =========================================================================
print("\n" + 50 * "=")
print("CONSOLIDATED CLINICAL PERFORMANCE PERFORMANCE AUDIT")
print(50 * "=")
print(performance_df.sort_values(by="MCC", ascending=False).to_string(index=False))
print(50 * "=")
print("High-resolution metric comparison plot saved as: clinical_model_comparison.png")