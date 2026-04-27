"""
Project: Heart Disease Prediction (Multi-Hospital Integration)
Phase: 5 - Full Portfolio Optimization & Robustness Audit
Objective: Optimize and validate all five classification models using
           5-Fold Cross-Validation to identify the most robust clinical tool.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, matthews_corrcoef

# 1. DATA PREPARATION (Consistent Baseline)
df = pd.read_csv('heart_disease_cleaned.csv')
X = df.drop(columns=['target', 'hospital'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("--- Phase 5 Started: Full Portfolio Optimization ---")

# -------------------------------------------------------------------------
# SECTION 2: DEFINE HYPERPARAMETER GRIDS
# -------------------------------------------------------------------------
# We define specific 'tuning knobs' for every algorithm in our portfolio.
# -------------------------------------------------------------------------
model_params = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            'C': [0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs']
        }
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(),
        "params": {
            'n_neighbors': [3, 5, 11, 19],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            'n_estimators': [100, 300],
            'max_depth': [None, 10, 20],
            'criterion': ['gini', 'entropy']
        }
    },
    "SVM": {
        "model": SVC(probability=True, random_state=42),
        "params": {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'poly']
        }
    }
}

optimized_results = []

# -------------------------------------------------------------------------
# SECTION 3: THE GLOBAL SEARCH LOOP
# -------------------------------------------------------------------------
# We use 5-Fold Cross-Validation to ensure the results are robust
# and less prone to overfitting compared to a single data split.
# -------------------------------------------------------------------------
for name, mp in model_params.items():
    print(f"Optimizing {name}...")

    # We optimize for F1-Score to balance clinical Recall and Precision
    grid_search = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)

    optimized_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Best Params": grid_search.best_params_
    })

# -------------------------------------------------------------------------
# SECTION 4: FINAL AUDIT SUMMARY
# -------------------------------------------------------------------------
performance_df = pd.DataFrame(optimized_results)
print("\n" + 50 * "=")
print("FINAL ROBUST PERFORMANCE AUDIT (POST 5-FOLD CV)")
print(50 * "=")
# Sort by MCC to show the most clinically reliable model at the top
print(
    performance_df[['Model', 'Accuracy', 'Recall', 'F1-Score', 'MCC']].sort_values(by='MCC', ascending=False).to_string(
        index=False))
print(50 * "=")

# Detailed parameters for the report appendix
print("\nOptimal Clinical Configurations:")
for index, row in performance_df.iterrows():
    print(f"- {row['Model']}: {row['Best Params']}")