"""
Project: Heart Disease Prediction (Multi-Hospital Integration)
Phase: 6 - Final Portfolio Presentation & Clinical Audit
Objective: Generate all high-resolution visuals and metrics for the final report,
           comparing every optimized model side-by-side.
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
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, matthews_corrcoef

# 1. DATA PREPARATION (Ensuring scientific consistency)
df = pd.read_csv('heart_disease_cleaned.csv')
X = df.drop(columns=['target', 'hospital'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. INITIALIZE ALL MODELS WITH THE CORRECT OPTIMAL PARAMETERS (Matched to Phase 5)
optimized_models = {
    "Logistic Regression": LogisticRegression(C=0.1, solver='lbfgs', max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(metric='manhattan', n_neighbors=19, weights='uniform'),
    "Decision Tree": DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=10, random_state=42),
    "Random Forest": RandomForestClassifier(criterion='gini', max_depth=None, n_estimators=100, random_state=42),
    "SVM": SVC(C=0.1, gamma='scale', kernel='rbf', probability=True, random_state=42) # Fixed C and Gamma
}

results = []
cms = {}

# 3. THE FINAL CLINICAL AUDIT LOOP
print("--- Phase 6: Generating Final Portfolio Visuals ---")
for name, model in optimized_models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Capture Metrics
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    })

    # Store Confusion Matrix for the Gallery
    cms[name] = confusion_matrix(y_test, y_pred)
    print(f"Finalizing Audit: {name}")

performance_df = pd.DataFrame(results)

# -------------------------------------------------------------------------
# VISUAL 1: THE PERFORMANCE GALLERY (ACCURACY VS RECALL VS F1 VS MCC)
# -------------------------------------------------------------------------
performance_melted = performance_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
plt.figure(figsize=(14, 8))
sns.barplot(data=performance_melted, x="Model", y="Score", hue="Metric", palette="viridis")
plt.title("Final Comparative Performance of Heart Disease Models", fontsize=16)
plt.ylim(0.6, 1.0)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('final_performance_comparison.png')

# -------------------------------------------------------------------------
# VISUAL 2: THE CONFUSION MATRIX PORTFOLIO (MULTI-GRID)
# -------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, (name, matrix) in enumerate(cms.items()):
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
    axes[i].set_title(f'{name}', fontsize=14)
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

# Hide the unused 6th subplot
axes[5].axis('off')
plt.suptitle('Diagnostic Portfolio: Comparison of Clinical Classification Gaps', fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('final_confusion_matrix_portfolio.png')

# -------------------------------------------------------------------------
# VISUAL 3: FEATURE IMPORTANCE (CHAMPION MODEL INSIGHT)
# -------------------------------------------------------------------------
rf = optimized_models["Random Forest"]
feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_df, hue='Feature', palette='magma', legend=False)
plt.title('Clinical Feature Importance (Final Predictive Power)', fontsize=15)
plt.savefig('final_feature_importance.png')

# 4. FINAL CONSOLE OUTPUT
print("\n" + 50 * "=")
print("FINAL ROBUST PERFORMANCE REPORT")
print(50 * "=")
print(performance_df.sort_values(by='MCC', ascending=False).to_string(index=False))
print(50 * "=")
print("All final images saved in your project folder.")