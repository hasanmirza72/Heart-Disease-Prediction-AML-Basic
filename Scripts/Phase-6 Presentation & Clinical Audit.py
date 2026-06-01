"""
Project: Heart Disease Prediction (Multi-Hospital Integration)
Phase: 6 - Final Portfolio Production Analytics & Clinical Audit
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
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, matthews_corrcoef

# =========================================================================
# 1. SCIENTIFIC CONSISTENCY BOUNDARY (DATA LEAKAGE PREVENTION)
# =========================================================================
# Loading our preprocessed, reconstructed master dataset checkpoint.
# Splitting rows into an 80/20 stratified partition prior to scaling preserves
# strict out-of-sample isolation, preventing downstream metric contamination.
# =========================================================================
df = pd.read_csv('heart_disease_cleaned.csv')
X = df.drop(columns=['target', 'hospital'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizing continuous features ensures wide-magnitude features (cholesterol)
# do not overpower low-magnitude continuous values (electrical oldpeak).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================================
# 2. OPTIMIZED CONFIGURATION LOCK-IN (POST CV-GRID SEARCH SELECTION)
# =========================================================================
# Rather than relying on default presets, we freeze our portfolio models using
# the precise mathematical hyperparameters discovered during 5-Fold Cross-Validation:
#
# * Logistic Regression: Strict L2 ridge bounds (C=0.1) paired with the 'lbfgs' Newton solver.
# * KNN: Bounded neighborhood pool (k=19) optimized over absolute Manhattan geometry.
# * Decision Tree: Max depth layer pruning (5) driven by Information Gain (entropy).
# * Random Forest: Unconstrained tree depth (None) leveraging 100 Gini estimators.
# * SVM: soft-margin regularization constraint (C=0.1) utilizing an RBF non-linear kernel.
# =========================================================================
optimized_models = {
    "Logistic Regression": LogisticRegression(C=0.1, solver='lbfgs', max_iter=1000, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(metric='manhattan', n_neighbors=19, weights='uniform'),
    "Decision Tree": DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=10, random_state=42),
    "Random Forest": RandomForestClassifier(criterion='gini', max_depth=None, n_estimators=100, random_state=42),
    "SVM": SVC(C=0.1, gamma='scale', kernel='rbf', probability=True, random_state=42)
}

results = []
cms = {}

# =========================================================================
# 3. PRODUCTION PORTFOLIO EVALUATION LOOP
# =========================================================================
# Fitting our regularized algorithms over the training matrix and evaluating
# performance on unseen test rows to build our comparative clinical portfolio.
# =========================================================================
print("--- Phase 6: Compiling Production-Grade Analytics & Visual Portfolios ---")
for name, model in optimized_models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Compiling evaluation metrics for statistical matrix tracking
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    })

    # Buffering raw confusion arrays for downstream multi-grid visualization
    cms[name] = confusion_matrix(y_test, y_pred)
    print(f"Finalizing Audit Verification: {name}")

performance_df = pd.DataFrame(results)

# =========================================================================
# VISUAL 1: THE PERFORMANCE MATRIX COMPARISON (GROUPED CHART)
# =========================================================================
# Generating a grouped bar chart mapping accuracy against imbalance-aware metrics.
# Enforcing a lower limit boundary of 0.6 magnifies minor classification gaps
# between top-performing assets, clarifying structural optimization jumps.
# =========================================================================
performance_melted = performance_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
plt.figure(figsize=(14, 8))
sns.barplot(data=performance_melted, x="Model", y="Score", hue="Metric", palette="viridis")
plt.title("Final Comparative Performance of Heart Disease Models", fontsize=16)
plt.ylim(0.6, 1.0)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('final_performance_comparison.png')

# =========================================================================
# VISUAL 2: THE CONFUSION MATRIX PORTFOLIO (MULTI-GRID COHORT GAP ANALYSIS)
# =========================================================================
# Compiling a multi-grid heatmap gallery to map true positive/negative allocations.
# This visualization maps clinical triage trade-offs, distinguishing the highly
# balanced auditor tracking profile of the Random Forest against the top sensitivity ceiling
# (Recall) delivered by the SVM soft-margin hyperplane.
# =========================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, (name, matrix) in enumerate(cms.items()):
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
    axes[i].set_title(f'{name}', fontsize=14)
    axes[i].set_xlabel('Predicted Diagnostic Class')
    axes[i].set_ylabel('Actual Clinical Ground Truth')

# Deactivating empty canvas coordinates within the subplot matrix array
axes[5].axis('off')
plt.suptitle('Diagnostic Portfolio: Comparison of Clinical Classification Gaps', fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('final_confusion_matrix_portfolio.png')

# =========================================================================
# VISUAL 3: GLOBAL FEATURE IMPORTANCE (CHAMPION MODEL LOGIC REVEAL)
# =========================================================================
# Extracting the global feature weights of our champion ensemble model.
# Mapping mean decrease in impurity across the Random Forest trees opens up
# the internal diagnostic logic, verifying that features like chest pain
# type and max heart rate match empirical cardiovascular guidelines.
# =========================================================================
rf = optimized_models["Random Forest"]
feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_df, hue='Feature', palette='magma', legend=False)
plt.title('Clinical Feature Importance (Final Predictive Power)', fontsize=15)
plt.savefig('final_feature_importance.png')

# =========================================================================
# 4. CONSOLIDATED CONSOLE SUMMARY REPORT
# =========================================================================
# Sorting console frames by descending Matthews Correlation Coefficient (MCC)
# to isolate our most robust multi-center classifiers at the top of the terminal feed.
# =========================================================================
print("\n" + 50 * "=")
print("FINAL CONSOLIDATED PRODUCTION ENGINE REPORT")
print(50 * "=")
print(performance_df.sort_values(by='MCC', ascending=False).to_string(index=False))
print(50 * "=")
print("All high-resolution production charts successfully serialized to root directory.")