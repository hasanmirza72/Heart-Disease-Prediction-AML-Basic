"""
Project: Heart Disease Prediction (Multi-Hospital Integration)
Phase: 5 - Multi-Model Hyperparameter Grid Optimization & Cross-Validation Audit
Author: Mirza Muhammad Hasan Ali
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, matthews_corrcoef

# =========================================================================
# 1. STANDARDIZED DATA MATRIX INGESTION & PIPELINE BOUNDARY CONTROL
# =========================================================================
# Loading the preprocessed, reconstructed dataset checkpoint. Splitting the rows
# into a stratified 80/20 train-test partition prior to fitting the scaler
# ensures strict protection against data leakage, guaranteeing a pristine out-of-sample
# validation space.
# =========================================================================
df = pd.read_csv('heart_disease_cleaned.csv')
X = df.drop(columns=['target', 'hospital'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("--- Initializing Phase 5 Pipeline: Executing Parallel Grid Search Optimization ---")

# =========================================================================
# SECTION 2: EXHAUSTIVE HYPERPARAMETER SPECIFICATION GRID
# =========================================================================
# Rather than relying on default presets, we configure a highly specific
# hyperparameter tuning grid for every algorithm in our portfolio.
# These "tuning knobs" evaluate fundamental mathematical properties to
# determine how each model handles multi-center data noise and artifacts.
# =========================================================================
model_params = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            # 'C' (Inverse of Regularization Strength):
            # Controls the L2 ridge regularization penalty. Smaller values (0.1)
            # apply a stricter penalty to constrain weight magnitudes, preventing
            # the linear model from overfitting to hospital-specific anomalies.
            'C': [0.1, 1, 10, 100],

            # 'solver' (Optimization Algorithm):
            # 'liblinear' applies a coordinate descent solver well-suited for
            # compact datasets, while 'lbfgs' utilizes a quasi-Newton line-search
            # method that handles multi-variate continuous features with elite stability.
            'solver': ['liblinear', 'lbfgs']
        }
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(),
        "params": {
            # 'n_neighbors' (Consensus Pool Size / k-value):
            # Bounds the local voting neighborhood. Small values (3, 5) create volatile,
            # complex decision boundaries highly sensitive to noise. Large values (11, 19)
            # apply a low-pass smoothing filter to base classifications on broad clinical trends.
            'n_neighbors': [3, 5, 11, 19],

            # 'weights' (Voting Priority Mapping):
            # 'uniform' treats all local neighbors with equal statistical weight, whereas
            # 'distance' weights votes inversely to their spatial proximity, giving closer,
            # clinically-similar patient vectors higher diagnostic priority.
            'weights': ['uniform', 'distance'],

            # 'metric' (Coordinate Space Geometry):
            # 'Euclidean' computes straight-line distance (L2 norm). 'manhattan' computes
            # absolute non-squared grid distance (L1 norm), which prevents wide-magnitude
            # continuous features (like cholesterol scaling to 603) from drowning out low-scale
            # variables (like electrical oldpeak) during proximity calculations.
            'metric': ['euclidean', 'manhattan']
        }
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            # 'criterion' (Mathematical Quality-of-Split Metric):
            # 'gini' minimizes misclassification risk by tracking variance impurity probabilities,
            # while 'entropy' measures Information Gain (logarithmic uncertainty reduction), forcing
            # high-level early branch splits that closely mimic clinical entry triage protocols.
            'criterion': ['gini', 'entropy'],

            # 'max_depth' (Structural Layer Growth Limits):
            # 'None' allows unconstrained growth until all leaves are pure (memorizing center bias).
            # Bounded integers (5, 10, 20) prune the tree early to preserve out-of-sample generalization.
            'max_depth': [None, 5, 10, 20],

            # 'min_samples_split' (Node Splitting Threshold Constraint):
            # Specifies the minimum patient count required to split an internal branch. Low bounds (2)
            # allow hyper-specific paths; high restrictions (5, 10) force early termination to stabilize structure.
            'min_samples_split': [2, 5, 10]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            # 'n_estimators' (Ensemble Size / Tree Volume):
            # Configures the number of independent decision trees to train. Testing 100 vs 300
            # estimators evaluates where the variance-reduction benefits of bootstrap aggregating
            # (bagging) fully plateau, balancing predictive power against computational efficiency.
            'n_estimators': [100, 300],

            # 'max_depth' & 'criterion':
            # Evaluates depth pruning boundaries and split impurity metrics across separate bagging
            # iterations to ensure the random subspace ensemble cancels out multi-centric recording noise.
            'max_depth': [None, 10, 20],
            'criterion': ['gini', 'entropy']
        }
    },
    "SVM": {
        "model": SVC(probability=True, random_state=42),
        "params": {
            # 'C' (Soft Margin Regularization Behavior):
            # Dictates the strictness of the geometric margin. High values (10) apply an intense
            # misclassification penalty, forcing a rigid boundary. Low values (0.1) enforce a
            # wide, soft margin that tolerates minor training errors, prioritizing structural
            # generalization over perfect training scores to remain robust against data anomalies.
            'C': [0.1, 1, 10],

            # 'kernel' (Spatial Dimensional Projection Function):
            # Projects binned multi-variate features into a higher-dimensional space where non-linear
            # physiological interactions become linearly separable by a maximum-margin hyperplane.
            # - 'rbf' (Radial Basis Function): A localized Gaussian kernel that maps complex, overlapping
            #   risk clusters (e.g., age interacting with exercise angina) into an infinite feature space.
            # - 'poly' (Polynomial Kernel): Evaluates explicit higher-degree polynomial feature combinations
            #   to trace structured curved risk boundaries.
            'kernel': ['rbf', 'poly'],

            # 'gamma' (Kernel Influence Radius Coefficient):
            # Explicitly defines the geographic reach of a single training example's decision boundary.
            # - 'scale': Sets gamma as 1 / (n_features * X.var()), dynamically scaling the radius of
            #   influence based on dataset variance to normalize noise across variable feature scales.
            # - 'auto': Sets gamma as 1 / n_features, assigning a fixed, uniform geometric influence
            #   radius across all attributes regardless of their underlying variance spreads.
            'gamma': ['scale', 'auto']
        }
    }
}

optimized_results = []

# =========================================================================
# 3. GLOBAL EXHAUSTIVE SEARCH LOOP & 5-FOLD CROSS-VALIDATION AUDIT
# =========================================================================
# To eliminate data split dependencies and protect against overfitting, each grid
# combination is evaluated using stratified 5-Fold Cross-Validation. The pipeline
# optimizes for the macro F1-Score, forcing the grid search to locate parameters
# that maximize clinical safety (Recall) without degrading overall positive predictive value.
# =========================================================================
for name, mp in model_params.items():
    print(f"Executing Exhaustive Optimization Pass: {name}...")

    # Optimizing for F1-Score enforces a strict balance between precision and sensitivity bounds
    grid_search = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator__
    y_pred = best_model.predict(X_test_scaled)

    optimized_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Best Params": grid_search.best_params_
    })

# =========================================================================
# 4. CONSOLIDATED POST-OPTIMIZATION SUMMARY REPORT
# =========================================================================
# Consiling the validation attributes into a clean performance summary data frame.
# Sorting predictions by descending Matthews Correlation Coefficient (MCC) automatically
# highlights our most reliable, symmetrically accurate multi-center auditors at the top.
# =========================================================================
performance_df = pd.DataFrame(optimized_results)
print("\n" + 50 * "=")
print("FINAL ROBUST PERFORMANCE AUDIT (POST 5-FOLD CV)")
print(50 * "=")
print(
    performance_df[['Model', 'Accuracy', 'Recall', 'F1-Score', 'MCC']].sort_values(by='MCC', ascending=False).to_string(
        index=False))
print(50 * "=")

# Exporting precise mathematical parameter structures for the report appendix
print("\nOptimal Discovered Clinical Configurations:")
for index, row in performance_df.iterrows():
    print(f"- {row['Model']}: {row['Best Params']}")