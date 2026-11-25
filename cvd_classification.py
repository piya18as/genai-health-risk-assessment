
# ================================================================
# CVD RISK LEVEL PREDICTION USING SUPERVISED MACHINE LEARNING
#
# Algorithms:
# 1. Support Vector Machine (SVM)
# 2. K-Nearest Neighbors (KNN)
# 3. Logistic Regression
# 4. Decision Tree
# 5. Random Forest
# 6. XGBoost (optional)
#
# Hyperparameter Tuning: GridSearchCV
# Model Selection: Cross-Validation Macro F1
#
# Dataset: CVD Dataset (2).csv
# Target: CVD Risk Level
# ================================================================

import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")


# ================================================================
# 1. LOAD DATASET
# ================================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "CVD Dataset (2).csv"

OUTPUT_DIR = BASE_DIR / "CVD_Results"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("\n========================================")
print("       CVD DATASET INFORMATION")
print("========================================")

print("Dataset Shape:", df.shape)

print("\nFirst Five Records:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())


# ================================================================
# 2. DATA CLEANING
# ================================================================

df = df.drop_duplicates().copy()

df = df.dropna(subset=["CVD Risk Level"]).copy()

df["CVD Risk Level"] = (
    df["CVD Risk Level"]
    .astype(str)
    .str.strip()
    .str.upper()
)

print("\nTarget Class Distribution:")
print(df["CVD Risk Level"].value_counts())


# ================================================================
# 3. BLOOD PRESSURE PROCESSING
# ================================================================

bp = df["Blood Pressure (mmHg)"].astype(str).str.extract(
    r"^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$"
)

parsed_systolic = pd.to_numeric(bp[0], errors="coerce")
parsed_diastolic = pd.to_numeric(bp[1], errors="coerce")

df["Systolic BP"] = pd.to_numeric(
    df["Systolic BP"], errors="coerce"
).fillna(parsed_systolic)

df["Diastolic BP"] = pd.to_numeric(
    df["Diastolic BP"], errors="coerce"
).fillna(parsed_diastolic)


# ================================================================
# 4. FEATURE SELECTION
# ================================================================

# CVD Risk Score is excluded to avoid direct target leakage.
# Derived/redundant columns are excluded initially.

features = [
    "Sex",
    "Age",
    "Weight (kg)",
    "Height (m)",
    "Abdominal Circumference (cm)",
    "Systolic BP",
    "Diastolic BP",
    "Total Cholesterol (mg/dL)",
    "HDL (mg/dL)",
    "Fasting Blood Sugar (mg/dL)",
    "Smoking Status",
    "Diabetes Status",
    "Physical Activity Level",
    "Family History of CVD"
]

target = "CVD Risk Level"

missing_columns = [
    col for col in features + [target]
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns in dataset: {missing_columns}"
    )

X = df[features].copy()
y = df[target].copy()

print("\nNumber of Input Features:", len(features))
print("Target Variable:", target)


# ================================================================
# 5. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ================================================================

numeric_features = [
    "Age",
    "Weight (kg)",
    "Height (m)",
    "Abdominal Circumference (cm)",
    "Systolic BP",
    "Diastolic BP",
    "Total Cholesterol (mg/dL)",
    "HDL (mg/dL)",
    "Fasting Blood Sugar (mg/dL)"
]

categorical_features = [
    "Sex",
    "Smoking Status",
    "Diabetes Status",
    "Physical Activity Level",
    "Family History of CVD"
]

for col in numeric_features:
    X[col] = pd.to_numeric(X[col], errors="coerce")

for col in categorical_features:
    X[col] = (
        X[col]
        .astype("string")
        .str.strip()
        .str.upper()
    )

print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)


# ================================================================
# 6. ENCODE TARGET LABELS
# ================================================================

# XGBoost requires numerical class labels.
# LabelEncoder is also used consistently for all algorithms.

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

class_labels = ["LOW", "INTERMEDIARY", "HIGH"]

if set(label_encoder.classes_) != set(class_labels):
    raise ValueError(
        f"Unexpected target classes: {label_encoder.classes_}"
    )

print("\nEncoded Target Classes:")

for index, label in enumerate(label_encoder.classes_):
    print(f"{label} = {index}")


# ================================================================
# 7. TRAIN-TEST SPLIT
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\n========================================")
print("          TRAIN-TEST SPLIT")
print("========================================")

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

print("\nTraining Class Distribution:")
print(pd.Series(
    label_encoder.inverse_transform(y_train)
).value_counts())


# ================================================================
# 8. PREPROCESSING PIPELINE
# ================================================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


def create_preprocessor():

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])


# ================================================================
# 9. DEFINE MODELS AND PARAMETER GRIDS
# ================================================================

models = {

    "SVM": (
        SVC(class_weight="balanced"),
        {
            "classifier__C": [0.1, 1, 10, 100],
            "classifier__kernel": ["linear", "rbf"],
            "classifier__gamma": ["scale", "auto"]
        }
    ),

    "KNN": (
        KNeighborsClassifier(),
        {
            "classifier__n_neighbors": [3, 5, 7, 9, 11],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2]
        }
    ),

    "Logistic Regression": (
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced"
        ),
        {
            "classifier__C": [0.01, 0.1, 1, 10, 100]
        }
    ),

    "Decision Tree": (
        DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),
        {
            "classifier__max_depth": [3, 5, 7, 10, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 5],
            "classifier__criterion": ["gini", "entropy"]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=1
        ),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [5, 10, 15, None],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2]
        }
    )
}


# ================================================================
# 10. ADD XGBOOST IF INSTALLED
# ================================================================

try:

    from xgboost import XGBClassifier

    models["XGBoost"] = (
        XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=1
        ),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.03, 0.1],
            "classifier__subsample": [0.8, 1.0]
        }
    )

    print("\nXGBoost is available.")

except ImportError:

    print("\nXGBoost is not installed. Skipping XGBoost.")


# ================================================================
# 11. CROSS-VALIDATION SETUP
# ================================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision_weighted": "precision_weighted",
    "recall_weighted": "recall_weighted",
    "f1_weighted": "f1_weighted",
    "f1_macro": "f1_macro",
    "balanced_accuracy": "balanced_accuracy"
}

results = []
best_models = {}


# ================================================================
# 12. HYPERPARAMETER TUNING AND MODEL TRAINING
# ================================================================

for model_name, (classifier, param_grid) in models.items():

    print("\n")
    print("=" * 65)
    print("TRAINING MODEL:", model_name.upper())
    print("=" * 65)

    pipeline = Pipeline(steps=[
        ("preprocessor", create_preprocessor()),
        ("classifier", classifier)
    ])

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        refit="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )

    # Train using training data only
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    best_models[model_name] = best_model

    # Predictions on the untouched test set
    y_pred = best_model.predict(X_test)

    # Convert encoded predictions back to original labels
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    # ------------------------------------------------------------
    # PERFORMANCE METRICS
    # ------------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    balanced_acc = balanced_accuracy_score(
        y_test,
        y_pred
    )

    # Best cross-validation results
    best_index = grid_search.best_index_

    cv_accuracy = grid_search.cv_results_[
        "mean_test_accuracy"
    ][best_index]

    cv_macro_f1 = grid_search.cv_results_[
        "mean_test_f1_macro"
    ][best_index]

    # ------------------------------------------------------------
    # STORE RESULTS
    # ------------------------------------------------------------

    results.append({

        "Algorithm": model_name,

        "Accuracy (%)": round(accuracy * 100, 2),

        "Precision (%)": round(precision * 100, 2),

        "Recall (%)": round(recall * 100, 2),

        "Weighted F1 (%)": round(weighted_f1 * 100, 2),

        "Macro F1 (%)": round(macro_f1 * 100, 2),

        "Balanced Accuracy (%)": round(balanced_acc * 100, 2),

        "CV Accuracy (%)": round(cv_accuracy * 100, 2),

        "CV Macro F1 (%)": round(cv_macro_f1 * 100, 2)

    })

    # ------------------------------------------------------------
    # PRINT RESULTS
    # ------------------------------------------------------------

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"Weighted F1: {weighted_f1 * 100:.2f}%")
    print(f"Macro F1: {macro_f1 * 100:.2f}%")
    print(f"Balanced Accuracy: {balanced_acc * 100:.2f}%")
    print(f"CV Accuracy: {cv_accuracy * 100:.2f}%")
    print(f"CV Macro F1: {cv_macro_f1 * 100:.2f}%")

    print("\nClassification Report:")

    print(classification_report(
        y_test_labels,
        y_pred_labels,
        labels=class_labels,
        zero_division=0
    ))

    # ------------------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------------------

    cm = confusion_matrix(
        y_test_labels,
        y_pred_labels,
        labels=class_labels
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_labels
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d"
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")

    plt.savefig(
        OUTPUT_DIR / f"{safe_name}_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    print(f"\n{model_name} evaluation completed.")


# ================================================================
# 13. FINAL MODEL COMPARISON
# ================================================================

results_df = pd.DataFrame(results)

# Rank by cross-validation Macro F1
results_df = results_df.sort_values(
    by="CV Macro F1 (%)",
    ascending=False
).reset_index(drop=True)

print("\n")
print("=" * 110)
print("                    FINAL MODEL COMPARISON")
print("=" * 110)

print(results_df.round(2).to_string(index=False))

results_df.to_csv(
    OUTPUT_DIR / "CVD_Model_Comparison_Tuned.csv",
    index=False
)


# ================================================================
# 14. MODEL COMPARISON GRAPH
# ================================================================

plot_df = results_df.set_index("Algorithm")[
    [
        "Accuracy (%)",
        "Macro F1 (%)",
        "Balanced Accuracy (%)"
    ]
]

plot_df.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("CVD Classification - Tuned Model Comparison")
plt.ylabel("Score (%)")
plt.xlabel("Algorithm")
plt.ylim(0, 100)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "Tuned_Model_Comparison.png",
    dpi=300
)

plt.show()


# ================================================================
# 15. SAVE BEST PERFORMING MODEL
# ================================================================

best_algorithm = results_df.iloc[0]["Algorithm"]

best_model = best_models[best_algorithm]

joblib.dump(
    best_model,
    OUTPUT_DIR / "best_cvd_model_tuned.pkl"
)

# Save the label encoder too, because the model predicts
# encoded numerical classes.
joblib.dump(
    label_encoder,
    OUTPUT_DIR / "cvd_label_encoder.pkl"
)

print("\n")
print("=" * 65)
print("              FINAL MODEL SELECTION")
print("=" * 65)

print("Best Algorithm:", best_algorithm)

print(
    "Test Accuracy:",
    results_df.iloc[0]["Accuracy (%)"],
    "%"
)

print(
    "Test Macro F1:",
    results_df.iloc[0]["Macro F1 (%)"],
    "%"
)

print(
    "Cross Validation Macro F1:",
    results_df.iloc[0]["CV Macro F1 (%)"],
    "%"
)

print("\nBest model saved successfully!")

print("Model:", OUTPUT_DIR / "best_cvd_model_tuned.pkl")
print("Label Encoder:", OUTPUT_DIR / "cvd_label_encoder.pkl")

print("\nAll results saved inside the CVD_Results folder.")

print("=" * 65)