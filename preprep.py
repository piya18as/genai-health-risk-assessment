
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
df = pd.read_csv("CVD Dataset (2).csv")

# 2. Remove duplicate records
df = df.drop_duplicates()

# 3. Remove rows without target labels
df = df.dropna(subset=["CVD Risk Level"])

# 4. Clean target labels
df["CVD Risk Level"] = (
    df["CVD Risk Level"].astype(str).str.strip().str.upper()
)

# 5. Binary encoding
binary_columns = [
    "Sex",
    "Smoking Status",
    "Diabetes Status",
    "Family History of CVD"
]

binary_mappings = {
    "Sex": {"F": 0, "M": 1},
    "Smoking Status": {"N": 0, "Y": 1},
    "Diabetes Status": {"N": 0, "Y": 1},
    "Family History of CVD": {"N": 0, "Y": 1}
}

for column in binary_columns:
    df[column] = (
        df[column].astype(str).str.strip().str.upper()
        .map(binary_mappings[column])
    )

# 6. Ordinal encoding
activity_mapping = {
    "Low": 0,
    "Moderate": 1,
    "High": 2
}

df["Physical Activity Level"] = (
    df["Physical Activity Level"]
    .astype(str).str.strip().str.title()
    .map(activity_mapping)
)

# 7. Select independent features
# Exclude CVD Risk Score to prevent target leakage.
# Exclude derived/redundant columns.

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

X = df[features].copy()
y = df["CVD Risk Level"].copy()

# 8. Convert features to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# 9. Split dataset BEFORE fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 10. Handle missing values
imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=features,
    index=X_train.index
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=features,
    index=X_test.index
)

# 11. Feature scaling
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=features,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=features,
    index=X_test.index
)

# 12. Verify processed dataset
print("\nProcessed Training Data:")
print(X_train_scaled.head())

print("\nMissing Values:")
print(X_train_scaled.isnull().sum().sum())

print("\nTraining Shape:", X_train_scaled.shape)
print("Testing Shape:", X_test_scaled.shape)

print("\nTarget Distribution:")
print(y.value_counts())

print("\nPreprocessing Completed Successfully!")