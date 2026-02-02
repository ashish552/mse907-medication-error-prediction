import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

import joblib

IN_PATH = Path("data/processed/model_dataset_labeled_v1.csv")
MODEL_OUT = Path("models/logreg_baseline_v1.joblib")

RANDOM_STATE = 42

def main():
    df = pd.read_csv(IN_PATH)

    # --- Target ---
    if "label_high_risk" not in df.columns:
        raise ValueError("label_high_risk not found. Run proxy label script first.")

    y = df["label_high_risk"].astype(int)

    # --- Features (keep it simple + robust for v1) ---
    # Numeric features
    numeric_features = [
        "polypharmacy_active_meds",
        "anchor_age",
        "creatinine", "bun", "alt", "ast", "bilirubin_total",
        "dose_val_rx_num"
    ]
    # Some may not exist depending on your dataset; keep only existing
    numeric_features = [c for c in numeric_features if c in df.columns]

    # Categorical features
    categorical_features = ["drug", "gender", "admission_type"]
    categorical_features = [c for c in categorical_features if c in df.columns]

    X = df[numeric_features + categorical_features].copy()

    # Train/test split (stratified so label ratio is preserved)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Preprocess:
    # - numeric: median imputation
    # - categorical: most_frequent imputation + one-hot encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]), numeric_features),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features),
        ],
        remainder="drop"
    )

    # Model
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",   # helps if classes are imbalanced
        n_jobs=None
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", clf)
    ])

    model.fit(X_train, y_train)

    # Save model + split indices (so evaluation uses same test set)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "X_test": X_test,
            "y_test": y_test
        },
        MODEL_OUT
    )

    print("✅ Trained Logistic Regression baseline")
    print("✅ Saved model bundle:", MODEL_OUT.resolve())
    print("Features used:")
    print(" - numeric:", numeric_features)
    print(" - categorical:", categorical_features)
    print("Train size:", len(X_train), "| Test size:", len(X_test))
    print("Positive rate (overall):", f"{y.mean():.2%}")
    print("Positive rate (test):", f"{y_test.mean():.2%}")

if __name__ == "__main__":
    main()
