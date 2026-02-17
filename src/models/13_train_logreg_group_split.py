import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

import joblib

IN_PATH = Path("data/processed/model_dataset_labeled_v1.csv")
MODEL_OUT = Path("models/logreg_group_baseline_v1.joblib")

RANDOM_STATE = 42

def main():
    df = pd.read_csv(IN_PATH)

    if "label_high_risk" not in df.columns:
        raise ValueError("label_high_risk not found. Run proxy label script first.")
    if "hadm_id" not in df.columns:
        raise ValueError("hadm_id not found. Group split requires hadm_id.")

    y = df["label_high_risk"].astype(int)

    # Features (same as before)
    numeric_features = [
        "polypharmacy_active_meds",
        "anchor_age",
        "creatinine", "bun", "alt", "ast", "bilirubin_total",
        "dose_val_rx_num"
    ]
    numeric_features = [c for c in numeric_features if c in df.columns]

    categorical_features = ["drug", "gender", "admission_type"]
    categorical_features = [c for c in categorical_features if c in df.columns]

    feature_cols = numeric_features + categorical_features
    X = df[feature_cols].copy()

    groups = df["hadm_id"]

    # GROUP split (by hadm_id)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Preprocess:
    # - numeric: median impute + standard scale
    # - categorical: most_frequent + one-hot
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_features),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features),
        ],
        remainder="drop"
    )

    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs"
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", clf)
    ])

    model.fit(X_train, y_train)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "X_test": X_test,
            "y_test": y_test,
            "split_type": "GroupShuffleSplit(hadm_id)"
        },
        MODEL_OUT
    )

    print("✅ Trained Logistic Regression (GROUP SPLIT by hadm_id)")
    print("✅ Saved model bundle:", MODEL_OUT.resolve())
    print("Train size:", len(X_train), "| Test size:", len(X_test))
    print("Positive rate (train):", f"{y_train.mean():.2%}")
    print("Positive rate (test):", f"{y_test.mean():.2%}")
    print("Features used:")
    print(" - numeric:", numeric_features)
    print(" - categorical:", categorical_features)

if __name__ == "__main__":
    main()
