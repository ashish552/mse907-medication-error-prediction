import pandas as pd
from pathlib import Path
import inspect

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
import joblib

IN_PATH = Path("data/processed/model_dataset_labeled_v1.csv")
MODEL_OUT = Path("models/xgb_mlp_stacking_group_v1.joblib")

RANDOM_STATE = 42


def make_dense_onehot_encoder() -> OneHotEncoder:
    """
    Return a dense OneHotEncoder across scikit-learn versions.

    - Older sklearn uses: sparse=False
    - Newer sklearn uses: sparse_output=False
    """
    params = inspect.signature(OneHotEncoder).parameters
    if "sparse_output" in params:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def main():
    df = pd.read_csv(IN_PATH)

    if "label_high_risk" not in df.columns:
        raise ValueError("label_high_risk not found. Run proxy label script first.")
    if "hadm_id" not in df.columns:
        raise ValueError("hadm_id not found. Group split requires hadm_id.")

    y = df["label_high_risk"].astype(int)

    numeric_features = [
        "polypharmacy_active_meds",
        "anchor_age",
        "creatinine", "bun", "alt", "ast", "bilirubin_total",
        "dose_val_rx_num",
    ]
    numeric_features = [c for c in numeric_features if c in df.columns]

    categorical_features = ["drug", "gender", "admission_type"]
    categorical_features = [c for c in categorical_features if c in df.columns]

    feature_cols = numeric_features + categorical_features
    X = df[feature_cols].copy()
    groups = df["hadm_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Dense preprocessing so both MLP and XGBoost can consume the same transformed matrix.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", make_dense_onehot_encoder()),
                ]),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    # Handle class imbalance for XGBoost.
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    xgb = XGBClassifier(
        n_estimators=700,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )

    # "Deep learning" component for tabular data (multi-layer neural net).
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
    )

    stack = StackingClassifier(
        estimators=[
            ("xgb", xgb),
            ("mlp", mlp),
        ],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
        stack_method="predict_proba",
        passthrough=True,
        n_jobs=-1,
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", stack),
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
            "split_type": "GroupShuffleSplit(hadm_id)",
            "model_type": "Stacking(XGBoost + MLP)",
        },
        MODEL_OUT,
    )

    print("✅ Trained stacking model: XGBoost + MLP (GROUP SPLIT by hadm_id)")
    print("✅ Saved model bundle:", MODEL_OUT.resolve())
    print("Train size:", len(X_train), "| Test size:", len(X_test))
    print("Positive rate (train):", f"{y_train.mean():.2%}")
    print("Positive rate (test):", f"{y_test.mean():.2%}")
    print("scale_pos_weight (XGB):", round(scale_pos_weight, 3))
    print("Features used:")
    print(" - numeric:", numeric_features)
    print(" - categorical:", categorical_features)


if __name__ == "__main__":
    main()