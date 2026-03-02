"""
Demo UI (Frontend) for Medication Error Risk Prediction (Hybrid Stacking).

Run:
    pip install streamlit joblib pandas scikit-learn xgboost
    streamlit run src/app/streamlit_demo_ui.py
"""

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st


DEFAULT_MODEL_PATH = Path("models/xgb_mlp_stacking_group_v1.joblib")
DEFAULT_DATA_PATH = Path("data/processed/model_dataset_labeled_v1.csv")

DEFAULT_NUMERIC_FEATURES = [
    "polypharmacy_active_meds",
    "anchor_age",
    "creatinine",
    "bun",
    "alt",
    "ast",
    "bilirubin_total",
    "dose_val_rx_num",
]
DEFAULT_CATEGORICAL_FEATURES = ["drug", "gender", "admission_type"]

st.set_page_config(page_title="Medication Error Risk Demo", layout="centered")


@st.cache_resource
def load_bundle(model_path: str):
    path = Path(model_path)
    bundle = joblib.load(path)
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle, bundle["model"]
    return {"model": bundle}, bundle


@st.cache_data
def load_reference_data(data_path: str) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def available_features(bundle: Dict) -> Tuple[List[str], List[str]]:
    numeric = bundle.get("numeric_features", DEFAULT_NUMERIC_FEATURES)
    categorical = bundle.get("categorical_features", DEFAULT_CATEGORICAL_FEATURES)
    return list(numeric), list(categorical)


def risk_band(prob: float) -> str:
    if prob < 0.30:
        return "Low"
    if prob < 0.70:
        return "Medium"
    return "High"


def get_dropdowns(df: pd.DataFrame, top_n_drugs: int = 200) -> Dict[str, List[str]]:
    out = {"drug": [], "gender": [], "admission_type": []}
    if df.empty:
        return out

    if "drug" in df.columns:
        out["drug"] = (
            df["drug"].astype(str).dropna().value_counts().head(top_n_drugs).index.tolist()
        )
    if "gender" in df.columns:
        out["gender"] = sorted(df["gender"].astype(str).dropna().unique().tolist())
    if "admission_type" in df.columns:
        out["admission_type"] = sorted(df["admission_type"].astype(str).dropna().unique().tolist())
    return out


def make_input_row(input_values: Dict[str, object], feature_order: List[str]) -> pd.DataFrame:
    row = {feature: input_values.get(feature, np.nan) for feature in feature_order}
    return pd.DataFrame([row])


def local_numeric_explain(
    X_input: pd.DataFrame, ref_df: pd.DataFrame, numeric_features: List[str]
) -> pd.DataFrame:
    """
    Demo-friendly local summary:
    compares numeric inputs against dataset medians (not model attribution).
    """
    rows = []
    if ref_df.empty:
        return pd.DataFrame(columns=["feature", "value", "reference_median", "direction_hint", "distance"])

    for feature in numeric_features:
        if feature not in X_input.columns or feature not in ref_df.columns:
            continue
        try:
            value = float(X_input.iloc[0][feature])
            ref_series = pd.to_numeric(ref_df[feature], errors="coerce").dropna()
            if ref_series.empty:
                continue
            median = float(ref_series.median())
            std = float(ref_series.std()) if float(ref_series.std()) > 0 else 1.0
            z = (value - median) / std
            direction = "higher than typical" if value > median else "lower than typical"
            rows.append(
                {
                    "feature": feature,
                    "value": round(value, 4),
                    "reference_median": round(median, 4),
                    "direction_hint": direction,
                    "distance": abs(z),
                }
            )
        except (TypeError, ValueError):
            continue

    if not rows:
        return pd.DataFrame(columns=["feature", "value", "reference_median", "direction_hint", "distance"])
    return pd.DataFrame(rows).sort_values("distance", ascending=False)


def try_base_probabilities(pipeline_model, X_input: pd.DataFrame):
    """
    Snapshot explanation for Hybrid Stacking:
    tries to extract probabilities from base models (xgb, mlp) inside the stacking classifier.

    Returns:
      (p_xgb, p_mlp, p_final)
    If extraction fails:
      (None, None, p_final)
    """
    # Final hybrid probability is always available
    p_final = float(pipeline_model.predict_proba(X_input)[:, 1][0])

    try:
        # Expecting Pipeline steps like: preprocess -> clf (StackingClassifier)
        pre = pipeline_model.named_steps.get("preprocess")
        stack = pipeline_model.named_steps.get("clf")

        if pre is None or stack is None:
            return None, None, p_final

        # Transform input to matrix used by base models
        Xt = pre.transform(X_input)

        # named_estimators_ is available after fit
        named = getattr(stack, "named_estimators_", None)
        if not named:
            return None, None, p_final

        xgb_est = named.get("xgb")
        mlp_est = named.get("mlp")

        p_xgb = float(xgb_est.predict_proba(Xt)[:, 1][0]) if xgb_est is not None else None
        p_mlp = float(mlp_est.predict_proba(Xt)[:, 1][0]) if mlp_est is not None else None

        return p_xgb, p_mlp, p_final
    except Exception:
        return None, None, p_final


def main():
    st.title("Medication Error Risk Prediction (Demo UI)")
    st.write(
        "Interactive prototype that loads your trained **Hybrid Stacking model (XGBoost + MLP)** "
        "and predicts **high-risk medication error probability**."
    )

    st.sidebar.header("Settings")
    model_path = st.sidebar.text_input("Model bundle path (.joblib)", str(DEFAULT_MODEL_PATH))
    data_path = st.sidebar.text_input("Dataset path (for dropdown/reference)", str(DEFAULT_DATA_PATH))
    threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
    st.sidebar.caption("Threshold changes only the High/Low class label (probability stays the same).")

    if not Path(model_path).exists():
        st.error(f"Model file not found: {model_path}")
        st.stop()

    try:
        bundle, model = load_bundle(model_path)
    except Exception as exc:
        st.error(f"Failed to load model bundle: {exc}")
        st.stop()

    numeric_features, categorical_features = available_features(bundle)
    feature_order = numeric_features + categorical_features

    ref_df = load_reference_data(data_path)
    dropdowns = get_dropdowns(ref_df)

    st.subheader("Enter input features")

    defaults_num = {
        "polypharmacy_active_meds": 8,
        "anchor_age": 55,
        "creatinine": 1.0,
        "bun": 12.0,
        "alt": 25.0,
        "ast": 22.0,
        "bilirubin_total": 0.8,
        "dose_val_rx_num": 1.0,
    }
    defaults_cat = {
        "gender": "M",
        "admission_type": "EMERGENCY",
        "drug": "VANCOMYCIN",
    }

    with st.form("predict_form"):
        input_values: Dict[str, object] = {}
        col1, col2 = st.columns(2)

        with col1:
            for feature in numeric_features[: (len(numeric_features) + 1) // 2]:
                val = st.number_input(
                    feature,
                    value=float(defaults_num.get(feature, 0.0)),
                    step=1.0 if feature in {"polypharmacy_active_meds", "anchor_age"} else 0.1,
                    min_value=0.0,
                )
                input_values[feature] = int(val) if feature in {"polypharmacy_active_meds", "anchor_age"} else float(val)

        with col2:
            for feature in numeric_features[(len(numeric_features) + 1) // 2 :]:
                val = st.number_input(
                    feature,
                    value=float(defaults_num.get(feature, 0.0)),
                    step=1.0 if feature in {"polypharmacy_active_meds", "anchor_age"} else 0.1,
                    min_value=0.0,
                )
                input_values[feature] = int(val) if feature in {"polypharmacy_active_meds", "anchor_age"} else float(val)

        for feature in categorical_features:
            options = dropdowns.get(feature, [])
            fallback = defaults_cat.get(feature, "UNKNOWN")
            if not options:
                options = [fallback]
            default_index = options.index(fallback) if fallback in options else 0
            input_values[feature] = st.selectbox(feature, options, index=default_index)

        submitted = st.form_submit_button("Predict risk")

    if submitted:
        X_input = make_input_row(input_values, feature_order)

        # Final probability (hybrid)
        try:
            proba = model.predict_proba(X_input)
            if proba.shape[1] < 2:
                raise ValueError(f"Unexpected predict_proba shape: {proba.shape}")
            prob = float(proba[:, 1][0])
        except Exception as exc:
            st.error(
                "Prediction failed. Check that the loaded model includes preprocessing and expects these feature names."
            )
            st.code(f"Error: {exc}")
            st.write("Columns sent to model:", list(X_input.columns))
            st.stop()

        pred = int(prob >= threshold)

        st.subheader("Prediction output")
        st.metric("Predicted risk probability (P=High Risk)", f"{prob:.4f}")
        st.write(f"**Risk band:** {risk_band(prob)}")
        st.write(f"**Predicted class (threshold={threshold:.2f}):** {'HIGH RISK (1)' if pred == 1 else 'LOW RISK (0)'}")
        st.caption("High/Low class is decided by: probability ≥ threshold.")

        with st.expander("Show input row sent to model"):
            st.dataframe(X_input)

        # -------- Hybrid snapshot explanation --------
        st.subheader("Hybrid snapshot explanation (Stacking)")
        p_xgb, p_mlp, p_final = try_base_probabilities(model, X_input)

        if p_xgb is None or p_mlp is None:
            st.info(
                "Base-model probabilities could not be extracted from the stacking object in this run. "
                "Final hybrid probability above is still correct."
            )
        else:
            st.write(f"- **XGBoost probability:** {p_xgb:.4f}")
            st.write(f"- **MLP probability:** {p_mlp:.4f}")
            st.write(f"- **Final stacked probability:** {p_final:.4f}")
            st.caption("The meta-learner combines base model probabilities (and features) to output the final risk.")

        # -------- Demo-friendly numeric explanation --------
        st.subheader("Explainability (demo-friendly numeric summary)")
        st.write(
            "This section gives a transparent local summary by comparing your numeric values "
            "against dataset medians. It helps interpret inputs, but it is not a causal explanation."
        )

        local_df = local_numeric_explain(X_input, ref_df, numeric_features)
        local_df = local_df.dropna(subset=["distance"]) if not local_df.empty else local_df

        if local_df.empty:
            st.info("Reference dataset not available (or missing numeric columns), so local summary is unavailable.")
        else:
            st.dataframe(local_df[["feature", "value", "reference_median", "direction_hint"]], hide_index=True)
            st.bar_chart(local_df.set_index("feature")["distance"])

        st.caption("For formal global explainability, use your scripts for coefficients and feature-importance reports.")


if __name__ == "__main__":
    main()