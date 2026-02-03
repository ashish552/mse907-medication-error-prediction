import numpy as np
import pandas as pd
from pathlib import Path
import joblib

MODEL_IN = Path("models/logreg_group_baseline_v1.joblib")
OUT_CSV = Path("reports/logreg_group_top_coefficients_v1.csv")

TOP_N = 25  # change to 20 if you want shorter output

def main():
    bundle = joblib.load(MODEL_IN)
    pipe = bundle["model"]

    # Get fitted preprocessor + classifier
    pre = pipe.named_steps["preprocess"]
    clf = pipe.named_steps["clf"]

    # Feature names from ColumnTransformer (works with sklearn >=1.0)
    feature_names = pre.get_feature_names_out()

    coefs = clf.coef_.ravel()  # binary classifier => shape (1, n_features)

    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs)
    }).sort_values("abs_coefficient", ascending=False)

    # Save full coefficients (sorted by magnitude)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # Print summary
    print("=== Logistic Regression Explainability (GROUP SPLIT) ===")
    print("Model:", MODEL_IN)
    print("Saved coefficients to:", OUT_CSV.resolve())
    print("\nTop features pushing risk UP (positive coefficients):")
    up = df.sort_values("coefficient", ascending=False).head(TOP_N)
    print(up[["feature", "coefficient"]].to_string(index=False))

    print("\nTop features pushing risk DOWN (negative coefficients):")
    down = df.sort_values("coefficient", ascending=True).head(TOP_N)
    print(down[["feature", "coefficient"]].to_string(index=False))

    # Optional: show if polypharmacy dominates
    poly = df[df["feature"].str.contains("polypharmacy_active_meds", na=False)]
    if len(poly) > 0:
        print("\nPolypharmacy coefficient:")
        print(poly[["feature", "coefficient"]].to_string(index=False))

if __name__ == "__main__":
    main()
