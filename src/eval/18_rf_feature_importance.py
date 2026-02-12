import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

MODEL_IN = Path("models/rf_group_v1.joblib")
OUT_CSV = Path("reports/rf_group_feature_importance_v1.csv")
OUT_PNG = Path("reports/figures/rf_group_feature_importance_v1.png")

TOP_N = 25  # show top 25 features

def main():
    if not MODEL_IN.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_IN}. Run Step 16 first.")

    bundle = joblib.load(MODEL_IN)
    pipe = bundle["model"]

    pre = pipe.named_steps["preprocess"]
    rf = pipe.named_steps["clf"]

    # Get transformed feature names (numeric + one-hot encoded categorical)
    feature_names = pre.get_feature_names_out()

    importances = rf.feature_importances_
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    # Save full importance table
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # Print top N
    print("=== Random Forest Feature Importance (GROUP SPLIT v1) ===")
    print("Saved CSV:", OUT_CSV.resolve())
    print("\nTop features:")
    print(df.head(TOP_N).to_string(index=False))

    # Plot top N
    top = df.head(TOP_N).iloc[::-1]  # reverse for nicer bar chart
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["importance"])
    plt.title(f"RF Feature Importance (Top {TOP_N}) - GROUP v1")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)

    print("\n✅ Saved plot:", OUT_PNG.resolve())

if __name__ == "__main__":
    main()
