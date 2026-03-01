import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

MODEL_IN = Path("models/xgb_group_v1.joblib")
OUT_CSV = Path("reports/xgb_group_feature_importance_v1.csv")
OUT_PNG = Path("reports/figures/xgb_group_feature_importance_v1.png")

TOP_N = 25

def main():
    if not MODEL_IN.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_IN}. Run Step 20 first.")

    bundle = joblib.load(MODEL_IN)
    pipe = bundle["model"]

    pre = pipe.named_steps["preprocess"]
    xgb = pipe.named_steps["clf"]

    feature_names = pre.get_feature_names_out()

    # XGBoost feature importance (gain-based)
    booster = xgb.get_booster()
    score = booster.get_score(importance_type="gain")

    # Map f0,f1,... -> actual feature names
    rows = []
    for k, v in score.items():
        idx = int(k[1:])  # "f12" -> 12
        if idx < len(feature_names):
            rows.append((feature_names[idx], float(v)))

    df = pd.DataFrame(rows, columns=["feature", "gain"]).sort_values("gain", ascending=False)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("=== XGBoost Feature Importance (GAIN) - GROUP v1 ===")
    print("Saved CSV:", OUT_CSV.resolve())
    print("\nTop features:")
    print(df.head(TOP_N).to_string(index=False))

    # Plot top N
    top = df.head(TOP_N).iloc[::-1]
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["gain"])
    plt.title(f"XGBoost Feature Importance (GAIN) Top {TOP_N} - GROUP v1")
    plt.xlabel("Gain")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)

    print("\n✅ Saved plot:", OUT_PNG.resolve())

if __name__ == "__main__":
    main()
