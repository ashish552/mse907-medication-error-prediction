import re
from pathlib import Path
import pandas as pd

METRICS_FILES = [
    Path("reports/metrics_logreg_group_baseline_v1.txt"),
    Path("reports/metrics_rf_group_v1.txt"),
    Path("reports/metrics_xgb_group_v1.txt"),
]

OUT_CSV = Path("reports/model_comparison_v2.csv")
OUT_MD = Path("reports/model_comparison_v2.md")

NAME_MAP = {
    "metrics_logreg_group_baseline_v1": "LogReg (Group Split)",
    "metrics_rf_group_v1": "Random Forest (Group Split)",
    "metrics_xgb_group_v1": "XGBoost (Group Split)",
}

def parse_metrics(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")

    def grab(key):
        m = re.search(rf"{key}:\s*([0-9.]+)", text)
        return float(m.group(1)) if m else None

    stem = path.stem
    return {
        "Model": NAME_MAP.get(stem, stem),
        "AUROC": grab("AUROC"),
        "AUPRC": grab("AUPRC"),
        "Precision": grab("Precision"),
        "Recall": grab("Recall"),
        "F1": grab("F1"),
    }

def main():
    for f in METRICS_FILES:
        if not f.exists():
            raise FileNotFoundError(f"Missing metrics file: {f}")

    df = pd.DataFrame([parse_metrics(f) for f in METRICS_FILES])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # Write markdown manually (no tabulate needed)
    md = "| " + " | ".join(df.columns) + " |\n"
    md += "| " + " | ".join(["---"] * len(df.columns)) + " |\n"
    for _, row in df.iterrows():
        md += "| " + " | ".join([str(x) for x in row.values]) + " |\n"

    OUT_MD.write_text(md, encoding="utf-8")

    print("=== Model Comparison (v2) ===")
    print(md)
    print("\n✅ Saved CSV:", OUT_CSV.resolve())
    print("✅ Saved MD:", OUT_MD.resolve())

if __name__ == "__main__":
    main()
