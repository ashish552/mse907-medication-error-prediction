import re
from pathlib import Path
import pandas as pd

LOGREG_METRICS = Path("reports/metrics_logreg_group_baseline_v1.txt")
RF_METRICS = Path("reports/metrics_rf_group_v1.txt")

OUT_CSV = Path("reports/model_comparison_v1.csv")
OUT_MD = Path("reports/model_comparison_v1.md")

def parse_metrics(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    def grab(key):
        m = re.search(rf"{key}:\s*([0-9.]+)", text)
        return float(m.group(1)) if m else None

    return {
        "Model": path.stem.replace("metrics_", ""),
        "AUROC": grab("AUROC"),
        "AUPRC": grab("AUPRC"),
        "Precision": grab("Precision"),
        "Recall": grab("Recall"),
        "F1": grab("F1"),
    }

def main():
    if not LOGREG_METRICS.exists():
        raise FileNotFoundError(f"Missing: {LOGREG_METRICS}")
    if not RF_METRICS.exists():
        raise FileNotFoundError(f"Missing: {RF_METRICS}")

    rows = [parse_metrics(LOGREG_METRICS), parse_metrics(RF_METRICS)]
    df = pd.DataFrame(rows)

    # Make model names nicer
    df["Model"] = df["Model"].replace({
        "logreg_group_baseline_v1": "LogReg (Group Split)",
        "rf_group_v1": "Random Forest (Group Split)"
    })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # Markdown table for easy copy/paste into report
    md = df.to_markdown(index=False)
    OUT_MD.write_text(md, encoding="utf-8")

    print("=== Model Comparison (v1) ===")
    print(md)
    print("\n✅ Saved CSV:", OUT_CSV.resolve())
    print("✅ Saved MD:", OUT_MD.resolve())

if __name__ == "__main__":
    main()
