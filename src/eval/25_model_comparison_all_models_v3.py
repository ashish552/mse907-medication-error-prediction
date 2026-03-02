import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METRIC_FILES = {
    "LogReg (Group Split)": Path("reports/metrics_logreg_group_baseline_v1.txt"),
    "Random Forest (Group Split)": Path("reports/metrics_rf_group_v1.txt"),
    "XGBoost (Group Split)": Path("reports/metrics_xgb_group_v1.txt"),
    "XGBoost + MLP Stacking (Group Split)": Path("reports/metrics_xgb_mlp_stacking_group_v1.txt"),
}

OUT_CSV = Path("reports/model_comparison_v3.csv")
OUT_MD = Path("reports/model_comparison_v3.md")
OUT_BAR = Path("reports/figures/model_comparison_bar_chart_v3.png")
OUT_LINE = Path("reports/figures/model_comparison_line_chart_v3.png")
OUT_HEATMAP = Path("reports/figures/model_comparison_correlation_heatmap_v3.png")
OUT_F1_HBAR = Path("reports/figures/model_comparison_f1_hbar_v3.png")
OUT_PREC_HBAR = Path("reports/figures/model_comparison_precision_hbar_v3.png")

METRIC_KEYS = ["AUROC", "AUPRC", "Precision", "Recall", "F1"]


def parse_metrics(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")

    def grab(key: str):
        m = re.search(rf"{key}:\s*([0-9.]+)", text)
        return float(m.group(1)) if m else np.nan

    return {k: grab(k) for k in METRIC_KEYS}


def build_comparison_table() -> pd.DataFrame:
    rows = []
    missing = []

    for model_name, metric_path in METRIC_FILES.items():
        if not metric_path.exists():
            missing.append((model_name, str(metric_path)))
            continue

        metrics = parse_metrics(metric_path)
        rows.append({"Model": model_name, **metrics})

    if not rows:
        raise FileNotFoundError(
            "No metrics files found. Run evaluation scripts first. Expected files:\n"
            + "\n".join(str(p) for p in METRIC_FILES.values())
        )

    df = pd.DataFrame(rows)

    # Keep row order aligned with METRIC_FILES order
    model_order = [m for m in METRIC_FILES.keys() if m in set(df["Model"])]
    df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
    df = df.sort_values("Model").reset_index(drop=True)

    if missing:
        print("⚠️ Missing metric files (these models were skipped):")
        for model_name, metric_path in missing:
            print(f" - {model_name}: {metric_path}")

    return df


def save_table_outputs(df: pd.DataFrame) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    md = df.to_markdown(index=False, floatfmt=".4f")
    OUT_MD.write_text(md, encoding="utf-8")

    print("✅ Saved table CSV:", OUT_CSV.resolve())
    print("✅ Saved table MD:", OUT_MD.resolve())
    print("\n=== Model Comparison (v3) ===")
    print(md)


def plot_bar_chart(df: pd.DataFrame) -> None:
    metric_colors = {
        "AUROC": "#1f77b4",
        "AUPRC": "#ff7f0e",
        "Precision": "#2ca02c",
        "Recall": "#d62728",
        "F1": "#9467bd",
    }

    x = np.arange(len(df))
    width = 0.15

    plt.figure(figsize=(14, 7))
    for i, m in enumerate(METRIC_KEYS):
        plt.bar(x + (i - 2) * width, df[m].astype(float), width=width, label=m, color=metric_colors[m])

    plt.xticks(x, df["Model"], rotation=20, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    plt.title("Model Comparison (v3) - Bar Chart")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    OUT_BAR.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_BAR, dpi=200)
    plt.close()
    print("✅ Saved bar chart:", OUT_BAR.resolve())


def plot_line_chart(df: pd.DataFrame) -> None:
    metric_colors = {
        "AUROC": "#1f77b4",
        "AUPRC": "#ff7f0e",
        "Precision": "#2ca02c",
        "Recall": "#d62728",
        "F1": "#9467bd",
    }

    plt.figure(figsize=(14, 7))
    for m in METRIC_KEYS:
        plt.plot(df["Model"], df[m].astype(float), marker="o", linewidth=2, color=metric_colors[m], label=m)

    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    plt.title("Model Comparison (v3) - Line Chart")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()

    OUT_LINE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_LINE, dpi=200)
    plt.close()
    print("✅ Saved line chart:", OUT_LINE.resolve())


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    metric_df = df[METRIC_KEYS].astype(float)
    corr = metric_df.corr()

    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(np.arange(len(METRIC_KEYS)), METRIC_KEYS, rotation=30, ha="right")
    plt.yticks(np.arange(len(METRIC_KEYS)), METRIC_KEYS)
    plt.title("Metric Correlation Heatmap (v3)")

    for i in range(len(METRIC_KEYS)):
        for j in range(len(METRIC_KEYS)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")

    plt.tight_layout()

    OUT_HEATMAP.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_HEATMAP, dpi=200)
    plt.close()
    print("✅ Saved correlation heatmap:", OUT_HEATMAP.resolve())


def plot_single_metric_horizontal_bar(df: pd.DataFrame, metric: str, out_path: Path, color: str) -> None:
    plot_df = df[["Model", metric]].copy().sort_values(metric, ascending=True)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(plot_df["Model"], plot_df[metric].astype(float), color=color)

    plt.xlim(0.0, 1.05)
    plt.xlabel(metric)
    plt.title(f"{metric} Comparison Across Models (v3)")
    plt.grid(axis="x", linestyle="--", alpha=0.3)

    for bar, val in zip(bars, plot_df[metric].astype(float)):
        plt.text(min(val + 0.01, 1.02), bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✅ Saved {metric} horizontal bar chart:", out_path.resolve())


def main():
    df = build_comparison_table()
    save_table_outputs(df)
    plot_bar_chart(df)
    plot_line_chart(df)
    plot_correlation_heatmap(df)
    plot_single_metric_horizontal_bar(df, metric="F1", out_path=OUT_F1_HBAR, color="#9467bd")
    plot_single_metric_horizontal_bar(df, metric="Precision", out_path=OUT_PREC_HBAR, color="#2ca02c")


if __name__ == "__main__":
    main()