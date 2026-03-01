from pathlib import Path
import csv
import argparse

DEFAULT_MD = Path("reports/model_comparison_v3.md")
DEFAULT_CSV = Path("reports/model_comparison_v3.csv")
OUT_BAR = Path("reports/figures/model_comparison_bar_chart_v1.svg")
OUT_LINE = Path("reports/figures/model_comparison_line_chart_v1.svg")
METRICS = ["AUROC", "AUPRC", "Precision", "Recall", "F1"]
COLORS = {
    "AUROC": "#1f77b4",
    "AUPRC": "#ff7f0e",
    "Precision": "#2ca02c",
    "Recall": "#d62728",
    "F1": "#9467bd",
}


def _parse_markdown_table(path: Path):
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows = []
    for line in lines:
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if not parts:
            continue
        if parts[0] == "Model":
            continue
        if set("".join(parts)) <= {"-", ":"}:
            continue
        if len(parts) != 6:
            continue
        row = {"Model": parts[0]}
        for k, v in zip(METRICS, parts[1:]):
            row[k] = float(v)
        rows.append(row)
    if not rows:
        raise ValueError(f"No rows parsed from markdown table: {path}")
    return rows


def _parse_csv(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            row = {"Model": r["Model"]}
            for m in METRICS:
                row[m] = float(r[m])
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows parsed from csv: {path}")
    return rows


def load_rows(csv_path: Path, md_path: Path):
    if csv_path.exists():
        return _parse_csv(csv_path), str(csv_path)
    if md_path.exists():
        return _parse_markdown_table(md_path), str(md_path)
    raise FileNotFoundError("No model comparison file found (CSV or MD).")


def _scale(value, y_min, y_max, plot_h):
    # map score to SVG y offset from top of plot
    return plot_h * (1.0 - (value - y_min) / (y_max - y_min))


def plot_bar_svg(rows, out_path: Path):
    width, height = 1200, 700
    left, top, right, bottom = 120, 70, 40, 140
    plot_w = width - left - right
    plot_h = height - top - bottom

    y_min, y_max = 0.9, 1.01
    model_count = len(rows)
    group_w = plot_w / model_count
    bar_w = group_w / (len(METRICS) + 1)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{width/2}" y="35" text-anchor="middle" font-size="24" font-family="Arial">Model Comparison - Bar Chart</text>')

    # axes
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="black"/>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="black"/>')

    # y ticks
    for t in [0.90, 0.92, 0.94, 0.96, 0.98, 1.00]:
        y = top + _scale(t, y_min, y_max, plot_h)
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left-10}" y="{y+5:.1f}" text-anchor="end" font-size="12" font-family="Arial">{t:.2f}</text>')

    # bars
    for i, row in enumerate(rows):
        group_x = left + i * group_w
        for j, metric in enumerate(METRICS):
            val = row[metric]
            h = plot_h - _scale(val, y_min, y_max, plot_h)
            x = group_x + j * bar_w + bar_w * 0.5
            y = top + plot_h - h
            color = COLORS[metric]
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.8:.1f}" height="{h:.1f}" fill="{color}"/>')

        # model label
        label_x = group_x + group_w / 2
        label_y = top + plot_h + 65
        parts.append(
            f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="end" transform="rotate(-25 {label_x:.1f} {label_y:.1f})" font-size="12" font-family="Arial">{row["Model"]}</text>'
        )

    # legend
    lx, ly = width - 250, 90
    for idx, metric in enumerate(METRICS):
        yy = ly + idx * 24
        parts.append(f'<rect x="{lx}" y="{yy-10}" width="14" height="14" fill="{COLORS[metric]}"/>')
        parts.append(f'<text x="{lx+22}" y="{yy+2}" font-size="12" font-family="Arial">{metric}</text>')

    parts.append('</svg>')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def plot_line_svg(rows, out_path: Path):
    width, height = 1200, 700
    left, top, right, bottom = 120, 70, 40, 140
    plot_w = width - left - right
    plot_h = height - top - bottom

    y_min, y_max = 0.9, 1.01
    model_count = len(rows)
    step_x = plot_w / max(1, model_count - 1)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{width/2}" y="35" text-anchor="middle" font-size="24" font-family="Arial">Model Comparison - Line Chart</text>')

    # axes
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="black"/>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="black"/>')

    # y ticks
    for t in [0.90, 0.92, 0.94, 0.96, 0.98, 1.00]:
        y = top + _scale(t, y_min, y_max, plot_h)
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left-10}" y="{y+5:.1f}" text-anchor="end" font-size="12" font-family="Arial">{t:.2f}</text>')

    # x labels
    for i, row in enumerate(rows):
        x = left + i * step_x
        y = top + plot_h + 65
        parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="end" transform="rotate(-25 {x:.1f} {y:.1f})" font-size="12" font-family="Arial">{row["Model"]}</text>'
        )

    # lines
    for metric in METRICS:
        points = []
        for i, row in enumerate(rows):
            x = left + i * step_x
            y = top + _scale(row[metric], y_min, y_max, plot_h)
            points.append((x, y))
        path = " ".join([f"{x:.1f},{y:.1f}" for x, y in points])
        color = COLORS[metric]
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{path}"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')

    # legend
    lx, ly = width - 250, 90
    for idx, metric in enumerate(METRICS):
        yy = ly + idx * 24
        parts.append(f'<line x1="{lx}" y1="{yy-3}" x2="{lx+18}" y2="{yy-3}" stroke="{COLORS[metric]}" stroke-width="3"/>')
        parts.append(f'<text x="{lx+24}" y="{yy+1}" font-size="12" font-family="Arial">{metric}</text>')

    parts.append('</svg>')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Plot model comparison charts (SVG, no external deps).")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--out-bar", type=Path, default=OUT_BAR)
    parser.add_argument("--out-line", type=Path, default=OUT_LINE)
    args = parser.parse_args()

    rows, source = load_rows(args.csv, args.md)
    plot_bar_svg(rows, args.out_bar)
    plot_line_svg(rows, args.out_line)

    print("Loaded model comparison from:", source)
    print("✅ Saved bar chart:", args.out_bar.resolve())
    print("✅ Saved line chart:", args.out_line.resolve())


if __name__ == "__main__":
    main()