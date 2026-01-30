import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("data/raw/mimic_demo_2.2")
RX_IN = Path("data/processed/base_rx_polypharm.csv")
OUT = Path("data/processed/base_rx_with_labs.csv")

# Lab label keywords to match in d_labitems.label
LAB_TARGETS = {
    "creatinine": ["CREATININE"],
    "bun": ["UREA NITROGEN", "BUN"],
    "alt": ["ALT", "ALANINE AMINOTRANSFERASE"],
    "ast": ["AST", "ASPARTATE AMINOTRANSFERASE"],
    "bilirubin_total": ["BILIRUBIN, TOTAL", "BILIRUBIN TOTAL"],
}

def get_dataset_root(base: Path) -> Path:
    """Handle nested folder structure after zip extraction."""
    if (base / "hosp").exists() or (base / "icu").exists():
        return base
    for c in base.iterdir():
        if c.is_dir() and ((c / "hosp").exists() or (c / "icu").exists()):
            return c
    return base

def add_latest_lab_per_hadm(rx: pd.DataFrame, lab: pd.DataFrame, feat: str) -> pd.DataFrame:
    """
    For each hadm_id, attach latest lab value with charttime <= starttime.
    Uses merge_asof per hadm_id to avoid global sorting issues.
    """
    rx_feat = pd.Series(np.nan, index=rx.index, dtype="float64")
    rx_since = pd.Series(np.nan, index=rx.index, dtype="float64")

    for hadm_id, idx in rx.groupby("hadm_id", sort=False).groups.items():
        g_rx = rx.loc[idx].sort_values("starttime")
        g_lab = lab[lab["hadm_id"] == hadm_id].sort_values("charttime")

        if g_lab.empty:
            continue

        merged = pd.merge_asof(
            g_rx[["starttime"]],
            g_lab[["charttime", "valuenum"]],
            left_on="starttime",
            right_on="charttime",
            direction="backward",
            allow_exact_matches=True,
        )

        rx_feat.loc[g_rx.index] = merged["valuenum"].to_numpy()
        rx_since.loc[g_rx.index] = (
            (g_rx["starttime"].to_numpy() - merged["charttime"].to_numpy())
            / pd.Timedelta(hours=1)
        )

    rx[feat] = rx_feat
    rx[f"{feat}_hours_since"] = rx_since
    rx[f"{feat}_missing"] = rx[feat].isna().astype(int)
    return rx

def main():
    root = get_dataset_root(BASE)
    print("✅ Using dataset root:", root.resolve())

    # Load prescription dataset (with polypharmacy already)
    rx = pd.read_csv(RX_IN)
    rx["starttime"] = pd.to_datetime(rx["starttime"], errors="coerce")

    # Ensure merge keys are clean
    rx = rx.dropna(subset=["hadm_id", "starttime"]).copy()
    rx["hadm_id"] = rx["hadm_id"].astype("int64")
    rx = rx.sort_values(["hadm_id", "starttime"]).reset_index(drop=True)

    # Load lab dictionary
    d_lab = pd.read_csv(root / "hosp/d_labitems.csv.gz")
    d_lab["label_up"] = d_lab["label"].astype(str).str.upper()

    # Identify itemids for each target lab group
    target_itemids = {}
    for feat, keywords in LAB_TARGETS.items():
        mask = False
        for kw in keywords:
            mask = mask | d_lab["label_up"].str.contains(kw, na=False)
        itemids = d_lab.loc[mask, "itemid"].dropna().unique().tolist()
        target_itemids[feat] = itemids
        print(f"🔎 {feat}: found {len(itemids)} itemids")

    all_itemids = sorted({iid for ids in target_itemids.values() for iid in ids})
    if not all_itemids:
        raise RuntimeError("No lab itemids matched. Check d_labitems labels/keywords.")

    # Load labevents (only columns we need)
    labs = pd.read_csv(
        root / "hosp/labevents.csv.gz",
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
    )
    labs["charttime"] = pd.to_datetime(labs["charttime"], errors="coerce")

    # Filter to relevant itemids + remove missing
    labs = labs[labs["itemid"].isin(all_itemids)].copy()
    labs = labs.dropna(subset=["hadm_id", "charttime", "valuenum"]).copy()
    labs["hadm_id"] = labs["hadm_id"].astype("int64")
    labs = labs.sort_values(["hadm_id", "charttime"]).reset_index(drop=True)

    print("✅ Filtered labevents rows:", len(labs))

    out = rx.copy()

    # Add each lab feature group
    for feat, itemids in target_itemids.items():
        if not itemids:
            out[feat] = np.nan
            out[f"{feat}_hours_since"] = np.nan
            out[f"{feat}_missing"] = 1
            continue

        lab_g = labs[labs["itemid"].isin(itemids)].copy()
        out = add_latest_lab_per_hadm(out, lab_g, feat)
        print(f"✅ Added feature: {feat} (missing={out[f'{feat}_missing'].mean():.2%})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print("\n✅ Saved:", OUT.resolve())
    cols_preview = ["subject_id", "hadm_id", "starttime", "drug",
                    "creatinine", "bun", "alt", "ast", "bilirubin_total"]
    existing = [c for c in cols_preview if c in out.columns]
    print(out[existing].head(5).to_string(index=False))

if __name__ == "__main__":
    main()
