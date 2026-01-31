import pandas as pd
import numpy as np
from pathlib import Path

IN_PATH = Path("data/processed/base_rx_with_labs.csv")
OUT_PATH = Path("data/processed/model_dataset_clean_v1.csv")

def clean_numeric(series: pd.Series) -> pd.Series:
    # Convert common messy numeric strings to float
    s = series.astype(str).str.strip()
    s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    # Keep only numeric-looking content
    return pd.to_numeric(s, errors="coerce")

def main():
    df = pd.read_csv(IN_PATH)

    # Parse times if loaded as strings
    for c in ["starttime", "stoptime", "admittime", "dischtime"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    before = len(df)

    # 1) Drop rows missing essentials
    df = df.dropna(subset=["hadm_id", "subject_id", "starttime", "drug"]).copy()

    # 2) Remove exact duplicates (common in prescriptions)
    # key chosen for "same prescription event"
    dup_key = ["hadm_id", "starttime", "drug"]
    dups_before = df.duplicated(subset=dup_key).sum()
    df = df.drop_duplicates(subset=dup_key).copy()

    # 3) Clean dose values if present
    if "dose_val_rx" in df.columns:
        df["dose_val_rx_num"] = clean_numeric(df["dose_val_rx"])
        df["dose_val_rx_missing"] = df["dose_val_rx_num"].isna().astype(int)

    # 4) Basic outlier flags (not removing, just flagging)
    # (you can later use these as features or exclude in modeling)
    if "creatinine" in df.columns:
        df["creatinine_outlier_flag"] = ((df["creatinine"] < 0) | (df["creatinine"] > 20)).astype(int)

    if "bun" in df.columns:
        df["bun_outlier_flag"] = ((df["bun"] < 0) | (df["bun"] > 200)).astype(int)

    after = len(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("✅ Saved:", OUT_PATH.resolve())
    print(f"Rows: {before} -> {after}")
    print(f"Duplicates removed (by {dup_key}): {dups_before}")
    if "dose_val_rx_num" in df.columns:
        print("Dose numeric coverage (non-missing):",
              f"{(1-df['dose_val_rx_missing'].mean()):.2%}")
    print("Preview:")
    cols = [c for c in ["subject_id", "hadm_id", "starttime", "drug", "polypharmacy_active_meds",
                        "creatinine", "bun", "alt", "ast", "bilirubin_total",
                        "dose_val_rx", "dose_val_rx_num"] if c in df.columns]
    print(df[cols].head(8).to_string(index=False))

if __name__ == "__main__":
    main()
