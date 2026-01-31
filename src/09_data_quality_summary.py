import pandas as pd
from pathlib import Path

IN_PATH = Path("data/processed/model_dataset_clean_v1.csv")

def main():
    df = pd.read_csv(IN_PATH)

    print("✅ Loaded:", IN_PATH)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print()

    # Missingness for labs
    lab_cols = ["creatinine", "bun", "alt", "ast", "bilirubin_total"]
    existing_labs = [c for c in lab_cols if c in df.columns]
    if existing_labs:
        print("=== % Missing lab values ===")
        for c in existing_labs:
            miss = df[c].isna().mean() * 100
            print(f"{c}: {miss:.2f}% missing")
        print()
    else:
        print("No lab columns found.\n")

    # Top drugs
    if "drug" in df.columns:
        print("=== Top 15 drugs by frequency ===")
        top = df["drug"].astype(str).value_counts().head(15)
        print(top.to_string())
        print()

    # Polypharmacy distribution
    if "polypharmacy_active_meds" in df.columns:
        print("=== Polypharmacy distribution ===")
        print(df["polypharmacy_active_meds"].describe().to_string())
        print()

    # Dose numeric distribution (if present)
    if "dose_val_rx_num" in df.columns:
        print("=== Dose numeric coverage ===")
        non_missing = df["dose_val_rx_num"].notna().mean() * 100
        print(f"dose_val_rx_num: {non_missing:.2f}% non-missing")
        print("Dose (numeric) summary (non-missing):")
        print(df["dose_val_rx_num"].dropna().describe().to_string())
        print()

if __name__ == "__main__":
    main()
