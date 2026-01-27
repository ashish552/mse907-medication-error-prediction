import pandas as pd
from pathlib import Path

BASE = Path("data/raw/mimic_demo_2.2")

def get_dataset_root(base: Path) -> Path:
    # data/raw/mimic_demo_2.2 might contain nested folder
    if (base / "hosp").exists() or (base / "icu").exists():
        return base
    for c in base.iterdir():
        if c.is_dir() and ((c / "hosp").exists() or (c / "icu").exists()):
            return c
    return base

def read_table(root: Path, rel_path: str) -> pd.DataFrame:
    path = root / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def main():
    root = get_dataset_root(BASE)
    print("✅ Using dataset root:", root.resolve())

    patients = read_table(root, "hosp/patients.csv.gz")
    admissions = read_table(root, "hosp/admissions.csv.gz")
    prescriptions = read_table(root, "hosp/prescriptions.csv.gz")

    print("\n--- Row counts ---")
    print("patients:", len(patients), "| unique subject_id:", patients["subject_id"].nunique())
    print("admissions:", len(admissions), "| unique hadm_id:", admissions["hadm_id"].nunique())
    print("prescriptions:", len(prescriptions), "| unique hadm_id:", prescriptions["hadm_id"].nunique())

    print("\n--- Columns (first 25) ---")
    print("patients:", list(patients.columns)[:25])
    print("admissions:", list(admissions.columns)[:25])
    print("prescriptions:", list(prescriptions.columns)[:25])

if __name__ == "__main__":
    main()
