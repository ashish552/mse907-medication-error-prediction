import pandas as pd
from pathlib import Path

BASE = Path("data/raw/mimic_demo_2.2")
OUT_PATH = Path("data/processed/base_rx_clean.csv")

def get_dataset_root(base: Path) -> Path:
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

def to_dt(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

def main():
    root = get_dataset_root(BASE)
    print("✅ Using dataset root:", root.resolve())

    patients = read_table(root, "hosp/patients.csv.gz")
    admissions = read_table(root, "hosp/admissions.csv.gz")
    rx = read_table(root, "hosp/prescriptions.csv.gz")

    # Convert datetime columns
    to_dt(admissions, ["admittime", "dischtime", "deathtime"])
    to_dt(rx, ["starttime", "stoptime"])

    # Keep a focused set of columns for v1
    keep_patients = [c for c in ["subject_id", "gender", "anchor_age", "anchor_year"] if c in patients.columns]
    keep_adm = [c for c in ["subject_id", "hadm_id", "admittime", "dischtime", "admission_type"] if c in admissions.columns]
    keep_rx = [c for c in ["subject_id", "hadm_id", "starttime", "stoptime", "drug", "route",
                           "dose_val_rx", "dose_unit_rx", "form_val_disp", "form_unit_disp"] if c in rx.columns]

    patients = patients[keep_patients].copy()
    admissions = admissions[keep_adm].copy()
    rx = rx[keep_rx].copy()

    # Basic cleaning: standardize drug names
    if "drug" in rx.columns:
        rx["drug"] = rx["drug"].astype(str).str.strip().str.upper()
        rx.loc[rx["drug"].isin(["NAN", "NONE", ""]), "drug"] = pd.NA

    before = len(rx)

    # Drop rows missing critical info
    rx = rx.dropna(subset=["subject_id", "hadm_id"])
    if "starttime" in rx.columns:
        rx = rx.dropna(subset=["starttime"])
    if "drug" in rx.columns:
        rx = rx.dropna(subset=["drug"])

    after = len(rx)
    print(f"🧹 Cleaned prescriptions: {before} -> {after} rows")

    # Join tables
    df = rx.merge(admissions, on=["subject_id", "hadm_id"], how="left")
    df = df.merge(patients, on=["subject_id"], how="left")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("✅ Saved:", OUT_PATH.resolve())
    print("Final rows:", len(df), "| cols:", len(df.columns))
    print(df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
