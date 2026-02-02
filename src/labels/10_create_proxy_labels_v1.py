import pandas as pd
from pathlib import Path

IN_PATH = Path("data/processed/model_dataset_clean_v1.csv")
OUT_PATH = Path("data/processed/model_dataset_labeled_v1.csv")

# Rule settings (v1)
POLYPHARM_THRESHOLD = 40          # defensible: above your 75th percentile (~37)
CREATININE_THRESHOLD = 2.0        # simple renal dysfunction proxy

RENAL_RISK_KEYWORDS = [
    "VANCOMYCIN",
    "GENTAMICIN",
    "IBUPROFEN",
    "NAPROXEN",
    "DICLOFENAC",
    "METFORMIN",
]

def contains_any_keyword(drug: str, keywords) -> bool:
    if pd.isna(drug):
        return False
    d = str(drug).upper()
    return any(k in d for k in keywords)

def main():
    df = pd.read_csv(IN_PATH)

    # Make sure needed columns exist
    required = ["drug", "polypharmacy_active_meds", "creatinine"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Basic cleanup
    df["drug"] = df["drug"].astype(str).str.upper().str.strip()

    # --- RULE A: Polypharmacy high ---
    rule_poly = df["polypharmacy_active_meds"] >= POLYPHARM_THRESHOLD

    # --- RULE B: Renal dysfunction + renal-risk drug ---
    renal_risk_drug = df["drug"].apply(lambda x: contains_any_keyword(x, RENAL_RISK_KEYWORDS))
    rule_renal = (df["creatinine"] >= CREATININE_THRESHOLD) & renal_risk_drug

    # Combine label (OR of rules)
    df["label_high_risk"] = (rule_poly | rule_renal).astype(int)

    # Reason columns (helpful for explainability + reporting)
    df["rule_polypharmacy"] = rule_poly.astype(int)
    df["rule_renal_risk_drug"] = rule_renal.astype(int)

    def build_reason(row):
        reasons = []
        if row["rule_polypharmacy"] == 1:
            reasons.append(f"polypharmacy>= {POLYPHARM_THRESHOLD}")
        if row["rule_renal_risk_drug"] == 1:
            reasons.append(f"creatinine>= {CREATININE_THRESHOLD} + renal-risk drug")
        return "; ".join(reasons) if reasons else "low-risk"

    df["risk_reason"] = df.apply(build_reason, axis=1)

    # Print summary
    print("✅ Loaded:", IN_PATH)
    print("Rows:", len(df))
    print("\n=== Label counts ===")
    print(df["label_high_risk"].value_counts(dropna=False).to_string())

    print("\n=== Rule trigger counts ===")
    print("polypharmacy rule:", int(df["rule_polypharmacy"].sum()))
    print("renal-risk rule:", int(df["rule_renal_risk_drug"].sum()))

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print("\n✅ Saved:", OUT_PATH.resolve())

    # Show some examples of high-risk
    print("\n=== Sample high-risk rows (first 10) ===")
    show_cols = ["subject_id", "hadm_id", "starttime", "drug", "polypharmacy_active_meds",
                 "creatinine", "label_high_risk", "risk_reason"]
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[df["label_high_risk"] == 1][show_cols].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
