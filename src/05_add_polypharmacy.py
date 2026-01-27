import pandas as pd
from pathlib import Path

IN_PATH = Path("data/processed/base_rx_clean.csv")
OUT_PATH = Path("data/processed/base_rx_polypharm.csv")

def main():
    df = pd.read_csv(IN_PATH, parse_dates=["starttime", "stoptime", "admittime", "dischtime"])

    # If stoptime missing, treat as active until discharge time
    df["stoptime_filled"] = df["stoptime"]
    df.loc[df["stoptime_filled"].isna(), "stoptime_filled"] = df["dischtime"]

    # Sort for efficient processing
    df = df.sort_values(["hadm_id", "starttime"]).reset_index(drop=True)

    # Polypharmacy count: number of meds active at the starttime within same admission
    # Active if other.starttime <= this.starttime <= other.stoptime_filled
    poly_counts = []
    for hadm_id, g in df.groupby("hadm_id", sort=False):
        starts = g["starttime"].values
        stops = g["stoptime_filled"].values

        # For each row i, count active meds j
        # (This is okay for demo dataset size)
        for i in range(len(g)):
            t = starts[i]
            active = ((starts <= t) & (stops >= t)).sum()
            poly_counts.append(int(active))

    df["polypharmacy_active_meds"] = poly_counts

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["stoptime_filled"]).to_csv(OUT_PATH, index=False)

    print("✅ Saved:", OUT_PATH.resolve())
    print(df[["subject_id", "hadm_id", "starttime", "drug", "polypharmacy_active_meds"]].head(10).to_string(index=False))
    print("\nPolypharmacy summary:")
    print(df["polypharmacy_active_meds"].describe().to_string())

if __name__ == "__main__":
    main()
