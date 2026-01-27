import pandas as pd
import numpy as np
from pathlib import Path

IN_PATH = Path("data/processed/base_rx_clean.csv")
OUT_PATH = Path("data/processed/base_rx_polypharm.csv")

def main():
    df = pd.read_csv(IN_PATH, parse_dates=["starttime", "stoptime", "admittime", "dischtime"])

    # If stoptime missing, assume active until discharge time
    df["stoptime_filled"] = df["stoptime"].where(df["stoptime"].notna(), df["dischtime"])

    # Keep original order index to restore after group calculations
    df = df.reset_index(drop=False).rename(columns={"index": "orig_idx"})
    df = df.sort_values(["hadm_id", "starttime"]).reset_index(drop=True)

    poly = np.zeros(len(df), dtype=int)

    for hadm_id, g in df.groupby("hadm_id", sort=False):
        starts = g["starttime"].to_numpy()
        stops  = g["stoptime_filled"].to_numpy()
        g_idx = g.index.to_numpy()

        for k, row_i in enumerate(g_idx):
            t = starts[k]
            poly[row_i] = int(((starts <= t) & (stops >= t)).sum())

    df["polypharmacy_active_meds"] = poly
    df["polypharmacy_active_meds"] = df["polypharmacy_active_meds"].clip(lower=1)

    # Restore original order
    df = df.sort_values("orig_idx").drop(columns=["orig_idx", "stoptime_filled"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("✅ Saved:", OUT_PATH.resolve())
    print("\nPolypharmacy summary:")
    print(df["polypharmacy_active_meds"].describe().to_string())

if __name__ == "__main__":
    main()
