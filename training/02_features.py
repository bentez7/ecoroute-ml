"""
Step 2: Extract rolling-window features from cleaned trip data.

Reads:  training/data/trips_clean.parquet
Writes: training/data/windows_features.parquet

Each row = one 10-second window (step 5s) with 12 features + trip_id + t_start.
8 speed-based features + 4 grade-aware features (fallback to flat terrain if gradient unavailable).

Usage:
  python training/02_features.py
  python training/02_features.py --window 10 --step 5
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Allow running from repo root or from training/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.features import extract_features, FEATURE_COLS


def extract_windows_for_trip(
    trip_df: pd.DataFrame, window_s: int = 10, step_s: int = 5
) -> list[dict]:
    """
    Extract overlapping windows from a single trip.

    eVED is sampled at ~1 Hz, so window_s ≈ window_n samples.
    We use index-based slicing (not time-based) because some rows
    have small gaps; index slicing gives exactly `window_s` speeds.
    """
    speeds     = trip_df["spd"].values
    times      = trip_df["t"].values
    fuel_rates = trip_df["fuel_rate"].values  if "fuel_rate"  in trip_df.columns else None
    gradients  = trip_df["gradient"].values   if "gradient"   in trip_df.columns else None
    elevations = trip_df["elevation"].values  if "elevation"  in trip_df.columns else None
    n = len(speeds)
    records = []

    for start in range(0, n - window_s + 1, step_s):
        end   = start + window_s
        grads = gradients[start:end] if gradients is not None else None
        feats = extract_features(speeds[start:end], grads)
        feats["trip_id"] = trip_df["trip_id"].iloc[0]
        feats["t_start"] = float(times[start])
        # Mean fuel rate for this window — NaN if column not present
        if fuel_rates is not None:
            window_fuel = fuel_rates[start:end]
            valid = window_fuel[~np.isnan(window_fuel)]
            feats["fuel_rate_mean"] = float(np.mean(valid)) if len(valid) > 0 else float("nan")
        else:
            feats["fuel_rate_mean"] = float("nan")

        # Mean elevation for this window — informational only, not a model feature
        if elevations is not None:
            window_elev = elevations[start:end]
            valid_elev  = window_elev[~np.isnan(window_elev)]
            feats["elevation_mean"] = float(np.mean(valid_elev)) if len(valid_elev) > 0 else float("nan")
        else:
            feats["elevation_mean"] = float("nan")

        records.append(feats)

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp",    default="training/data/trips_clean.parquet")
    parser.add_argument("--out",    default="training/data/windows_features.parquet")
    parser.add_argument("--window", type=int, default=10, help="Window size in seconds")
    parser.add_argument("--step",   type=int, default=5,  help="Step size in seconds")
    args = parser.parse_args()

    print(f"Loading {args.inp} ...")
    df = pd.read_parquet(args.inp)
    trips = df["trip_id"].unique()
    print(f"  {len(df):,} rows, {len(trips)} trips")

    all_records = []
    for i, tid in enumerate(trips):
        trip_df  = df[df["trip_id"] == tid].copy()
        records  = extract_windows_for_trip(trip_df, args.window, args.step)
        all_records.extend(records)
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(trips)} trips, {len(all_records):,} windows so far")

    windows = pd.DataFrame(all_records)
    # Reorder columns
    col_order = ["trip_id", "t_start"] + FEATURE_COLS + ["fuel_rate_mean", "elevation_mean"]
    windows = windows[col_order]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    windows.to_parquet(args.out, index=False)

    print(f"\nSaved {len(windows):,} windows → {args.out}")
    print(windows.describe().T[["mean", "std", "min", "max"]].round(3))


if __name__ == "__main__":
    main()
