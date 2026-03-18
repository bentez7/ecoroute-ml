"""
Step 1: Load and clean eVED CSV files.

eVED schema (relevant columns):
  VehId               - vehicle identifier
  Trip                - trip number (within vehicle)
  Timestamp(ms)       - elapsed ms since trip start
  Vehicle Speed[km/h] - speed (converted to m/s here)
  Fuel Rate[L/hr]     - used for K-Means labelling in 03_label.py

Output: training/data/trips_clean.parquet
  Columns: trip_id, t, spd (m/s), fuel_rate (L/hr, NaN if unavailable)

Usage:
  python training/01_load.py
  python training/01_load.py --data_dir training/data/eved --out training/data/trips_clean.parquet
"""

import argparse
import glob
import os
import pandas as pd

KMH_TO_MPS = 1 / 3.6
MIN_TRIP_SECONDS = 60    # drop trips shorter than this
MIN_TRIP_POINTS  = 30    # drop trips with too few GPS points


def load_eved_files(data_dir: str) -> pd.DataFrame:
    pattern = os.path.join(data_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'.\n"
            "Download eVED from: https://bitbucket.org/datarepo/eved-dataset\n"
            "and place the weekly CSVs in that folder."
        )
    print(f"Found {len(files)} CSV file(s) in {data_dir}")

    frames = []
    for f in files:
        print(f"  Loading {os.path.basename(f)} ...", end=" ")
        df = pd.read_csv(f, low_memory=False)
        print(f"{len(df):,} rows")
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    # Normalise column names (strip whitespace)
    raw.columns = raw.columns.str.strip()

    required = ["VehId", "Trip", "Timestamp(ms)", "Vehicle Speed[km/h]"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}\nFound: {list(raw.columns)}")

    # Fuel Rate is optional — present in most eVED files but not all
    has_fuel = "Fuel Rate[L/hr]" in raw.columns
    cols = required + (["Fuel Rate[L/hr]"] if has_fuel else [])

    df = raw[cols].copy()
    df.rename(columns={
        "Timestamp(ms)"      : "t_ms",
        "Vehicle Speed[km/h]": "spd_kmh",
    }, inplace=True)
    if not has_fuel:
        df["Fuel Rate[L/hr]"] = float("nan")
        print("  NOTE: 'Fuel Rate[L/hr]' not found — fuel_rate will be NaN (K-Means labelling will fall back to rule-based)")

    # Build a unique trip_id string
    df["trip_id"] = df["VehId"].astype(str) + "_" + df["Trip"].astype(str)

    # Convert units
    df["spd"] = df["spd_kmh"] * KMH_TO_MPS          # km/h → m/s
    df["t"]   = df["t_ms"] / 1000.0                  # ms → seconds

    # Drop rows with missing or negative speed/time
    df = df.dropna(subset=["spd", "t"])
    df = df[df["spd"] >= 0]
    df = df[df["t"]   >= 0]

    # Sort within each trip
    df = df.sort_values(["trip_id", "t"]).reset_index(drop=True)

    # --- Filter out short/sparse trips ---
    trip_stats = df.groupby("trip_id").agg(
        duration=("t", lambda x: x.max() - x.min()),
        n_points =("t", "count"),
    )
    valid_trips = trip_stats[
        (trip_stats["duration"] >= MIN_TRIP_SECONDS) &
        (trip_stats["n_points"]  >= MIN_TRIP_POINTS)
    ].index

    before = df["trip_id"].nunique()
    df = df[df["trip_id"].isin(valid_trips)]
    after  = df["trip_id"].nunique()
    print(f"Trips retained: {after} / {before} "
          f"(dropped {before - after} short/sparse trips)")

    df.rename(columns={"Fuel Rate[L/hr]": "fuel_rate"}, inplace=True)
    return df[["trip_id", "t", "spd", "fuel_rate"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="training/data/eved",
                        help="Folder containing eVED weekly CSV files")
    parser.add_argument("--out", default="training/data/trips_clean.parquet",
                        help="Output parquet path")
    args = parser.parse_args()

    raw  = load_eved_files(args.data_dir)
    df   = clean(raw)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"\nSaved {len(df):,} rows, {df['trip_id'].nunique()} trips → {args.out}")
    print(df.head())


if __name__ == "__main__":
    main()
