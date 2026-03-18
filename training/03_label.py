"""
Step 3: Label feature windows using K-Means clustering on fuel rate.

Strategy (from paper: "Driving Behaviour Analysis Using ML and DL on Streaming Data"):
  1. K-Means (k=3) clusters windows by mean fuel rate → data-driven labels
  2. Clusters are ordered by mean fuel consumption:
       lowest fuel  → 0 (smooth)
       middle fuel  → 1 (moderate)
       highest fuel → 2 (aggressive)

Fallback (if fuel_rate_mean is all NaN):
  Rule-based thresholds from FYP Table 2 are used instead.
  This allows the pipeline to run on datasets without fuel rate data.

Why K-Means instead of rules:
  - Labels come from actual measured fuel consumption, not hand-crafted thresholds
  - Breaks the circular training problem (rules → labels → model learns rules)
  - F1 score now measures real predictive power, not rule memorisation

Reads:  training/data/windows_features.parquet
Writes: training/data/windows_labelled.parquet  (adds 'label' + 'label_method' columns)

Usage:
  python training/03_label.py
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Rule-based fallback (used when fuel rate is unavailable)
# ---------------------------------------------------------------------------

def label_rule_based(row) -> int:
    aggressive = (
        row["hard_brake_n"] >= 2   or
        row["hard_accel_n"] >= 2   or
        row["accel_var"]    >  2.5 or
        row["jerk_mean"]    >  1.0
    )
    moderate = (
        row["hard_brake_n"] >= 1  or
        row["accel_var"]    >  1.0 or
        row["jerk_mean"]    >  0.5
    )
    if aggressive: return 2
    if moderate:   return 1
    return 0


# ---------------------------------------------------------------------------
# K-Means labelling on fuel rate
# ---------------------------------------------------------------------------

def label_kmeans(df: pd.DataFrame) -> pd.Series:
    """
    Cluster windows into 3 groups by fuel_rate_mean.
    Returns a Series of labels (0=smooth, 1=moderate, 2=aggressive).
    """
    fuel = df["fuel_rate_mean"].values.reshape(-1, 1)

    # Normalise before clustering
    scaler = StandardScaler()
    fuel_scaled = scaler.fit_transform(fuel)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(fuel_scaled)

    # Map cluster IDs → 0/1/2 ordered by mean fuel rate (lowest = smooth)
    cluster_fuel_mean = (
        pd.Series(clusters)
        .groupby(clusters)
        .apply(lambda idx: df["fuel_rate_mean"].iloc[idx.index].mean())
    )
    ordered = cluster_fuel_mean.sort_values().index.tolist()
    label_map = {ordered[0]: 0, ordered[1]: 1, ordered[2]: 2}

    return pd.Series(clusters).map(label_map)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", default="training/data/windows_features.parquet")
    parser.add_argument("--out", default="training/data/windows_labelled.parquet")
    args = parser.parse_args()

    print(f"Loading {args.inp} ...")
    df = pd.read_parquet(args.inp)
    print(f"  {len(df):,} windows")

    fuel_available = (
        "fuel_rate_mean" in df.columns and
        df["fuel_rate_mean"].notna().sum() > len(df) * 0.5  # need >50% non-NaN
    )

    if fuel_available:
        print(f"\nUsing K-Means labelling on fuel_rate_mean "
              f"({df['fuel_rate_mean'].notna().sum():,} / {len(df):,} windows have fuel data)")

        # Drop windows with missing fuel for clustering, label them separately
        has_fuel = df["fuel_rate_mean"].notna()
        df.loc[has_fuel,  "label"] = label_kmeans(df[has_fuel]).values
        df.loc[~has_fuel, "label"] = df[~has_fuel].apply(label_rule_based, axis=1)
        df["label"] = df["label"].astype(int)
        df["label_method"] = has_fuel.map({True: "kmeans", False: "rule_fallback"})

        # Show cluster fuel rate stats
        print("\nCluster fuel rate means:")
        for lbl, name in [(0, "smooth"), (1, "moderate"), (2, "aggressive")]:
            mean_f = df.loc[df["label"] == lbl, "fuel_rate_mean"].mean()
            print(f"  {lbl} {name:12s}: {mean_f:.3f} L/hr avg")

    else:
        print("\nFuel rate not available — using rule-based labelling (fallback)")
        df["label"] = df.apply(label_rule_based, axis=1)
        df["label_method"] = "rule_based"

    # --- Distribution check ---
    dist = df["label"].value_counts(normalize=True).sort_index()
    print("\nLabel distribution:")
    for lbl, name in [(0, "smooth"), (1, "moderate"), (2, "aggressive")]:
        pct = dist.get(lbl, 0) * 100
        bar = "█" * int(pct / 2)
        print(f"  {lbl} {name:12s}: {pct:5.1f}%  {bar}")
        if pct < 10:
            print(f"    WARNING: class {lbl} is < 10% — model may struggle with this class")
        if pct > 70:
            print(f"    WARNING: class {lbl} is > 70% — consider adjusting k-Means or thresholds")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\nSaved {len(df):,} labelled windows → {args.out}")
    print(f"Label method used: {df['label_method'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
