"""
Step 4: Train XGBoost classifier on labelled windows.

Target: CV weighted-F1 > 0.75

Outputs:
  ml/models/clf_c1.pkl     - trained XGBoost classifier
  ml/models/scaler_c1.pkl  - fitted StandardScaler (use same at inference)
  training/shap_summary.png

Usage:
  python training/04_train.py
  python training/04_train.py --inp training/data/windows_labelled.parquet
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.features import FEATURE_COLS


def sanity_check(clf, scaler):
    """Quick smoke test with known-smooth and known-aggressive speed traces."""
    from ml.features import extract_features

    smooth_spd     = np.array([10.0] * 10)
    aggressive_spd = np.array([0, 5, 12, 20, 14, 6, 0, 0, 8, 18], dtype=float)

    for name, spd in [("smooth", smooth_spd), ("aggressive", aggressive_spd)]:
        feats = extract_features(spd)
        X = np.array([[feats[c] for c in FEATURE_COLS]])
        X_scaled = scaler.transform(X)
        label = int(clf.predict(X_scaled)[0])
        prob  = float(clf.predict_proba(X_scaled)[0].max())
        names = {0: "smooth", 1: "moderate", 2: "aggressive"}
        print(f"  Sanity [{name:10s}] → predicted: {names[label]:12s}  (conf={prob:.2f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp",       default="training/data/windows_labelled.parquet")
    parser.add_argument("--model_out", default="ml/models/clf_c1.pkl")
    parser.add_argument("--scaler_out",default="ml/models/scaler_c1.pkl")
    parser.add_argument("--shap_out",  default="training/shap_summary.png")
    args = parser.parse_args()

    print(f"Loading {args.inp} ...")
    df = pd.read_parquet(args.inp)
    print(f"  {len(df):,} windows, label dist: {df['label'].value_counts().to_dict()}")

    X = df[FEATURE_COLS].values
    y = df["label"].values

    # --- Scale features ---
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Train / test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # --- XGBoost ---
    clf = xgb.XGBClassifier(
        n_estimators       = 300,
        max_depth          = 4,
        learning_rate      = 0.05,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        eval_metric        = "mlogloss",
        early_stopping_rounds = 20,
        random_state       = 42,
        n_jobs             = -1,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # --- Evaluation ---
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["smooth", "moderate", "aggressive"]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # CV needs a separate estimator without early_stopping_rounds
    clf_cv = xgb.XGBClassifier(
        n_estimators       = clf.best_iteration + 1,  # use the best n_estimators found
        max_depth          = 4,
        learning_rate      = 0.05,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        eval_metric        = "mlogloss",
        random_state       = 42,
        n_jobs             = -1,
    )
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(clf_cv, X_scaled, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
    print(f"\nCV Weighted-F1: {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")
    if cv_f1.mean() < 0.75:
        print("  WARNING: CV F1 below 0.75 target — check label distribution or features")
    else:
        print("  ✓ CV F1 meets target (> 0.75)")

    # --- SHAP feature importance ---
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        explainer  = shap.TreeExplainer(clf)
        shap_vals  = explainer.shap_values(X_test)

        # Newer SHAP returns 3D array (n_samples, n_features, n_classes);
        # older SHAP returns a list of 2D arrays [class0, class1, class2].
        shap_arr = np.array(shap_vals)  # shape: (3, n_samples, n_features) or (n_samples, n_features, 3)
        if shap_arr.ndim == 3 and shap_arr.shape[0] == 3:
            # old-style list → (n_classes, n_samples, n_features)
            mean_abs   = np.abs(shap_arr).mean(axis=(0, 1))   # (n_features,)
            shap_class2 = shap_arr[2]                          # (n_samples, n_features)
        elif shap_arr.ndim == 3:
            # new-style → (n_samples, n_features, n_classes)
            mean_abs   = np.abs(shap_arr).mean(axis=(0, 2))   # (n_features,)
            shap_class2 = shap_arr[:, :, 2]                    # (n_samples, n_features)
        else:
            mean_abs   = np.abs(shap_arr).mean(axis=0)
            shap_class2 = shap_arr

        top_feat = FEATURE_COLS[int(np.argmax(mean_abs))]
        print(f"\nTop SHAP feature: {top_feat}")
        if top_feat not in ("accel_var", "hard_brake_n"):
            print("  NOTE: Expected accel_var or hard_brake_n to be top feature — verify data quality")

        # Summary plot (beeswarm for class 2 = aggressive)
        shap.summary_plot(shap_class2, X_test, feature_names=FEATURE_COLS, show=False)
        plt.tight_layout()
        plt.savefig(args.shap_out, dpi=150)
        plt.close()
        print(f"SHAP plot saved → {args.shap_out}")

    except ImportError:
        print("shap not installed — skipping SHAP plots (pip install shap)")

    # --- Save artefacts ---
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    with open(args.model_out, "wb") as f:
        pickle.dump(clf, f)
    with open(args.scaler_out, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\nModel  saved → {args.model_out}")
    print(f"Scaler saved → {args.scaler_out}")

    # --- Sanity check ---
    print("\nSanity checks:")
    sanity_check(clf, scaler)


if __name__ == "__main__":
    main()
