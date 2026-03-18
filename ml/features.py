import numpy as np
import pandas as pd

FEATURE_COLS = [
    'mean_speed', 'speed_var', 'accel_var', 'accel_mean_pos',
    'hard_accel_n', 'hard_brake_n', 'jerk_mean', 'idle_frac'
]


def extract_features(speeds: np.ndarray) -> dict:
    spd  = np.array(speeds, dtype=float)
    acc  = np.diff(spd)
    jerk = np.diff(acc)
    pos_acc = acc[acc > 0]
    return {
        'mean_speed'    : float(np.mean(spd)),
        'speed_var'     : float(np.var(spd)),
        'accel_var'     : float(np.var(acc)),
        'accel_mean_pos': float(np.mean(pos_acc)) if len(pos_acc) > 0 else 0.0,
        'hard_accel_n'  : int(np.sum(acc >  2.0)),
        'hard_brake_n'  : int(np.sum(acc < -2.0)),
        'jerk_mean'     : float(np.mean(np.abs(jerk))) if len(jerk) > 0 else 0.0,
        'idle_frac'     : float(np.mean(spd < 0.5)),
    }


def extract_all_windows(df: pd.DataFrame, window_s=10, step_s=5) -> pd.DataFrame:
    records = []
    n = len(df)
    for start in range(0, n - window_s + 1, step_s):
        end   = start + window_s
        feats = extract_features(df['spd'].iloc[start:end].values)
        feats['t_start'] = float(df['t'].iloc[start])
        records.append(feats)
    return pd.DataFrame(records)
