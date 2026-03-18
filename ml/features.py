import numpy as np
import pandas as pd

GRAVITY = 9.81  # m/s²
UPHILL_THRESHOLD   =  0.03   # 3% grade — hard accel here is terrain-justified
DOWNHILL_THRESHOLD = -0.03   # -3% grade — hard braking here is terrain-justified

FEATURE_COLS = [
    # Speed-based (original)
    'mean_speed', 'speed_var', 'accel_var', 'accel_mean_pos',
    'hard_accel_n', 'hard_brake_n', 'jerk_mean', 'idle_frac',
    # Elevation/gradient (new)
    'mean_gradient', 'grade_adj_accel_var',
    'terrain_excuse_accel_n', 'terrain_excuse_brake_n',
]


def extract_features(speeds: np.ndarray, gradients: np.ndarray = None) -> dict:
    """
    Extract driving behaviour features from a speed window.

    Args:
        speeds:    Array of speed values (m/s), length = window size.
        gradients: Array of road gradient values (dimensionless rise/run),
                   same length as speeds. If None, flat terrain is assumed
                   and grade-adjusted features fall back to speed-only values.
    """
    spd  = np.array(speeds, dtype=float)
    acc  = np.diff(spd)
    jerk = np.diff(acc)
    pos_acc = acc[acc > 0]

    feats = {
        'mean_speed'    : float(np.mean(spd)),
        'speed_var'     : float(np.var(spd)),
        'accel_var'     : float(np.var(acc)),
        'accel_mean_pos': float(np.mean(pos_acc)) if len(pos_acc) > 0 else 0.0,
        'hard_accel_n'  : int(np.sum(acc >  2.0)),
        'hard_brake_n'  : int(np.sum(acc < -2.0)),
        'jerk_mean'     : float(np.mean(np.abs(jerk))) if len(jerk) > 0 else 0.0,
        'idle_frac'     : float(np.mean(spd < 0.5)),
    }

    if gradients is not None:
        grad = np.array(gradients, dtype=float)
        # Align gradients with acceleration samples (acc[i] = spd[i+1] - spd[i])
        grad_acc = grad[1:]  # shape matches acc
        # Remove gravity component from acceleration
        grade_adj_acc = acc - GRAVITY * grad_acc
        feats['mean_gradient']         = float(np.mean(grad))
        feats['grade_adj_accel_var']   = float(np.var(grade_adj_acc))
        feats['terrain_excuse_accel_n']= int(np.sum((acc > 2.0) & (grad_acc > UPHILL_THRESHOLD)))
        feats['terrain_excuse_brake_n']= int(np.sum((acc < -2.0) & (grad_acc < DOWNHILL_THRESHOLD)))
    else:
        # Flat terrain fallback — grade features are neutral/zero
        feats['mean_gradient']         = 0.0
        feats['grade_adj_accel_var']   = feats['accel_var']
        feats['terrain_excuse_accel_n']= 0
        feats['terrain_excuse_brake_n']= 0

    return feats


def extract_all_windows(df: pd.DataFrame, window_s=10, step_s=5) -> pd.DataFrame:
    has_gradient = 'gradient' in df.columns
    records = []
    n = len(df)
    for start in range(0, n - window_s + 1, step_s):
        end   = start + window_s
        grads = df['gradient'].iloc[start:end].values if has_gradient else None
        feats = extract_features(df['spd'].iloc[start:end].values, grads)
        feats['t_start'] = float(df['t'].iloc[start])
        records.append(feats)
    return pd.DataFrame(records)
