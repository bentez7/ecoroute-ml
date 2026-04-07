import pickle
import numpy as np
from collections import deque
from ml.features import extract_features, FEATURE_COLS

LABEL_MAP  = {0: 'smooth', 1: 'moderate', 2: 'aggressive'}
ALERT_MSGS = {2: "Ease off — aggressive driving detected"}


def diagnose_aggressive(feats: dict) -> str:
    """
    Inspect a feature dict (from extract_features) and return a specific
    human-readable reason why the window was classified as aggressive.

    Called only when label == 2. Priority: braking > acceleration > variance > jerk > generic.
    """
    if feats.get('hard_brake_n', 0) >= 2:
        return f"Avoid sudden braking — {feats['hard_brake_n']} hard brake event(s) detected"
    if feats.get('hard_accel_n', 0) >= 2:
        return f"Avoid heavy acceleration — {feats['hard_accel_n']} hard acceleration event(s) detected"
    if feats.get('hard_brake_n', 0) == 1:
        return "Avoid sudden braking — hard braking detected"
    if feats.get('hard_accel_n', 0) == 1:
        return "Avoid heavy acceleration — hard acceleration detected"
    if feats.get('grade_adj_accel_var', 0) > 4.0:
        return "Smooth out your speed — very erratic acceleration detected"
    if feats.get('accel_var', 0) > 3.0:
        return "Smooth out your speed — erratic acceleration detected"
    if feats.get('jerk_mean', 0) > 1.5:
        return "Drive more smoothly — sudden speed changes detected"
    return ALERT_MSGS[2]


class RealTimeDetector:
    def __init__(self, clf, scaler=None, window=10, alert_cooldown=15):
        self.clf            = clf
        self.scaler         = scaler          # StandardScaler (required at inference)
        self.buffer         = deque(maxlen=window)
        self.last_alert_t   = -float('inf')
        self.alert_cooldown = alert_cooldown
        self.last_label     = 0
        self.log            = []

    def _predict(self, speeds: np.ndarray, gradients: np.ndarray = None) -> tuple:
        feats = extract_features(speeds, gradients)
        X = np.array([[feats[c] for c in FEATURE_COLS]])
        if self.scaler is not None:
            X = self.scaler.transform(X)
        label = int(self.clf.predict(X)[0])
        prob  = float(self.clf.predict_proba(X)[0].max())
        return label, prob, feats

    def predict_window(self, speeds: list, gradients: list = None) -> tuple:
        """Stateless prediction — does not modify buffer, log, or cooldown state."""
        grads = np.array(gradients) if gradients is not None else None
        return self._predict(np.array(speeds), grads)

    def push(self, speed_mps: float, timestamp: float):
        """Push one speed sample; returns alert string or None.
        Note: single-sample push cannot provide gradient data.
        Use push_window() for grade-aware inference.
        """
        self.buffer.append(speed_mps)
        if len(self.buffer) < 10:
            return None
        label, prob, feats = self._predict(np.array(self.buffer))
        self.last_label = label
        self.log.append({'t': timestamp, 'label': label, 'prob': prob})
        if label == 2 and (timestamp - self.last_alert_t) > self.alert_cooldown and prob > 0.70:
            self.last_alert_t = timestamp
            return diagnose_aggressive(feats)
        return None

    def push_window(self, speeds: list, timestamp: float, gradients: list = None):
        """Receive a full 10-second window at once.

        Args:
            speeds:    Speed values in m/s (10 samples).
            timestamp: Current time in seconds.
            gradients: Optional road gradient values (same length as speeds).
                       Pass these for grade-aware classification that forgives
                       terrain-forced acceleration/braking events.
        """
        grads = np.array(gradients) if gradients is not None else None
        label, prob, feats = self._predict(np.array(speeds), grads)
        self.last_label = label
        self.log.append({'t': timestamp, 'label': label, 'prob': prob})
        if label == 2 and (timestamp - self.last_alert_t) > self.alert_cooldown and prob > 0.70:
            self.last_alert_t = timestamp
            return diagnose_aggressive(feats)
        return None

    def get_trip_summary(self) -> dict:
        labels = [r['label'] for r in self.log]
        return {
            'total_windows'   : len(labels),
            'smooth_pct'      : labels.count(0) / max(len(labels), 1),
            'moderate_pct'    : labels.count(1) / max(len(labels), 1),
            'aggressive_pct'  : labels.count(2) / max(len(labels), 1),
            'aggressive_times': [r['t'] for r in self.log if r['label'] == 2],
        }


SHAP_FEEDBACK_MAP = {
    'hard_brake_n':           'braking_frequency',
    'terrain_excuse_brake_n': 'braking_frequency',
    'accel_var':              'accel_variance',
    'grade_adj_accel_var':    'accel_variance',
    'hard_accel_n':           'accel_variance',
    'jerk_mean':              'accel_variance',
    'accel_mean_pos':         'accel_variance',
    'idle_frac':              'idle_time_pct',
    'speed_var':              'speed_variance',
    'mean_speed':             'speed_variance',
    'mean_gradient':          'accel_variance',
    'terrain_excuse_accel_n': 'accel_variance',
}


def load_explainer(model_path: str = "ml/models/clf_c1.pkl"):
    """
    Load a SHAP TreeExplainer from the saved XGBoost model.
    Returns None if shap is not installed (graceful degradation).
    """
    try:
        import shap
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
        return shap.TreeExplainer(clf)
    except ImportError:
        return None


def compute_shap_top_feature(explainer, X_scaled: np.ndarray) -> str:
    """
    Return the feedback category name for the most impactful SHAP feature
    in a single-sample prediction.

    Handles both old SHAP (list of 2D arrays) and new SHAP (3D array) output formats,
    using the same version-detection logic as training/04_train.py.
    """
    shap_arr = np.array(explainer.shap_values(X_scaled))
    if shap_arr.ndim == 3 and shap_arr.shape[0] == 3:
        # Old SHAP: (n_classes, n_samples, n_features)
        mean_abs = np.abs(shap_arr).mean(axis=(0, 1))
    elif shap_arr.ndim == 3:
        # New SHAP: (n_samples, n_features, n_classes)
        mean_abs = np.abs(shap_arr).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(shap_arr).mean(axis=0)
    top_feat = FEATURE_COLS[int(np.argmax(mean_abs))]
    return SHAP_FEEDBACK_MAP.get(top_feat, 'accel_variance')


def load_detector(model_path="ml/models/clf_c1.pkl",
                  scaler_path="ml/models/scaler_c1.pkl") -> RealTimeDetector:
    """Convenience loader used by FastAPI lifespan and tests."""
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    scaler = None
    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        pass
    return RealTimeDetector(clf, scaler=scaler)
