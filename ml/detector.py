import pickle
import numpy as np
from collections import deque
from ml.features import extract_features, FEATURE_COLS

LABEL_MAP  = {0: 'smooth', 1: 'moderate', 2: 'aggressive'}
ALERT_MSGS = {2: "Ease off — aggressive driving detected"}


class RealTimeDetector:
    def __init__(self, clf, scaler=None, window=10, alert_cooldown=15):
        self.clf            = clf
        self.scaler         = scaler          # StandardScaler (required at inference)
        self.buffer         = deque(maxlen=window)
        self.last_alert_t   = -float('inf')
        self.alert_cooldown = alert_cooldown
        self.last_label     = 0
        self.log            = []

    def _predict(self, speeds: np.ndarray):
        feats = extract_features(speeds)
        X = np.array([[feats[c] for c in FEATURE_COLS]])
        if self.scaler is not None:
            X = self.scaler.transform(X)
        label = int(self.clf.predict(X)[0])
        prob  = float(self.clf.predict_proba(X)[0].max())
        return label, prob

    def push(self, speed_mps: float, timestamp: float):
        """Push one speed sample; returns alert string or None."""
        self.buffer.append(speed_mps)
        if len(self.buffer) < 10:
            return None
        label, prob = self._predict(np.array(self.buffer))
        self.last_label = label
        self.log.append({'t': timestamp, 'label': label, 'prob': prob})
        if label == 2 and (timestamp - self.last_alert_t) > self.alert_cooldown and prob > 0.70:
            self.last_alert_t = timestamp
            return ALERT_MSGS[2]
        return None

    def push_window(self, speeds: list, timestamp: float):
        """Receive a full 10-second window at once."""
        label, prob = self._predict(np.array(speeds))
        self.last_label = label
        self.log.append({'t': timestamp, 'label': label, 'prob': prob})
        if label == 2 and (timestamp - self.last_alert_t) > self.alert_cooldown and prob > 0.70:
            self.last_alert_t = timestamp
            return ALERT_MSGS[2]
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
