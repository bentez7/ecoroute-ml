"""
Tests for RealTimeDetector (Component 1).

Run from ecoroute_ml/:
    python -m pytest tests/test_detector.py -v
    python tests/test_detector.py          # without pytest
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.detector import load_detector

MODEL_PATH  = "ml/models/clf_c1.pkl"
SCALER_PATH = "ml/models/scaler_c1.pkl"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_detector():
    return load_detector(MODEL_PATH, SCALER_PATH)

# Speed traces (m/s, 10 samples = 10 seconds at 1 Hz)
SMOOTH_WINDOW     = [10.0] * 10                                      # constant cruise
MODERATE_WINDOW   = [8.0, 8.5, 9.0, 9.5, 9.0, 8.5, 8.0, 8.5, 9.0, 9.5]  # gentle variation
AGGRESSIVE_WINDOW = [0.0, 5.0, 12.0, 20.0, 14.0, 6.0, 0.0, 0.0, 8.0, 18.0]  # hard accel/brake

LABEL_NAMES = {0: "smooth", 1: "moderate", 2: "aggressive"}

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run_window(detector, speeds, timestamp=10.0):
    alert = detector.push_window(speeds, timestamp)
    label = detector.last_label
    return label, alert


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_smooth_window():
    """Constant cruise speed should be classified as smooth (0)."""
    det = make_detector()
    label, alert = run_window(det, SMOOTH_WINDOW)
    assert label == 0, f"Expected smooth (0), got {LABEL_NAMES.get(label, label)}"
    assert alert is None, f"No alert expected for smooth driving, got: {alert}"
    print(f"  PASS  smooth window → label={label} ({LABEL_NAMES[label]}), alert={alert}")


def test_aggressive_window():
    """Hard accel/brake trace should be classified as aggressive (2)."""
    det = make_detector()
    label, alert = run_window(det, AGGRESSIVE_WINDOW, timestamp=10.0)
    assert label == 2, f"Expected aggressive (2), got {LABEL_NAMES.get(label, label)}"
    print(f"  PASS  aggressive window → label={label} ({LABEL_NAMES[label]}), alert={alert!r}")


def test_alert_fires_for_aggressive():
    """Alert string should be returned on first aggressive window."""
    det = make_detector()
    _, alert = run_window(det, AGGRESSIVE_WINDOW, timestamp=10.0)
    assert alert is not None, "Expected an alert string for aggressive driving"
    assert isinstance(alert, str)
    print(f"  PASS  alert fires: {alert!r}")


def test_alert_cooldown():
    """Alert should NOT fire again within the cooldown window (15s)."""
    det = make_detector()
    _, first_alert  = run_window(det, AGGRESSIVE_WINDOW, timestamp=10.0)
    _, second_alert = run_window(det, AGGRESSIVE_WINDOW, timestamp=20.0)  # only 10s later
    assert first_alert  is not None, "First alert should fire"
    assert second_alert is None,     "Second alert within cooldown should be suppressed"
    print(f"  PASS  cooldown suppressed second alert at t=20s")


def test_alert_fires_after_cooldown():
    """Alert SHOULD fire again once cooldown (15s) has elapsed."""
    det = make_detector()
    _, first_alert = run_window(det, AGGRESSIVE_WINDOW, timestamp=10.0)
    _, later_alert = run_window(det, AGGRESSIVE_WINDOW, timestamp=30.0)  # 20s later
    assert first_alert is not None, "First alert should fire"
    assert later_alert is not None, "Alert should fire again after cooldown"
    print(f"  PASS  alert re-fires at t=30s (after 20s cooldown)")


def test_push_single_samples():
    """push() one sample at a time should behave the same as push_window()."""
    det = make_detector()
    alert = None
    for i, spd in enumerate(AGGRESSIVE_WINDOW):
        result = det.push(spd, timestamp=float(i + 1))
        if result:
            alert = result
    assert det.last_label == 2, f"Expected aggressive (2), got {det.last_label}"
    print(f"  PASS  single-sample push → label={det.last_label}, alert={alert!r}")


def test_trip_summary_counts():
    """Trip summary percentages should sum to 1.0 and counts should match."""
    det = make_detector()
    windows = [SMOOTH_WINDOW, SMOOTH_WINDOW, AGGRESSIVE_WINDOW, MODERATE_WINDOW]
    for i, w in enumerate(windows):
        det.push_window(w, timestamp=float((i + 1) * 10))

    summary = det.get_trip_summary()

    assert summary["total_windows"] == 4
    total_pct = summary["smooth_pct"] + summary["moderate_pct"] + summary["aggressive_pct"]
    assert abs(total_pct - 1.0) < 1e-6, f"Percentages don't sum to 1: {total_pct}"
    print(f"  PASS  trip summary: {summary}")


def test_trip_summary_aggressive_times():
    """aggressive_times should list timestamps of aggressive windows only."""
    det = make_detector()
    det.push_window(SMOOTH_WINDOW,     timestamp=10.0)
    det.push_window(AGGRESSIVE_WINDOW, timestamp=20.0)
    det.push_window(SMOOTH_WINDOW,     timestamp=30.0)
    det.push_window(AGGRESSIVE_WINDOW, timestamp=40.0)

    summary = det.get_trip_summary()
    assert 20.0 in summary["aggressive_times"]
    assert 40.0 in summary["aggressive_times"]
    assert 10.0 not in summary["aggressive_times"]
    print(f"  PASS  aggressive_times: {summary['aggressive_times']}")


# ---------------------------------------------------------------------------
# Manual runner (no pytest needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_smooth_window,
        test_aggressive_window,
        test_alert_fires_for_aggressive,
        test_alert_cooldown,
        test_alert_fires_after_cooldown,
        test_push_single_samples,
        test_trip_summary_counts,
        test_trip_summary_aggressive_times,
    ]

    passed = failed = 0
    for t in tests:
        try:
            print(f"\n{t.__name__}")
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
