"""
EcoRoute ML Service — FastAPI entry point.

Run from repo root:
    uvicorn api.main:app --reload

Endpoints:
    GET  /health                      — liveness check
    POST /analyse/segment             — primary: stream telemetry points, get segment results
    POST /classify                    — dev utility: stateless single-window classification
    POST /trip/start                  — dev utility: create stateful trip session
    POST /trip/{trip_id}/window       — dev utility: push pre-extracted window
    GET  /trip/{trip_id}/summary      — dev utility: trip behaviour summary
    DELETE /trip/{trip_id}            — dev utility: end trip session
"""

import math
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from time import time
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml.detector import (
    RealTimeDetector, LABEL_MAP, FEATURE_COLS,
    diagnose_aggressive, load_detector,
    load_explainer, compute_shap_top_feature,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SESSIONS     = 50
SESSION_TTL_S    = 3600   # 1 hour
ALERT_COOLDOWN_S = 15
WINDOW_SIZE      = 10     # seconds / samples (1 Hz telemetry)


# ---------------------------------------------------------------------------
# GPS utilities
# ---------------------------------------------------------------------------

def haversine_distance_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in metres between two GPS coordinates."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_gradients_from_points(points: list) -> Optional[np.ndarray]:
    """
    Derive a (N,) road gradient array (dimensionless rise/run) from a list of
    TelemetryPoint objects.  Returns None if lat/lng/altitude_m are missing on
    any point — extract_features() will then assume flat terrain.
    """
    if any(p.lat is None or p.lng is None or p.altitude_m is None for p in points):
        return None
    n = len(points)
    grads = np.zeros(n)
    for i in range(1, n):
        dist = haversine_distance_m(points[i - 1].lat, points[i - 1].lng,
                                    points[i].lat,     points[i].lng)
        if dist > 0.1:   # ignore sub-decimetre jitter
            grads[i] = (points[i].altitude_m - points[i - 1].altitude_m) / dist
        # else grads[i] stays 0 (flat assumption for stationary points)
    grads[0] = grads[1]  # backfill first sample from its neighbour
    return grads


def _parse_timestamp(iso_str: str) -> float:
    """Parse ISO 8601 string to Unix timestamp (float seconds)."""
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return dt.timestamp()


# ---------------------------------------------------------------------------
# Session / buffer storage
# ---------------------------------------------------------------------------

@dataclass
class TripSession:
    """Used by dev-utility /trip/* endpoints."""
    detector:   RealTimeDetector
    created_at: float = field(default_factory=time)
    last_used:  float = field(default_factory=time)


@dataclass
class TripBuffer:
    """Per-trip accumulation buffer for /analyse/segment."""
    detector:      RealTimeDetector
    points:        list  = field(default_factory=list)
    segment_index: int   = 0
    last_alert_t:  float = field(default_factory=lambda: -float('inf'))
    last_used:     float = field(default_factory=time)


def _evict_stale(store: dict) -> None:
    now   = time()
    stale = [k for k, v in store.items() if now - v.last_used > SESSION_TTL_S]
    for k in stale:
        del store[k]


def _get_session(request: Request, trip_id: str) -> TripSession:
    session = request.app.state.trips.get(trip_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Trip not found")
    return session


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector  = load_detector()   # shared detector for /classify
    app.state.explainer = load_explainer()  # SHAP TreeExplainer; None if shap not installed
    app.state.trips   = {}
    app.state.buffers = {}
    yield
    app.state.trips.clear()
    app.state.buffers.clear()


# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EcoRoute ML Service",
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "Production",
            "description": "Endpoints called by the EcoRoute backend during active trips.",
        },
        {
            "name": "Dev Utilities",
            "description": (
                "**For testing and development only.** "
                "These endpoints accept pre-extracted speed arrays instead of raw telemetry. "
                "The backend does not call these in production."
            ),
        },
    ],
)

ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models — /analyse/segment
# ---------------------------------------------------------------------------

class TelemetryPoint(BaseModel):
    recorded_at: str                     # ISO 8601 — "2024-01-15T08:30:00.000Z"
    speed_ms:    float
    altitude_m:  Optional[float] = None
    lat:         Optional[float] = None
    lng:         Optional[float] = None


class AnalyseSegmentRequest(BaseModel):
    trip_id: str
    points:  list[TelemetryPoint] = Field(..., min_length=1)


class SegmentResult(BaseModel):
    segment_index:     int
    started_at:        str
    ended_at:          str
    behaviour_label:   str            # smooth / moderate / aggressive
    confidence:        float
    alert:             Optional[str]  # specific reason or null (15s cooldown per trip)
    avg_speed_kmh:     float
    accel_variance:    float
    braking_frequency: float          # hard braking events / km  (0 if near-stationary)
    idle_time_pct:     float          # % of window where speed < 0.5 m/s
    shap_top_feature:  str            # accel_variance | braking_frequency | idle_time_pct | speed_variance


class AnalyseSegmentResponse(BaseModel):
    segments: list[SegmentResult]     # empty if fewer than 10 points accumulated so far


# ---------------------------------------------------------------------------
# Pydantic models — dev-utility endpoints
# ---------------------------------------------------------------------------

class WindowPayload(BaseModel):
    speeds:    list[float] = Field(..., min_length=10, max_length=10)
    gradients: Optional[list[float]] = Field(None, min_length=10, max_length=10)
    timestamp: float


class ClassifyResponse(BaseModel):
    label:      int
    label_name: str
    confidence: float
    alert:      Optional[str]


class TripStartResponse(BaseModel):
    trip_id: str


class TripSummaryResponse(BaseModel):
    trip_id:          str
    total_windows:    int
    smooth_pct:       float
    moderate_pct:     float
    aggressive_pct:   float
    aggressive_times: list[float]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Production"])
def health():
    return {"status": "ok"}


@app.post("/analyse/segment", response_model=AnalyseSegmentResponse, tags=["Production"])
def analyse_segment(body: AnalyseSegmentRequest, request: Request):
    """
    Primary endpoint. The backend calls this with each batch of raw telemetry
    points for an active trip. The ML service accumulates points per trip_id
    and returns a SegmentResult for every complete 10-second window.

    Returns {"segments": []} while the buffer is still filling up.
    """
    buffers   = request.app.state.buffers
    explainer = request.app.state.explainer

    _evict_stale(buffers)
    if body.trip_id not in buffers:
        buffers[body.trip_id] = TripBuffer(detector=load_detector())

    buf = buffers[body.trip_id]
    buf.points.extend(body.points)
    buf.last_used = time()

    results = []

    while len(buf.points) >= WINDOW_SIZE:
        window     = buf.points[:WINDOW_SIZE]
        buf.points = buf.points[WINDOW_SIZE:]   # tumbling window — no overlap

        speeds    = np.array([p.speed_ms for p in window])
        gradients = compute_gradients_from_points(window)
        timestamp = _parse_timestamp(window[-1].recorded_at)

        label, prob, feats = buf.detector.predict_window(speeds, gradients)

        # Alert with per-trip cooldown (does not use detector's internal cooldown)
        alert = None
        if label == 2 and prob > 0.70 and (timestamp - buf.last_alert_t) > ALERT_COOLDOWN_S:
            buf.last_alert_t = timestamp
            alert = diagnose_aggressive(feats)

        # SHAP top feature — gracefully skipped if shap not installed
        shap_top = 'accel_variance'
        if explainer is not None:
            X        = np.array([[feats[c] for c in FEATURE_COLS]])
            X_scaled = buf.detector.scaler.transform(X) if buf.detector.scaler else X
            shap_top = compute_shap_top_feature(explainer, X_scaled)

        # Derived stats
        mean_spd_kmh = feats['mean_speed'] * 3.6
        dist_km      = feats['mean_speed'] * WINDOW_SIZE / 1000.0
        braking_freq = feats['hard_brake_n'] / dist_km if dist_km > 0.01 else 0.0

        results.append(SegmentResult(
            segment_index     = buf.segment_index,
            started_at        = window[0].recorded_at,
            ended_at          = window[-1].recorded_at,
            behaviour_label   = LABEL_MAP[label],
            confidence        = round(prob, 4),
            alert             = alert,
            avg_speed_kmh     = round(mean_spd_kmh, 2),
            accel_variance    = round(feats['accel_var'], 4),
            braking_frequency = round(braking_freq, 4),
            idle_time_pct     = round(feats['idle_frac'] * 100, 2),
            shap_top_feature  = shap_top,
        ))
        buf.segment_index += 1

    return AnalyseSegmentResponse(segments=results)


# ---------------------------------------------------------------------------
# Dev-utility endpoints (kept for testing / internal tooling)
# ---------------------------------------------------------------------------

@app.post("/classify", response_model=ClassifyResponse, tags=["Dev Utilities"])
def classify(body: WindowPayload, request: Request):
    """Stateless classification — no cooldown, no session state."""
    det   = request.app.state.detector
    label, prob, feats = det.predict_window(body.speeds, body.gradients)
    alert = diagnose_aggressive(feats) if label == 2 and prob > 0.70 else None
    return ClassifyResponse(
        label=label,
        label_name=LABEL_MAP[label],
        confidence=round(prob, 4),
        alert=alert,
    )


@app.post("/trip/start", response_model=TripStartResponse, status_code=201, tags=["Dev Utilities"])
def trip_start(request: Request):
    trips = request.app.state.trips
    _evict_stale(trips)
    if len(trips) >= MAX_SESSIONS:
        raise HTTPException(status_code=503, detail="Too many active sessions — try again later")
    trip_id = str(uuid.uuid4())
    trips[trip_id] = TripSession(detector=load_detector())
    return TripStartResponse(trip_id=trip_id)


@app.post("/trip/{trip_id}/window", response_model=ClassifyResponse, tags=["Dev Utilities"])
def trip_window(trip_id: str, body: WindowPayload, request: Request):
    session = _get_session(request, trip_id)
    session.last_used = time()
    alert = session.detector.push_window(body.speeds, body.timestamp, gradients=body.gradients)
    label = session.detector.last_label
    prob  = session.detector.log[-1]["prob"] if session.detector.log else 0.0
    return ClassifyResponse(
        label=label,
        label_name=LABEL_MAP[label],
        confidence=round(prob, 4),
        alert=alert,
    )


@app.get("/trip/{trip_id}/summary", response_model=TripSummaryResponse, tags=["Dev Utilities"])
def trip_summary(trip_id: str, request: Request):
    session = _get_session(request, trip_id)
    summary = session.detector.get_trip_summary()
    return TripSummaryResponse(trip_id=trip_id, **summary)


@app.delete("/trip/{trip_id}", status_code=204, tags=["Dev Utilities"])
def trip_delete(trip_id: str, request: Request):
    trips = request.app.state.trips
    if trip_id not in trips:
        raise HTTPException(status_code=404, detail="Trip not found")
    del trips[trip_id]
