# EcoRoute ML Service — API Reference

The ML service is a passive HTTP server. It does not call the backend. The backend calls it and writes results to the database.

Base URL: `http://localhost:8000` (local dev). Set `ML_SERVICE_URL` in the backend environment.

---

## Endpoints

### `GET /health`

Liveness check.

**Response `200`:**
```json
{ "status": "ok" }
```

---

### `POST /analyse/segment`

The primary endpoint. Call this every time the mobile app posts a batch of telemetry points for an active trip.

The ML service accumulates points per `trip_id` in memory. For every complete 10-second window (10 points at 1 Hz), it returns a `SegmentResult`. Returns an empty list while the buffer is still filling up.

**The buffer auto-creates on the first call for a new `trip_id`** — no session setup needed.

**Request body:**
```json
{
  "trip_id": "uuid",
  "points": [
    {
      "recorded_at": "2024-01-15T08:30:00.000Z",
      "speed_ms": 13.8,
      "altitude_m": 45.2,
      "lat": 3.1412,
      "lng": 101.6865
    }
  ]
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `trip_id` | string | yes | Must match the active trip ID — used to route points to the correct buffer |
| `points` | array | yes | One or more telemetry points; at least 1 required |
| `recorded_at` | ISO 8601 string | yes | Timestamp of the sample |
| `speed_ms` | float | yes | Speed in m/s |
| `altitude_m` | float | no | GPS altitude in metres — enables grade-aware classification |
| `lat` | float | no | GPS latitude — required together with `lng` and `altitude_m` for gradient |
| `lng` | float | no | GPS longitude — required together with `lat` and `altitude_m` for gradient |

> If `lat`, `lng`, and `altitude_m` are all present, the ML service derives the road gradient and uses it to forgive terrain-forced acceleration/braking (e.g. hard braking downhill is not penalised). If any are missing, flat terrain is assumed.

**Response `200` — buffer still filling (fewer than 10 points accumulated):**
```json
{ "segments": [] }
```

**Response `200` — one or more windows complete:**
```json
{
  "segments": [
    {
      "segment_index": 0,
      "started_at": "2024-01-15T08:30:00.000Z",
      "ended_at": "2024-01-15T08:30:09.000Z",
      "behaviour_label": "aggressive",
      "confidence": 0.94,
      "alert": "Avoid sudden braking — 3 hard brake event(s) detected",
      "avg_speed_kmh": 49.7,
      "accel_variance": 0.82,
      "braking_frequency": 1.4,
      "idle_time_pct": 5.2,
      "shap_top_feature": "braking_frequency"
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `segment_index` | int | 0-based, monotonically increasing per `trip_id` |
| `started_at` | ISO 8601 | Timestamp of the first point in this window |
| `ended_at` | ISO 8601 | Timestamp of the last point in this window |
| `behaviour_label` | string | `"smooth"`, `"moderate"`, or `"aggressive"` |
| `confidence` | float | Model confidence (0–1) |
| `alert` | string or null | Human-readable alert for the driver. Only set when `behaviour_label == "aggressive"` and confidence > 0.70. Suppressed if an alert was already sent within the last 15 seconds for this trip. |
| `avg_speed_kmh` | float | Mean speed over the window in km/h |
| `accel_variance` | float | Variance of acceleration (m/s²) — most predictive feature |
| `braking_frequency` | float | Hard braking events per km |
| `idle_time_pct` | float | % of window where speed < 0.5 m/s |
| `shap_top_feature` | string | Most impactful feature driving the classification — see mapping below |

**`shap_top_feature` values and corresponding feedback event types:**

| `shap_top_feature` | Suggested `event_type` |
|---|---|
| `accel_variance` | `harsh_accel` |
| `braking_frequency` | `harsh_brake` |
| `idle_time_pct` | `idling` |
| `speed_variance` | `speed_variance` |

**`alert` message examples:**

| Cause | Message |
|---|---|
| Multiple hard brakes | `"Avoid sudden braking — N hard brake event(s) detected"` |
| Multiple hard accelerations | `"Avoid heavy acceleration — N hard acceleration event(s) detected"` |
| Single hard brake | `"Avoid sudden braking — hard braking detected"` |
| Single hard acceleration | `"Avoid heavy acceleration — hard acceleration detected"` |
| Erratic speed changes | `"Smooth out your speed — erratic acceleration detected"` |
| High jerk | `"Drive more smoothly — sudden speed changes detected"` |

---

## Notes

- **State is in-memory.** The ML service holds per-trip buffers in RAM. If the service restarts, all buffers are lost. Telemetry for an in-progress trip must be re-streamed from the last incomplete window.
- **Buffers are evicted** after 1 hour of inactivity. Active trips should post telemetry at least once per hour (in practice, every few seconds).
- **Multiple segments per response** is possible. If a single request contains 20+ points, two or more complete windows will be returned in one response.
- **No authentication** is required between backend and ML service on the local network. Add a shared secret header (`X-ML-Secret`) if deploying to a public network.
- **CORS** is open by default (`*`). Set the `CORS_ORIGINS` environment variable to restrict origins in production (e.g. `CORS_ORIGINS=https://ecoroute.app`).

---

## Running the service

```bash
# From the ecoroute-ml repo root
uvicorn api.main:app --host 0.0.0.0 --port 8000

# With auto-reload (development)
uvicorn api.main:app --reload
```

Interactive API docs (Swagger UI): `http://localhost:8000/docs`
