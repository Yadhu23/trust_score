"""
api.py
------
TrustLayer – Real-Time Data Ingestion API
FastAPI server that accepts data points from any source,
runs them through the TrustLayer engine, and returns a trust report.

Run with:
    uvicorn api:app --reload

Endpoints:
    POST /submit-data   → ingest a new data point, get trust result
    GET  /status        → current trust state for all sources
    POST /reset         → clear in-memory state (start fresh)
    GET  /              → health check / welcome
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import the real-time functions from trust_engine (no ML, no DB)
from trust_engine import process_new_data, get_current_trust_state, reset_realtime_state, compute_recent_insight

# ─────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="TrustLayer API",
    description=(
        "Statistical Multi-Source Data Trust Engine — Real-Time Ingestion.\n\n"
        "POST data points from Source_A / Source_B / Source_C and receive "
        "an instant trust score, anomaly score, consensus score, and decision."
    ),
    version="1.0.0",
)


# ─────────────────────────────────────────────────────────────
# REQUEST & RESPONSE MODELS
# ─────────────────────────────────────────────────────────────

class DataPoint(BaseModel):
    """Schema for a single incoming data point."""
    source_id: str = Field(
        ...,
        description="Must be one of: Source_A, Source_B, Source_C",
        examples=["Source_A"],
    )
    value: float = Field(
        ...,
        description="The numerical sensor / stream reading.",
        examples=[123.45],
    )
    timestamp: int = Field(
        ...,
        description="Integer step number (used for tracking, not computation).",
        examples=[101],
    )


class TrustResult(BaseModel):
    """Schema for the trust engine's response."""
    source_id:          str
    timestamp:          int
    value:              float

    # ── Short-term scores ─────────────────────────────────────────
    anomaly_score:      float   # 0 = normal, 1 = extreme outlier
    consensus_score:    float   # 0 = outlier vs peers, 1 = fully aligned
    historical_trust:   float   # EMA trust (0.8·old + 0.2·perf)
    final_score:        float   # raw single-step weighted score
    smoothed_score:     float   # 3-point rolling average (USE THIS for decisions)
    trust_score:        float   # alias for smoothed_score (backward compat)
    raw_trust_score:    float   # alias for final_score (backward compat)

    # ── Decision ──────────────────────────────────────────────────
    decision:           str     # "✅ Trusted" | "⚠️ Monitor" | "🚨 Isolate"
    decision_plain:     str     # Same without emoji
    trend:              str     # "improving" | "stable" | "degrading"
    confidence:         float   # 0–1: distance from nearest threshold
    confidence_volatility: float # rolling std of confidence
    reason:             str     # Plain-English explanation

    # ── Long-term Reliability Index ───────────────────────────────
    reliability_index:  float   # Slow EMA (0.98/0.02), init 0.7
    reliability_status: str     # "Reliable" | "Under Observation" | "Unreliable"

    # ── Historic Trust (cumulative success ratio) ─────────────────
    historic_trust:     float   # successful_events / total_events
    historic_status:    str     # "Highly Reliable" | "Reliable" | "Unstable" | "Unreliable"


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def root():
    """Simple health-check endpoint."""
    return {
        "status": "ok",
        "message": "TrustLayer API is running. POST to /submit-data to begin.",
    }


@app.post(
    "/submit-data",
    response_model=TrustResult,
    summary="Submit a new data point and receive a trust report",
)
def submit_data(point: DataPoint) -> TrustResult:
    """
    Accept a new data point from one of the three sources.

    - Appends the value to the in-memory buffer for that source.
    - Recomputes rolling Z-score anomaly detection (window = 5).
    - Recomputes cross-source consensus against the other sources.
    - Updates the source's historical trust via EMA.
    - Returns the final trust score and decision label.

    **Example request body:**
    ```json
    {
      "source_id": "Source_A",
      "value": 123.45,
      "timestamp": 101
    }
    ```
    """
    try:
        result = process_new_data(
            source_id=point.source_id,
            value=point.value,
            timestamp=point.timestamp,
        )
    except ValueError as e:
        # Unknown source_id
        raise HTTPException(status_code=422, detail=str(e))

    return TrustResult(**result)


@app.get(
    "/status",
    summary="Current trust state for all sources",
)
def get_status():
    """
    Returns the current historical trust level, buffer size,
    and latest ingested value for all three sources.

    Useful for monitoring without submitting new data.
    """
    return {
        "trust_state":    get_current_trust_state(),
        "recent_insight": compute_recent_insight(last_n=40),
        "note": (
            "historical_trust is the EMA trust score. "
            "buffer_length is how many data points have been submitted."
        ),
    }


@app.post(
    "/reset",
    summary="Reset all in-memory state",
)
def reset():
    """
    Clears all in-memory buffers and resets trust to 0.5 for all sources.
    Call this to start a fresh session without restarting the server.
    """
    reset_realtime_state()
    return {
        "status": "reset",
        "message": "All buffers cleared. Trust reset to 0.5 for all sources.",
    }
