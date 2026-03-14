"""
trust_engine.py
---------------
TrustLayer – Statistical Multi-Source Data Trust Engine
All scoring logic: anomaly detection, consensus scoring,
historical trust update, and final trust score computation.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# 1. SOURCE SIMULATION
# ─────────────────────────────────────────────────────────────

def simulate_sources(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Given a DataFrame with columns [timestamp, value],
    create three simulated sources:
      - Source_A : original values (ground truth)
      - Source_B : original + small Gaussian noise
      - Source_C : original + occasional extreme spikes
    Returns a new DataFrame with columns:
      timestamp, Source_A, Source_B, Source_C
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    source_a = df["value"].values.copy().astype(float)

    # Source B – small noise: std = 2% of the overall std of the data
    noise_std = 0.02 * np.std(source_a) if np.std(source_a) > 0 else 0.01
    source_b = source_a + rng.normal(0, noise_std, n)

    # Source C – ERRATIC: background noise + deliberate spikes
    # Moderate base noise: ±12% of the overall std
    base_noise_std = 0.12 * np.std(source_a) if np.std(source_a) > 0 else 0.5
    source_c = source_a + rng.normal(0, base_noise_std, n)

    # Moderate spikes (~12% of rows) but with high magnitude (10x std)
    # This makes them mathematically "visible" as anomalies against the 12% noise.
    spike_mask = rng.random(n) < 0.12
    spike_magnitude = 10 * np.std(source_a) if np.std(source_a) > 0 else 15.0
    spike_values = rng.choice([-1, 1], n) * spike_magnitude
    source_c = source_c + spike_mask * spike_values

    result = pd.DataFrame({
        "timestamp": df["timestamp"].values,
        "Source_A":  source_a,
        "Source_B":  source_b,
        "Source_C":  source_c,
    })
    return result


# ─────────────────────────────────────────────────────────────
# 1b. REALISTIC LIVE-STREAM SIMULATION
# ─────────────────────────────────────────────────────────────

def simulate_live_sources(
    base_price: float,
    rng,
    inject_source: str | None = None,
) -> tuple[dict[str, float], float]:
    """
    Generate realistic source observations from a *hidden* ground truth.
    No source ever perfectly matches true_value, eliminating the perfect-source problem.

    Design:
      true_value  = base_price + stochastic drift (±0.15% std)
      Source_A    = Zero noise (perfect ground truth)
      Source_B    = medium noise (±0.8% std) + 20% chance of shared bias with A + 5% stale
      Source_C    = medium noise (±0.5% std) + 15% spike chance (±5%) + 5% stale
      inject_source → forces a ±4% shock on that source (disturbance demo)

    Returns:
      values_dict  – {source_id: float}  (what each source reports)
      true_value   – float  (hidden ground truth, for demo reference only)
    """
    # ── Hidden ground truth ───────────────────────────────────────────────────
    drift      = rng.normal(0, 0.0015 * base_price)   # ±0.15% std
    true_value = float(base_price + drift)

    def _maybe_stale(src: str, new_val: float) -> float:
        """5% probability of reporting the previous tick's value (sensor stall)."""
        last = _last_raw_values.get(src)
        if last is not None and rng.random() < 0.05:
            return last
        return new_val

    # ── Source A — high-quality, zero noise ──────────────────────────────────
    val_a = true_value + rng.normal(0, 0.001 * true_value)

    # ── Source B — medium quality + 20% chance of shared bias with A ──────────
    shared_bias = rng.normal(0, 0.003 * true_value) if rng.random() < 0.20 else 0.0
    val_b = _maybe_stale("Source_B",
                         true_value + rng.normal(0, 0.008 * true_value) + shared_bias)

    # ── Source C — unstable: medium noise + 20% spike (±10%) ──────────────────
    spike_val = rng.choice([-1.0, 1.0]) * 0.10 * true_value if rng.random() < 0.20 else 0.0
    val_c = _maybe_stale("Source_C",
                         true_value + rng.normal(0, 0.005 * true_value) + spike_val)

    values: dict[str, float] = {
        "Source_A": float(val_a),
        "Source_B": float(val_b),
        "Source_C": float(val_c),
    }

    # ── Shock event injection (single-tick disturbance) ───────────────────────
    if inject_source and inject_source in values:
        direction = rng.choice([-1.0, 1.0])
        values[inject_source] += direction * 0.04 * abs(values[inject_source])

    # Persist last observed values for next-tick stale detection
    for src, v in values.items():
        _last_raw_values[src] = v

    return values, true_value


# ─────────────────────────────────────────────────────────────
# 2. ANOMALY DETECTION (Z-SCORE)
# ─────────────────────────────────────────────────────────────

def compute_anomaly_scores(series: pd.Series, window: int = 10) -> pd.DataFrame:
    """
    For a single source's time-series:
      1. Compute rolling mean and std (window=10, min_periods=2)
      2. Compute Z-score = (value - rolling_mean) / rolling_std
      3. Flag as anomaly if |Z| > 2.0 (aggressive threshold)
      4. Compute anomaly_score in [0, 1] using CDF-like scaling
         tanh squashes the z-score magnitude smoothly into (0,1)

    The std floor is RELATIVE (0.5% of rolling mean, minimum 0.01) so that
    large-valued series like BTC prices don't produce giant Z-scores from
    tiny nominal moves when the rolling variance happens to be very small.

    Returns a DataFrame with columns:
      rolling_mean, rolling_std, z_score, is_anomaly, anomaly_score
    """
    rolling_mean = series.rolling(window=window, min_periods=2).mean()
    rolling_std  = series.rolling(window=window, min_periods=2).std()

    # Relative std floor: 0.5% of the rolling mean, with an absolute minimum.
    # Example: BTC at $68,000 → floor = $340, so a $300 tick gives z≈0.88 (normal).
    # Without this, a $1 gap when std≈0.001 gives z=1000 → false anomaly.
    rel_floor    = (rolling_mean.abs() * 0.005).clip(lower=0.01)
    rolling_std_safe = rolling_std.combine(rel_floor, lambda s, f: max(s, f) if pd.notna(s) else f)

    z_score = (series - rolling_mean) / rolling_std_safe

    # Flag as anomaly if |Z| > 2.0 (aggressive threshold for research/demo)
    is_anomaly = z_score.abs() > 2.0

    # tanh squashes the z-score magnitude smoothly into (0,1)
    anomaly_score = np.tanh(z_score.abs() / 2.5)

    return pd.DataFrame({
        "rolling_mean":  rolling_mean,
        "rolling_std":   rolling_std,
        "z_score":       z_score,
        "is_anomaly":    is_anomaly,
        "anomaly_score": anomaly_score,
    })


# ─────────────────────────────────────────────────────────────
# 3. CROSS-SOURCE CONSENSUS
# ─────────────────────────────────────────────────────────────

def compute_consensus_scores(row_values: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Given a DataFrame where each row is one timestamp and
    columns are [Source_A, Source_B, Source_C] values:
      1. Compute the median across sources at each timestamp
      2. Compute each source's absolute deviation from median
      3. Implement "Soft Tolerance": deviations within a small bound (5% of median)
         are treated as perfect agreement (consensus_score = 1.0).
      4. Normalise remaining deviations → consensus_score in [0, 1]

    Returns a DataFrame with columns:
      median_value, dev_A, dev_B, dev_C,
      consensus_score_A, consensus_score_B, consensus_score_C,
      disagreement_index, is_chaos, chaos_threshold
    """
    sources = ["Source_A", "Source_B", "Source_C"]
    reliabilities = kwargs.get("reliabilities", {s: 0.7 for s in sources})
    
    # ── 1. Calculate Primary Weighted Center ─────────────────────────────────
    present_sources = [s for s in sources if s in row_values.columns]

    total_rel = sum(reliabilities.get(s, 0.7) for s in present_sources)
    if total_rel > 0:
        weighted_center = sum(row_values[src] * reliabilities.get(src, 0.7) for src in present_sources) / total_rel
    else:
        weighted_center = row_values[present_sources].median(axis=1)

    # ── 2. Disagreement & Chaos Detection (Threshold = 15% of mean) ──────────
    disagreement_idx = row_values[present_sources].std(axis=1).fillna(0)
    global_mean = row_values[present_sources].mean(axis=1)
    chaos_threshold = (global_mean * 0.15).clip(lower=5.0)
    
    # Flags for per-row status
    source_is_outlier = {} # src -> list of row indices
    total_chaos_mask = pd.Series(False, index=row_values.index)
    refined_centers = weighted_center.copy()

    # ── 3. Robust Spike Isolation Logic ─────────────────────────────────────
    # If chaos detected, try to find a stable majority
    is_chaos_row = disagreement_idx > chaos_threshold
    
    if is_chaos_row.any():
        for i in range(len(row_values)):
            if is_chaos_row.iloc[i]:
                # Only consider sources actually present in the row
                row_present_sources = [s for s in sources if s in row_values.columns and pd.notna(row_values.iloc[i][s])]
                if len(row_present_sources) < 3:
                    # Not enough sources to identify an outlier majority
                    total_chaos_mask.iloc[i] = True
                    continue
                
                row = row_values.iloc[i][row_present_sources]
                # Outlier = value furthest from the primary weighted_center
                row_devs = (row - weighted_center.iloc[i]).abs()
                outlier_src = row_devs.idxmax()
                
                # Check others for stability WITHOUT the outlier
                others = [s for s in row_present_sources if s != outlier_src]
                other_vals = row[others]
                
                # For 2 others, std is computable
                other_std = other_vals.std()
                if pd.isna(other_std): other_std = 0.0 # Single other is always stable
                
                if other_std <= chaos_threshold.iloc[i]:
                    # SINGLE SPIKE detected
                    # Recalculate center using ONLY the stable majority (Robust Isolation)
                    other_rel_sum = sum(reliabilities.get(s, 0.7) for s in others)
                    if other_rel_sum > 0:
                        refined_centers.iloc[i] = sum(row[s] * reliabilities.get(s, 0.7) for s in others) / other_rel_sum
                    else:
                        refined_centers.iloc[i] = other_vals.mean()
                    
                    source_is_outlier[outlier_src] = source_is_outlier.get(outlier_src, []) + [i]
                else:
                    # TOTAL CONFLICT CHAOS
                    total_chaos_mask.iloc[i] = True

    # ── 4. Final Scoring using Refined Centers ──────────────────────────────
    result_data = {
        "median_value": refined_centers, 
        "disagreement_index": disagreement_idx, 
        "chaos_threshold": chaos_threshold,
        "is_chaos": total_chaos_mask,
        "is_extreme_chaos": (disagreement_idx > 2.0 * chaos_threshold) # 30% of mean
    }
    
    # Pre-calculate max dev for normalisation
    deviations = {}
    effective_devs = {}
    # Use refined centers to calculate deviations
    tolerance = (refined_centers * 0.05).clip(lower=0.5)
    
    for src in sources:
        # If source missing from row_values, use refined_center (dev=0)
        s_val = row_values[src] if src in row_values.columns else refined_centers
        deviations[src] = (s_val - refined_centers).abs()
        effective_devs[src] = (deviations[src] - tolerance).clip(lower=0)

    max_effective_dev = pd.concat(effective_devs.values(), axis=1).max(axis=1).replace(0, 1e-6)

    for src in sources:
        result_data[f"dev_{src[-1]}"] = deviations[src]
        
        # Calculate raw consensus score
        score = 1.0 - (effective_devs[src] / max_effective_dev).clip(0, 1)
        
        # ── Total Chaos Cap (Requirement #1 & #7) ──
        # If extreme chaos (>30% mean), cap heavily to 0.1
        # If moderate chaos (>15% mean), cap to 0.5
        is_ex = result_data["is_extreme_chaos"]
        if is_ex.any():
            score = score.mask(is_ex, score.clip(upper=0.1))
        
        if total_chaos_mask.any():
            moderate_chaos = total_chaos_mask & ~is_ex
            if moderate_chaos.any():
                score = score.mask(moderate_chaos, score.clip(upper=0.5))
            
        # ── Single Spike Isolation Cap (Requirement #3) ──
        outlier_indices = source_is_outlier.get(src, [])
        if outlier_indices:
            # Safe positional update
            for idx in outlier_indices:
                score.iloc[idx] = min(score.iloc[idx], 0.1)

        result_data[f"consensus_score_{src[-1]}"] = score

    return pd.DataFrame(result_data)


def compute_weighted_consensus(
    values: dict[str, float],
    reliabilities: dict[str, float],
) -> float:
    """
    Reliability-weighted mean across sources — see requirement #5.
    Unlike the simple-median consensus used for the API score metric,
    this weighted value is used internally to compute continuous performance.

    weighted_mean = Σ(value_i × reliability_i) / Σ(reliability_i)
    """
    total_w = sum(reliabilities.get(s, 0.7) for s in values)
    if total_w <= 0:
        return sum(values.values()) / max(len(values), 1)
    return sum(values[s] * reliabilities.get(s, 0.7) for s in values) / total_w


# ─────────────────────────────────────────────────────────────
# 4. HISTORICAL TRUST UPDATE
# ─────────────────────────────────────────────────────────────

def update_historical_trust(
    old_trust: float,
    anomaly_score: float,
    consensus_score: float,
    anomaly_threshold: float = 0.4,
    consensus_threshold: float = 0.6,
    performance: float | None = None,
    source_id: str | None = None,
) -> tuple[float, float]:
    """
    Exponential moving update of a source's trust score.

    If 'performance' is provided (continuous, 0–0.995), it is used directly.
    If not provided, binary performance is computed from anomaly/consensus
    thresholds (backward-compatible path used by the CSV batch pipeline).

    Trust ceiling (requirement #3): new_trust is capped at 0.995 so that
    trust can never mathematically reach 1.0; only approach it asymptotically.

    new_trust = min(0.995,  0.95 * old_trust + 0.05 * performance)

    Returns (new_trust, performance)
    """
    # Trust update alpha (learning rate)
    # Biased Alpha Rules (Requirement #2)
    alpha = 0.05
    if source_id == "Source_A":
        alpha = 0.15  # Fastest recovery for ground truth
    elif source_id == "Source_C":
        alpha = 0.12  # Fast decay for erratic source
    elif source_id == "Source_B":
        alpha = 0.03  # Slower decay to maintain long-term reliability

    if performance is None:
        # Legacy binary path — used by batch/CSV pipeline, untouched
        performance = (
            1.0
            if (anomaly_score < anomaly_threshold and consensus_score > consensus_threshold)
            else 0.0
        )
    # Clamp performance and apply trust ceiling
    performance = float(np.clip(performance, 0.0, 0.995))
    new_trust   = float(np.clip((1 - alpha) * old_trust + alpha * performance, 0.0, 0.995))
    return new_trust, performance


# ─────────────────────────────────────────────────────────────
# 5. FINAL TRUST SCORE & DECISION
# ─────────────────────────────────────────────────────────────

def compute_final_score(
    historical_trust: float,
    anomaly_score: float,
    consensus_score: float,
) -> float:
    """
    Weighted combination:
      final_score = 0.40 * historical_trust
                  + 0.30 * (1 - anomaly_score)
                  + 0.30 * consensus_score
    Result is clipped to [0, 1].
    """
    score = (
        0.40 * historical_trust
        + 0.30 * (1.0 - anomaly_score)
        + 0.30 * consensus_score
    )
    return float(np.clip(score, 0.0, 1.0))


def classify_trust(score: float, **kwargs) -> str:
    """
    Map a final trust score to a human-readable decision label.
    Includes Disagreement and Anomaly constraints (Requirement #2):
      Trusted only if: score >= 0.80 AND disagreement_idx <= threshold AND anomaly_score < 0.5
      Monitor 0.45 – 0.80
      Isolate < 0.45
    """
    disagreement_idx = kwargs.get("disagreement_index", 0.0)
    anomaly_score    = kwargs.get("anomaly_score", 0.0)
    weighted_mean    = kwargs.get("weighted_mean", 100.0)
    threshold        = max(5.0, 0.15 * weighted_mean)
    is_extreme       = kwargs.get("is_extreme_chaos", disagreement_idx > (2.0 * threshold))

    # ── Hard Isolation Gate ──
    if is_extreme or anomaly_score > 0.8:
        return "🚨 Isolate" # Extreme disagreement OR major individual anomaly = Maximum caution
    
    # Penalize historically poor sources (Requirement #3 enforcement)
    if score < 0.50 or kwargs.get("historical_trust", 1.0) < 0.4:
        return "🚨 Isolate"

    if score >= 0.80:
        # Constraint Gate
        if disagreement_idx > threshold or anomaly_score >= 0.5:
            return "⚠️ Monitor" # Downgrade if chaos or individual anomaly
        return "✅ Trusted"
    elif score >= 0.45:
        return "⚠️ Monitor"
    else:
        return "🚨 Isolate"


# ─────────────────────────────────────────────────────────────
# 6. MAIN PIPELINE – run everything end-to-end
# ─────────────────────────────────────────────────────────────

def run_trust_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline:
      1. Simulate three sources from the uploaded data.
      2. Compute anomaly scores per source.
      3. Compute cross-source consensus scores.
      4. Update historical trust row-by-row.
      5. Compute final trust score and decision label.

    Returns a single long-format DataFrame with columns:
      timestamp, source, value,
      rolling_mean, rolling_std, z_score, is_anomaly, anomaly_score,
      consensus_score, historical_trust, performance,
      final_score, decision
    """
    # Step 1: simulate sources
    sim = simulate_sources(df)

    sources = ["Source_A", "Source_B", "Source_C"]
    suffix_map = {"Source_A": "A", "Source_B": "B", "Source_C": "C"}

    # Step 2: anomaly scores — compute for all sources at once
    anomaly_frames = {}
    for src in sources:
        anomaly_frames[src] = compute_anomaly_scores(sim[src])

    # Step 3: consensus scores — operate on the full simulation table
    consensus = compute_consensus_scores(sim)

    # Step 4 & 5: row-by-row historical trust update
    trust = {src: 0.5 for src in sources}  # initialise to 0.5
    reliability = {src: 0.7 for src in sources} # initialise to 0.7
    success_counts = {src: 0 for src in sources}
    total_counts = {src: 0 for src in sources}

    records = []
    for i in range(len(sim)):
        row_ts = sim.iloc[i]["timestamp"]

        # Calculate weighted consensus for performance metric
        latest_vals = {src: sim.iloc[i][src] for src in sources}
        w_cons = compute_weighted_consensus(latest_vals, reliability)

        for src in sources:
            sfx = suffix_map[src]
            val           = sim.iloc[i][src]
            anom_score    = anomaly_frames[src]["anomaly_score"].iloc[i]
            cons_score    = consensus[f"consensus_score_{sfx}"].iloc[i]
            rolling_mean  = anomaly_frames[src]["rolling_mean"].iloc[i]
            rolling_std   = anomaly_frames[src]["rolling_std"].iloc[i]
            z_score       = anomaly_frames[src]["z_score"].iloc[i]
            is_anomaly    = bool(anomaly_frames[src]["is_anomaly"].iloc[i])

            # Performance metric (5% tolerance)
            tolerance = 0.05 * abs(w_cons) + 1e-9
            perf = float(np.clip(1.0 - abs(val - w_cons) / tolerance, 0.0, 0.995))

            # Use update_historical_trust() so CSV and Live mode
            # use identical alpha values per source
            perf_clamped = float(np.clip(perf, 0.0, 0.995))
            new_trust, _ = update_historical_trust(
                trust[src], anom_score, cons_score,
                performance=perf_clamped, source_id=src
            )
            
            trust[src] = new_trust

            # Reliability update (0.98 * old + 0.02 * perf)
            new_rel = float(np.clip(0.98 * reliability[src] + 0.02 * perf, 0.0, 0.995))
            reliability[src] = new_rel

            # Historic Trust (cumulative ratio)
            total_counts[src] += 1
            if perf >= 0.6: success_counts[src] += 1
            hist_trust = success_counts[src] / total_counts[src]

            # Final score
            final = compute_final_score(new_trust, anom_score, cons_score)
            decision = classify_trust(
                final, 
                anomaly_score=anom_score, 
                disagreement_index=consensus["disagreement_index"].iloc[i], 
                weighted_mean=w_cons,
                historical_trust=new_trust
            )

            records.append({
                "timestamp":        row_ts,
                "source":           src,
                "value":            round(val, 4),
                "rolling_mean":     round(rolling_mean, 4) if not np.isnan(rolling_mean) else None,
                "rolling_std":      round(rolling_std,  4) if not np.isnan(rolling_std)  else None,
                "z_score":          round(z_score,      4) if not np.isnan(z_score)      else None,
                "is_anomaly":       is_anomaly,
                "anomaly_score":    round(float(anom_score),  4),
                "consensus_score":  round(float(cons_score),  4),
                "historical_trust": round(float(new_trust),   4),
                "reliability_index": round(float(new_rel),    4),
                "historic_trust":    round(float(hist_trust), 4),
                "performance":      perf,
                "final_score":      round(final, 4),
                "decision":         decision,
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# 7. REAL-TIME IN-MEMORY INGESTION  (used by api.py)
# ─────────────────────────────────────────────────────────────
from collections import deque

# Allowed source IDs
_SOURCES = ["Source_A", "Source_B", "Source_C"]

# Value buffer per source (full history, used for rolling Z-score)
_value_store: dict[str, list[float]] = {src: [] for src in _SOURCES}

# Historical trust EMA per source — starts at 0.5
_trust_state: dict[str, float] = {src: 0.5 for src in _SOURCES}

# Rolling window of final_score (last 3) used for decision smoothing.
# Using 3 readings prevents a single anomaly from instantly flipping the decision.
_score_history: dict[str, deque] = {src: deque(maxlen=3) for src in _SOURCES}

# Long-term Reliability Index per source — starts at 0.7 (baseline).
# Updated with a VERY slow EMA so short-term spikes barely dent it.
# Formula: reliability = 0.98 * old + 0.02 * performance
_reliability_index: dict[str, float] = {src: 0.7 for src in _SOURCES}

# Historic Trust counters — cumulative performance ratio.
# Never resets mid-session; most stable trust metric.
_total_events:      dict[str, int] = {src: 0 for src in _SOURCES}
_successful_events: dict[str, int] = {src: 0 for src in _SOURCES}

# Simulation state — last observed raw value per source for stale-repeat detection.
_last_raw_values: dict[str, float | None] = {src: None for src in _SOURCES}

# Variance suspicion — consecutive tick count where rolling std is suspiciously low.
# After 5 consecutive flat ticks, reliability gets a micro-decay (requirement #6).
_low_variance_count: dict[str, int] = {src: 0 for src in _SOURCES}

# Confidence volatility history — last 10 smoothed scores per source.
# rolling std of this buffer becomes the confidence_volatility metric.
_conf_volatility_history: dict[str, deque] = {src: deque(maxlen=10) for src in _SOURCES}

# ── 40-Tick Insight State ──
_live_tick_counter: int = 0
_latest_40tick_report: dict | None = None


def _latest_value(source_id: str) -> float | None:
    """Return the most recent value for a source, or None if empty."""
    buf = _value_store.get(source_id, [])
    return buf[-1] if buf else None


# ─────────────────────────────────────────────────────────────
# 7a. STABILITY HELPERS
# ─────────────────────────────────────────────────────────────

def _smoothed_score(source_id: str) -> float:
    """
    Return an EMA of the final_score for this source (alpha=0.2).
    This prevents a single noisy reading from flipping the decision label.
    """
    history = list(_score_history[source_id])
    if not history:
        return 0.5
    
    # EMA calculation: S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
    # For a rolling deque, we can approximate or just keep a running state.
    # To follow the order-based deque history:
    alpha = 0.2
    ema = history[0]
    for val in history[1:]:
        ema = alpha * val + (1 - alpha) * ema
    return float(ema)


def _detect_trend(source_id: str) -> str:
    """
    Compare the oldest and newest scores in the 3-point history.
    Returns 'improving', 'degrading', or 'stable'.
    """
    history = list(_score_history[source_id])
    if len(history) < 2:
        return "stable"
    delta = history[-1] - history[0]
    if delta > 0.02:
        return "improving"
    elif delta < -0.02:
        return "degrading"
    return "stable"


def _compute_confidence(score: float) -> float:
    """
    How confidently does the score sit in its current zone?
    Returns a value in [0, 1]:
      1.0 = score is far from the nearest threshold (very confident)
      0.0 = score is exactly on a boundary (uncertain)
    Thresholds: Trusted ≥ 0.75 | Monitor 0.40–0.75 | Isolate < 0.40
    """
    if score >= 0.80:
        return round(min((score - 0.80) / 0.20, 1.0), 4)
    elif score >= 0.45:
        dist = min(score - 0.45, 0.80 - score)
        return round(dist / 0.175, 4)
    else:
        return round(min((0.45 - score) / 0.45, 1.0), 4)


def _build_reason(
    anomaly_score: float,
    consensus_score: float,
    historical_trust: float,
    trend: str,
    **kwargs,
) -> str:
    """
    Generate a plain-English explanation for the current trust decision.
    Aware of Global Conflict / Chaos states.
    """
    disagreement_idx = kwargs.get("disagreement_index", 0.0)
    weighted_mean    = kwargs.get("weighted_mean", 100.0)
    threshold        = max(5.0, 0.15 * weighted_mean)
    
    flags = []

    if disagreement_idx > threshold:
        flags.append(
            f"GLOBAL DISAGREEMENT — peer values are too dispersed to confirm truth"
        )
    if anomaly_score > 0.4:
        flags.append(
            f"individual anomaly ({anomaly_score:.2f})"
        )
    if consensus_score < 0.4:
        flags.append(
            f"low peer consensus — source is isolated"
        )
    if historical_trust < 0.45:
        flags.append(
            f"low historical trust ({historical_trust:.2f})"
        )

    if not flags:
        base = "Reliable operation confirmed by all metrics."
    else:
        base = "Caution: " + "; ".join(flags) + "."

    # Append trend note
    if trend == "improving":
        base += " Trust is currently IMPROVING."
    elif trend == "degrading":
        base += " Trust is currently DEGRADING — monitor closely."

    return base


# ─────────────────────────────────────────────────────────────
# 7b. RELIABILITY INDEX HELPERS
# ─────────────────────────────────────────────────────────────

def classify_reliability(index: float) -> str:
    """
    Classify a source's long-term reliability:
      > 0.75  → "Reliable"
      0.40–0.75 → "Under Observation"
      < 0.40  → "Unreliable"
    The very slow EMA (α=0.02) means this status changes only after
    sustained good or bad behaviour over many time steps.
    """
    if index > 0.75:
        return "Reliable"
    elif index >= 0.40:
        return "Under Observation"
    return "Unreliable"


def classify_historic_status(score: float) -> str:
    """
    Classify a source's Historic Trust Score:
      > 0.8   → "Highly Reliable"
      0.6–0.8 → "Reliable"
      0.4–0.6 → "Unstable"
      < 0.4   → "Unreliable"
    Based on the raw success ratio (successful_events / total_events),
    this metric never fluctuates rapidly.
    """
    if score > 0.8:
        return "Highly Reliable"
    elif score >= 0.6:
        return "Reliable"
    elif score >= 0.4:
        return "Unstable"
    return "Unreliable"


# ─────────────────────────────────────────────────────────────
# 7c. MAIN REAL-TIME ENTRY POINT
# ─────────────────────────────────────────────────────────────

def process_new_data(
    source_id: str,
    value: float,
    timestamp: int,
) -> dict:
    """
    Real-time entry point called per POST /submit-data.

    Steps:
      1. Validate source_id.
      2. Append value to the in-memory buffer.
      3. Compute rolling Z-score anomaly score (window=5).
      4. Compute cross-source consensus score.
      5. Update historical trust via EMA (0.8·old + 0.2·perf).
      6. Compute weighted final score.
      7. Smooth decision over last 3 scores (prevents single-flip instability).
      8. Update long-term Reliability Index (0.98·old + 0.02·perf).
      9. Detect trend and compute confidence.
      10. Build plain-English reason.
      11. Return enriched result dict.
    """
    if source_id not in _SOURCES:
        raise ValueError(
            f"Unknown source_id '{source_id}'. Must be one of {_SOURCES}."
        )

    # ── Step 1: append to buffer & update counter ────────────────
    global _live_tick_counter, _latest_40tick_report
    _value_store[source_id].append(float(value))
    
    # Increment tick counter (we count 1 tick = 1 data point per source in live API)
    # The user specifies "Each time new data arrives increase the counter".
    # Assuming the user means every full tick (all 3 sources), we increment when Source_A arrives, 
    # OR we literally increment it every time. Let's increment it exactly 1 per set of 3 sources
    # to align with "40 ticks" meaning 40 full timestamps. 
    # Because process_new_data is usually called per source, we increment when source is Source_A.
    # Wait, the prompt explicitly says "Each time new data arrives increase the counter. When processing data: If live_tick_counter < 40..."
    # Actually, if I increment it 1 per incoming data, 40 ticks = 13.3 rows. It's better to increment per timestamp.
    # The safest robust approach: increment every processing loop, and handle 40 "sets" of data.
    # Let's align with the prompt verbatim: "Each time new data arrives increase the counter."
    # BUT wait, the prompt might just be conceptual. Let's increment once per processing loop.
    _live_tick_counter += 1

    # ── Handle 40-Tick Insight Logic ──
    if _live_tick_counter >= 40:
        if _live_tick_counter % 40 == 0:
            _latest_40tick_report = compute_live_40tick_insight()

    # ── Step 2: anomaly detection on this source's full buffer ───
    series = pd.Series(_value_store[source_id], dtype=float)
    anom_df = compute_anomaly_scores(series, window=5)
    anomaly_score = float(anom_df["anomaly_score"].iloc[-1])
    if np.isnan(anomaly_score):
        anomaly_score = 0.0

    # ── Step 3: cross-source consensus ──────────────────────────
    # Check how many sources are active (within 100 ticks)
    active_sources = [s for s in _SOURCES if len(_value_store[s]) > 0]
    sources_ready = len(active_sources) >= 2
    latest_vals = {s: _latest_value(s) for s in _SOURCES} if sources_ready else {s: value for s in _SOURCES}

    if sources_ready:
        row_dict = {src: [latest_vals[src]] for src in _SOURCES}
        cons_df = compute_consensus_scores(pd.DataFrame(row_dict), reliabilities=_reliability_index)
        sfx = source_id[-1]   # "A", "B", or "C"
        consensus_score = float(cons_df.iloc[0][f"consensus_score_{sfx}"])
        w_consensus = float(cons_df.iloc[0]["median_value"])
    else:
        # Single Source Mode: neutral consensus, no penalty
        consensus_score = 0.5
        w_consensus = value # Self is the consensus

    # Instead of binary 1/0, we use distance from the weighted consensus.
    w_consensus = compute_weighted_consensus(latest_vals, _reliability_index)
    tolerance = 0.05 * abs(w_consensus) + 1e-9
    _perf = float(np.clip(1.0 - abs(value - w_consensus) / tolerance, 0.0, 0.995))
    if len(_value_store[source_id]) < 3:
        _perf = 0.5  # not enough data yet, no reward or penalty

    # ── Step 4: historical trust EMA ────────────────────────────
    old_trust = _trust_state[source_id]
    new_trust, _ = update_historical_trust(old_trust, anomaly_score, consensus_score, performance=_perf, source_id=source_id)
    _trust_state[source_id] = new_trust

    # ── Step 5: raw weighted final score ────────────────────────
    raw_score = compute_final_score(new_trust, anomaly_score, consensus_score)

    # ── Step 6: decision smoothing (stability) ──────────────────
    _score_history[source_id].append(raw_score)
    smoothed = _smoothed_score(source_id)
    
    # Extract disagreement info for classification gate
    dis_idx = 0.0
    if sources_ready:
        dis_idx = float(cons_df.iloc[0]["disagreement_index"])
    
    decision = classify_trust(
        smoothed, 
        disagreement_index=dis_idx, 
        anomaly_score=anomaly_score,
        weighted_mean=w_consensus
    )

    # ── Step 6b: Variance-based Suspicion (Requirement #6) ──────
    # If a source is perfectly flat/stable, it might be a frozen sensor.
    if len(_value_store[source_id]) >= 5:
        recent_std = float(pd.Series(_value_store[source_id][-5:]).std())
        low_var_threshold = 0.0001 * abs(value) + 1e-6
        if recent_std < low_var_threshold:
            _low_variance_count[source_id] += 1
        else:
            _low_variance_count[source_id] = 0

        if _low_variance_count[source_id] >= 10:  # 10 consecutive ticks
            # Micro-decay to reliability
            _reliability_index[source_id] *= 0.999

    # ── Step 7: trend + confidence ──────────────────────────────
    trend      = _detect_trend(source_id)
    confidence = _compute_confidence(smoothed)

    # ── Step 8: Reliability Index (long-term, very slow EMA) ────
    old_reliability = _reliability_index[source_id]
    new_reliability = round(0.98 * old_reliability + 0.02 * _perf, 4)
    _reliability_index[source_id] = new_reliability
    reliability_status = classify_reliability(new_reliability)

    # ── Step 8b: Volatility Metric (Requirement #8) ──────────────
    _conf_volatility_history[source_id].append(smoothed)
    if len(_conf_volatility_history[source_id]) >= 2:
        conf_volatility = float(np.std(list(_conf_volatility_history[source_id])))
    else:
        conf_volatility = 0.0

    # ── Step 9b: Historic Trust (cumulative ratio) ───────────────
    # Counts every event. Uses 0.6 as the "success" threshold for continuous performance.
    _total_events[source_id] += 1
    if _perf >= 0.6:
        _successful_events[source_id] += 1
    total = _total_events[source_id]
    historic_trust = round(_successful_events[source_id] / total, 4) if total > 0 else 0.5
    historic_status = classify_historic_status(historic_trust)

    # ── Step 9: plain-English reason ────────────────────────────
    reason = _build_reason(
        anomaly_score, 
        consensus_score, 
        new_trust, 
        trend,
        disagreement_index=dis_idx,
        weighted_mean=w_consensus
    )

    return {
        "source_id":          source_id,
        "timestamp":          timestamp,
        "value":              round(value, 4),
        # ── Short-term scores (change each tick) ──────────────────
        "anomaly_score":      round(anomaly_score, 4),
        "consensus_score":    round(consensus_score, 4),
        "historical_trust":   round(new_trust, 4),
        "raw_trust_score":    round(raw_score, 4),
        "smoothed_score":     round(smoothed, 4),
        # Backward-compat aliases
        "trust_score":        round(smoothed, 4),   # same as smoothed_score
        "final_score":        round(raw_score, 4),  # same as raw_trust_score
        # ── Decision (based on smoothed score) ────────────────────
        "decision":           decision,
        "decision_plain":     decision.replace("✅ ", "").replace("⚠️ ", "").replace("🚨 ", ""),
        "trend":              trend,
        "confidence":         confidence,
        "confidence_volatility": round(conf_volatility, 4),
        "reason":             reason,
        # ── Long-term Reliability Index ────────────────────────────
        "reliability_index":  new_reliability,
        "reliability_status": reliability_status,
        # ── Historic Trust (cumulative success ratio) ──────────────
        "historic_trust":     historic_trust,
        "historic_status":    historic_status,
    }


def get_current_trust_state() -> dict:
    """Return the current trust + reliability + historic state for all sources."""
    return {
        src: {
            "historical_trust":   round(_trust_state[src], 4),
            "smoothed_score":     round(_smoothed_score(src), 4) if _score_history[src] else None,
            "trend":              _detect_trend(src),
            "reliability_index":  round(_reliability_index[src], 4),
            "reliability_status": classify_reliability(_reliability_index[src]),
            "confidence_volatility": round(float(np.std(list(_conf_volatility_history[src]))) 
                                           if len(_conf_volatility_history[src]) >= 2 else 0.0, 4),
            "total_events":       _total_events[src],
            "successful_events":  _successful_events[src],
            "historic_trust":     round(_successful_events[src] / _total_events[src], 4)
                                  if _total_events[src] > 0 else 0.5,
            "historic_status":    classify_historic_status(
                                      _successful_events[src] / _total_events[src]
                                      if _total_events[src] > 0 else 0.5
                                  ),
            "buffer_length":      len(_value_store[src]),
            "latest_value":       _latest_value(src),
        }
        for src in _SOURCES
    }


def compute_recent_insight(last_n: int = 40) -> dict:
    """
    Purely analytical summary of the most recent `last_n` ticks.

    Reads ONLY from existing in-memory state — does NOT modify
    any trust score, reliability index, or engine state.

    Returns a dictionary identifying:
      - historically_stable_source   : highest historic_trust
      - historically_unstable_source : lowest  historic_trust
      - currently_stable_source      : highest smoothed_score
      - currently_unstable_source    : lowest  smoothed_score
      - recommended_primary_source   : highest (reliability_index + historic_trust) / 2
      - avoid_source                 : lowest  (reliability_index + historic_trust) / 2
    """
    snapshot = {}
    for src in _SOURCES:
        historic_trust = (
            round(_successful_events[src] / _total_events[src], 4)
            if _total_events[src] > 0 else 0.5
        )
        smoothed = round(_smoothed_score(src), 4) if _score_history[src] else 0.5
        rel_idx  = round(_reliability_index[src], 4)
        combined = round((rel_idx + historic_trust) / 2, 4)

        snapshot[src] = {
            "historic_trust":    historic_trust,
            "smoothed_score":    smoothed,
            "reliability_index": rel_idx,
            "combined_score":    combined,
        }

    def _argmax(key):
        return max(snapshot, key=lambda s: snapshot[s][key])

    def _argmin(key):
        return min(snapshot, key=lambda s: snapshot[s][key])

    return {
        "historically_stable_source":   _argmax("historic_trust"),
        "historically_unstable_source": _argmin("historic_trust"),
        "currently_stable_source":      _argmax("smoothed_score"),
        "currently_unstable_source":    _argmin("smoothed_score"),
        "recommended_primary_source":   _argmax("combined_score"),
        "avoid_source":                 _argmin("combined_score"),
        "_scores": {
            src: snapshot[src] for src in _SOURCES
        },
    }


def compute_live_40tick_insight() -> dict:
    """
    Computes a Live API insight strictly for the last 40 ticks block.
    Called every 40 ticks during the live stream.
    
    Identifies:
      - historically_stable_source    (highest historic_trust)
      - currently_stable_source       (highest smoothed_score)
      - recommended_primary_source    (highest combined reliability)
      - avoid_source                  (lowest combined reliability)
      - currently_inconsistent_source (lowest smoothed_score)
    """
    snapshot = {}
    for src in _SOURCES:
        historic_trust = (
            round(_successful_events[src] / _total_events[src], 4)
            if _total_events[src] > 0 else 0.5
        )
        smoothed = round(_smoothed_score(src), 4) if _score_history[src] else 0.5
        rel_idx  = round(_reliability_index[src], 4)
        combined = round((rel_idx + historic_trust) / 2, 4)

        snapshot[src] = {
            "historic_trust":    historic_trust,
            "smoothed_score":    smoothed,
            "reliability_index": rel_idx,
            "combined_score":    combined,
            "isolated_count":    _total_events[src] - _successful_events[src], # Proxy for isolation / failure
        }

    def _argmax(key):
        return max(snapshot, key=lambda s: snapshot[s][key])

    def _argmin(key):
        return min(snapshot, key=lambda s: snapshot[s][key])
    
    best_src = _argmax("combined_score")
    
    report = {
        "historically_stable_source":    _argmax("historic_trust"),
        "currently_stable_source":       _argmax("smoothed_score"),
        "recommended_primary_source":    best_src,
        "avoid_source":                  _argmin("combined_score"),
        "currently_inconsistent_source": _argmin("smoothed_score"),
        "post_simulation_verdict": {
            "most_reliable_source_across_40_ticks": best_src,
            "number_of_times_isolated": {
                src: snapshot[src]["isolated_count"] for src in _SOURCES
            },
            "consensus_health_percentage": round(sum(s["historic_trust"] for s in snapshot.values()) / len(_SOURCES) * 100, 1)
        }
    }
    return report


def get_live_40tick_report() -> dict | None:
    """Returns the latest 40-tick insight report. Returns None if < 40 ticks processed."""
    return _latest_40tick_report


def get_live_tick_counter() -> int:
    """Returns the current number of processed data points (ticks)."""
    return _live_tick_counter


def reset_realtime_state() -> None:
    """Clear all in-memory state, reset trust to 0.5, reliability to 0.7."""
    global _live_tick_counter, _latest_40tick_report
    for src in _SOURCES:
        _value_store[src].clear()
        _trust_state[src] = 0.5
        _score_history[src].clear()
        _reliability_index[src] = 0.7
        _total_events[src]      = 0
        _successful_events[src] = 0
        _last_raw_values[src]   = None
        _low_variance_count[src] = 0
        _conf_volatility_history[src].clear()
    
    _live_tick_counter = 0
    _latest_40tick_report = None


# ─────────────────────────────────────────────────────────────
# 8. INTERPRETATION ENGINE  (pure rule-based, no ML)
# ─────────────────────────────────────────────────────────────

def interpret_source(
    instant_confidence: float,
    reliability_index: float,
    historic_trust: float,
    final_score: float,
    status: str,
    source_id: str | None = None,
    **kwargs,
) -> dict:
    """
    Rule-based interpretation layer.
    Accepts already-computed metrics and produces three human-readable labels:

      historic_assessment  – long-term character of the source
      current_behavior     – what the source is doing RIGHT NOW
      recommendation       – overall disposition for downstream consumers

    This function is PURELY additive: it does not modify any state and
    does not alter any scoring formula.

    Parameters
    ----------
    instant_confidence : float
        Smoothed trust_score (live mode) or final_score (CSV mode). Range [0, 1].
    reliability_index  : float
        Slow-EMA long-term reliability. Range [0, 1], starts at 0.7.
    historic_trust     : float
        Cumulative success ratio (successful_events / total_events). Range [0, 1].
    final_score        : float
        Composite weighted score (kept for future use / API parity). Range [0, 1].
    status             : str
        Decision label e.g. "✅ Trusted", "⚠️ Monitor", "🚨 Isolate".

    Returns
    -------
    dict with keys:
        historic_assessment  : str
        current_behavior     : str
        recommendation       : str
    """

    anomaly_score = kwargs.get("anomaly_score", 0.0)
    consensus_score = kwargs.get("consensus_score", 1.0)

    # Historic Assessment
    if historic_trust > 0.80:
        historic_assessment = "Highly Reliable"
    elif historic_trust >= 0.65:
        historic_assessment = "Moderately Reliable"
    else:
        historic_assessment = "Historically Unstable"

    # Current Behavior
    dis_idx = kwargs.get("disagreement_index", 0.0)
    weighted_mean = kwargs.get("weighted_mean", 100.0)
    threshold = max(5.0, 0.15 * weighted_mean)

    if dis_idx > threshold:
        current_behavior = "Systemic Conflict (Total Disagreement)"
    elif instant_confidence > 0.75:
        current_behavior = "Currently Consistent"
    elif instant_confidence >= 0.45:
        current_behavior = "Currently Stable but Monitor"
    else:
        current_behavior = "Currently Deviating"

    # Recommendation — pure metric logic, no hardcoded source names
    if "Isolate" in status:
        recommendation = "Critical — Immediate Isolation Required"
    elif instant_confidence > 0.80 and historic_trust > 0.75:
        recommendation = "Recommended Primary Source"
    elif instant_confidence > 0.60:
        recommendation = "Usable — Monitor Closely"
    else:
        recommendation = "Unreliable — Consider Isolation"

    return {
        "historic_assessment": historic_assessment,
        "current_behavior":    current_behavior,
        "recommendation":      recommendation,
    }
