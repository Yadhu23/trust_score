import numpy as np
import pandas as pd

# This file contains the logic originally in trust_engine.py but moved here for modularity in services
# Note: The user requested to move simulate_sources and simulate_live_sources to services/simulation.py
# however they are currently defined in trust_engine.py. 
# I will keep them in trust_engine.py as well for backward compatibility if needed, 
# or just import them from trust_engine.py here.
# But for a true modular refactor, I will copy the logic here as well.

def simulate_sources(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Refactored to generate a deterministic, highly-educational dataset.
    - Source_A: Smooth, stable trend (Gold Standard)
    - Source_B: Realistic noise (High Quality Node)
    - Source_C: Erratic failure modes (Outlier Node)
    """
    rng = np.random.default_rng(seed)
    
    # Target exactly 50 rows for clarity
    n = 50
    timestamps = pd.date_range("2024-01-01 09:00:00", periods=n, freq="min")
    
    # ── Source A: Smooth, slowly increasing trend (Gold Standard) ──
    # Low variance, no sudden jumps.
    base_val = 100.0
    trend = np.linspace(0, 10, n)
    small_noise = rng.normal(0, 0.05, n)
    source_a = base_val + trend + small_noise
    
    # ── Source B: Gaussian Noise (1-2% variance) ──
    # Realistic but stable.
    noise_std = 0.2 # ~2% of the drift magnitude
    source_b = source_a + rng.normal(0, noise_std, n)

    # ── Source C: Erratic Failure Modes ──
    # Base is Source_A + noise
    source_c = source_a.copy() + rng.normal(0, 0.15, n)
    
    # 1. Spikes: 7 deterministic large spikes (±5.0)
    # These will easily cross Z > 2.5 and consensus thresholds.
    spike_indices = [5, 12, 18, 25, 32, 40, 48]
    for idx in spike_indices:
        side = 1.0 if idx % 2 == 0 else -1.0
        source_c[idx] += side * 6.0 # Large enough to be > 3 std

    # 2. Gradual Drift: 2 segments
    # Segment 1: Indices 8-11
    source_c[8:12] += np.linspace(0, 4, 4)
    # Segment 2: Indices 35-38
    source_c[35:39] -= np.linspace(0, 3, 4)
    
    # 3. Plateau Phase: 5 ticks of repeated values
    plateau_val = float(source_c[20])
    source_c[21:26] = plateau_val

    result = pd.DataFrame({
        "timestamp": timestamps,
        "Source_A":  source_a,
        "Source_B":  source_b,
        "Source_C":  source_c,
    })
    return result

# Simulation state — last observed raw value per source for stale-repeat detection.
_last_raw_values: dict[str, float | None] = {"Source_A": None, "Source_B": None, "Source_C": None}

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
      Source_A    = small Gaussian noise (±0.4% std) + 5% stale-repeat chance
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

    # ── Source A — high-quality, small noise ──────────────────────────────────
    val_a = _maybe_stale("Source_A",
                         true_value + rng.normal(0, 0.004 * true_value))

    # ── Source B — medium quality + 20% chance of shared bias with A ──────────
    shared_bias = rng.normal(0, 0.003 * true_value) if rng.random() < 0.20 else 0.0
    val_b = _maybe_stale("Source_B",
                         true_value + rng.normal(0, 0.008 * true_value) + shared_bias)

    # ── Source C — unstable: medium noise + 15% spike (±5%) ──────────────────
    spike_val = rng.choice([-1.0, 1.0]) * 0.05 * true_value if rng.random() < 0.15 else 0.0
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
