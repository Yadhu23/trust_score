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

    # Source C – occasional extreme spikes (~10% of rows)
    spike_mask = rng.random(n) < 0.10
    spike_magnitude = 5 * np.std(source_a) if np.std(source_a) > 0 else 5.0
    spike_values = rng.choice([-1, 1], n) * spike_magnitude
    source_c = source_a + spike_mask * spike_values

    result = pd.DataFrame({
        "timestamp": df["timestamp"].values,
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
