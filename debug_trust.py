import pandas as pd
from trust_engine import run_trust_pipeline

# Create dummy data
df = pd.DataFrame({
    "timestamp": ["2026-01-01 00:00:00", "2026-01-01 00:00:01", "2026-01-01 00:00:02", "2026-01-01 00:00:03", "2026-01-01 00:00:04", "2026-01-01 00:00:05"],
    "value": [100.0, 101.0, 102.0, 99.0, 100.0, 101.0]
})

results = run_trust_pipeline(df)
print("Columns in results:", results.columns)
print("Unique sources in results:", results["source"].unique())
print("Source_A historical_trust sample:", results[results["source"] == "Source_A"]["historical_trust"].head())
print("Source_A row count:", len(results[results["source"] == "Source_A"]))

hist_pivot = results.pivot(index="timestamp", columns="source", values="historical_trust")
print("Pivot columns:", hist_pivot.columns)
print("Pivot head:\n", hist_pivot.head())
