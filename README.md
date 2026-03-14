# 🛡️ TrustLayer Lab – Statistical Multi-Source Data Trust Engine

> A real-time data reliability analysis laboratory. No machine learning — pure statistics.

---

## What Is TrustLayer?

TrustLayer evaluates the reliability of multiple data sources using statistical trust scoring. It cross-references three independent data streams, detects statistical drift and anomalies, computes consensus values, and isolates untrustworthy sources — all in real time.

Built during a 20-hour hackathon. No ML. Pure math.

---

## Features

| Feature | Description |
|---------|-------------|
| 📂 **CSV Historical Analysis** | Upload any time-series CSV — TrustLayer simulates 3 sources and runs full trust analysis |
| 📡 **Live API Mode** | Streams real-world Bitcoin price from 3 APIs (Binance, CoinGecko, CoinCap) |
| 🧪 **Stress Testing Lab** | 10 deterministic scenarios: stable baseline, gradual drift, malicious spike, chaos, recovery, and more |
| 🛡️ **Trust Score Engine** | EMA-smoothed composite trust score per source updated every tick |
| 🤝 **Consensus Engine** | Median-based consensus with deviation scoring |
| 📡 **Deviation Monitoring** | Per-tick deviation heatmap with outlier risk badges |
| 📊 **Historical Reliability Tracking** | Stability score, reliability %, and average deviation per source |
| 🔍 **Explainable Trust Scoring** | Per-source breakdown of deviation penalty, variance penalty, and consistency bonus |
| 🔬 **Research Lab** | 8-tab deep analysis dashboard: reliability, consensus viz, explanation, formula viewer, data export |
| 📤 **Data Export** | Download full trust report as CSV or JSON |

---

## How to Run

```bash
git clone <your-repo-url>
cd trustscore
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Project Structure

```
trustscore/
├── services/           # BTC API fetcher and CSV simulator
├── ui/                 # All Streamlit UI modules
│   ├── csv_ui.py       # CSV mode
│   ├── live_ui.py      # Live API mode
│   ├── test_ui.py      # Stress Lab
│   ├── research_lab.py # Research Lab (8 analytical tabs)
│   └── insights.py     # Post-simulation insights
├── trust_engine.py     # ★ Core scoring engine
├── app.py              # ★ Main entry point
├── api.py              # FastAPI REST endpoints
├── sample_data.csv     # Sample dataset for demo
└── requirements.txt    # Python dependencies
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for full architecture documentation.

---

## Example Dataset

A `sample_data.csv` is included in the root. It contains timestamped numerical values that TrustLayer uses to simulate Sources A, B, and C (clean, noisy, and spike-prone respectively).

Upload it in **CSV Mode** to see a full analysis immediately.

---

## Trust Score Logic

```
Trust Score ∈ [0.0, 1.0]

✅ ≥ 0.80  →  Trusted
⚠️ ≥ 0.45  →  Monitor
🚨 < 0.45  →  Isolate
```

Computed from: `anomaly_score × consensus_score × historical_performance_EMA`

---

## Tech Stack

- **Python** – Core logic
- **Pandas / NumPy** – Statistical computations
- **Streamlit** – Dashboard UI
- **Plotly** – Interactive charts
- **FastAPI** – Optional REST API layer
- **Requests** – Live price feed integration

---

## License

MIT License — see [LICENSE](LICENSE).
