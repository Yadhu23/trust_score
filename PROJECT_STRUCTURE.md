# TrustLayer Lab – Project Architecture

## Folder Structure

```
trustscore/
│
├── services/           # Backend algorithms and API integrations
│   ├── btc_api.py      # Live Bitcoin price fetcher (Binance, CoinGecko, CoinCap)
│   └── simulation.py   # CSV-based multi-source data simulator
│
├── ui/                 # Streamlit dashboard modules
│   ├── app_ui.py       # Shared UI components
│   ├── components.py   # Reusable styled components (badges, cards)
│   ├── csv_ui.py       # CSV Historical Analysis mode
│   ├── home_ui.py      # Landing/home page
│   ├── insights.py     # Post-simulation insights panel
│   ├── live_ui.py      # Live API Mode (real-time BTC feed)
│   ├── research_lab.py # Research Lab: 8 analytical panels
│   └── test_ui.py      # Stress Testing Lab scenarios
│
├── trust_engine.py     # ★ Core trust scoring engine (anomaly, consensus, EMA)
├── api.py              # FastAPI REST endpoints (optional backend)
├── app.py              # ★ Main Streamlit entry point — run this
├── test_app.py         # Standalone legacy entry point (older structure)
├── debug_trust.py      # Debugging and introspection utilities
├── sample_data.csv     # Sample dataset for CSV Mode demo
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
└── LICENSE             # MIT License
```

---

## Data Flow

```
┌─────────────────┐
│  Data Ingestion │  CSV upload / Live BTC API / Stress scenario generator
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  Source Simulation  │  Derives Source_A (clean), Source_B (noise), Source_C (spikes)
└────────┬────────────┘
         │
         ▼
┌──────────────────┐
│  Consensus Engine│  Median-based consensus per tick. Z-score outlier detection.
└────────┬─────────┘
         │
         ▼
┌──────────────────────┐
│  Trust Scoring Engine│  EMA-smoothed trust state. Combines anomaly + consensus +
│  (trust_engine.py)   │  historical performance into a [0–1] composite score.
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Visualization Layer │  Streamlit + Plotly. Live placeholders via while loops +
│  (ui/ modules)       │  st.empty() — no page reloads, pure statistical inference.
└──────────────────────┘
```

---

## Key Modules

| Module | Role |
|--------|------|
| `trust_engine.py` | Anomaly detection, EMA trust updates, consensus scoring, classification |
| `ui/research_lab.py` | 8-tab deep analysis: reliability, consensus viz, deviation monitor, explainability, export |
| `ui/live_ui.py` | Real-time simulation loop with per-tick chart/metric updates |
| `ui/test_ui.py` | Deterministic scenario-driven simulation with validation rules |
| `ui/csv_ui.py` | Static historical analysis with highlighted anomaly tables |
| `ui/insights.py` | Post-run narrative insights across 6 analytical sections |
| `services/btc_api.py` | Multi-endpoint BTC price fetcher with offline fallback |

---

## Trust Score Formula

```
Trust Score = f(anomaly_score, consensus_score, historical_performance)

Where:
  anomaly_score       → rolling z-score based deviation detection
  consensus_score     → deviation from median of all sources
  historical_perf     → EMA of past performance (reliability index)

Final score ∈ [0.0, 1.0]
  ≥ 0.80 → ✅ Trusted
  ≥ 0.45 → ⚠️ Monitor
  < 0.45 → 🚨 Isolate
```
