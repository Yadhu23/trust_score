"""
test_app.py -> Unified TrustLayer Lab
-------------------------------------
TrustLayer – Statistical Multi-Source Data Trust Engine
Combined Entry Point:
1. 📂 CSV Mode (Static Analysis)
2. 📡 Live API Mode (Real-world BTC feed)
3. 🧪 Stress Testing Lab (Automated Scenarios)

Run with:
    streamlit run test_app.py
"""

import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

from trust_engine import (
    run_trust_pipeline,
    simulate_sources,
    simulate_live_sources,
    compute_anomaly_scores,
    compute_consensus_scores,
    update_historical_trust,
    compute_final_score,
    classify_trust,
    reset_realtime_state,
    interpret_source,
    # Real-time internals for deep integration in Live Mode
    _reliability_index,
    _trust_state,
    _score_history,
    _total_events,
    _successful_events,
    _conf_volatility_history,
    _low_variance_count,
    _SOURCES,
    _smoothed_score,
    classify_historic_status,
    compute_weighted_consensus,
    _value_store,
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & CSS
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrustLayer Lab",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    [data-testid="metric-container"] {
        background: #1e2130;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #2d3148;
    }
    h1 { color: #7eb8f7 !important; }
    h2, h3 { color: #a8c4e0 !important; }
    .badge-trusted { color: #2ecc71; font-weight: 700; }
    .badge-monitor { color: #f39c12; font-weight: 700; }
    .badge-isolate { color: #e74c3c; font-weight: 700; }
    .info-card {
        background: #1a1e2e;
        border-left: 4px solid #7eb8f7;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        font-size: 0.93rem;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# CONSTANTS & SCENARIOS
# ─────────────────────────────────────────────────────────────
SCENARIOS = {
    "🧪 1: Stable Baseline": {
        "desc": "Ensure system behaves calmly under normal noise. true_value = 100",
        "expected": "All trust scores remain high; no isolation.",
        "max_ticks": 50
    },
    "🧪 2: Sudden Spike Event": {
        "desc": "Ticks 1-50 normal, Tick 51 spike A (1.5x), then back to normal.",
        "expected": "A instant confidence drops sharply; recovers quickly.",
        "max_ticks": 70
    },
    "🧪 3: Two Weak Sources Colluding": {
        "desc": "A(high) vs B/C(low). Then B/C collude at 140 for 20 ticks.",
        "expected": "Weighted consensus protects A; B/C penalized.",
        "max_ticks": 40
    },
    "🧪 4: Gradual Drift": {
        "desc": "A = 100 + tick * 0.5. Slow decay over 50 ticks.",
        "expected": "A trust decays gradually; eventual isolation.",
        "max_ticks": 60
    },
    "🧪 5: Recovery Mode": {
        "desc": "B is noisy for 30 ticks, then stabilizes.",
        "expected": "B trust climbs back slowly after fixing itself.",
        "max_ticks": 70
    },
    "🧪 6: Single Source Mode": {
        "desc": "A runs alone. Spike at tick 40.",
        "expected": "Anomaly detection works without consensus.",
        "max_ticks": 60
    },
    "🧪 7: Total Conflict Chaos": {
        "desc": "A=100, B=150, C=200 constant conflict.",
        "expected": "System isolates all; no random fallback.",
        "max_ticks": 30
    },
    "🧪 8: Perfect Agreement": {
        "desc": "All sources identical for 100 ticks.",
        "expected": "Trust rise steadily to ceiling.",
        "max_ticks": 100
    }
}

BTC_SOURCES = [
    {"url": "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", "extract": lambda j: float(j["price"])},
    {"url": "https://api.coincap.io/v2/assets/bitcoin", "extract": lambda j: float(j["data"]["priceUsd"])},
    {"url": "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd", "extract": lambda j: float(j["bitcoin"]["usd"])},
]

# ─────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────
for key in ["mode", "running", "tick", "records", "value_buffers", "trust_state", "rng", "streaming", "live_records", "live_tick", "reliability_state", "smoothed_score_history"]:
    if key not in st.session_state:
        if key == "records" or key == "live_records": st.session_state[key] = []
        elif key == "tick" or key == "live_tick": st.session_state[key] = 0
        elif key == "running" or key == "streaming": st.session_state[key] = False
        elif key == "value_buffers": st.session_state[key] = {"Source_A": [], "Source_B": [], "Source_C": []}
        elif key == "trust_state": st.session_state[key] = {"Source_A": 0.5, "Source_B": 0.5, "Source_C": 0.5}
        elif key == "reliability_state": st.session_state[key] = {"Source_A": 0.7, "Source_B": 0.7, "Source_C": 0.7}
        elif key == "smoothed_score_history": st.session_state[key] = {"Source_A": [], "Source_B": [], "Source_C": []}
        elif key == "rng": st.session_state[key] = np.random.default_rng()
        elif key == "mode": st.session_state[key] = "📂 CSV Mode"

# ─────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────
def _render_interp_card(src, val, final, decision, conclusion):
    """Common UI block for source status."""
    if "Trusted" in decision: border = "#22c55e"
    elif "Monitor" in decision: border = "#f59e0b"
    else: border = "#ef4444"
    
    st.markdown(f"""
        <div style="border:2px solid {border}; border-radius:10px; padding:12px; margin-bottom:10px;">
            <div style="font-weight:700;">{src}</div>
            <div style="font-size:1.4rem; font-weight:800;">{val}</div>
            <div><span class="badge-{'trusted' if border=='#22c55e' else 'monitor' if border=='#f59e0b' else 'isolate'}">{decision}</span></div>
        </div>
        <div style="background:#12172a; border-left:3px solid #7eb8f7; border-radius:4px; padding:8px; font-size:0.8rem;">
            🧠 <b>Engine Conclusion:</b><br>{conclusion}
        </div>
    """, unsafe_allow_html=True)

def fetch_btc_price() -> float:
    for source in BTC_SOURCES:
        try:
            resp = requests.get(source["url"], timeout=3)
            resp.raise_for_status()
            return source["extract"](resp.json())
        except: continue
    # Fallback simulation if offline
    if "fall_btc" not in st.session_state: st.session_state.fall_btc = 65000.0
    st.session_state.fall_btc *= (1 + st.session_state.rng.normal(0, 0.001))
    return st.session_state.fall_btc

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ TrustLayer Lab")
    mode = st.radio("Ingestion Mode", ["📂 CSV Mode", "📡 Live API Mode", "🧪 Stress Lab"])
    st.session_state.mode = mode
    st.divider()
    
    if mode == "🧪 Stress Lab":
        st.header("🔬 Test Scenario")
        sc_name = st.selectbox("Select Scenario", list(SCENARIOS.keys()))
        sc_cfg = SCENARIOS[sc_name]
        st.info(f"**Goal:** {sc_cfg['desc']}")
        st.success(f"**Expectation:** {sc_cfg['expected']}")
    elif mode == "📡 Live API Mode":
        st.info("Streaming real-world BTC/USD price data across 3 simulated sources.")
        if st.button("🗑️ Reset Live Stream", use_container_width=True):
            st.session_state.live_records = []
            st.session_state.live_tick = 0
            st.session_state.streaming = False
            reset_realtime_state()
            st.rerun()

# ─────────────────────────────────────────────────────────────
# MAIN APP BODY
# ─────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════
# 1. CSV MODE
# ══════════════════════════════════════════════════════════════
if mode == "📂 CSV Mode":
    st.header("📂 CSV Mode — Historical Analysis")
    uploaded_file = st.file_uploader("Upload CSV (timestamp, value)", type=["csv"])
    
    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
        if {"timestamp", "value"}.issubset(raw_df.columns):
            results = run_trust_pipeline(raw_df)
            last_ts = results["timestamp"].iloc[-1]
            latest = results[results["timestamp"] == last_ts]
            
            st.subheader("Latest Timestamp Snapshot")
            cols = st.columns(3)
            for i, src in enumerate(["Source_A", "Source_B", "Source_C"]):
                row = latest[latest["source"] == src].iloc[0]
                interp = interpret_source(row["final_score"], 0.7, row["historical_trust"], row["final_score"], row["decision"])
                with cols[i]:
                    _render_interp_card(src, f"{row['value']:.3f}", row["final_score"], row["decision"], interp["recommendation"])
            
            st.divider()
            st.subheader("Trust Evolution")
            chart_data = results.pivot(index="timestamp", columns="source", values="final_score")
            st.line_chart(chart_data)
        else:
            st.error("CSV must have 'timestamp' and 'value' columns.")
    else:
        st.info("Upload a CSV to process historical datasets.")

# ══════════════════════════════════════════════════════════════
# 2. LIVE API MODE
# ══════════════════════════════════════════════════════════════
elif mode == "📡 Live API Mode":
    st.header("📡 Live API Mode — Bitcoin Feed")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        if not st.session_state.streaming:
            if st.button("▶️ Start Live Stream", use_container_width=True):
                reset_realtime_state()
                st.session_state.live_records = []
                st.session_state.live_tick = 0
                st.session_state.streaming = True
                st.rerun()
        else:
            if st.button("⏹️ Stop Live Stream", use_container_width=True):
                st.session_state.streaming = False
                st.rerun()
    
    if st.session_state.streaming:
        price = fetch_btc_price()
        sources_vals, _ = simulate_live_sources(price, st.session_state.rng)
        
        st.session_state.live_tick += 1
        tick = st.session_state.live_tick
        ts = time.strftime("%H:%M:%S")
        
        # 1. Update Buffers first
        for src, val in sources_vals.items():
            _value_store[src].append(float(val))

            # 2. Run Engine Logic
            series = pd.Series(_value_store[src])
            anom_df = compute_anomaly_scores(series, window=5)
            anom_score = float(anom_df["anomaly_score"].iloc[-1]) if not anom_df.empty else 0.0
            if np.isnan(anom_score): anom_score = 0.0
            
            # Weighted Consensus (Robust)
            active_srcs = [s for s in _SOURCES if len(_value_store[s]) > 0]
            sources_ready = len(active_srcs) >= 2
            
            if sources_ready:
                # Only include sources that have at least one value
                row_dict = {s: [_value_store[s][-1]] for s in active_srcs}
                cons_df = compute_consensus_scores(pd.DataFrame(row_dict), reliabilities=_reliability_index)
                # For sources not in active_srcs, use default cs
                if src in active_srcs:
                    cs = float(cons_df.iloc[0][f"consensus_score_{src[-1]}"])
                    w_cons = float(cons_df.iloc[0]["median_value"])
                else:
                    cs = 0.5
                    w_cons = val
            else:
                cs = 0.5
                w_cons = val

            # Continuous Performance
            tolerance = max(0.5, 0.08 * abs(w_cons)) + 1e-9
            perf = float(np.clip(1.0 - abs(val - w_cons) / tolerance, 0.0, 0.995))
            
            # Update EMA (with Minority Protection in trust_engine.py)
            old_trust = _trust_state[src]
            new_trust, _ = update_historical_trust(old_trust, anom_score, cs, performance=perf)
            _trust_state[src] = new_trust
            # Slower reliability update (alpha 0.005)
            _reliability_index[src] = 0.995 * _reliability_index[src] + 0.005 * perf
            
            # Final Score
            raw_score = compute_final_score(new_trust, anom_score, cs)
            _score_history[src].append(raw_score)
            smooth = _smoothed_score(src)
            
            # Extract disagreement info for classification gate
            dis_idx = float(cons_df.iloc[0]["disagreement_index"]) if cons_df is not None else 0.0
            is_ex = bool(cons_df.iloc[0]["is_extreme_chaos"]) if cons_df is not None and "is_extreme_chaos" in cons_df.columns else False
            
            decision = classify_trust(
                smooth, 
                disagreement_index=dis_idx, 
                anomaly_score=anom_score,
                weighted_mean=w_cons,
                is_extreme_chaos=is_ex
            )
            
            # Historic count
            _total_events[src] += 1
            if perf >= 0.6: _successful_events[src] += 1
            h_trust = _successful_events[src] / _total_events[src]
            
            st.session_state.live_records.append({
                "tick": tick, "ts": ts, "source": src, "value": val,
                "final": smooth, "decision": decision, "h_trust": h_trust
            })
            
        time.sleep(1.5)
        st.rerun()

    if st.session_state.live_records:
        df = pd.DataFrame(st.session_state.live_records)
        latest = df[df["tick"] == df["tick"].max()]
        st.subheader(f"Current BTC Price: ${latest.iloc[0]['value']:,.2f}")
        
        lcols = st.columns(3)
        for i, src in enumerate(["Source_A", "Source_B", "Source_C"]):
            row = latest[latest["source"] == src].iloc[0]
            interp = interpret_source(row["final"], 0.7, row["h_trust"], row["final"], row["decision"])
            with lcols[i]:
                _render_interp_card(src, f"${row['value']:,.2f}", row["final"], row["decision"], interp["recommendation"])
        
        st.divider()
        st.line_chart(df.pivot_table(index="tick", columns="source", values="final"))

# ══════════════════════════════════════════════════════════════
# 3. STRESS LAB
# ══════════════════════════════════════════════════════════════
elif mode == "🧪 Stress Lab":
    st.header(f"🧪 Stress Lab: {sc_name}")
    
    sl1, sl2 = st.columns(2)
    with sl1:
        if not st.session_state.running:
            if st.button("▶ Run Scenario", use_container_width=True):
                reset_realtime_state()
                st.session_state.records = []
                st.session_state.tick = 0
                st.session_state.value_buffers = {"Source_A": [], "Source_B": [], "Source_C": []}
                st.session_state.smoothed_score_history = {"Source_A": [], "Source_B": [], "Source_C": []}
                st.session_state.running = True
                st.rerun()
        else:
            if st.button("⏹ Stop Lab", use_container_width=True):
                st.session_state.running = False
                st.rerun()
    with sl2:
        if st.button("🔄 Clear Lab History", use_container_width=True):
            st.session_state.records = []
            st.session_state.tick = 0
            st.rerun()

    if st.session_state.running:
        st.session_state.tick += 1
        tick = st.session_state.tick
        if tick > sc_cfg["max_ticks"]:
            st.session_state.running = False
            st.rerun()
            
        # Data Generator Logic
        base = 100.0
        rng = st.session_state.rng
        if sc_name == "🧪 1: Stable Baseline": v = {"Source_A": base + rng.normal(0,0.5), "Source_B": base+rng.normal(0,0.8), "Source_C": base+rng.normal(0,1)}
        elif sc_name == "🧪 2: Sudden Spike Event": v = {"Source_A": base*1.5 if tick==51 else base+rng.normal(0,0.5), "Source_B": base+rng.normal(0,0.8), "Source_C": base+rng.normal(0,1)}
        elif sc_name == "🧪 3: Two Weak Sources Colluding":
            if tick <= 10: v = {"Source_A": base+rng.normal(0,0.1), "Source_B": base+rng.normal(0,10), "Source_C": base+rng.normal(0,12)}
            else: v = {"Source_A": 100.0, "Source_B": 140.0, "Source_C": 142.0}
        elif sc_name == "🧪 4: Gradual Drift": v = {"Source_A": 100.0+(tick*0.5), "Source_B": 100.0, "Source_C": 100.0}
        elif sc_name == "🧪 5: Recovery Mode": v = {"Source_A": base, "Source_B": base+rng.normal(0,20) if tick<=30 else base+rng.normal(0,0.5), "Source_C": base}
        elif sc_name == "🧪 6: Single Source Mode": v = {"Source_A": base*1.8 if tick==40 else base+rng.normal(0,0.5)}
        elif sc_name == "🧪 7: Total Conflict Chaos": v = {"Source_A": 100.0, "Source_B": 150.0, "Source_C": 200.0}
        elif sc_name == "🧪 8: Perfect Agreement": v = {"Source_A": base, "Source_B": base, "Source_C": base}
        
        # Lab Processing
        df_row = pd.DataFrame([v])
        # Use reliability if available in state
        reliabs = st.session_state.get("reliability_state", {s: 0.7 for s in v})
        cons_df = compute_consensus_scores(df_row, reliabilities=reliabs) if len(v) >= 2 else None
        
        for s, val in v.items():
            st.session_state.value_buffers[s].append(val)
            abuf = pd.Series(st.session_state.value_buffers[s])
            anom = float(compute_anomaly_scores(abuf).iloc[-1]["anomaly_score"]) if not abuf.empty else 0.0
            if np.isnan(anom): anom = 0.0
            
            cs = float(cons_df.iloc[0][f"consensus_score_{s[-1]}"]) if cons_df is not None else 0.5
            w_cons = float(cons_df.iloc[0]["median_value"]) if cons_df is not None else val
            
            # Performance for reliability
            tol = max(0.5, 0.08 * abs(w_cons)) + 1e-9
            perf = float(np.clip(1.0 - abs(val - w_cons) / tol, 0.0, 0.995))

            old_t = st.session_state.trust_state.get(s, 0.5)
            new_t, _ = update_historical_trust(old_t, anom, cs, performance=perf)
            st.session_state.trust_state[s] = new_t
            
            # Update reliability state in session
            old_rel = reliabs.get(s, 0.7)
            new_rel = 0.995 * old_rel + 0.005 * perf
            st.session_state.reliability_state[s] = new_rel
            
            final = compute_final_score(new_t, anom, cs)
            
            # Smoothing (EMA 0.2)
            hist = st.session_state.smoothed_score_history[s]
            if not hist:
                smoothed = final
            else:
                smoothed = 0.2 * final + 0.8 * hist[-1]
            st.session_state.smoothed_score_history[s].append(smoothed)
            if len(st.session_state.smoothed_score_history[s]) > 10: 
                st.session_state.smoothed_score_history[s].pop(0)

            # Extract disagreement info for classification gate
            dis_idx = float(cons_df.iloc[0]["disagreement_index"]) if cons_df is not None else 0.0
            is_ex = bool(cons_df.iloc[0]["is_extreme_chaos"]) if cons_df is not None and "is_extreme_chaos" in cons_df.columns else False
            
            st.session_state.records.append({
                "tick": tick, "source": s, "value": val, "final": final,
                "smoothed": smoothed,
                "decision": classify_trust(
                    smoothed, 
                    disagreement_index=dis_idx, 
                    anomaly_score=anom,
                    weighted_mean=w_cons,
                    is_extreme_chaos=is_ex
                ), 
                "historic": new_t,
                "disagreement_index": dis_idx,
                "anomaly_score": anom,
                "is_extreme_chaos": is_ex
            })
        time.sleep(0.1)
        st.rerun()

    if st.session_state.records:
        rdf = pd.DataFrame(st.session_state.records)
        rlast = rdf[rdf["tick"] == rdf["tick"].max()]
        st.subheader(f"Current Tick: {st.session_state.tick} / {sc_cfg['max_ticks']}")
        rcols = st.columns(3)
        for i, s in enumerate(["Source_A", "Source_B", "Source_C"]):
            if s in rlast["source"].values:
                row = rlast[rlast["source"] == s].iloc[0]
                interp = interpret_source(
                    row["final"], 
                    0.7, 
                    row["historic"], 
                    row["final"], 
                    row["decision"],
                    disagreement_index=row.get("disagreement_index", 0.0),
                    anomaly_score=row.get("anomaly_score", 0.0),
                    weighted_mean=row["value"] # Best proxy if we don't store w_cons
                )
                with rcols[i]:
                    _render_interp_card(s, f"{row['value']:.2f}", row["final"], row["decision"], interp["recommendation"])
            else:
                with rcols[i]: st.metric(s, "OFFLINE")
        st.divider()
        st.line_chart(rdf.pivot_table(index="tick", columns="source", values="final"))

st.sidebar.divider()
st.sidebar.caption("TrustLayer Lab Unified v1.0 🛡️")
