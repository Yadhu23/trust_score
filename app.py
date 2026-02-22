"""
app.py
------
TrustLayer – Statistical Multi-Source Data Trust Engine
Streamlit UI: CSV mode + Live API Mode (Bitcoin price feed).

Run with:
    streamlit run app.py
"""

import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

from trust_engine import (
    run_trust_pipeline,
    simulate_sources,
    simulate_live_sources,
    compute_anomaly_scores,
    compute_consensus_scores,
    process_new_data,
    reset_realtime_state,
    interpret_source,
)



# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrustLayer",
    page_icon="🛡️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS  (dark-card style, coloured badges)
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Load Inter Font from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@600&display=swap');

    /* Main background & Global Typography */
    .stApp { 
        background-color: #0d1117; 
        color: #FFFFFF; 
        font-family: 'Inter', 'Segoe UI', sans-serif; 
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #161b22;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #30363d;
        transition: all 0.2s ease;
    }
    [data-testid="metric-container"]:hover { 
        transform: translateY(-2px); 
        border-color: #58a6ff; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    /* Section headers */
    .section-header {
        font-size: 24px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #FFFFFF;
        margin: 40px 0 20px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .section-header::after {
        content: "";
        flex: 1;
        height: 2px;
        background: #30363d;
    }

    /* Trusted / Monitor / Isolate badge colours */
    .badge-trusted { 
        color: #3fb950; 
        font-size: 16px;
        font-weight: 800; 
        background: rgba(63, 185, 80, 0.15); 
        padding: 4px 12px; 
        border-radius: 6px; 
        border: 1px solid rgba(63, 185, 80, 0.3); 
    }
    .badge-monitor { 
        color: #d29922; 
        font-size: 16px;
        font-weight: 800; 
        background: rgba(210, 153, 34, 0.15); 
        padding: 4px 12px; 
        border-radius: 6px; 
        border: 1px solid rgba(210, 153, 34, 0.3); 
    }
    .badge-isolate { 
        color: #f85149; 
        font-size: 16px;
        font-weight: 800; 
        background: rgba(248, 81, 73, 0.15); 
        padding: 4px 12px; 
        border-radius: 6px; 
        border: 1px solid rgba(248, 81, 73, 0.3); 
    }

    /* Anomaly highlight */
    .anomaly-badge { 
        background: #f85149; 
        color: #FFFFFF; 
        font-size: 14px; 
        padding: 4px 10px; 
        border-radius: 6px; 
        font-weight: 800; 
        animation: pulse 2s infinite; 
    }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.7; } 100% { opacity: 1; } }

    /* Primary Source Glow */
    .primary-glow {
        border: 2px solid #3fb950 !important;
        box-shadow: 0 0 20px rgba(63, 185, 80, 0.4);
        position: relative;
    }
    .primary-badge {
        position: absolute;
        top: -14px;
        left: 16px;
        background: #3fb950;
        color: #FFFFFF;
        font-size: 12px;
        font-weight: 800;
        padding: 3px 12px;
        border-radius: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }

    /* Trust Bar */
    .trust-bar-container {
        width: 100%;
        height: 10px;
        background: #30363d;
        border-radius: 6px;
        margin: 14px 0;
        overflow: hidden;
    }
    .trust-bar-fill {
        height: 100%;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Info card */
    .info-card {
        background: #161b22;
        border-left: 5px solid #58a6ff;
        border-radius: 10px;
        padding: 18px 24px;
        margin: 12px 0;
        font-size: 16px;
        color: #e6edf3;
        line-height: 1.6;
        border: 1px solid #30363d;
    }

    /* Metric Alignment & Typography */
    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        font-size: 15px;
    }
    .metric-label { color: #B3B3B3; font-weight: 600; }
    .metric-value { 
        font-weight: 700; 
        font-size: 18px;
        color: #FFFFFF;
        font-family: 'JetBrains Mono', monospace; 
    }

    /* Typography Overrides for helper functions */
    .source-name-header {
        font-size: 20px;
        font-weight: 700;
        color: #8b949e;
    }
    .price-value-hero {
        font-size: 32px;
        font-weight: 800;
        color: #FFFFFF;
        margin: 12px 0;
    }
    .metric-sub-header {
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #8b949e;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.title("🛡️ TrustLayer")
st.subheader("Statistical Multi-Source Data Trust Engine")
st.markdown(
    """
    Upload a CSV with **`timestamp`** and **`value`** columns.  
    TrustLayer simulates **three independent data sources** from your dataset,
    detects anomalies with Z-score analysis, measures cross-source consensus,
    and tracks a live **trust score** for each source — no ML required.
    """
)
st.divider()

# ─────────────────────────────────────────────────────────────
# HELPER – Bitcoin price fetch
# ─────────────────────────────────────────────────────────────
BTC_SOURCES = [
    # CoinGecko (free, sometimes rate-limited)
    {
        "url": "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
        "extract": lambda j: float(j["bitcoin"]["usd"]),
    },
    # Binance (no key needed, very reliable)
    {
        "url": "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
        "extract": lambda j: float(j["price"]),
    },
    # CoinCap (no key needed)
    {
        "url": "https://api.coincap.io/v2/assets/bitcoin",
        "extract": lambda j: float(j["data"]["priceUsd"]),
    },
]

# Seed for the simulated fallback price (keeps it stable between calls)
_sim_price = {"value": 67000.0}

def fetch_btc_price() -> float:
    """
    Try three free price APIs in order.
    If all fail (offline / rate-limited), return a simulated price
    so the live stream keeps running in demo mode.
    """
    for source in BTC_SOURCES:
        try:
            resp = requests.get(
                source["url"],
                timeout=5,
                headers={"User-Agent": "TrustLayer/1.0"},
            )
            resp.raise_for_status()
            return source["extract"](resp.json())
        except Exception:
            continue   # try next source

    # All APIs failed → simulate a realistic BTC price walk
    rng = np.random.default_rng()
    _sim_price["value"] = _sim_price["value"] * (1 + rng.normal(0, 0.002))
    return round(_sim_price["value"], 2)


# ─────────────────────────────────────────────────────────────
# TEST SCENARIOS & HELPERS
# ─────────────────────────────────────────────────────────────
TEST_SCENARIOS = {
    "Normal": {
        "desc": "Stable Baseline: system behaves calmly under normal noise.",
        "expected": "All trust scores remain high.",
        "max_ticks": 40
    },
    "Single Source Failure": {
        "desc": "Sudden Spike: Source A spikes significantly at tick 20.",
        "expected": "Source A trust drops and recovers.",
        "max_ticks": 40
    },
    "Two Weak Collude": {
        "desc": "Sources B & C deviate together from the ground truth.",
        "expected": "System identifies anomaly in both; protects Source A.",
        "max_ticks": 40
    },
    "Gradual Drift": {
        "desc": "Source A slowly drifts away from the true value.",
        "expected": "Trust decays gradually; eventual isolation.",
        "max_ticks": 40
    },
    "Chaos": {
        "desc": "Total Conflict: all sources provide wildly different values.",
        "expected": "System isolates all; high disagreement index.",
        "max_ticks": 40
    },
    "Recovery": {
        "desc": "Source B is noisy initially, then stabilizes.",
        "expected": "Trust climbs back after behavior improves.",
        "max_ticks": 40
    }
}

def _render_interp_card(src, val, final, decision, conclusion):
    """Common UI block for source status (Test Mode)."""
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

def render_final_system_decision(latest_df):
    """
    Renders the Final System Decision block with reliability metrics.
    Requirement: Shows System Status, Recommended Value, Most Reliable Source, System Confidence, and Insight.
    """
    if latest_df is None or latest_df.empty:
        return

    # ── 1. Global System Confidence Meter ────────────────────────
    # mean(trust metrics of active sources)
    score_col = 'trust_score' if 'trust_score' in latest_df.columns else 'smoothed' if 'smoothed' in latest_df.columns else 'final_score'
    avg_conf = latest_df[score_col].mean()
    
    if avg_conf >= 0.80:   conf_lvl, conf_icon, conf_col = "High Confidence", "🟢", "#2ecc71"
    elif avg_conf >= 0.50: conf_lvl, conf_icon, conf_col = "Moderate Confidence", "🟡", "#f1c40f"
    else:                  conf_lvl, conf_icon, conf_col = "Low Confidence", "🔴", "#e74c3c"

    st.markdown(f"""
        <div style="background:#1e2130; border:1px solid #2d3148; border-radius:10px; padding:15px; margin-bottom:20px; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:0.85rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em; font-weight:700;">System Confidence Meter</span><br>
                <span style="font-size:1.2rem; font-weight:800; color:{conf_col};">{conf_icon} {conf_lvl} ({avg_conf*100:.1f}%)</span>
            </div>
            <div style="text-align:right;">
                <span style="font-size:0.75rem; color:#94a3b8;">AGGREGATE ENGINE HEALTH</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── 2. Final Decision Logic (Requirement PART 2 & 4) ────────────────────────
    # Determine Status and Recommended Value
    trusted = latest_df[latest_df['decision'].str.contains("Trusted")]
    monitors = latest_df[latest_df['decision'].str.contains("Monitor")]
    
    # Chaos Safety check
    is_chaos = any("Isolate" in d for d in latest_df['decision']) and len(trusted) == 0
    
    status = "System Uncertain"
    status_icon = "🔴"
    rec_val = None
    
    # Most Reliable Source Logic
    # active_sources = sources where classification != "Isolate"
    active_sources = latest_df[~latest_df['decision'].str.contains("Isolate")]
    best_source = None
    
    if is_chaos:
        status = "System Uncertain"
        status_icon = "❓"
        rec_val = None
        best_source = None
    else:
        # Determine status and weighted mean
        if not trusted.empty:
            status = "Operational"
            status_icon = "🟢"
            rel_col = 'reliability_index' if 'reliability_index' in latest_df.columns else None
            if rel_col:
                total_rel = trusted[rel_col].sum()
                rec_val = (trusted['value'] * trusted[rel_col]).sum() / total_rel if total_rel > 0 else trusted['value'].mean()
            else:
                rec_val = trusted['value'].mean()
        elif not monitors.empty:
            status = "Caution"
            status_icon = "🟡"
            rel_col = 'reliability_index' if 'reliability_index' in latest_df.columns else None
            if rel_col:
                total_rel = monitors[rel_col].sum()
                rec_val = (monitors['value'] * monitors[rel_col]).sum() / total_rel if total_rel > 0 else monitors['value'].mean()
            else:
                rec_val = monitors['value'].mean()
        
        # Determine Best Source based on NEW logic
        # historical_score = 0.7 * historic_trust + 0.3 * reliability_index
        if not active_sources.empty:
            # Ensure we have the required columns
            if 'historic_trust' in active_sources.columns and 'reliability_index' in active_sources.columns:
                temp_sources = active_sources.copy()
                temp_sources['historical_score'] = 0.7 * temp_sources['historic_trust'] + 0.3 * temp_sources['reliability_index']
                best_source = temp_sources.loc[temp_sources['historical_score'].idxmax()]
            else:
                # Fallback to score_col if metrics are missing
                best_source = active_sources.loc[active_sources[score_col].idxmax()]

    # ── 3. Render Final Display Block (Requirement PART 3) ──────────────────────
    st.subheader("🏆 Final System Decision")
    
    val_str = f"${rec_val:,.2f}" if rec_val is not None else "None (System Uncertain)"
    if best_source is not None:
        # Determine source name
        if 'source' in best_source and pd.notna(best_source['source']):
            src_name = best_source['source']
        elif hasattr(best_source, 'name') and best_source.name and "Source_" in str(best_source.name):
            src_name = best_source.name
        else:
            src_name = "Unknown Source"
            
        # Display the score if available, otherwise just the name
        h_score = best_source.get('historical_score')
        if pd.notna(h_score):
            best_source_display = f"{src_name} ({h_score:.3f} HIST)"
        else:
            best_source_display = f"{src_name} ({best_source[score_col]:.3f})"
    else:
        best_source_display = "None (System Uncertain)"

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown(f"""
            <div style="background:#1e2130; padding:20px; border-radius:10px; border:2px solid {conf_col}; height: 120px;">
                <span style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">System Status</span><br>
                <span style="font-size:1.4rem; font-weight:800;">{status_icon} {status}</span>
            </div>
        """, unsafe_allow_html=True)
    with sc2:
        st.markdown(f"""
            <div style="background:#1e2130; padding:20px; border-radius:10px; border:1px solid #2d3148; height: 120px;">
                <span style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">Recommended Value</span><br>
                <span style="font-size:1.4rem; font-weight:800; color:#7eb8f7;">{val_str}</span>
            </div>
        """, unsafe_allow_html=True)
    with sc3:
        st.markdown(f"""
            <div style="background:#1e2130; padding:20px; border-radius:10px; border:1px solid #2d3148; height: 120px;">
                <span style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">Most Reliable Source</span><br>
                <span style="font-size:1.2rem; font-weight:800; color:#a8c4e0;">{best_source_display}</span>
            </div>
        """, unsafe_allow_html=True)

    # Human-Readable Explanation
    is_spike = any("Isolate" in d for d in latest_df['decision']) and not trusted.empty
    is_single = len(latest_df) < 2
    
    if is_chaos:
        explanation = "Extreme cross-source disagreement detected. No reliable consensus."
    elif is_spike:
        explanation = "Outlier deviation detected and isolated."
    elif is_single:
        explanation = "Operating without cross-source validation."
    else:
        explanation = "Sources are statistically aligned. Trust stabilizing."

    st.info(f"🧬 **System Insight:** {explanation}")


# ─────────────────────────────────────────────────────────────
# SESSION STATE – initialise once per session
# ─────────────────────────────────────────────────────────────
if "streaming" not in st.session_state:
    st.session_state.streaming = False
if "live_records" not in st.session_state:
    # Each entry: {tick, timestamp_label, Source_A, Source_B, Source_C,
    #              trust_A, trust_B, trust_C, decision_A, decision_B, decision_C}
    st.session_state.live_records = []
if "live_tick" not in st.session_state:
    st.session_state.live_tick = 0
if "live_rng" not in st.session_state:
    st.session_state.live_rng = np.random.default_rng()

# 🧪 TEST SCENARIO STATE (Strict Separation)
if "test_records" not in st.session_state:
    st.session_state.test_records = []
if "test_tick" not in st.session_state:
    st.session_state.test_tick = 0
if "test_rng" not in st.session_state:
    st.session_state.test_rng = np.random.default_rng()
if "test_value_buffers" not in st.session_state:
    st.session_state.test_value_buffers = {"Source_A": [], "Source_B": [], "Source_C": []}
if "test_trust_state" not in st.session_state:
    st.session_state.test_trust_state = {"Source_A": 0.5, "Source_B": 0.5, "Source_C": 0.5}
if "test_reliability_state" not in st.session_state:
    st.session_state.test_reliability_state = {"Source_A": 0.7, "Source_B": 0.7, "Source_C": 0.7}
if "test_smoothed_score_history" not in st.session_state:
    st.session_state.test_smoothed_score_history = {"Source_A": [], "Source_B": [], "Source_C": []}
if "test_successful_events" not in st.session_state:
    st.session_state.test_successful_events = {"Source_A": 0, "Source_B": 0, "Source_C": 0}
if "test_total_events" not in st.session_state:
    st.session_state.test_total_events = {"Source_A": 0, "Source_B": 0, "Source_C": 0}
if "test_running" not in st.session_state:
    st.session_state.test_running = False

# 🎮 DEMO TRIGGER STATE (Stress Lab Only)
if "trigger_spike" not in st.session_state:
    st.session_state.trigger_spike = False
if "trigger_collusion" not in st.session_state:
    st.session_state.trigger_collusion = False
if "trigger_chaos" not in st.session_state:
    st.session_state.trigger_chaos = False

# 🛡️ MODE SWITCH GUARD
if "last_mode" not in st.session_state:
    st.session_state.last_mode = None


# ─────────────────────────────────────────────────────────────
# SIDEBAR – controls & mode selector
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")

    # ── Mode selector ─────────────────────────────────────────
    mode = st.radio(
        "Ingestion Mode",
        ["📂 CSV Mode", "📡 Live API Mode", "🧪 Test Scenario Mode"],
        help="CSV Mode: upload a file. Live API Mode: stream real BTC prices. Test Mode: automated scenarios.",
    )
    st.divider()

    # Reset detection
    if st.session_state.last_mode is not None and st.session_state.last_mode != mode:
        st.session_state.live_records = []
        st.session_state.live_tick = 0
        st.session_state.streaming = False
        st.session_state.test_records = []
        st.session_state.test_tick = 0
        st.session_state.test_running = False
        st.session_state.trigger_spike = False
        st.session_state.trigger_collusion = False
        st.session_state.trigger_chaos = False
        reset_realtime_state()
        st.toast(f"Mode switched to {mode}. State reset.", icon="🔄")
    
    st.session_state.last_mode = mode

    if mode == "📂 CSV Mode":
        uploaded_file = st.file_uploader(
            "Upload CSV (timestamp, value)",
            type=["csv"],
        )
    else:
        uploaded_file = None
        st.info(
            "📡 **Live API Mode** active.\n\n"
            "Fetches real Bitcoin (BTC/USD) price from CoinGecko "
            "every 2 seconds and runs it through the TrustLayer engine."
        )
        if st.button("🗑️ Reset Live Data", width="stretch"):
            st.session_state.live_records = []
            st.session_state.live_tick = 0
            st.session_state.streaming = False
            reset_realtime_state()
            st.rerun()
    st.divider()

    st.header("📖 How It Works")
    st.markdown(
        """
        <div class="info-card">
        <b>1. Source Simulation</b><br>
        Three sources are derived from your CSV:<br>
        • <b>Source A</b> – original data<br>
        • <b>Source B</b> – data + tiny noise<br>
        • <b>Source C</b> – data + occasional spikes
        </div>

        <div class="info-card">
        <b>2. Anomaly Detection</b><br>
        Rolling Z-score (window=5).
        |Z| > 2.5 → anomaly flag.
        </div>

        <div class="info-card">
        <b>3. Cross-Source Consensus</b><br>
        Deviation from the median value
        across all 3 sources at each timestamp.
        </div>

        <div class="info-card">
        <b>4. Historical Trust</b><br>
        Exponential smoothing:<br>
        <code>trust = 0.8·old + 0.2·perf</code>
        </div>

        <div class="info-card">
        <b>5. Final Score</b><br>
        <code>0.4·trust + 0.3·(1-anom) + 0.3·cons</code><br>
        ✅ ≥ 0.75 Trusted &nbsp;
        ⚠️ 0.4–0.75 Monitor &nbsp;
        🚨 &lt; 0.4 Isolate
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# MAIN CONTENT – branch on mode
# ─────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════
# ██  LIVE API MODE
# ══════════════════════════════════════════════════════════════
if mode == "📡 Live API Mode":
    # ── Top Navigation & Status ──────────────────────────────
    nav_c1, nav_c2 = st.columns([2, 1])
    with nav_c1:
        st.markdown(
            """
            <div style="background:#161b22; padding:10px 16px; border-radius:8px; border:1px solid #30363d;">
                <span style="color:#8b949e; font-size:0.8rem; text-transform:uppercase; font-weight:700;">Current Mode</span><br>
                <span style="font-size:1.1rem; font-weight:700; color:#58a6ff;">📡 LIVE API STREAMING</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with nav_c2:
        # Market Status calculation
        market_status = "Stable"
        status_color = "#3fb950"
        if "live_records" in st.session_state and st.session_state.live_records:
            df_m = pd.DataFrame(st.session_state.live_records)
            latest_tick_m = df_m["tick"].max()
            avg_anom = df_m[df_m["tick"] == latest_tick_m]["anomaly_score"].mean()
            if avg_anom > 0.4:
                market_status = "Volatile"
                status_color = "#d29922"
            if avg_anom > 0.7:
                market_status = "Extreme"
                status_color = "#f85149"
        
        st.markdown(
            f"""
            <div style="background:#161b22; padding:10px 16px; border-radius:8px; border:1px solid #30363d;">
                <span style="color:#8b949e; font-size:0.8rem; text-transform:uppercase; font-weight:700;">Market Status</span><br>
                <span style="font-size:1.1rem; font-weight:700; color:{status_color};">● {market_status}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── View Controls ──────────────────────────────────────────
    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([2, 1, 1])
    with ctrl_c1:
        compact_view = st.toggle("Compact View", value=False, help="Hide details for a high-level summary")
    
    

    # ── Start / Stop / Inject buttons ───────────────────────────
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn1:
        if not st.session_state.streaming:
            if st.button("▶️ Start Live Stream", width="stretch"):
                reset_realtime_state()
                st.session_state.live_records = []
                st.session_state.live_tick = 0
                st.session_state.streaming = True
                st.session_state.inject_source = None
                st.rerun()
        else:
            if st.button("⏹️ Stop Live Stream", width="stretch"):
                st.session_state.streaming = False
                st.rerun()
    with col_btn2:
        if st.session_state.streaming:
            if st.button("🔥 Inject Spike (Source_C)", width="stretch", help="Trigger a deterministic +20% spike on Source_C"):
                st.session_state.inject_source = "Source_C"
                st.toast("Spike queued for Source_C (+20%)!", icon="🔥")
    with col_btn3:
        tick_count = len(st.session_state.live_records) // 3  # 3 sources per tick
        st.metric("Ticks collected", tick_count)

    st.divider()

    # ── Placeholders for dynamic content ───────────────────────
    status_placeholder   = st.empty()
    metrics_placeholder  = st.empty()
    chart_placeholder    = st.empty()
    table_placeholder    = st.empty()

    # ── Helper: render interpretation panel ──────────────────
    def _render_interpretation(
        instant_confidence: float,
        reliability_index: float,
        historic_trust: float,
        final_score: float,
        status: str,
    ):
        """
        Renders the rule-based interpretation panel for one source.
        Calls interpret_source() and displays the three labels with
        colour-coded icons in a compact styled block.
        """
        interp = interpret_source(
            instant_confidence=instant_confidence,
            reliability_index=reliability_index,
            historic_trust=historic_trust,
            final_score=final_score,
            status=status,
        )

        # Icon mapping for recommendation
        rec_icons = {
            "Recommended Primary Source":         "🟢",
            "Temporarily Deviating – Monitor":    "🟡",
            "Short-Term Agreement, Long-Term Risk": "🟠",
            "Unreliable – Consider Isolation":    "🔴",
        }
        # Icon mapping for historic assessment
        hist_icons = {
            "Historically Reliable":   "✅",
            "Moderately Reliable":     "🔵",
            "Historically Unstable":   "⚠️",
        }
        # Icon mapping for current behavior
        behav_icons = {
            "Currently Consistent":          "✅",
            "Currently Stable but Monitor":  "⚠️",
            "Currently Deviating":           "🚨",
        }

        rec  = interp["recommendation"]
        hist = interp["historic_assessment"]
        behv = interp["current_behavior"]

        st.markdown(
            f"""
            <div style="background:#12172a; border-left:3px solid #7eb8f7;
                        border-radius:10px; padding:18px 24px; margin-top:12px;
                        font-size:16px; line-height:1.8; border:1px solid #30363d;">
              <div class="metric-sub-header">
                📋 Engine Interpretation
              </div>
              <div style="margin-bottom:8px;">🕰 <b>Historic Assessment:</b>&nbsp;&nbsp;
                {hist_icons.get(hist, '—')} {hist}
              </div>
              <div style="margin-bottom:8px;">📡 <b>Current Behavior:</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                {behav_icons.get(behv, '—')} {behv}
              </div>
              <div style="margin-bottom:4px;">🧠 <b>Engine Conclusion:</b>&nbsp;&nbsp;&nbsp;&nbsp;
                {rec_icons.get(rec, '—')} <b style="color:#FFFFFF">{rec}</b>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Helper: render current live state ──────────────────────
    def render_live_state():
        records = st.session_state.live_records
        if not records:
            status_placeholder.info("No data yet. Press ▶️ Start Live Stream.")
            return

        df = pd.DataFrame(records)

        # Latest tick values
        latest_tick = df["tick"].max()
        latest = df[df["tick"] == latest_tick].set_index("source")

        # Status bar
        btc_price = latest.loc["Source_A", "value"] if "Source_A" in latest.index else 0
        status_placeholder.markdown(
            f"""
            <div style="background:rgba(63, 185, 80, 0.1); border:1px solid rgba(63, 185, 80, 0.3); padding:10px 20px; border-radius:8px; margin-bottom:20px;">
                <span style="color:#3fb950; font-weight:700;">🟢 Streaming</span> · 
                <span style="color:#FFFFFF;">Tick {int(latest_tick)}</span> · 
                <span style="color:#FFFFFF;">BTC/USD = <b>${btc_price:,.2f}</b></span> · 
                <span style="color:#8b949e;">Last update: {latest.iloc[0]['ts_label']}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ── Per-source trust layer cards ─────────────────────────
    
        
        # Determine recommended source for glow effect
        rec_src_name = None
        if "final_decision_data" in st.session_state:
            fd_data = st.session_state.final_decision_data
            active = fd_data[~fd_data['decision'].str.contains("Isolate")]
            if not active.empty:
                if 'historic_trust' in active.columns and 'reliability_index' in active.columns:
                    active['h_score'] = 0.7 * active['historic_trust'] + 0.3 * active['reliability_index']
                    rec_src_row = active.loc[active['h_score'].idxmax()]
                    rec_src_name = rec_src_row['source']

        with metrics_placeholder.container():
            mc1, mc2, mc3 = st.columns(3)
            source_icons = {"Source_A": "🟢", "Source_B": "🔵", "Source_C": "🔴"}

            for col, src in zip([mc1, mc2, mc3], ["Source_A", "Source_B", "Source_C"]):
                if src not in latest.index:
                    continue
                row = latest.loc[src]
                badge = row["decision"]
                icon  = source_icons.get(src, "⚪")
                is_primary = (src == rec_src_name)

                # Styling logic
                if "Trusted" in badge:
                    border_col = "#3fb950"
                    badge_cls  = "badge-trusted"
                elif "Monitor" in badge:
                    border_col = "#d29922"
                    badge_cls  = "badge-monitor"
                else:
                    border_col = "#f85149"
                    badge_cls  = "badge-isolate"

                # Metrics
                inst_conf     = row.get("trust_score", 0)
                hist_pct     = row.get("historic_trust", 0.5)
                anom_score   = row.get("anomaly_score", 0)
                cons_score   = row.get("consensus_score", 0)
                
                # Trust Bar Color
                bar_color = "#f85149" if hist_pct < 0.5 else "#d29922" if hist_pct < 0.75 else "#3fb950"

                with col:
                    # Header card
                    st.markdown(
                        f"""
                        <div class="{'primary-glow' if is_primary else ''}" style="border:1px solid {border_col}; border-radius:12px;
                                     padding:20px; margin-bottom:16px; background:#161b22; min-height:180px;">
                          {f'<div class="primary-badge">⭐ PRIMARY SOURCE</div>' if is_primary else ''}
                          <div style="display:flex; justify-content:space-between; align-items:start;">
                            <div class="source-name-header">{icon} {src}</div>
                            <span class="{badge_cls}">{badge}</span>
                          </div>
                          <div class="price-value-hero">
                            ${row['value']:,.2f}
                          </div>
                          <div style="display:flex; justify-content:center;">
                            {f'<span class="anomaly-badge">🚨 ANOMALY DETECTED</span>' if anom_score > 0.5 else ''}
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if not compact_view:
                        # Detailed metrics
                        st.markdown(
                            f"""
                            <div style="padding:0 4px;">
                                <div class="metric-row">
                                    <span class="metric-label">⚡ Instant Confidence</span>
                                    <span class="metric-value">{inst_conf:.3f}</span>
                                </div>
                                <div class="metric-row" style="margin-top:10px;">
                                    <span class="metric-label">📜 Historic Trust</span>
                                    <span class="metric-value">{hist_pct * 100:.1f}%</span>
                                </div>
                                <div class="trust-bar-container">
                                    <div class="trust-bar-fill" style="width:{hist_pct * 100}%; background:{bar_color};"></div>
                                </div>
                                <div class="metric-row">
                                    <span class="metric-label">🔍 Anomaly Score</span>
                                    <span class="metric-value">{anom_score:.3f}</span>
                                </div>
                                <div class="metric-row">
                                    <span class="metric-label">🤝 Consensus Match</span>
                                    <span class="metric-value">{cons_score:.3f}</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        # ── Engine Interpretation panel ───────────────────
                        st.markdown('<div class="metric-sub-header" style="margin:24px 0 12px 0;">🧠 Engine Interpretation</div>', unsafe_allow_html=True)
                        _render_interpretation(
                            instant_confidence=inst_conf,
                            reliability_index=row.get("reliability_index", 0.7),
                            historic_trust=hist_pct,
                            final_score=row.get("trust_score", 0),
                            status=row.get("decision", "—"),
                        )
                        st.markdown("<div style='margin-bottom:24px;'></div>", unsafe_allow_html=True)

        # Trust evolution chart
        if len(df["tick"].unique()) >= 2:
            pivot = df.pivot(index="tick", columns="source", values="trust_score")
            with chart_placeholder.container():
                st.markdown("**📈 Live Trust Score Evolution (Instant Confidence)**")
                st.line_chart(pivot, height=250)

        # Recent data table (last 15 ticks × 3 sources = 45 rows)
        recent = df[df["tick"] >= max(0, latest_tick - 14)].copy()
        
        # ── Update Session State for Final Decision (Requirement #1) ───
        st.session_state.final_decision_data = latest.reset_index()

        display_cols = [c for c in [
            "tick", "ts_label", "source", "value",
            "anomaly_score", "consensus_score", "trust_score",
            "reliability_index", "historic_trust", "decision",
        ] if c in recent.columns]
        with table_placeholder.container():
            st.markdown("**Recent Data (last 15 ticks)**")
            st.dataframe(
                recent[display_cols].rename(columns={"ts_label": "time"}),
                hide_index=True,
                width="stretch",
            )

    # ── Initial render (if paused with data) ───────────────────
    render_live_state()

    # ── Streaming loop ─────────────────────────────────────────
    if st.session_state.streaming:
        rng = st.session_state.live_rng
        price = fetch_btc_price()

        # ── Step 1: Real-Time Simulation (Requirement #1 & #2) ────────────
        # Derives values from a hidden ground truth + noise + stale chance.
        inject = st.session_state.get("inject_source")
        # Call simulation WITHOUT engine injection to keep it deterministic in app logic
        sources_vals, true_value = simulate_live_sources(price, rng, inject_source=None)
        
        # Apply deterministic +20% spike if Source_C is targeted
        if inject == "Source_C":
            sources_vals["Source_C"] *= 1.20
            st.toast("🔥 Deterministic +20% Spike applied to Source_C!", icon="🔥")
        
        # Reset injection after one tick
        st.session_state.inject_source = None

        tick  = st.session_state.live_tick + 1
        ts_label = time.strftime("%H:%M:%S")
        st.session_state.live_tick = tick

        # ── Step 2: Phase 1 — Append to buffers ───────────────────────────
        from trust_engine import _value_store
        for src, val in sources_vals.items():
            _value_store[src].append(float(val))

        # ── Step 3: Phase 2 — Run full trust model logic ──────────────────
        from trust_engine import (
            compute_anomaly_scores, compute_consensus_scores,
            update_historical_trust, compute_final_score, classify_trust,
            _trust_state, _score_history, _reliability_index,
            _total_events, _successful_events,
            _smoothed_score, _detect_trend, _compute_confidence,
            _build_reason, classify_reliability, classify_historic_status,
            compute_weighted_consensus, _conf_volatility_history,
            _low_variance_count, _SOURCES,
        )
        import pandas as _pd, numpy as _np

        for src, val in sources_vals.items():
            # Anomaly
            series = _pd.Series(_value_store[src], dtype=float)
            anom_df = compute_anomaly_scores(series, window=5)
            anomaly_score = float(anom_df["anomaly_score"].iloc[-1])
            if _np.isnan(anomaly_score): anomaly_score = 0.0

            # Consensus logic
            latest_vals = {s: _value_store[s][-1] for s in _SOURCES}
            row_dict = {s: [latest_vals[s]] for s in _SOURCES}
            cons_df = compute_consensus_scores(_pd.DataFrame(row_dict))
            sfx = src[-1]
            consensus_score = float(cons_df.iloc[0][f"consensus_score_{sfx}"])

            # ── Weighted internal consensus for performance ──────────────
            w_consensus = compute_weighted_consensus(latest_vals, _reliability_index)
            tolerance   = 0.02 * abs(w_consensus) + 1e-9
            _perf       = float(_np.clip(1.0 - abs(val - w_consensus) / tolerance, 0.0, 0.995))

            # EMA Update with ceiling
            old_trust = _trust_state[src]
            new_trust, _ = update_historical_trust(old_trust, anomaly_score, consensus_score, performance=_perf)
            _trust_state[src] = new_trust

            # Scoring & Stability
            raw_score = compute_final_score(new_trust, anomaly_score, consensus_score)
            _score_history[src].append(raw_score)
            smoothed = _smoothed_score(src)
            decision = classify_trust(smoothed)

            # Volatility tracking
            _conf_volatility_history[src].append(smoothed)
            conf_vol = float(_np.std(list(_conf_volatility_history[src]))) if len(_conf_volatility_history[src]) >= 2 else 0.0

            # Reliability & Variance Penalty
            old_rel = _reliability_index[src]
            new_rel = round(0.98 * old_rel + 0.02 * _perf, 4)
            _reliability_index[src] = new_rel
            
            # Low variance check
            if len(_value_store[src]) >= 5:
                # Use numpy for std to avoid direct pandas series creation if possible, but series is fine too
                if _pd.Series(_value_store[src][-5:]).std() < (0.0001 * abs(val)):
                    _low_variance_count[src] += 1
                else:
                    _low_variance_count[src] = 0
                if _low_variance_count[src] >= 10:
                    _reliability_index[src] *= 0.999

            # Historic trust
            _total_events[src] += 1
            if _perf >= 0.6: _successful_events[src] += 1
            hist_trust = round(_successful_events[src] / _total_events[src], 4)
            hist_status = classify_historic_status(hist_trust)

            st.session_state.live_records.append({
                "tick":              tick,
                "ts_label":          ts_label,
                "source":            src,
                "value":             round(val, 2),
                "anomaly_score":     round(anomaly_score, 4),
                "consensus_score":   round(consensus_score, 4),
                "trust_score":       round(smoothed, 4),
                "confidence_volatility": round(conf_vol, 4),
                "reliability_index": round(new_rel, 4),
                "historic_trust":    hist_trust,
                "historic_status":   hist_status,
                "decision":          decision,
            })

        # Re-render with new data
        render_live_state()

        # Wait 2 seconds then trigger Streamlit rerun
        # No st.stop() here to allow reaching the final decision section


# ══════════════════════════════════════════════════════════════
# ██  TEST SCENARIO MODE
# ══════════════════════════════════════════════════════════════
if mode == "🧪 Test Scenario Mode":
    st.warning("🧪 TEST SCENARIO MODE")
    st.header("🧪 Test Scenario Mode — Stress Testing Lab")
    st.markdown(
        """
        Deterministic simulation of edge-case behaviors. 
        Each scenario runs for exactly **40 ticks**.
        """
    )

    # ── Scenario Selection ─────────────────────────────────────
    sc_name = st.selectbox("Select Scenario", list(TEST_SCENARIOS.keys()), help="Choose a behavior pattern to simulate.")
    sc_cfg = TEST_SCENARIOS[sc_name]
    
    sc_col1, sc_col2 = st.columns([2, 1])
    with sc_col1:
        st.info(f"**Description:** {sc_cfg['desc']}")
        st.success(f"**Expected Outcome:** {sc_cfg['expected']}")
    with sc_col2:
        if not st.session_state.test_running:
            if st.button("▶ Run Scenario", width="stretch"):
                # NOTE: We DO NOT call reset_realtime_state() here to keep separation
                st.session_state.test_records = []
                st.session_state.test_tick = 0
                st.session_state.test_value_buffers = {"Source_A": [], "Source_B": [], "Source_C": []}
                st.session_state.test_smoothed_score_history = {"Source_A": [], "Source_B": [], "Source_C": []}
                st.session_state.test_trust_state = {"Source_A": 0.5, "Source_B": 0.5, "Source_C": 0.5}
                st.session_state.test_reliability_state = {"Source_A": 0.7, "Source_B": 0.7, "Source_C": 0.7}
                st.session_state.test_successful_events = {"Source_A": 0, "Source_B": 0, "Source_C": 0}
                st.session_state.test_total_events = {"Source_A": 0, "Source_B": 0, "Source_C": 0}
                st.session_state.test_running = True
                st.rerun()
        else:
            if st.button("⏹ Stop Scenario", width="stretch"):
                st.session_state.test_running = False
                st.rerun()

    # Triggers removed as per user request

    st.divider()

    # ── Simulation Loop ────────────────────────────────────────
    if st.session_state.test_running:
        st.session_state.test_tick += 1
        tick = st.session_state.test_tick
        
        if tick > sc_cfg["max_ticks"]:
            st.session_state.test_running = False
            st.rerun()

        # Data Generator Logic (Deterministic per Scenario)
        base = 100.0
        rng = st.session_state.test_rng

        # ── Manual Trigger Injection (High Priority) ─────────────
        if st.session_state.trigger_spike:
            v = {"Source_A": 250.0, "Source_B": base + rng.normal(0,0.5), "Source_C": base + rng.normal(0,0.5)}
            st.session_state.trigger_spike = False
            st.toast("Manual Spike Injected into Source A!", icon="💥")
        elif st.session_state.trigger_collusion:
            v = {"Source_A": base, "Source_B": 150.0, "Source_C": 152.0}
            st.session_state.trigger_collusion = False
            st.toast("Manual Collusion Injected (B & C)!", icon="🤝")
        elif st.session_state.trigger_chaos:
            v = {"Source_A": 100.0, "Source_B": 180.0, "Source_C": 260.0}
            st.session_state.trigger_chaos = False
            st.toast("Manual Chaos Injected!", icon="🌪️")
        elif sc_name == "Normal": 
            v = {"Source_A": base + rng.normal(0,0.5), "Source_B": base+rng.normal(0,0.8), "Source_C": base+rng.normal(0,1)}
        elif sc_name == "Single Source Failure": 
            v = {"Source_A": base*1.6 if tick==20 else base+rng.normal(0,0.5), "Source_B": base+rng.normal(0,0.8), "Source_C": base+rng.normal(0,1)}
        elif sc_name == "Two Weak Collude":
            if tick <= 10: v = {"Source_A": base+rng.normal(0,0.1), "Source_B": base+rng.normal(0,0.5), "Source_C": base+rng.normal(0,0.5)}
            else: v = {"Source_A": 100.0, "Source_B": 140.0, "Source_C": 142.0}
        elif sc_name == "Gradual Drift": 
            v = {"Source_A": 100.0+(tick*0.6), "Source_B": 100.0, "Source_C": 100.0}
        elif sc_name == "Chaos": 
            v = {"Source_A": 100.0, "Source_B": 150.0, "Source_C": 200.0}
        elif sc_name == "Recovery": 
            v = {"Source_B": base+rng.normal(0,20) if tick<=20 else base+rng.normal(0,0.5), "Source_A": base, "Source_C": base}
        
        # Note: No spike buttons or manual overrides in Test Mode (Requirement #3.3 & #3.4)

        # ── Processing Pipeline ────────────────────────────────
        from trust_engine import (
            compute_anomaly_scores, compute_consensus_scores,
            update_historical_trust, compute_final_score, classify_trust
        )
        import pandas as pd
        import numpy as np

        df_row = pd.DataFrame([v])
        reliabs = st.session_state.test_reliability_state
        cons_df = compute_consensus_scores(df_row, reliabilities=reliabs) if len(v) >= 2 else None
        
        for s, val in v.items():
            st.session_state.test_value_buffers[s].append(val)
            abuf = pd.Series(st.session_state.test_value_buffers[s])
            anom = float(compute_anomaly_scores(abuf).iloc[-1]["anomaly_score"]) if not abuf.empty else 0.0
            if np.isnan(anom): anom = 0.0
            
            cs = float(cons_df.iloc[0][f"consensus_score_{s[-1]}"]) if cons_df is not None else 0.5
            w_cons = float(cons_df.iloc[0]["median_value"]) if cons_df is not None else val
            
            # Performance for reliability
            tol = max(0.5, 0.08 * abs(w_cons)) + 1e-9
            perf = float(np.clip(1.0 - abs(val - w_cons) / tol, 0.0, 0.995))

            old_t = st.session_state.test_trust_state.get(s, 0.5)
            new_t, _ = update_historical_trust(old_t, anom, cs, performance=perf)
            st.session_state.test_trust_state[s] = new_t
            
            # Update reliability state in session
            old_rel = reliabs.get(s, 0.7)
            new_rel = 0.995 * old_rel + 0.005 * perf
            st.session_state.test_reliability_state[s] = new_rel
            
            final = compute_final_score(new_t, anom, cs)
            
            # Smoothing (EMA 0.2)
            hist = st.session_state.test_smoothed_score_history[s]
            if not hist:
                smoothed = final
            else:
                smoothed = 0.2 * final + 0.8 * hist[-1]
            st.session_state.test_smoothed_score_history[s].append(smoothed)
            if len(st.session_state.test_smoothed_score_history[s]) > 10: 
                st.session_state.test_smoothed_score_history[s].pop(0)

            # Classification
            dis_idx = float(cons_df.iloc[0]["disagreement_index"]) if cons_df is not None else 0.0
            is_ex = bool(cons_df.iloc[0]["is_extreme_chaos"]) if cons_df is not None and "is_extreme_chaos" in cons_df.columns else False
            
            # Historic trust update
            st.session_state.test_total_events[s] += 1
            if perf >= 0.6:
                st.session_state.test_successful_events[s] += 1
            h_trust = st.session_state.test_successful_events[s] / st.session_state.test_total_events[s]

            st.session_state.test_records.append({
                "tick": tick, "source": s, "value": val, "final_score": final,
                "smoothed": smoothed,
                "decision": classify_trust(
                    smoothed, 
                    disagreement_index=dis_idx, 
                    anomaly_score=anom,
                    weighted_mean=w_cons,
                    is_extreme_chaos=is_ex
                ),
                "historical_trust": new_t,
                "historic_trust": h_trust,
                "anomaly_score": anom,
                "consensus_score": cs,
                "reliability_index": new_rel,
                "is_anomaly": anom > 0.4
            })
        
        # Control speed
        # No st.rerun here, moved to the bottom
        pass

    # ── Render Results ─────────────────────────────────────────
    if st.session_state.test_records:
        rdf = pd.DataFrame(st.session_state.test_records)
        rlast = rdf[rdf["tick"] == rdf["tick"].max()]
        
        st.subheader(f"Tick: {st.session_state.test_tick} / {sc_cfg['max_ticks']}")
        
        # ── Update Session State for Final Decision (Requirement #1) ───
        st.session_state.final_decision_data = rlast

        # 1. Source cards (Match original test_app.py layout)
        rcols = st.columns(3)
        for i, s in enumerate(["Source_A", "Source_B", "Source_C"]):
            if s in rlast["source"].values:
                row = rlast[rlast["source"] == s].iloc[0]
                interp = interpret_source(
                    row["smoothed"], 0.7, row["historical_trust"], row["final_score"], row["decision"]
                )
                with rcols[i]:
                    _render_interp_card(s, f"{row['value']:.2f}", row["final_score"], row["decision"], interp["recommendation"])
            else:
                with rcols[i]: st.metric(s, "OFFLINE")
        
        st.divider()
        
        # 2. Trust evolution chart
        st.subheader("📈 Trust Score Evolution")
        st.line_chart(rdf.pivot_table(index="tick", columns="source", values="smoothed"))

        # 3. Ranking Table
        st.subheader("🏆 Source Ranking")
        ranking = rlast[["source", "final_score", "historical_trust", "anomaly_score", "consensus_score", "decision"]].copy()
        ranking = ranking.sort_values("final_score", ascending=False).reset_index(drop=True)
        ranking.index = ranking.index + 1
        st.dataframe(ranking.style.format({
            "final_score": "{:.3f}", "historical_trust": "{:.3f}", 
            "anomaly_score": "{:.3f}", "consensus_score": "{:.3f}"
        }), width="stretch")

        # No st.stop() here to allow combined UI render at bottom


if mode == "📂 CSV Mode":
    st.warning("📂 CSV ANALYSIS MODE")

    # ─────────────────────────────────────────────────────────────
    # MAIN CONTENT
    # ─────────────────────────────────────────────────────────────
    if uploaded_file is None:
        st.info(
            "👈 Upload a CSV file using the sidebar to get started.  \n"
            "A `sample_data.csv` file is included in the project for a quick demo."
        )
    else:
        # ── Load & validate ──────────────────────────────────────────
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        required_cols = {"timestamp", "value"}
        if not required_cols.issubset(raw_df.columns):
            st.error(
                f"CSV must contain columns: {required_cols}.  "
                f"Found: {list(raw_df.columns)}"
            )
            st.stop()

        raw_df["value"] = pd.to_numeric(raw_df["value"], errors="coerce")
        raw_df.dropna(subset=["value"], inplace=True)

        if len(raw_df) < 5:
            st.error("Need at least 5 rows for rolling statistics.")
            st.stop()

        # ── Run the trust pipeline ───────────────────────────────────
        with st.spinner("Simulating sources and running TrustLayer analysis…"):
            sim_df = simulate_sources(raw_df)
            results = run_trust_pipeline(raw_df)

        sources = ["Source_A", "Source_B", "Source_C"]

        # Compute per-source anomaly frames (rolling stats + Z-score + anomaly_score)
        anom_frames = {src: compute_anomaly_scores(sim_df[src]) for src in sources}

        # Compute cross-source consensus (median, deviations, consensus scores)
        consensus_df = compute_consensus_scores(sim_df)

        # ─────────────────────────────────────────────────────────────
        # SECTION 0 – Simulated Sources Table
        # ─────────────────────────────────────────────────────────────
        st.header("🔬 Simulated Data Sources")
        st.markdown(
            """
            From your uploaded CSV, TrustLayer derives three independent sources:
            - **Source A** – original values (ground truth)
            - **Source B** – original + small Gaussian noise (~2% of std)
            - **Source C** – original + occasional extreme spikes (~10% of rows)
            """
        )

        # Style the table: highlight Source_C spikes by comparing to Source_A
        def highlight_spikes(row):
            """Light red background on rows where Source_C deviates a lot from Source_A."""
            diff = abs(row["Source_C"] - row["Source_A"])
            threshold = 2 * sim_df["Source_A"].std()
            if diff > threshold:
                return ["background-color: rgba(231,76,60,0.18)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            sim_df.style.apply(highlight_spikes, axis=1)
                       .format({"Source_A": "{:.4f}", "Source_B": "{:.4f}", "Source_C": "{:.4f}"}),
            width="stretch",
            hide_index=True,
        )
        st.caption("🔴 Red rows = Source C spike detected (deviation > 2× std of Source A)")

        st.divider()

        # ─────────────────────────────────────────────────────────────
        # SECTION 1 – Simulated Source Values  (last timestamp)
        # ─────────────────────────────────────────────────────────────
        st.header("📊 Current Source Snapshot (Latest Timestamp)")

        last_ts = results["timestamp"].iloc[-1]
        last_rows = results[results["timestamp"] == last_ts]

        cols = st.columns(3)
        for col, src in zip(cols, sources):
            row = last_rows[last_rows["source"] == src].iloc[0]
            decision_text = row["decision"]
            trust_pct = row["final_score"] * 100

            # Colour class
            if "Trusted" in decision_text:
                badge_class = "badge-trusted"
            elif "Monitor" in decision_text:
                badge_class = "badge-monitor"
            else:
                badge_class = "badge-isolate"

            with col:
                st.metric(
                    label=f"🔵 {src}",
                    value=f"{row['value']:.3f}",
                    delta=f"Trust: {trust_pct:.1f}%",
                )
                st.markdown(
                    f"<span class='{badge_class}'>{decision_text}</span>",
                    unsafe_allow_html=True,
                )
                anomaly_label = "🔴 Anomaly detected!" if row["is_anomaly"] else "🟢 Normal"
                st.caption(anomaly_label)

                # ── Engine Interpretation panel ───────────────────────
                _interp = interpret_source(
                    instant_confidence=float(row["final_score"]),
                    reliability_index=0.7,           # CSV mode has no session state; use default
                    historic_trust=float(row["historical_trust"]),
                    final_score=float(row["final_score"]),
                    status=row["decision"],
                )
                _rec_icons  = {
                    "Recommended Primary Source":           "🟢",
                    "Temporarily Deviating – Monitor":      "🟡",
                    "Short-Term Agreement, Long-Term Risk":  "🟠",
                    "Unreliable – Consider Isolation":      "🔴",
                }
                _hist_icons  = {
                    "Historically Reliable":   "✅",
                    "Moderately Reliable":     "🔵",
                    "Historically Unstable":   "⚠️",
                }
                _behav_icons = {
                    "Currently Consistent":          "✅",
                    "Currently Stable but Monitor":  "⚠️",
                    "Currently Deviating":           "🚨",
                }
                _rec  = _interp["recommendation"]
                _hist = _interp["historic_assessment"]
                _behv = _interp["current_behavior"]
                st.markdown(
                    f"""
                    <div style="background:#12172a; border-left:3px solid #7eb8f7;
                                border-radius:6px; padding:10px 14px; margin-top:6px;
                                font-size:0.82rem; line-height:1.8;">
                      <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase;
                                  letter-spacing:0.08em; color:#7eb8f7; margin-bottom:5px;">
                        📋 Engine Interpretation
                      </div>
                      <div>🕰 <b>Historic Assessment:</b>&nbsp;&nbsp;
                        {_hist_icons.get(_hist, '—')} {_hist}
                      </div>
                      <div>📡 <b>Current Behavior:</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                        {_behav_icons.get(_behv, '—')} {_behv}
                      </div>
                      <div>🧠 <b>Engine Conclusion:</b>&nbsp;&nbsp;&nbsp;&nbsp;
                        {_rec_icons.get(_rec, '—')} <b>{_rec}</b>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.divider()

        # ─────────────────────────────────────────────────────────────
        # SECTION 1.5 – Rolling Anomaly Detection (per source)
        # ─────────────────────────────────────────────────────────────
        st.header("🔎 Rolling Anomaly Detection")
        st.markdown(
            """
            For each source, a **rolling window of 5** computes the local mean and standard deviation.
            The **Z-score** measures how many standard deviations a value is from its recent history.
            Values with |Z| > 2.5 are flagged as anomalies and highlighted in red.
            The **anomaly_score** (0–1) is derived via `tanh(|Z| / 2.5)` — higher means more suspicious.
            """
        )

        # ── Z-score line chart (all sources overlaid) ──────────────
        z_chart_data = pd.DataFrame(
            {src: anom_frames[src]["z_score"].values for src in sources},
            index=sim_df["timestamp"],
        )
        st.markdown("**Z-Score Over Time** (dashed reference lines at ±2.5 not shown, but values > ±2.5 are anomalies)")
        st.line_chart(z_chart_data, height=250)

        # ── Anomaly score chart ────────────────────────────────────
        anom_score_chart = pd.DataFrame(
            {src: anom_frames[src]["anomaly_score"].values for src in sources},
            index=sim_df["timestamp"],
        )
        st.markdown("**Anomaly Score Over Time** (0 = normal, 1 = extreme outlier)")
        st.line_chart(anom_score_chart, height=220)

        # ── Per-source detail tables in tabs ──────────────────────
        st.markdown("**Per-Source Anomaly Detail Tables** (🔴 = anomaly row)")
        tabs = st.tabs(["Source A", "Source B", "Source C"])

        for tab, src in zip(tabs, sources):
            with tab:
                af = anom_frames[src].copy()
                af.insert(0, "timestamp", sim_df["timestamp"].values)
                af.insert(1, "value",     sim_df[src].values)

                # Count summary
                n_anom = int(af["is_anomaly"].sum())
                if n_anom == 0:
                    st.success(f"No anomalies detected in {src}.")
                else:
                    st.warning(f"**{n_anom} anomaly row(s)** detected in {src}.")

                # Row highlighter
                def _highlight_anom(row):
                    if row["is_anomaly"]:
                        return ["background-color: rgba(231,76,60,0.22); color: #ff8080"] * len(row)
                    return [""] * len(row)

                display_af = af[["timestamp", "value", "rolling_mean", "rolling_std",
                                  "z_score", "is_anomaly", "anomaly_score"]]
                st.dataframe(
                    display_af.style
                        .apply(_highlight_anom, axis=1)
                        .format({
                            "value":        "{:.4f}",
                            "rolling_mean": "{:.4f}",
                            "rolling_std":  "{:.4f}",
                            "z_score":      "{:.4f}",
                            "anomaly_score":"{:.4f}",
                        }),
                    width="stretch",
                    hide_index=True,
                )

        st.divider()

        # ─────────────────────────────────────────────────────────────
        # SECTION 1.7 – Cross-Source Consensus Scoring
        # ─────────────────────────────────────────────────────────────
        st.header("🤝 Cross-Source Consensus Scoring")
        st.markdown(
            """
            At every timestamp, TrustLayer asks: **"Do all sources agree?"**
            - The **median** of Source A/B/C is used as the reference value.
            - Each source's **deviation** from that median is computed and normalised.
            - **consensus_score** (0–1): `1.0` = perfectly aligned with peers · `0.0` = complete outlier.

            > 💡 When **Source C spikes**, its deviation from the median is large → its consensus score
            > drops near **0**. Source A and B, being close to the median, keep high consensus scores.
            > This is exactly how the engine knows something is wrong — without any ML.
            """
        )

        # ── Consensus score line chart ──────────────────────────────
        chart_cons = pd.DataFrame(
            {
                "Source_A": consensus_df["consensus_score_A"].values,
                "Source_B": consensus_df["consensus_score_B"].values,
                "Source_C": consensus_df["consensus_score_C"].values,
            },
            index=sim_df["timestamp"],
        )
        st.markdown("**Consensus Score Over Time** (1 = fully agrees with peers · 0 = outlier)")
        st.line_chart(chart_cons, height=260)

        # ── Summary metric cards ────────────────────────────────────
        LOW_CONS_THRESHOLD = 0.3
        low_cons_c = int((consensus_df["consensus_score_C"] < LOW_CONS_THRESHOLD).sum())
        low_cons_b = int((consensus_df["consensus_score_B"] < LOW_CONS_THRESHOLD).sum())

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric(
                "Avg Consensus – A",
                f"{consensus_df['consensus_score_A'].mean():.3f}",
                help="Source A is the ground truth — almost always near 1.0",
            )
        with mc2:
            st.metric(
                "Avg Consensus – B",
                f"{consensus_df['consensus_score_B'].mean():.3f}",
                delta=f"{low_cons_b} rows below {LOW_CONS_THRESHOLD}" if low_cons_b else "All good",
                delta_color="inverse",
                help="Source B adds tiny noise — should stay high",
            )
        with mc3:
            st.metric(
                "Avg Consensus – C",
                f"{consensus_df['consensus_score_C'].mean():.3f}",
                delta=f"{low_cons_c} spike row(s) < {LOW_CONS_THRESHOLD}",
                delta_color="inverse",
                help="Source C has spikes — expect a lower average consensus",
            )

        # ── Full colour-coded detail table ──────────────────────────
        cons_display = pd.DataFrame(
            {
                "timestamp":   sim_df["timestamp"].values,
                "Source_A":    sim_df["Source_A"].round(4),
                "Source_B":    sim_df["Source_B"].round(4),
                "Source_C":    sim_df["Source_C"].round(4),
                "median":      consensus_df["median_value"].round(4),
                "consensus_A": consensus_df["consensus_score_A"].round(4),
                "consensus_B": consensus_df["consensus_score_B"].round(4),
                "consensus_C": consensus_df["consensus_score_C"].round(4),
            }
        )

        def _highlight_consensus(row):
            """Red if Source C consensus is very low (spike), yellow if Source B is low."""
            if row["consensus_C"] < LOW_CONS_THRESHOLD:
                return ["background-color: rgba(231,76,60,0.22); color: #ff8080"] * len(row)
            if row["consensus_B"] < LOW_CONS_THRESHOLD:
                return ["background-color: rgba(243,156,18,0.15)"] * len(row)
            return [""] * len(row)

        st.markdown(
            f"**Full Consensus Table** "
            f"(🔴 = Source C consensus < {LOW_CONS_THRESHOLD} · 🟡 = Source B consensus < {LOW_CONS_THRESHOLD})"
        )
        st.dataframe(
            cons_display.style
                .apply(_highlight_consensus, axis=1)
                .format(
                    {
                        "Source_A":   "{:.4f}",
                        "Source_B":   "{:.4f}",
                        "Source_C":   "{:.4f}",
                        "median":     "{:.4f}",
                        "consensus_A": "{:.4f}",
                        "consensus_B": "{:.4f}",
                        "consensus_C": "{:.4f}",
                    }
                ),
            width="stretch",
            hide_index=True,
        )

        st.divider()

        # ─────────────────────────────────────────────────────────────
        # SECTION 2 – Trust Score Evolution (line chart)
        # ─────────────────────────────────────────────────────────────
        st.header("📈 Trust Score Over Time")

        trust_pivot = results.pivot(
            index="timestamp", columns="source", values="final_score"
        )
        st.line_chart(trust_pivot, height=300)

        st.caption(
            "Green zone ≥ 0.75 = Trusted · Yellow 0.40–0.75 = Monitor · Red < 0.40 = Isolate"
        )

        st.divider()

        # ─────────────────────────────────────────────────────────────
        # SECTION 2 – Historical Trust Update + Final Score Evolution
        # ─────────────────────────────────────────────────────────────
        st.header("📈 Historical Trust Update")
        st.markdown(
            """
            Trust is **adaptive** — it changes over time based on each source’s behaviour:

            | Component | Formula |
            |---|---|
            | **Performance** | `1` if anomaly_score < 0.4 AND consensus_score > 0.6, else `0` |
            | **Historical Trust** | `new = 0.8 × old + 0.2 × performance` (EMA) |
            | **Final Score** | `0.4 × hist_trust + 0.3 × (1 − anomaly) + 0.3 × consensus` |
            | **Decision** | ✅ ≥ 0.75 Trusted  ·  ⚠️ 0.40–0.75 Monitor  ·  🚨 < 0.40 Isolate |

            Starting at **0.5**, trust drifts upward when a source is consistent and agreeable,
            and erodes when it produces anomalies or disagrees with its peers.
            """
        )

        # ── Chart 1: Historical trust evolution ──────────────────────
        st.markdown("**Historical Trust (EMA) Over Time** — starts at 0.5 for all sources")
        hist_pivot = results.pivot(
            index="timestamp", columns="source", values="historical_trust"
        )
        st.line_chart(hist_pivot, height=260)

        # ── Chart 2: Final score evolution (weighted combo) ───────────
        # Add threshold reference line by compositing with a constant series
        st.markdown(
            "**Final Trust Score Over Time** — "
            "Dashed thresholds: ✅ ≥ 0.75 Trusted · ⚠️ 0.40 Monitor/Isolate boundary"
        )
        trust_final_pivot = results.pivot(
            index="timestamp", columns="source", values="final_score"
        )
        st.line_chart(trust_final_pivot, height=260)

        # ── Chart 3: Performance (binary 0/1) per source ──────────────
        st.markdown(
            "**Performance Score per Step** — "
            "`1` = source was reliable (low anomaly + high consensus), `0` = unreliable"
        )
        perf_pivot = results.pivot(
            index="timestamp", columns="source", values="performance"
        )
        st.bar_chart(perf_pivot, height=200)

        # ── Per-source drilldown tabs ────────────────────────────────
        st.markdown("**Historical Trust Step-by-Step — Per Source** (🔴 = anomaly | 🟡 = poor performance)")
        trust_tabs = st.tabs(["Source A", "Source B", "Source C"])

        def _trust_row_style(row):
            if row["is_anomaly"]:
                return ["background-color: rgba(231,76,60,0.22); color: #ff8080"] * len(row)
            if row["performance"] == 0:
                return ["background-color: rgba(243,156,18,0.15)"] * len(row)
            return ["background-color: rgba(46,204,113,0.07)"] * len(row)

        for tab, src in zip(trust_tabs, sources):
            with tab:
                src_trust = results[results["source"] == src][[
                    "timestamp", "value", "anomaly_score", "consensus_score",
                    "performance", "historical_trust", "final_score", "decision", "is_anomaly"
                ]].copy()

                # Stat summary
                tca, tcb, tcc = st.columns(3)
                with tca:
                    st.metric("Final Trust (latest)",
                              f"{src_trust['final_score'].iloc[-1]:.3f}")
                with tcb:
                    st.metric("Hist Trust (latest)",
                              f"{src_trust['historical_trust'].iloc[-1]:.3f}")
                with tcc:
                    good_steps = int((src_trust["performance"] == 1.0).sum())
                    st.metric("Good Steps",
                              f"{good_steps} / {len(src_trust)}",
                              delta=f"{good_steps/len(src_trust)*100:.0f}% reliable")

                st.dataframe(
                    src_trust.style
                        .apply(_trust_row_style, axis=1)
                        .hide(["is_anomaly"], axis="columns")
                        .format({
                            "value":            "{:.4f}",
                            "anomaly_score":    "{:.4f}",
                            "consensus_score":  "{:.4f}",
                            "performance":      "{:.0f}",
                            "historical_trust": "{:.4f}",
                            "final_score":      "{:.4f}",
                        }),
                    width='stretch',
                    hide_index=True,
                )

        # ── Final score weight breakdown (static explainer cards) ──
        _WA, _WB, _WC = st.columns(3)
        with _WA:
            st.markdown(
                """
                <div class="info-card">
                <strong>40% Historical Trust</strong><br>
                The EMA of past reliability.
                Builds slowly; erosion is gradual.<br>
                <code>0.8 × old + 0.2 × perf</code>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with _WB:
            st.markdown(
                """
                <div class="info-card">
                <strong>30% Anomaly Health</strong><br>
                <code>1 − anomaly_score</code><br>
                High Z-score → lower contribution.
                </div>
                """,
                unsafe_allow_html=True,
            )
        with _WC:
            st.markdown(
                """
                <div class="info-card">
                <strong>30% Consensus</strong><br>
                Agreement with peer sources.<br>
                Spike → near-zero consensus.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # ─────────────────────────────────────────────────────────────
        # SECTION 3 – Anomaly Highlights
        # ─────────────────────────────────────────────────────────────
        st.header("🚨 Detected Anomalies")

        anomalies = results[results["is_anomaly"] == True].copy()

        if anomalies.empty:
            st.success("No anomalies detected in the dataset.")
        else:
            st.warning(f"{len(anomalies)} anomalous data point(s) found across all sources.")

            display_cols = [
                "timestamp", "source", "value",
                "z_score", "anomaly_score", "consensus_score",
                "final_score", "decision",
            ]
            def _highlight_anom_row(row):
                return ["background-color: rgba(231,76,60,0.22); color: #ff8080"] * len(row)

            st.dataframe(
                anomalies[display_cols].style.apply(_highlight_anom_row, axis=1),
                width='stretch',
                hide_index=True,
            )

        st.divider()

        # ─────────────────────────────────────────────────────────────
        # SECTION 4 – Ranked Trust Table
        # ─────────────────────────────────────────────────────────────
        st.header("🏆 Source Rankings (by Final Trust Score)")

        # Aggregate: last trust score per source
        ranking = (
            results.groupby("source", group_keys=False)
            .apply(lambda g: g.sort_values("timestamp").iloc[[-1]])
            .reset_index(drop=True)[["source", "final_score", "historical_trust",
                                       "anomaly_score", "consensus_score", "decision"]]
            .sort_values("final_score", ascending=False)
            .reset_index(drop=True)
        )

        ranking.index = ranking.index + 1   # 1-based rank
        ranking.columns = [
            "Source", "Final Score", "Historical Trust",
            "Anomaly Score", "Consensus Score", "Decision"
        ]

        # Colour the Decision column
        def colour_decision(val):
            if "Trusted" in str(val):
                return "color: #2ecc71; font-weight: bold"
            elif "Monitor" in str(val):
                return "color: #f39c12; font-weight: bold"
            else:
                return "color: #e74c3c; font-weight: bold"

        st.dataframe(
            ranking.style.map(colour_decision, subset=["Decision"])
                       .format({
                           "Final Score":      "{:.3f}",
                           "Historical Trust": "{:.3f}",
                           "Anomaly Score":    "{:.3f}",
                           "Consensus Score":  "{:.3f}",
                       }),
            width='stretch',
        )

        st.divider()

        # ── Update Session State for Final Decision (CSV Mode) ─────────
        st.session_state.final_decision_data = ranking.rename(columns={
            "Source": "source", "Final Score": "final_score", "Historical Trust": "historical_trust",
            "Anomaly Score": "anomaly_score", "Consensus Score": "consensus_score", "Decision": "decision"
        })
        # Note: we add 'value' to ranking earlier if needed, but ranking here has no value.
        # Let's get the latest values for the weighted mean.
        csv_latest_vals = results[results["timestamp"] == results["timestamp"].max()]
        st.session_state.final_decision_data = pd.merge(st.session_state.final_decision_data, csv_latest_vals[["source", "value"]], on="source")

        # ─────────────────────────────────────────────────────────────
        # SECTION 5 – Explanation Panel
        # ─────────────────────────────────────────────────────────────
        st.header("🔍 Why Was a Source Flagged?")

        selected_source = st.selectbox("Select a source to inspect:", sources)
        src_df = results[results["source"] == selected_source].copy()

        # Most recent row
        latest = src_df.iloc[-1]

        # Summary stats
        total_rows      = len(src_df)
        anomaly_count   = src_df["is_anomaly"].sum()
        avg_final_score = src_df["final_score"].mean()
        low_perf_count  = (src_df["performance"] == 0.0).sum()

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(
                f"""
                <div class="info-card">
                <b>Source:</b> {selected_source}<br>
                <b>Current Decision:</b> {latest['decision']}<br>
                <b>Current Final Score:</b> {latest['final_score']:.3f}<br>
                <b>Anomaly Score (latest):</b> {latest['anomaly_score']:.3f}
                  {'🔴 HIGH' if latest['anomaly_score'] > 0.4 else '🟢 LOW'}<br>
                <b>Consensus Score (latest):</b> {latest['consensus_score']:.3f}
                  {'🟢 HIGH' if latest['consensus_score'] > 0.6 else '🔴 LOW'}<br>
                <b>Historical Trust (latest):</b> {latest['historical_trust']:.3f}<br>
                <b>Z-Score (latest):</b> {latest['z_score']:.3f}
                  {'— anomalous' if latest['is_anomaly'] else '— normal'}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Plain-English explanation
            reasons = []
            if anomaly_count > 0:
                reasons.append(
                    f"🔴 **{anomaly_count} anomalous row(s)** were detected (|Z-score| > 2.5). "
                    "This means the source produced values statistically far from its own recent history."
                )
            if low_perf_count > total_rows * 0.4:
                reasons.append(
                    f"⚠️ **Performance was poor in {low_perf_count} of {total_rows} steps**, "
                    "indicating repeated divergence from the consensus or high anomaly scores."
                )
            if avg_final_score < 0.4:
                reasons.append(
                    f"🚨 **Average final score ({avg_final_score:.3f}) is below the Isolate threshold (0.40)**. "
                    "This source has consistently underperformed across the session."
                )
            if not reasons:
                reasons.append(
                    "✅ **No significant issues detected.** This source is behaving as expected — "
                    "its values are close to peers and within expected statistical bounds."
                )

            st.markdown("#### Plain-English Diagnosis")
            for r in reasons:
                st.markdown(f"- {r}")

        with col2:
            st.markdown("#### Score Breakdown (latest)")
            st.metric("Final Score",      f"{latest['final_score']:.3f}")
            st.metric("Historical Trust", f"{latest['historical_trust']:.3f}")
            st.metric("1 - Anomaly",      f"{1 - latest['anomaly_score']:.3f}")
            st.metric("Consensus",        f"{latest['consensus_score']:.3f}")

        # Show the source's own timeline
        st.markdown("#### Timeline for selected source")
        timeline_cols = ["timestamp", "value", "z_score", "anomaly_score",
                         "consensus_score", "historical_trust", "final_score", "decision"]

        def highlight_anomaly(row):
            if row["is_anomaly"]:
                return ["background-color: rgba(231,76,60,0.15)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            src_df[timeline_cols + ["is_anomaly"]]
                .style.apply(highlight_anomaly, axis=1),
            width="stretch",
            hide_index=True,
        )

st.divider()

# ─────────────────────────────────────────────────────────────
# FINAL SYSTEM DECISION (EXACTLY ONCE)
# ─────────────────────────────────────────────────────────────
if st.session_state.get("final_decision_data") is not None:
    render_final_system_decision(st.session_state.final_decision_data)

st.divider()

# ─────────────────────────────────────────────────────────────
# CONTINUOUS UPDATE TRIGGER (Live / Test)
# ─────────────────────────────────────────────────────────────
if mode == "📡 Live API Mode" and st.session_state.get("streaming"):
    time.sleep(2)
    st.rerun()
elif mode == "🧪 Test Scenario Mode" and st.session_state.get("test_running"):
    time.sleep(0.1)
    st.rerun()

st.caption("TrustLayer · Built for a 20-hour hackathon · No ML · Pure stats 🛡️")


