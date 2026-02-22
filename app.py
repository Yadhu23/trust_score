"""
app.py
------
TrustLayer – Statistical Multi-Source Data Trust Engine
Clean Controller Architecture.

Run with:
    streamlit run app.py
"""

import time
import numpy as np
import pandas as pd
import streamlit as st

# Components & Services
from trust_engine import reset_realtime_state
from ui.components import render_final_system_decision
from ui.live_ui import render_live_ui
from ui.csv_ui import render_csv_ui
from ui.test_ui import render_test_ui
from ui.home_ui import render_home_ui

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrustLayer | Data Trust Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# MODERN SAAS CSS
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

    :root {
        --bg-dark: #0a0c10;
        --card-bg: rgba(22, 27, 34, 0.7);
        --glass-bg: rgba(13, 17, 23, 0.7);
        --accent-blue: #58a6ff;
        --accent-purple: #bc8cff;
        --accent-cyan: #39d353;
        --border-color: rgba(48, 54, 61, 0.6);
        --text-main: #e6edf3;
        --text-dim: #8b949e;
        --trusted-green: #3fb950;
        --monitor-yellow: #d29922;
        --isolate-red: #f85149;
    }

    .stApp {
        background: radial-gradient(circle at top right, #161b22, #0d1117);
        color: var(--text-main);
        font-family: 'Outfit', sans-serif;
    }

    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        border-color: var(--accent-blue);
        transform: translateY(-4px);
        box-shadow: 0 12px 48px 0 rgba(0, 0, 0, 0.4);
    }

    [data-testid="stSidebar"] {
        background-color: var(--bg-dark);
        border-right: 1px solid var(--border-color);
    }

    [data-testid="metric-container"] {
        background: var(--card-bg);
        backdrop-filter: blur(8px);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 16px;
    }

    .gradient-text {
        background: linear-gradient(90deg, #58a6ff, #bc8cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    .dashboard-header {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .dashboard-tagline {
        color: var(--text-dim);
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .badge {
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .badge-trusted { background: rgba(63, 185, 80, 0.15); color: var(--trusted-green); border: 1px solid var(--trusted-green); }
    .badge-monitor { background: rgba(210, 153, 34, 0.15); color: var(--monitor-yellow); border: 1px solid var(--monitor-yellow); }
    .badge-isolate { background: rgba(248, 81, 73, 0.15); color: var(--isolate-red); border: 1px solid var(--isolate-red); }

    @keyframes pulse-glow {
        0% { box-shadow: 0 0 5px rgba(88, 166, 255, 0.2); }
        50% { box-shadow: 0 0 20px rgba(88, 166, 255, 0.5); }
        100% { box-shadow: 0 0 5px rgba(88, 166, 255, 0.2); }
    }

    .live-pulse {
        width: 8px;
        height: 8px;
        background-color: var(--accent-cyan);
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        animation: pulse-glow 2s infinite;
    }

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-dark); }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #484f58; }

    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# LANDING HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align: left; margin-bottom: 3rem;">
        <h1 class="dashboard-header">🛡️ <span class="gradient-text">TrustLayer</span></h1>
        <p class="dashboard-tagline">Statistical Multi-Source Data Trust Engine</p>
        <div class="glass-card" style="padding: 20px; border-left: 4px solid #58a6ff;">
            <p style="margin:0; font-size: 1rem; line-height: 1.6;">
                High-fidelity data validation without machine learning. TrustLayer cross-references multiple data sources, 
                detects statistical drift, and isolates anomalous streams in real-time.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.divider()

# ─────────────────────────────────────────────────────────────
# CONSTANTS & CONFIG
# ─────────────────────────────────────────────────────────────
TEST_SCENARIOS = {
    "Normal": {
        "desc": "Stable Baseline: system behaves calmly under normal noise.",
        "expected": "All trust scores remain high.",
        "max_ticks": 40
    },
    # ... rest of scenarios mapped below for brevity, but I will include all to be safe.
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

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
def init_session_state():
    if "streaming" not in st.session_state: st.session_state.streaming = False
    if "live_records" not in st.session_state: st.session_state.live_records = []
    if "live_tick" not in st.session_state: st.session_state.live_tick = 0
    if "live_rng" not in st.session_state: st.session_state.live_rng = np.random.default_rng()

    if "test_records" not in st.session_state: st.session_state.test_records = []
    if "test_tick" not in st.session_state: st.session_state.test_tick = 0
    if "test_rng" not in st.session_state: st.session_state.test_rng = np.random.default_rng()
    if "test_value_buffers" not in st.session_state: st.session_state.test_value_buffers = {"Source_A": [], "Source_B": [], "Source_C": []}
    if "test_trust_state" not in st.session_state: st.session_state.test_trust_state = {"Source_A": 0.5, "Source_B": 0.5, "Source_C": 0.5}
    if "test_reliability_state" not in st.session_state: st.session_state.test_reliability_state = {"Source_A": 0.7, "Source_B": 0.7, "Source_C": 0.7}
    if "test_smoothed_score_history" not in st.session_state: st.session_state.test_smoothed_score_history = {"Source_A": [], "Source_B": [], "Source_C": []}
    if "test_successful_events" not in st.session_state: st.session_state.test_successful_events = {"Source_A": 0, "Source_B": 0, "Source_C": 0}
    if "test_total_events" not in st.session_state: st.session_state.test_total_events = {"Source_A": 0, "Source_B": 0, "Source_C": 0}
    if "test_running" not in st.session_state: st.session_state.test_running = False

    if "trigger_spike" not in st.session_state: st.session_state.trigger_spike = False
    if "trigger_collusion" not in st.session_state: st.session_state.trigger_collusion = False
    if "trigger_chaos" not in st.session_state: st.session_state.trigger_chaos = False
    if "last_mode" not in st.session_state: st.session_state.last_mode = None
    if "final_decision_data" not in st.session_state: st.session_state.final_decision_data = None

init_session_state()

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<h2 style="font-weight:800; margin-bottom:1.5rem;">⚙️ <span class="gradient-text">Engine Control</span></h2>', unsafe_allow_html=True)
    mode = st.radio("Navigation Mode", ["🏠 Home", "📂 CSV Mode", "📡 Live API Mode", "🧪 Test Scenario Mode"])
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
        st.session_state.trigger_chaos = False
        st.session_state.final_decision_data = None
        reset_realtime_state()
        st.toast(f"Mode switched to {mode}.", icon="🔄")
    st.session_state.last_mode = mode

    if mode == "📂 CSV Mode":
        uploaded_file = st.file_uploader("Target CSV", type=["csv"])
    else:
        uploaded_file = None
        st.markdown(f'<div class="glass-card" style="padding:15px; margin-bottom:20px;"><p style="margin:0; font-size:0.85rem; font-weight:600;">📡 {mode} Active</p></div>', unsafe_allow_html=True)
        if st.button("🗑️ Reset Engine State", width="stretch"):
            st.session_state.live_records = []
            st.session_state.live_tick = 0
            st.session_state.streaming = False
            st.session_state.final_decision_data = None
            reset_realtime_state()
            st.rerun()

# ─────────────────────────────────────────────────────────────
# MAIN ROUTING
# ─────────────────────────────────────────────────────────────
if mode == "🏠 Home":
    render_home_ui()
elif mode == "📡 Live API Mode":
    render_live_ui()
elif mode == "🧪 Test Scenario Mode":
    render_test_ui(TEST_SCENARIOS)
elif mode == "📂 CSV Mode":
    render_csv_ui(uploaded_file)

# ─────────────────────────────────────────────────────────────
# POST-RENDER (Global Components)
# ─────────────────────────────────────────────────────────────
if mode != "🧪 Test Scenario Mode":
    st.divider()
    if st.session_state.get("final_decision_data") is not None:
        render_final_system_decision(st.session_state.final_decision_data)

st.divider()
# Continuous Update Loop
if mode == "📡 Live API Mode" and st.session_state.get("streaming"):
    time.sleep(2)
    st.rerun()
elif mode == "🧪 Test Scenario Mode" and st.session_state.get("test_running"):
    time.sleep(0.1)
    st.rerun()

st.caption("TrustLayer · Built for a 20-hour hackathon · No ML · Pure stats 🛡️")
