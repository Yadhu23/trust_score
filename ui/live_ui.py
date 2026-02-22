import streamlit as st
import pandas as pd
import time
import numpy as np
import plotly.graph_objects as go
from services.btc_api import fetch_btc_price
from services.simulation import simulate_live_sources
from ui.components import _render_interpretation
from trust_engine import (
    compute_anomaly_scores, compute_consensus_scores,
    update_historical_trust, compute_final_score, classify_trust,
    _trust_state, _score_history, _reliability_index,
    _total_events, _successful_events,
    _smoothed_score, _detect_trend, _compute_confidence,
    _build_reason, classify_reliability, classify_historic_status,
    compute_weighted_consensus, _conf_volatility_history,
    _low_variance_count, _SOURCES, _value_store, reset_realtime_state
)

def render_live_ui():
    """Renders the Live API Mode UI."""
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

    # ── Helper: render current live state ──────────────────────
    def _render_live_state():
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
                <span style="color:#8b949e;">Last update: {latest.iloc[0]['ts_label'] if 'ts_label' in latest.columns else 'N/A'}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ── Per-source trust layer cards ─────────────────────────
        with metrics_placeholder.container():
            mc1, mc2, mc3 = st.columns(3)
            
            # Rec source for highlighting
            rec_src_name = None
            if "final_decision_data" in st.session_state:
                fd_data = st.session_state.final_decision_data
                active = fd_data[~fd_data['decision'].str.contains("Isolate", na=False)]
                if not active.empty:
                    if 'historic_trust' in active.columns and 'reliability_index' in active.columns:
                        active = active.copy()
                        active['h_score'] = 0.7 * active['historic_trust'] + 0.3 * active['reliability_index']
                        rec_src_row = active.loc[active['h_score'].idxmax()]
                        rec_src_name = rec_src_row['source']

            source_colors = {"Source_A": "#58a6ff", "Source_B": "#bc8cff", "Source_C": "#f85149"}

            for col, src in zip([mc1, mc2, mc3], ["Source_A", "Source_B", "Source_C"]):
                if src not in latest.index:
                    continue
                row = latest.loc[src]
                val = row["value"]
                score = row.get("trust_score", 0.5)
                decision = row["decision"]
                color = source_colors.get(src, "white")
                
                is_best = (src == rec_src_name)

                with col:
                    # Circular Gauge (Plotly)
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = score * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        number = {'suffix': "%", 'font': {'size': 24, 'color': color, 'family': "JetBrains Mono"}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.1)"},
                            'bar': {'color': color},
                            'bgcolor': "rgba(255,255,255,0.03)",
                            'borderwidth': 0,
                            'steps': [
                                {'range': [0, 40], 'color': "rgba(248, 81, 73, 0.1)"},
                                {'range': [40, 75], 'color': "rgba(210, 153, 34, 0.1)"},
                                {'range': [75, 100], 'color': "rgba(63, 185, 80, 0.1)"}
                            ],
                        }
                    ))
                    fig.update_layout(
                        height=160, margin=dict(l=10, r=10, t=30, b=10),
                        paper_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': "Outfit"}
                    )
                    
                    st.markdown(f'''
                        <div class="glass-card" style="padding: 20px; border-top: 4px solid {color}; position:relative; {'box-shadow: 0 0 20px '+color+'44;' if is_best else ''}">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span style="font-weight:800; color:{color}; letter-spacing:0.05em;">{src}</span>
                                <div style="display:flex; align-items:center; gap:8px;">
                                    <div class="live-pulse"></div>
                                    <span class="badge badge-{'trusted' if 'Trusted' in decision else 'monitor' if 'Monitor' in decision else 'isolate'}">{decision}</span>
                                </div>
                            </div>
                            <div style="margin-top:15px; text-align:center;">
                                <p style="margin:0; font-size:0.75rem; color:var(--text-dim); text-transform:uppercase; font-weight:700;">Live stream</p>
                                <p style="margin:5px 0; font-size:2rem; font-weight:800; font-family:'JetBrains Mono';">${val:,.2f}</p>
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)
                    st.plotly_chart(fig, key=f"gauge_{src}_{latest_tick}", width="stretch", config={'displayModeBar': False})

        # Trust evolution chart (Plotly)
        if len(df["tick"].unique()) >= 2:
            pivot = df.pivot(index="tick", columns="source", values="trust_score")
            with chart_placeholder.container():
                st.markdown('<h3 style="font-weight:700; margin:2rem 0 1rem 0;">📈 Trust Evolution (Instant Confidence)</h3>', unsafe_allow_html=True)
                
                fig_ev = go.Figure()
                for src_name in pivot.columns:
                    fig_ev.add_trace(go.Scatter(
                        x=pivot.index, y=pivot[src_name],
                        name=src_name,
                        line=dict(color=source_colors.get(src_name, "white"), width=3),
                        mode='lines'
                    ))
                
                fig_ev.update_layout(
                    height=300, 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=10, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font={'color': "white"}),
                    font={'color': "white", 'family': "Outfit"},
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False)
                )
                st.plotly_chart(fig_ev, key=f"evolution_chart_{latest_tick}", width="stretch", config={'displayModeBar': False})

        # Recent data table (last 15 ticks × 3 sources = 45 rows)
        recent = df[df["tick"] >= max(0, latest_tick - 14)].copy()
        
        # ── Update Session State for Final Decision ───
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
    _render_live_state()

    # ── Streaming Step ─────────────────────────────────────────
    if st.session_state.streaming:
        rng = st.session_state.live_rng
        price = fetch_btc_price()

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
        for src, val in sources_vals.items():
            _value_store[src].append(float(val))

        # ── Step 3: Phase 2 — Run full trust model logic ──────────────────
        for src, val in sources_vals.items():
            # Anomaly
            series = pd.Series(_value_store[src], dtype=float)
            anom_df = compute_anomaly_scores(series, window=5)
            anomaly_score = float(anom_df["anomaly_score"].iloc[-1])
            if np.isnan(anomaly_score): anomaly_score = 0.0

            # Consensus logic
            latest_vals = {s: _value_store[s][-1] for s in _SOURCES}
            row_dict = {s: [latest_vals[s]] for s in _SOURCES}
            cons_df = compute_consensus_scores(pd.DataFrame(row_dict))
            sfx = src[-1]
            consensus_score = float(cons_df.iloc[0][f"consensus_score_{sfx}"])

            # ── Weighted internal consensus for performance ──────────────
            w_consensus = compute_weighted_consensus(latest_vals, _reliability_index)
            tolerance   = 0.02 * abs(w_consensus) + 1e-9
            _perf       = float(np.clip(1.0 - abs(val - w_consensus) / tolerance, 0.0, 0.995))

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
            conf_vol = float(np.std(list(_conf_volatility_history[src]))) if len(_conf_volatility_history[src]) >= 2 else 0.0

            # Reliability & Variance Penalty
            old_rel = _reliability_index[src]
            new_rel = round(0.98 * old_rel + 0.02 * _perf, 4)
            _reliability_index[src] = new_rel
            
            # Low variance check
            if len(_value_store[src]) >= 5:
                if pd.Series(_value_store[src][-5:]).std() < (0.0001 * abs(val)):
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

        # Re-render with new data (will be called in next loop or after rerun)
        _render_live_state()
