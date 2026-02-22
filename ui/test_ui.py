import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ui.components import _render_interp_card
from trust_engine import (
    interpret_source, compute_anomaly_scores, compute_consensus_scores,
    update_historical_trust, compute_final_score, classify_trust
)

def render_test_ui(TEST_SCENARIOS):
    """Renders the Test Scenario Mode (Stress Lab) UI."""
    st.markdown('<h1 class="dashboard-header">🧪 <span class="gradient-text">Stress Lab</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="dashboard-tagline">Deterministic simulation of systemic failures and edge-case behaviors.</p>', unsafe_allow_html=True)

    # ── Internal Validation Logic ──────────────────────────────
    def _render_sc_validation(sc_name, rdf):
        """Computes and renders rule-based validation for scenarios."""
        st.markdown('<h2 style="font-weight:800; margin:3rem 0 1.5rem 0;">🧪 Scenario Validation Summary</h2>', unsafe_allow_html=True)
        
        rlast = rdf[rdf["tick"] == rdf["tick"].max()]
        sources = ["Source_A", "Source_B", "Source_C"]
        
        # Rule Processing
        rules = []
        
        if sc_name == "Normal":
            all_trusted = all("Trusted" in str(d) for d in rlast["decision"])
            rules.append({"name": "All sources maintained high trust", "pass": all_trusted})
        
        elif sc_name == "Single Source Failure":
            isolated = rlast[rlast["decision"].str.contains("Isolate", na=False)]
            trusted = rlast[rlast["decision"].str.contains("Trusted", na=False)]
            rules.append({"name": "Exactly one source isolated", "pass": len(isolated) == 1})
            rules.append({"name": "Two sources remained trusted", "pass": len(trusted) == 2})
            
        elif sc_name == "Two Weak Collude":
            flagged = rlast[rlast["decision"].str.contains("Monitor|Isolate", na=False)]
            rules.append({"name": "Collusion detected (at least one flagged)", "pass": len(flagged) >= 1})
            
        elif sc_name == "Gradual Drift":
            src_a = rlast[rlast["source"] == "Source_A"]
            is_monitor = src_a["decision"].str.contains("Monitor", na=False).any()
            rules.append({"name": "Drifting source identified as Monitor", "pass": is_monitor})
            
        elif sc_name == "Chaos":
            flagged = rlast[rlast["decision"].str.contains("Monitor|Isolate", na=False)]
            rules.append({"name": "Systemic failure recognized (majority flagged)", "pass": len(flagged) >= 2})
            
        elif sc_name == "Recovery":
            src_b = rlast[rlast["source"] == "Source_B"]
            recovered = src_b["decision"].str.contains("Trusted|Monitor", na=False).any()
            rules.append({"name": "Stabilization recognized (source B recovered)", "pass": recovered})

        # Render Logic
        all_pass = all(r["pass"] for r in rules) if rules else False
        
        inner_html = ""
        for r in rules:
            icon = "✅" if r["pass"] else "❌"
            inner_html += f'<div style="margin-bottom:10px; font-size:1.1rem;">Rule: {r["name"]} → {icon}</div>'
            
        summary_text = "🎉 Scenario behaving as expected." if all_pass else "⚠️ Scenario behavior deviates from expected logic."
        summary_color = "var(--trusted-green)" if all_pass else "var(--isolate-red)"

        st.markdown(
            f'''
            <div class="glass-card" style="border-left: 5px solid {summary_color};">
                {inner_html}
                <div style="margin-top:20px; padding-top:15px; border-top: 1px solid var(--border-color); font-weight:800; font-size:1.2rem; color:{summary_color};">
                    {summary_text}
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    # ── Scenario Selection ─────────────────────────────────────
    sc_name = st.selectbox("Select Scenario", list(TEST_SCENARIOS.keys()), help="Choose a behavior pattern to simulate.")
    sc_cfg = TEST_SCENARIOS[sc_name]
    
    st.markdown(
        f'''
        <div class="glass-card" style="margin-bottom:2rem; border-left: 4px solid var(--accent-blue);">
            <p style="margin:0; font-size:0.9rem; color:var(--text-dim); text-transform:uppercase; font-weight:700;">Scenario Blueprint</p>
            <p style="margin:10px 0; font-size:1.1rem; font-weight:600;">{sc_cfg['desc']}</p>
            <p style="margin:0; font-size:0.85rem; color:var(--trusted-green); font-weight:600;">🎯 Expected Outcome: {sc_cfg['expected']}</p>
        </div>
        ''',
        unsafe_allow_html=True
    )

    sc_col1, sc_col2 = st.columns([2, 1])
    with sc_col2:
        if not st.session_state.test_running:
            if st.button("▶ Start Simulation", width="stretch"):
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
            if st.button("⏹ Terminate", width="stretch"):
                st.session_state.test_running = False
                st.rerun()

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

        # Triggers
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
        
        # Processing Pipeline
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
            new_t, _ = update_historical_trust(old_t, anom, cs, performance=perf, source_id=s)
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
            decision = classify_trust(
                smoothed, 
                disagreement_index=dis_idx, 
                anomaly_score=anom,
                weighted_mean=w_cons,
                is_extreme_chaos=is_ex
            )
            
            # Historic trust update
            st.session_state.test_total_events[s] += 1
            if perf >= 0.6:
                st.session_state.test_successful_events[s] += 1
            h_trust = st.session_state.test_successful_events[s] / st.session_state.test_total_events[s]

            # Interpretation
            interp_data = interpret_source(
                instant_confidence=smoothed,
                reliability_index=new_rel,
                historic_trust=h_trust,
                final_score=final,
                status=decision,
                source_id=s,
            )

            st.session_state.test_records.append({
                "tick": tick, "source": s, "value": val, "final_score": final,
                "smoothed": smoothed,
                "decision": decision,
                "historical_trust": new_t,
                "historic_trust": h_trust,
                "anomaly_score": anom,
                "consensus_score": cs,
                "reliability_index": new_rel,
                "is_anomaly": anom > 0.4,
                "interpretation": interp_data
            })

    # ── Render Results ─────────────────────────────────────────
    if st.session_state.test_records:
        rdf = pd.DataFrame(st.session_state.test_records)
        rlast = rdf[rdf["tick"] == rdf["tick"].max()]
        
        st.markdown(f'<h2 style="font-weight:800; margin:2rem 0 1.5rem 0;">📊 Real-time Simulation Feed (Tick: {st.session_state.test_tick} / {sc_cfg["max_ticks"]})</h2>', unsafe_allow_html=True)
        
        # ── Update Session State for Final Decision ───
        st.session_state.final_decision_data = rlast

        # 1. Source cards
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
                with rcols[i]: 
                    st.markdown(f'<div class="glass-card" style="padding:20px; text-align:center;"><p style="margin:0; color:var(--text-dim);">{s}</p><p style="font-size:1.2rem; font-weight:800;">OFFLINE</p></div>', unsafe_allow_html=True)
        
        st.divider()
        
        # 2. Trust evolution chart
        st.markdown('<h3 style="font-weight:700; margin:2rem 0 1.5rem 0;">📈 Trust Evolution</h3>', unsafe_allow_html=True)
        pivot_test = rdf.pivot_table(index="tick", columns="source", values="smoothed")
        
        fig_test = go.Figure()
        source_colors = {"Source_A": "#58a6ff", "Source_B": "#bc8cff", "Source_C": "#f85149"}
        
        for src_name in pivot_test.columns:
            fig_test.add_trace(go.Scatter(
                x=pivot_test.index, y=pivot_test[src_name],
                name=src_name,
                line=dict(color=source_colors.get(src_name, "white"), width=3),
                mode='lines'
            ))
        
        fig_test.update_layout(
            height=350, 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font={'color': "white"}),
            font={'color': "white", 'family': "Outfit"},
            xaxis=dict(showgrid=False, zeroline=False, title="Simulation Tick"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, title="Trust Score")
        )
        st.plotly_chart(fig_test, key=f"test_chart_{st.session_state.test_tick}", width="stretch", config={'displayModeBar': False})

        # 3. Ranking Table
        st.markdown('<h3 style="font-weight:700; margin:2rem 0 1.5rem 0;">🏆 Intelligence Ranking</h3>', unsafe_allow_html=True)
        ranking = rlast[["source", "final_score", "historical_trust", "anomaly_score", "consensus_score", "decision"]].copy()
        ranking = ranking.sort_values("final_score", ascending=False).reset_index(drop=True)
        ranking.index = ranking.index + 1
        st.dataframe(ranking.style.format({
            "final_score": "{:.3f}", "historical_trust": "{:.3f}", 
            "anomaly_score": "{:.3f}", "consensus_score": "{:.3f}"
        }), width="stretch")

        # 4. Global Validation Summary
        if st.session_state.test_tick == sc_cfg["max_ticks"]:
            _render_sc_validation(sc_name, rdf)
