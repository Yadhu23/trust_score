import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from services.simulation import simulate_sources
from trust_engine import (
    run_trust_pipeline, compute_anomaly_scores, compute_consensus_scores, interpret_source
)

def render_csv_ui(uploaded_file):
    """Renders the CSV Mode UI."""
    st.warning("📂 CSV ANALYSIS MODE")

    if uploaded_file is None:
        st.session_state.final_decision_data = None
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

        # ── Run the trust pipeline (Cached in session state) ─────────
        if "csv_results_v3" not in st.session_state or st.session_state.get("csv_filename") != uploaded_file.name:
            with st.spinner("Simulating sources and running TrustLayer analysis…"):
                st.session_state.csv_results_v3 = run_trust_pipeline(raw_df)
                st.session_state.csv_filename = uploaded_file.name
                # Ensure timestamp is datetime for robust plotting
                st.session_state.csv_results_v3["timestamp"] = pd.to_datetime(st.session_state.csv_results_v3["timestamp"], errors="coerce")
        
        results = st.session_state.csv_results_v3
        
        # Reconstruct sim_df from long-format results for backward compatibility in the UI
        sim_df = results.pivot(index="timestamp", columns="source", values="value").reset_index()

        sources = ["Source_A", "Source_B", "Source_C"]
        source_colors = {"Source_A": "#58a6ff", "Source_B": "#bc8cff", "Source_C": "#f85149"}

        # Extract per-source detail frames from results
        anom_frames = {}
        for src in sources:
            anom_frames[src] = results[results["source"] == src].copy()

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

        def highlight_spikes(row):
            diff = abs(row["Source_C"] - row["Source_A"])
            std_a = sim_df["Source_A"].std()
            mean_a = sim_df["Source_A"].mean()
            if abs(row["Source_C"] - mean_a) > 2 * std_a:
                return ["background-color: rgba(248,81,73,0.3)"] * len(row)
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
        # SECTION 0.5 – Per-Source Anomaly Detail Tables (🔴 = anomaly row)
        # ─────────────────────────────────────────────────────────────
        st.markdown("**Per-Source Anomaly Detail Tables** (🔴 = anomaly row)")
        tabs = st.tabs(["Source A", "Source B", "Source C"])

        for tab, src in zip(tabs, sources):
            with tab:
                af = anom_frames[src]

                n_anom = int(af["is_anomaly"].sum())
                if n_anom == 0:
                    st.success(f"✅ No anomalies detected in {src}")
                else:
                    st.warning(f"**{n_anom} anomaly row(s)** detected in {src}.")

                def _highlight_anom(row):
                    if row["is_anomaly"]:
                        return ["background-color: rgba(248,81,73,0.25)"] * len(row)
                    return [""] * len(row)

                display_af = af[["timestamp", "value", "rolling_mean", "rolling_std",
                                  "z_score", "is_anomaly", "anomaly_score"]]
                st.dataframe(
                    display_af.style
                        .apply(_highlight_anom, axis=1)
                        .format({
                            "value":        "{:.4f}", "rolling_mean": "{:.4f}",
                            "rolling_std":  "{:.4f}", "z_score":      "{:.4f}",
                            "anomaly_score":"{:.4f}",
                        }),
                    width="stretch",
                    hide_index=True,
                )

        st.divider()

        # ─────────────────────────────────────────────────────────────
        # SECTION 1 – Source Intelligence (Latest)
        # ─────────────────────────────────────────────────────────────
        st.markdown('<h2 style="font-weight:800; margin:3rem 0 2rem 0;">📊 Source Intelligence <span style="font-size:0.9rem; color:var(--text-dim); font-weight:400;">/ Latest Snapshot</span></h2>', unsafe_allow_html=True)

        last_ts = results["timestamp"].iloc[-1]
        last_rows = results[results["timestamp"] == last_ts]

        cols = st.columns(3)
        for col, src in zip(cols, sources):
            row = last_rows[last_rows["source"] == src].iloc[0]
            decision = row["decision"]
            trust_val = row["final_score"]
            color = source_colors.get(src, "var(--accent-blue)")

            with col:
                st.markdown(f'''
                    <div style="background: rgba(22,27,34,0.7); border: 1px solid {color}; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
                            <span style="font-weight:800; color:{color}; letter-spacing:0.05em; font-size:1.2rem;">{src}</span>
                            <span class="badge badge-{'trusted' if 'Trusted' in decision else 'monitor' if 'Monitor' in decision else 'isolate'}">{decision}</span>
                        </div>
                        <p style="margin:0; font-size:0.8rem; color:var(--text-dim); text-transform:uppercase; font-weight:700;">Measured Value</p>
                        <p style="margin:5px 0; font-size:2.4rem; font-weight:800;">{row['value']:.3f}</p>
                        <div style="margin-top:20px;">
                             <p style="margin:0; font-size:0.8rem; color:var(--text-dim); text-transform:uppercase; font-weight:700;">Trust Score</p>
                             <div style="display:flex; align-items:center; gap:10px; margin-top:5px;">
                                <div style="flex:1; background:#1e1e2e; border-radius:4px; height:8px;">
                                    <div style="width:{trust_val*100:.1f}%; height:8px; background:{color}; border-radius:4px; box-shadow:0 0 10px {color}88;"></div>
                                </div>
                                <span style="font-size:1rem; font-weight:800; color:{color};">{trust_val*100:.1f}%</span>
                             </div>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)

                _interp = interpret_source(
                    instant_confidence=float(row["final_score"]), 
                    reliability_index=float(row["reliability_index"]), 
                    historic_trust=float(row["historical_trust"]), 
                    final_score=float(row["final_score"]), 
                    status=row["decision"], 
                    source_id=src,
                    weighted_mean=float(row["value"])
                )
                st.markdown(f'''
                    <div style="padding:15px; border-left: 3px solid {color}88; margin-bottom:10px;">
                        <p style="margin:0; font-size:0.8rem; color:var(--text-dim); text-transform:uppercase; font-weight:700; margin-bottom:10px;">Engine Drilldown</p>
                        <div style="display:flex; flex-direction:column; gap:8px;">
                            <div style="font-size:0.9rem;"><span style="color:var(--text-dim);">Historical:</span> <b>{_interp.get("historic_assessment", "")}</b></div>
                            <div style="font-size:0.9rem;"><span style="color:var(--text-dim);">Current:</span> <b>{_interp.get("current_behavior", "")}</b></div>
                            <div style="margin-top:5px; padding-top:10px; border-top:1px solid rgba(255,255,255,0.05); font-size:0.95rem; color:{color}; font-weight:700;">
                                {_interp.get("recommendation", "")}
                            </div>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)

        st.divider()




        # ─────────────────────────────────────────────────────────────
        # SECTION 1.7 – Cross-Source Consensus Scoring
        # ─────────────────────────────────────────────────────────────
        st.header("🤝 Cross-Source Consensus Scoring")
        st.markdown("> 💡 When **Source C spikes**, its deviation from the median is large → its consensus score drops near **0**.")

        consensus_df = compute_consensus_scores(sim_df)
        chart_cons = results.pivot(index="timestamp", columns="source", values="consensus_score")

        # Summary Metrics
        LOW_CONS_THRESHOLD = 0.3
        low_cons_c = int((consensus_df["consensus_score_C"] < LOW_CONS_THRESHOLD).sum())
        low_cons_b = int((consensus_df["consensus_score_B"] < LOW_CONS_THRESHOLD).sum())

        mc1, mc2, mc3 = st.columns(3)
        with mc1: st.metric("Avg Consensus – A", f"{consensus_df['consensus_score_A'].mean():.3f}")
        with mc2: st.metric("Avg Consensus – B", f"{consensus_df['consensus_score_B'].mean():.3f}", delta=f"{low_cons_b} rows low" if low_cons_b else "All good", delta_color="inverse")
        with mc3: st.metric("Avg Consensus – C", f"{consensus_df['consensus_score_C'].mean():.3f}", delta=f"{low_cons_c} rows low", delta_color="inverse")

        st.divider()

        # ─────────────────────────────────────────────────────────────
        # SECTION 1.8 – Historical Belief Evolution
        # ─────────────────────────────────────────────────────────────
        st.markdown('<h2 style="font-weight:800; margin:3rem 0 1.5rem 0;">📉 Historical Belief Evolution</h2>', unsafe_allow_html=True)
        st.markdown('<p style="color:var(--text-dim); margin-bottom:2rem;">Visualization of how the engine\'s trust in each source evolved over the course of the dataset.</p>', unsafe_allow_html=True)

        trust_final_pivot = results.pivot(index="timestamp", columns="source", values="final_score")
        fig_final = go.Figure()
        for src_name in trust_final_pivot.columns:
            fig_final.add_trace(go.Scatter(
                x=trust_final_pivot.index, 
                y=trust_final_pivot[src_name], 
                name=src_name, 
                line=dict(color=source_colors.get(src_name, "white"), width=3)
            ))
        
        # Horizontal lines for thresholds
        fig_final.add_hline(y=0.75, line_dash="dash", line_color="var(--trusted-green)", annotation={"text": "Trust Threshold", "font_color": "var(--trusted-green)"})
        fig_final.add_hline(y=0.40, line_dash="dash", line_color="var(--isolate-red)", annotation={"text": "Isolation Threshold", "font_color": "var(--isolate-red)"})
        
        fig_final.update_layout(
            height=400, 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font={'color': "white"}),
            font={'color': "white", 'family': "Inter"},
            xaxis=dict(showgrid=False), 
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", range=[0, 1.05])
        )
        st.plotly_chart(fig_final, key="belief_evolution_plotly", width="stretch", config={'displayModeBar': False})

        # ── Update Session State for Final Decision (CSV Mode) ─────────
        latest_records = []
        for src in sources:
            src_res = results[results["source"] == src].sort_values("timestamp")
            if not src_res.empty:
                latest_records.append(src_res.iloc[-1])
        
        ranking = pd.DataFrame(latest_records)
        ranking = ranking[["source", "final_score", "historical_trust", "anomaly_score", "consensus_score", "decision"]]
        ranking = ranking.sort_values("source").reset_index(drop=True)
        csv_latest_vals = results[results["timestamp"] == results["timestamp"].max()]
        fd_data = pd.merge(ranking, csv_latest_vals[["source", "value"]], on="source")
        st.session_state.final_decision_data = fd_data

        # Explanation panel
        st.header("🔍 Why Was a Source Flagged?")
        sel_src = st.selectbox("Select a source to inspect:", sources, key="csv_source_inspector")
        
        # Get latest data for this source for interpretation
        # Use .iloc[-1] for direct, reactive access to the latest state of the selected source
        src_all = results[results["source"] == sel_src].sort_values("timestamp")
        if not src_all.empty:
            row = src_all.iloc[-1]
            interp = interpret_source(
                instant_confidence=row["final_score"],
                reliability_index=row["reliability_index"],
                historic_trust=row["historic_trust"],
                final_score=row["final_score"],
                status=row["decision"],
                source_id=sel_src,
                weighted_mean=row["value"]
            )
            
            # Render interpretation in a premium box
            st.markdown(f"""
                <div style="background:#12172a; border-left:3px solid #7eb8f7; border-radius:10px; padding:18px 24px; margin-bottom:20px; font-size:16px; line-height:1.8; border:1px solid #30363d;">
                    <div class="metric-sub-header" style="color:var(--text-dim); font-size:0.75rem; font-weight:700; text-transform:uppercase; margin-bottom:10px;">📋 Engine Interpretation: {sel_src}</div>
                    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:15px;">
                        <div>
                            <p style="margin:0; font-size:0.75rem; color:var(--text-dim); text-transform:uppercase; font-weight:700;">Historical Assessment</p>
                            <p style="margin:2px 0 0 0; font-weight:700;">{interp['historic_assessment']}</p>
                        </div>
                        <div>
                            <p style="margin:0; font-size:0.75rem; color:var(--text-dim); text-transform:uppercase; font-weight:700;">Current Behavior</p>
                            <p style="margin:2px 0 0 0; font-weight:700;">{interp['current_behavior']}</p>
                        </div>
                    </div>
                    <div style="margin-top:15px; padding-top:15px; border-top:1px solid rgba(255,255,255,0.05);">
                        <p style="margin:0; font-size:0.75rem; color:var(--text-dim); text-transform:uppercase; font-weight:700;">Operational Recommendation</p>
                        <p style="margin:4px 0 0 0; font-size:1.1rem; font-weight:800; color:{'#f85149' if 'Isolate' in row['decision'] else '#58a6ff'};">
                            {interp['recommendation']}
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        src_df = results[results["source"] == sel_src].copy().sort_values("timestamp", ascending=False)
        st.dataframe(src_df, width="stretch")
