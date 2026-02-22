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
        source_colors = {"Source_A": "#58a6ff", "Source_B": "#bc8cff", "Source_C": "#f85149"}

        # Compute per-source anomaly frames
        anom_frames = {src: compute_anomaly_scores(sim_df[src]) for src in sources}

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
                    <div class="glass-card" style="padding: 20px; border-top: 4px solid {color}; margin-bottom: 20px;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
                            <span style="font-weight:800; color:{color}; letter-spacing:0.05em;">{src}</span>
                            <span class="badge badge-{'trusted' if 'Trusted' in decision else 'monitor' if 'Monitor' in decision else 'isolate'}">{decision}</span>
                        </div>
                        <p style="margin:0; font-size:0.75rem; color:var(--text-dim); text-transform:uppercase; font-weight:700;">Measured Value</p>
                        <p style="margin:5px 0; font-size:2.4rem; font-weight:800;">{row['value']:.3f}</p>
                        <div style="margin-top:20px; background:rgba(255,255,255,0.02); padding:12px; border-radius:8px; border:1px solid var(--border-color);">
                             <p style="margin:0; font-size:0.7rem; color:var(--text-dim); text-transform:uppercase; font-weight:700;">Trust Score</p>
                             <div style="display:flex; align-items:center; gap:10px; margin-top:5px;">
                                <div style="flex:1; height:6px; background:rgba(255,255,255,0.05); border-radius:3px;">
                                    <div style="width:{trust_val*100}%; height:100%; background:{color}; border-radius:3px; box-shadow:0 0 10px {color}88;"></div>
                                </div>
                                <span style="font-size:0.9rem; font-weight:800; color:{color};">{trust_val*100:.1f}%</span>
                             </div>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)

                _interp = interpret_source(float(row["final_score"]), 0.7, float(row["historical_trust"]), float(row["final_score"]), row["decision"])
                st.markdown(f'''
                    <div class="glass-card" style="padding:15px; border-left: 3px solid {color}88;">
                        <p style="margin:0; font-size:0.75rem; color:var(--text-dim); text-transform:uppercase; font-weight:700; margin-bottom:10px;">Engine Drilldown</p>
                        <div style="display:flex; flex-direction:column; gap:8px;">
                            <div style="font-size:0.85rem;"><span style="color:var(--text-dim);">Historical:</span> <b>{_interp["historic_assessment"]}</b></div>
                            <div style="font-size:0.85rem;"><span style="color:var(--text-dim);">Current:</span> <b>{_interp["current_behavior"]}</b></div>
                            <div style="margin-top:5px; padding-top:10px; border-top:1px solid var(--border-color); font-size:0.85rem; color:{color};">
                                <b>{_interp["recommendation"]}</b>
                            </div>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)

        st.divider()

        # ─────────────────────────────────────────────────────────────
        # SECTION 1.5 – Anomaly Analysis
        # ─────────────────────────────────────────────────────────────
        st.markdown('<h2 style="font-weight:800; margin:3rem 0 1.5rem 0;">🔎 Anomaly Detection Strategy</h2>', unsafe_allow_html=True)
        st.markdown('<p style="color:var(--text-dim); margin-bottom:2rem;">Rolling window Z-score analysis (window=5) to identify temporal drift and value spikes.</p>', unsafe_allow_html=True)

        anom_score_chart = pd.DataFrame(
            {src: anom_frames[src]["anomaly_score"].values for src in sources},
            index=sim_df["timestamp"],
        )
        
        fig_anom = go.Figure()
        for src_name in anom_score_chart.columns:
            fig_anom.add_trace(go.Scatter(
                x=anom_score_chart.index, y=anom_score_chart[src_name],
                name=src_name,
                line=dict(color=source_colors.get(src_name, "white"), width=2),
                mode='lines'
            ))
        
        fig_anom.update_layout(
            height=300, 
            title="Suspicion Index (0 = Normal, 1 = Anomaly)",
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font={'color': "white"}),
            font={'color': "white", 'family': "Outfit"},
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False)
        )
        st.plotly_chart(fig_anom, width="stretch", config={'displayModeBar': False})

        # Tabs for detail
        st.markdown("**Per-Source Anomaly Detail Tables** (🔴 = anomaly row)")
        tabs = st.tabs(["Source A", "Source B", "Source C"])

        for tab, src in zip(tabs, sources):
            with tab:
                af = anom_frames[src].copy()
                af.insert(0, "timestamp", sim_df["timestamp"].values)
                af.insert(1, "value",     sim_df[src].values)

                n_anom = int(af["is_anomaly"].sum())
                if n_anom == 0:
                    st.success(f"No anomalies detected in {src}.")
                else:
                    st.warning(f"**{n_anom} anomaly row(s)** detected in {src}.")

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
                            "value":        "{:.4f}", "rolling_mean": "{:.4f}",
                            "rolling_std":  "{:.4f}", "z_score":      "{:.4f}",
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
        st.markdown("> 💡 When **Source C spikes**, its deviation from the median is large → its consensus score drops near **0**.")

        consensus_df = compute_consensus_scores(sim_df)
        chart_cons = pd.DataFrame(
            {
                "Source_A": consensus_df["consensus_score_A"].values,
                "Source_B": consensus_df["consensus_score_B"].values,
                "Source_C": consensus_df["consensus_score_C"].values,
            },
            index=sim_df["timestamp"],
        )
        st.markdown('<h3 style="font-weight:700; margin:2.5rem 0 1rem 0;">🤝 Consensus Score Over Time</h3>', unsafe_allow_html=True)
        
        fig_cons = go.Figure()
        for src_name in chart_cons.columns:
            fig_cons.add_trace(go.Scatter(
                x=chart_cons.index, y=chart_cons[src_name],
                name=src_name,
                line=dict(color=source_colors.get(src_name, "white"), width=2, dash='dot' if src_name == 'Source_C' else 'solid'),
                mode='lines'
            ))
        
        fig_cons.update_layout(
            height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font={'color': "white"}),
            font={'color': "white", 'family': "Inter"},
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_cons, width="stretch", config={'displayModeBar': False})

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
        # HISTORICAL & FINAL SCORES
        # ─────────────────────────────────────────────────────────────
        st.header("📈 Historical Trust Update")
        
        # Chart 1: Historical trust
        hist_pivot = results.pivot(index="timestamp", columns="source", values="historical_trust")
        fig_hist = go.Figure()
        for src_name in hist_pivot.columns:
            fig_hist.add_trace(go.Scatter(x=hist_pivot.index, y=hist_pivot[src_name], name=src_name, line=dict(color=source_colors.get(src_name, "white"), width=3)))
        fig_hist.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=20, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_hist, key="hist_plotly", width="stretch")

        # Chart 2: Final trust
        trust_final_pivot = results.pivot(index="timestamp", columns="source", values="final_score")
        fig_final = go.Figure()
        for src_name in trust_final_pivot.columns:
            fig_final.add_trace(go.Scatter(x=trust_final_pivot.index, y=trust_final_pivot[src_name], name=src_name, line=dict(color=source_colors.get(src_name, "white"), width=3)))
        fig_final.add_hline(y=0.75, line_dash="dash", line_color="var(--trusted-green)")
        fig_final.add_hline(y=0.40, line_dash="dash", line_color="var(--isolate-red)")
        fig_final.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_final, key="final_plotly", width="stretch")

        # ── Update Session State for Final Decision (CSV Mode) ─────────
        ranking = (
            results.groupby("source", group_keys=False)
            .apply(lambda g: g.sort_values("timestamp").iloc[[-1]])
            .reset_index(drop=True)[["source", "final_score", "historical_trust",
                                       "anomaly_score", "consensus_score", "decision"]]
            .sort_values("final_score", ascending=False)
            .reset_index(drop=True)
        )
        csv_latest_vals = results[results["timestamp"] == results["timestamp"].max()]
        fd_data = pd.merge(ranking, csv_latest_vals[["source", "value"]], on="source")
        st.session_state.final_decision_data = fd_data

        # Explanation panel
        st.header("🔍 Why Was a Source Flagged?")
        sel_src = st.selectbox("Select a source to inspect:", sources)
        src_df = results[results["source"] == sel_src].copy()
        st.dataframe(src_df, width="stretch")
