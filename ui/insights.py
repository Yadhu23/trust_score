import streamlit as st
import pandas as pd

def render_insights_panel(records: list[dict]):
    """
    Renders a rich insights panel based on the completed simulation records.
    """
    if not records:
        return
        
    df = pd.DataFrame(records)
    
    src_col = "source_id" if "source_id" in df.columns else "source"
    tick_col = "tick" if "tick" in df.columns else "timestamp"
    
    score_col = next((c for c in ["smoothed_score", "trust_score", "smoothed", "final_score", "final"] if c in df.columns), None)
    anom_col = "anomaly_score" if "anomaly_score" in df.columns else "anomaly"
    cons_col = "consensus_score" if "consensus_score" in df.columns else "consensus"
    
    if not score_col:
        st.warning("Could not locate a trust score column for insights generation.")
        return

    # Approximate disagreement index if it was not recorded directly
    dis_idx_col = "disagreement_index"
    if dis_idx_col not in df.columns:
        df[dis_idx_col] = df.groupby(tick_col)["value"].transform("std").fillna(0)
        
    # CSS & Styles
    st.markdown("""
        <style>
        .insights-header {
            background: linear-gradient(90deg, #58a6ff, #bc8cff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            margin-bottom: 1rem;
            margin-top: 2rem;
            font-family: 'Outfit', sans-serif;
        }
        .insights-card {
            background: rgba(22,27,34,0.7);
            border: 1px solid rgba(48,54,61,0.6);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
            font-family: 'Outfit', sans-serif;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 class='insights-header'>🧠 Post-Simulation Insights</h2>", unsafe_allow_html=True)
    
    # ── SECTION 1 — SYSTEM VERDICT CARD ──
    avg_scores = df.groupby(src_col)[score_col].mean().to_dict()
    most_reliable = max(avg_scores, key=avg_scores.get)
    least_reliable = min(avg_scores, key=avg_scores.get)
    
    if anom_col in df.columns:
        total_anomalies = int((df[anom_col] > 0.5).sum())
    else:
        total_anomalies = 0
        
    ticks = df[tick_col].nunique()
    
    if cons_col in df.columns:
        mean_cons = df.groupby(tick_col)[cons_col].mean()
        healthy_ticks = int((mean_cons > 0.6).sum())
        health_pct = int(healthy_ticks / max(1, ticks) * 100)
    else:
        health_pct = 100

    verdict_html = f"""
    <div class="insights-card">
        <h3 style="margin-top:0; font-weight:800;">🛑 System Verdict</h3>
        <p style="font-size:1.1rem; margin-bottom:5px;">✅ <b>{most_reliable}</b> was the most reliable feed across {ticks} ticks.</p>
        <p style="font-size:1.1rem; margin-bottom:5px;">🚨 <b>{least_reliable}</b> was isolated {total_anomalies} times due to anomalies.</p>
        <p style="font-size:1.1rem; margin-bottom:0;">🤝 System consensus was healthy for {health_pct}% of the run.</p>
    </div>
    """
    st.markdown(verdict_html, unsafe_allow_html=True)
    
    # ── SECTION 2 — TRUST JOURNEY PER SOURCE ──
    st.markdown("<h3 class='insights-header'>🛣️ Trust Journey Per Source</h3>", unsafe_allow_html=True)
    colors = {"Source_A": "#58a6ff", "Source_B": "#bc8cff", "Source_C": "#ff7b72"}
    
    for src in sorted(df[src_col].unique()):
        src_df = df[df[src_col] == src].sort_values(tick_col)
        if src_df.empty: continue
        start_t = src_df[score_col].iloc[0]
        end_t = src_df[score_col].iloc[-1]
        peak_t = src_df[score_col].max()
        peak_tick = src_df.loc[src_df[score_col].idxmax(), tick_col]
        
        trend = "Stable"
        if end_t > start_t + 0.1:
            trend = "Improving"
        elif end_t < start_t - 0.1:
            trend = "Degrading"
            
        color = colors.get(src, "#ffffff")
        
        journey_html = f"""
        <div class="insights-card" style="border-left: 4px solid {color};">
            <p style="margin:0; font-size:1.05rem;">
                <b style="color:{color};">{src}</b> started at {start_t:.2f}, peaked at {peak_t:.2f} at tick {peak_tick}, ended at {end_t:.2f}. Currently {trend.lower()}.
            </p>
        </div>
        """
        st.markdown(journey_html, unsafe_allow_html=True)
        
    # ── SECTION 3 — ANOMALY EVENT LOG ──
    st.markdown("<h3 class='insights-header'>⚠️ Anomaly Event Log</h3>", unsafe_allow_html=True)
    if anom_col in df.columns:
        anom_df = df[df[anom_col] > 0.5].sort_values(tick_col).copy()
        if anom_df.empty:
            st.markdown("<div class='insights-card'>✅ No significant anomalies detected</div>", unsafe_allow_html=True)
        else:
            cols_to_show = [tick_col, src_col, "value"]
            if "z_score" in df.columns: cols_to_show.append("z_score")
            cols_to_show.extend([anom_col, "decision"])
            st.dataframe(anom_df[cols_to_show], width="stretch", hide_index=True)
            
    # ── SECTION 4 — CONSENSUS ANALYSIS ──
    st.markdown("<h3 class='insights-header'>🤝 Consensus Analysis</h3>", unsafe_allow_html=True)
    
    # group by tick so we don't overcount sources
    tick_dis = df.groupby(tick_col)[dis_idx_col].mean()
    avg_disagreement = tick_dis.mean()
    
    gb_val = df.groupby(tick_col)["value"].mean()
    chaos_ticks = sum(tick_dis > (gb_val * 0.15).clip(lower=5.0))
    chaos_pct = int(chaos_ticks / max(1, ticks) * 100)
    
    agree_ticks = sum(tick_dis < (gb_val * 0.05).clip(lower=0.5))
    agree_pct = int(agree_ticks / max(1, ticks) * 100)
        
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Disagreement", f"{avg_disagreement:.2f}")
    c2.metric("Time in Chaos", f"{chaos_pct}%")
    c3.metric("Perfect Agreement", f"{agree_pct}%")
    
    if chaos_pct > 20:
        consensus_verdict = "System experienced frequent conflict — data reliability is low."
    elif agree_pct > 60:
        consensus_verdict = f"Sources were in strong consensus for {agree_pct}% of the run."
    else:
        consensus_verdict = "System experienced moderate varying consensus."
        
    st.markdown(f"<div class='insights-card'><i style='color:var(--text-dim)'>{consensus_verdict}</i></div>", unsafe_allow_html=True)
    
    # ── SECTION 5 — RECOVERY DETECTION ──
    st.markdown("<h3 class='insights-header'>⚕️ Recovery Detection</h3>", unsafe_allow_html=True)
    recovery_detected = False
    
    for src in sorted(df[src_col].unique()):
        src_df = df[df[src_col] == src].sort_values(tick_col)
        consecutive_low = 0
        recovered_tick = None
        
        for _, row in src_df.iterrows():
            sc = row[score_col]
            if sc < 0.5:
                consecutive_low += 1
            elif sc > 0.65 and consecutive_low >= 5:
                recovered_tick = row[tick_col]
                break
            elif sc > 0.5:
                consecutive_low = 0
                
        if recovered_tick is not None:
            st.markdown(f"<div class='insights-card' style='border-left: 4px solid #3fb950;'>⬆️ <b>{src}</b> recovered at tick {recovered_tick} after {consecutive_low} ticks of instability</div>", unsafe_allow_html=True)
            recovery_detected = True
            
    if not recovery_detected:
        st.markdown("<div class='insights-card'>No recovery events detected</div>", unsafe_allow_html=True)
        
    # ── SECTION 6 — FINAL RECOMMENDATION ──
    st.markdown("<h3 class='insights-header'>💡 Final Recommendation</h3>", unsafe_allow_html=True)
    
    final_recs = []
    for src, avg_sc in avg_scores.items():
        if avg_sc > 0.80:
            final_recs.append(f"✅ Use {src} as your primary feed")
            break
            
    if max(avg_scores.values()) < 0.5:
        final_recs.append("🚨 All sources showed instability — cross-validate before trusting any single feed")
        
    if recovery_detected:
        final_recs.append("⚕️ Source self-healing behavior observed — consider adaptive weighting")
        
    if chaos_pct > 20:
        final_recs.append("⚠️ High conflict detected — implement redundancy or add a 4th source")
        
    if not final_recs:
        final_recs.append("✅ System operating nominally. Continue normal operations.")
        
    html_recs = "<br><br>".join([f"<b>{r}</b>" for r in final_recs])
    st.markdown(f"<div class='insights-card' style='font-size:1.15rem; border-color:#58a6ff;'>{html_recs}</div>", unsafe_allow_html=True)
