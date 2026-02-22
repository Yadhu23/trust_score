import streamlit as st
import time
import pandas as pd
from trust_engine import interpret_source

def _render_interp_card(src, val, final, decision, conclusion):
    """Common UI block for source status (Test Mode)."""
    source_colors = {"Source_A": "#58a6ff", "Source_B": "#bc8cff", "Source_C": "#f85149"}
    color = source_colors.get(src, "var(--accent-blue)")
    
    st.markdown(f"""
        <div class="glass-card" style="padding: 15px; border-top: 4px solid {color}; margin-bottom:15px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <span style="font-weight:700; color:{color};">{src}</span>
                <span class="badge badge-{'trusted' if 'Trusted' in decision else 'monitor' if 'Monitor' in decision else 'isolate'}">{decision}</span>
            </div>
            <p style="font-size:1.6rem; font-weight:800; margin:5px 0;">{val}</p>
            <div style="background:rgba(255,255,255,0.02); padding:10px; border-radius:8px; border:1px dashed var(--border-color);">
                 <p style="margin:0; font-size:0.75rem; font-weight:700; color:var(--text-dim); text-transform:uppercase;">Conclusion</p>
                 <p style="margin:5px 0 0 0; font-size:0.85rem; font-weight:600;">{conclusion}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_final_system_decision(latest_df):
    """
    Renders the Final System Decision block with a premium SaaS dashboard aesthetic.
    """
    if latest_df is None or latest_df.empty:
        return

    # ── 1. Aggregate Statistics & Calculations ──────────────────
    score_col = 'trust_score' if 'trust_score' in latest_df.columns else 'smoothed' if 'smoothed' in latest_df.columns else 'final_score'
    avg_conf = latest_df[score_col].mean()
    
    if avg_conf >= 0.80:   conf_lvl, conf_icon, conf_col = "Optimal", "🟢", "var(--trusted-green)"
    elif avg_conf >= 0.50: conf_lvl, conf_icon, conf_col = "Warning", "🟡", "var(--monitor-yellow)"
    else:                  conf_lvl, conf_icon, conf_col = "Critical", "🔴", "var(--isolate-red)"

    # Decision Logic
    trusted = latest_df[latest_df['decision'].str.contains("Trusted", na=False)]
    monitors = latest_df[latest_df['decision'].str.contains("Monitor", na=False)]
    is_chaos = any("Isolate" in str(d) for d in latest_df['decision']) and len(trusted) == 0
    
    status = "System Uncertain"
    status_icon = "⚪"
    rec_val = None
    
    active_sources = latest_df[~latest_df['decision'].str.contains("Isolate", na=False)]
    best_source = None
    
    if is_chaos or active_sources.empty:
        status = "System Uncertain"
        status_icon = "❓"
    else:
        if not trusted.empty:
            status = "Operational"
            status_icon = "🟢"
            rec_val = (trusted['value'] * trusted[score_col]).sum() / trusted[score_col].sum()
        else:
            status = "Under Pressure"
            status_icon = "🟡"
            if monitors.empty:
                rec_val = active_sources['value'].median()
            else:
                rec_val = monitors['value'].mean()
        
        if not active_sources.empty:
            if 'historic_trust' in active_sources.columns and 'reliability_index' in active_sources.columns:
                temp_sources = active_sources.copy()
                temp_sources['historical_score'] = 0.7 * temp_sources['historic_trust'] + 0.3 * temp_sources['reliability_index']
                best_source = temp_sources.loc[temp_sources['historical_score'].idxmax()]
            else:
                best_source = active_sources.loc[active_sources[score_col].idxmax()]

    # ── 2. Top Bar Navigation ──────────────────────────────────
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
            <div class="glass-card" style="padding: 12px 24px; flex: 1; margin-right: 15px; border-left: 4px solid {conf_col};">
                <p style="margin:0; font-size:0.75rem; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.1em;">System Status</p>
                <p style="margin:5px 0 0 0; font-size:1.2rem; font-weight:800;">{status_icon} {status}</p>
            </div>
            <div class="glass-card" style="padding: 12px 24px; flex: 1; margin-right: 15px;">
                <p style="margin:0; font-size:0.75rem; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.1em;">Engine Confidence</p>
                <p style="margin:5px 0 0 0; font-size:1.2rem; font-weight:800; color:{conf_col};">{avg_conf*100:.1f}%</p>
            </div>
            <div class="glass-card" style="padding: 12px 24px; flex: 1;">
                <p style="margin:0; font-size:0.75rem; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.1em;">System Clock</p>
                <p style="margin:5px 0 0 0; font-size:1.2rem; font-weight:800; font-family:'JetBrains Mono';">{time.strftime("%H:%M:%S")}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── 3. Main Decision Block ──────────────────────────────────
    st.markdown('<h3 style="font-weight:700; margin-bottom:1.5rem;">🏆 Consensus Intelligence</h3>', unsafe_allow_html=True)
    
    val_str = f"${rec_val:,.2f}" if rec_val is not None else "None"
    best_src_name = "None"
    best_src_score = ""
    
    if best_source is not None:
        if 'source' in best_source and pd.notna(best_source['source']):
            best_src_name = best_source['source']
        elif hasattr(best_source, 'name') and best_source.name and "Source_" in str(best_source.name):
            best_src_name = best_source.name
        
        h_score = best_source.get('historical_score')
        if pd.notna(h_score):
            best_src_score = f"{h_score:.3f} HIST"
        else:
            best_src_score = f"{best_source[score_col]:.3f}"

    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown(
            f"""
            <div class="glass-card" style="height: 100%; position: relative; overflow: hidden;">
                <div style="position: absolute; top: -20px; right: -20px; font-size: 8rem; opacity: 0.03;">💰</div>
                <p style="margin:0; font-size:0.85rem; color:var(--text-dim); text-transform:uppercase;">Recommended Consensus Value</p>
                <p style="margin:10px 0; font-size:3.5rem; font-weight:800; color:var(--accent-blue);">${val_str}</p>
                <div style="display:flex; align-items:center; gap:10px; margin-top:20px;">
                    <span class="badge badge-trusted">Validated Source</span>
                    <span style="font-size:0.9rem; color:var(--text-dim);">Weighted mean of trusted nodes</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="glass-card" style="height: 100%; border-top: 4px solid var(--accent-purple);">
                <p style="margin:0; font-size:0.85rem; color:var(--text-dim); text-transform:uppercase;">Top Performing Source</p>
                <p style="margin:15px 0; font-size:1.8rem; font-weight:800;">{best_src_name}</p>
                <p style="margin:0; font-size:1rem; font-weight:600; color:var(--accent-purple);">{best_src_score}</p>
                <div style="margin-top:25px; height:6px; background:rgba(255,255,255,0.05); border-radius:3px;">
                    <div style="width:{avg_conf*100}%; height:100%; background:var(--accent-purple); border-radius:3px; box-shadow: 0 0 10px var(--accent-purple);"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Human-Readable Explanation
    is_spike = any("Isolate" in str(d) for d in latest_df['decision']) and not trusted.empty
    is_single = len(latest_df) < 2
    
    if is_chaos:
        explanation = "Critical disagreement across all sources. Engine is unable to form a reliable consensus."
        insight_col = "var(--isolate-red)"
    elif is_spike:
        explanation = "Outlier detected in sub-sources. The system has isolated the noise to maintain data integrity."
        insight_col = "var(--monitor-yellow)"
    elif is_single:
        explanation = "Insufficient sources for cross-validation. Operating on single-stream data."
        insight_col = "var(--accent-blue)"
    else:
        explanation = "All sources are statistically aligned. Consensus is stable and highly reliable."
        insight_col = "var(--trusted-green)"

    st.markdown(
        f"""
        <div class="glass-card" style="margin-top:1.5rem; padding:15px 25px; display:flex; align-items:center; gap:20px; border-left: 4px solid {insight_col};">
            <div style="font-size:1.5rem;">🧬</div>
            <div>
                <p style="margin:0; font-size:0.75rem; color:var(--text-dim); text-transform:uppercase; font-weight:700;">Engine Insight</p>
                <p style="margin:0; font-size:1rem; font-weight:500;">{explanation}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

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
          <div class="metric-sub-header" style="color:var(--text-dim); font-size:0.75rem; font-weight:700; text-transform:uppercase; margin-bottom:10px;">
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
