"""
ui/research_lab.py
──────────────────────────────────────────────
TrustLayer Research Lab — 10 analytical panels.
Reads from st.session_state.live_records or
st.session_state.test_records (whichever is populated).
Pure read-only — never mutates engine state.
"""

import json
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─── helpers ──────────────────────────────────────────────────
SOURCES = ["Source_A", "Source_B", "Source_C"]
COLORS  = {"Source_A": "#58a6ff", "Source_B": "#bc8cff", "Source_C": "#ff7b72"}

def _section(title: str):
    st.markdown(
        f'<h3 style="background:linear-gradient(90deg,#58a6ff,#bc8cff);'
        f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        f'font-weight:800;margin:2.5rem 0 1.2rem 0;">{title}</h3>',
        unsafe_allow_html=True,
    )

def _card(html: str, border_color: str = "rgba(48,54,61,0.6)"):
    st.markdown(
        f'<div style="background:rgba(22,27,34,0.7);border:1px solid {border_color};'
        f'border-radius:16px;padding:22px;margin-bottom:14px;">{html}</div>',
        unsafe_allow_html=True,
    )

def _get_records() -> list[dict]:
    """Return whichever record list has data (live takes precedence)."""
    live = st.session_state.get("live_records", [])
    test = st.session_state.get("test_records", [])
    return live if live else test


def _build_df(records: list[dict]) -> pd.DataFrame | None:
    if not records:
        return None
    df = pd.DataFrame(records)
    # Normalize column names coming from live vs test modes
    if "source" in df.columns and "source_id" not in df.columns:
        df = df.rename(columns={"source": "source_id"})
    if "trust_score" in df.columns and "smoothed" not in df.columns:
        df = df.rename(columns={"trust_score": "smoothed"})
    if "tick" not in df.columns and "timestamp" in df.columns:
        df["tick"] = range(1, len(df) + 1)
    return df


# ──────────────────────────────────────────────────────────────
# PANEL 1 — Historical Reliability
# ──────────────────────────────────────────────────────────────
def _render_reliability_panel(df: pd.DataFrame):
    _section("📊 Historical Reliability Panel")

    # Per-source stats
    pivot = df.pivot_table(index="tick", columns="source_id", values="value", aggfunc="mean")
    consensus_vals = pivot.median(axis=1)

    metrics = {}
    for src in SOURCES:
        if src not in pivot.columns:
            continue
        vals = pivot[src].dropna()
        consensus = consensus_vals.reindex(vals.index).fillna(vals.mean())
        deviation = (vals - consensus).abs()
        metrics[src] = {
            "avg_deviation": deviation.mean(),
            "variance": vals.var(),
            "stability_score": float(np.clip(1 - (vals.std() / (vals.mean() + 1e-9)), 0, 1)),
            "reliability_pct": int(100 * (df[df["source_id"] == src]["anomaly_score"].lt(0.5).mean()))
                               if "anomaly_score" in df.columns else 75,
        }

    # Metric cards
    badge_map = [
        (0.80, "🟢 Highly Trusted", "#3fb950"),
        (0.45, "🟡 Monitor",        "#d29922"),
        (0.00, "🔴 Unreliable",     "#f85149"),
    ]
    def reliability_badge(pct):
        score = pct / 100
        for thr, label, col in badge_map:
            if score >= thr:
                return label, col
        return "🔴 Unreliable", "#f85149"

    cols = st.columns(len(metrics))
    for col, (src, m) in zip(cols, metrics.items()):
        label, color = reliability_badge(m["reliability_pct"])
        latest_src = df[df["source_id"] == src]
        latest_val = latest_src["value"].iloc[-1] if not latest_src.empty else 0
        latest_score = float(latest_src["smoothed"].iloc[-1]) if "smoothed" in latest_src.columns and not latest_src.empty else 0.5
        with col:
            _card(
                f'<div style="border-top:4px solid {COLORS[src]};padding-top:10px;">'
                f'<p style="font-weight:800;color:{COLORS[src]};font-size:1.1rem;margin:0;">{src}</p>'
                f'<p style="margin:2px 0;font-size:0.75rem;color:#8b949e;">RELIABILITY</p>'
                f'<p style="font-size:2rem;font-weight:800;margin:0;">{m["reliability_pct"]}%</p>'
                f'<div style="display:flex;gap:6px;margin-top:8px;flex-wrap:wrap;">'
                f'  <span style="padding:4px 10px;border-radius:20px;border:1px solid {color};color:{color};font-size:0.75rem;font-weight:700;">{label}</span>'
                f'</div>'
                f'<hr style="border-color:rgba(255,255,255,0.07);margin:10px 0;">'
                f'<div style="font-size:0.85rem;display:grid;grid-template-columns:1fr 1fr;gap:4px;">'
                f'  <span style="color:#8b949e;">Trust Score</span><b>{latest_score*100:.1f}%</b>'
                f'  <span style="color:#8b949e;">Avg Deviation</span><b>{m["avg_deviation"]:.3f}</b>'
                f'  <span style="color:#8b949e;">Stability</span><b>{m["stability_score"]*100:.1f}%</b>'
                f'  <span style="color:#8b949e;">Latest Value</span><b>{latest_val:.3f}</b>'
                f'</div></div>',
                border_color=COLORS[src],
            )

    # Average deviation chart
    if len(pivot) > 1:
        st.markdown("<p style='color:#8b949e;margin-top:1.5rem;'>Rolling Average Deviation from Consensus</p>", unsafe_allow_html=True)
        fig = go.Figure()
        for src in SOURCES:
            if src not in pivot.columns:
                continue
            vals = pivot[src]
            consensus = consensus_vals.reindex(vals.index).fillna(vals.mean())
            deviation = (vals - consensus).abs().rolling(3, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=deviation.index, y=deviation,
                name=src, line=dict(color=COLORS[src], width=2), mode="lines"
            ))
        fig.update_layout(
            height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.1, font={"color": "white"}),
            font={"color": "white", "family": "Outfit"},
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig, key="reliability_deviation_chart", use_container_width=True, config={"displayModeBar": False})


# ──────────────────────────────────────────────────────────────
# PANEL 2 — Consensus Engine Visualization
# ──────────────────────────────────────────────────────────────
def _render_consensus_viz(df: pd.DataFrame):
    _section("🤝 Consensus Engine Visualization")
    pivot = df.pivot_table(index="tick", columns="source_id", values="value", aggfunc="mean")
    pivot["Consensus"] = pivot.median(axis=1)

    fig = go.Figure()
    for src in SOURCES:
        if src not in pivot.columns:
            continue
        fig.add_trace(go.Scatter(
            x=pivot.index, y=pivot[src], name=src,
            line=dict(color=COLORS[src], width=2), mode="lines", opacity=0.8
        ))
    fig.add_trace(go.Scatter(
        x=pivot.index, y=pivot["Consensus"], name="Consensus (Median)",
        line=dict(color="white", width=3, dash="dash"), mode="lines"
    ))
    fig.update_layout(
        height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.1, font={"color": "white"}),
        font={"color": "white", "family": "Outfit"},
        xaxis=dict(showgrid=False, title="Tick"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Value")
    )
    st.plotly_chart(fig, key="consensus_viz_chart", use_container_width=True, config={"displayModeBar": False})


# ──────────────────────────────────────────────────────────────
# PANEL 3 — Source Deviation Monitor
# ──────────────────────────────────────────────────────────────
def _render_deviation_monitor(df: pd.DataFrame):
    _section("📡 Source Deviation Monitor")

    pivot = df.pivot_table(index="tick", columns="source_id", values="value", aggfunc="mean")
    consensus = pivot.median(axis=1)
    threshold = 2 * pivot.get("Source_A", pd.Series([100])).std() if "Source_A" in pivot.columns else 10

    fig = go.Figure()
    for src in SOURCES:
        if src not in pivot.columns:
            continue
        dev = (pivot[src] - consensus).abs()
        fig.add_trace(go.Scatter(
            x=dev.index, y=dev, name=src, fill="tozeroy",
            line=dict(color=COLORS[src], width=2),
            fillcolor=COLORS[src].replace("ff", "22").replace("#", "rgba(").rstrip(")") + ",0.12)",
            mode="lines"
        ))
    fig.add_hline(y=threshold, line_color="rgba(248,81,73,0.7)", line_dash="dash",
                  annotation_text=f"Outlier Threshold ({threshold:.2f})", annotation_font_color="#f85149")
    fig.update_layout(
        height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", y=1.1, font={"color": "white"}),
        font={"color": "white", "family": "Outfit"},
        xaxis=dict(showgrid=False, title="Tick"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Deviation")
    )
    st.plotly_chart(fig, key="deviation_monitor_chart", use_container_width=True, config={"displayModeBar": False})

    # Outlier badges
    outlier_cols = st.columns(3)
    for col, src in zip(outlier_cols, SOURCES):
        if src not in pivot.columns:
            continue
        latest_dev = float((pivot[src] - consensus).abs().iloc[-1])
        is_outlier = latest_dev > threshold
        badge_color = "#f85149" if is_outlier else "#3fb950"
        badge_label = "🚨 Outlier Risk" if is_outlier else "✅ Within Range"
        with col:
            _card(
                f'<p style="color:{COLORS[src]};font-weight:800;margin:0;">{src}</p>'
                f'<p style="margin:4px 0;font-size:0.85rem;color:#8b949e;">Latest Deviation: <b style="color:white;">{latest_dev:.4f}</b></p>'
                f'<span style="color:{badge_color};font-weight:700;">{badge_label}</span>',
                border_color=badge_color,
            )


# ──────────────────────────────────────────────────────────────
# PANEL 5 — Most Reliable Source
# ──────────────────────────────────────────────────────────────
def _render_best_source(df: pd.DataFrame):
    _section("🏆 Recommended Primary Source")

    pivot  = df.pivot_table(index="tick", columns="source_id", values="value", aggfunc="mean")
    consensus = pivot.median(axis=1)
    score_pivot = df.pivot_table(index="tick", columns="source_id", values="smoothed", aggfunc="mean") \
                  if "smoothed" in df.columns else pivot * 0 + 0.5

    scores = {}
    for src in SOURCES:
        if src not in pivot.columns:
            continue
        avg_trust   = score_pivot[src].mean() if src in score_pivot.columns else 0.5
        avg_dev     = (pivot[src] - consensus).abs().mean()
        stability   = float(np.clip(1 - pivot[src].std() / (pivot[src].mean() + 1e-9), 0, 1))
        # Normalise, higher is better
        scores[src] = avg_trust * 0.5 + (1 - avg_dev / (avg_dev + 1)) * 0.3 + stability * 0.2

    winner = max(scores, key=scores.get) if scores else "Source_A"
    color  = COLORS.get(winner, "#58a6ff")

    _card(
        f'<div style="text-align:center;padding:10px 0;">'
        f'  <p style="font-size:3rem;margin:0;">🏆</p>'
        f'  <p style="font-size:2rem;font-weight:800;color:{color};margin:5px 0;">{winner}</p>'
        f'  <p style="color:#8b949e;margin:0;">Most Reliable Source this session</p>'
        f'  <div style="display:flex;justify-content:center;gap:20px;margin-top:16px;">'
        + "".join(
            f'<div style="text-align:center;">'
            f'<p style="margin:0;font-size:0.75rem;color:#8b949e;">{s}</p>'
            f'<p style="margin:0;font-weight:700;color:{COLORS[s]};">{100*v:.1f}</p>'
            f'</div>'
            for s, v in sorted(scores.items(), key=lambda x: -x[1])
        ) +
        f'  </div>'
        f'</div>',
        border_color=color,
    )


# ──────────────────────────────────────────────────────────────
# PANEL 6 — Explainable Trust Engine
# ──────────────────────────────────────────────────────────────
def _render_explanation(df: pd.DataFrame):
    _section("🔍 Explainable Trust Engine")

    pivot    = df.pivot_table(index="tick", columns="source_id", values="value", aggfunc="mean")
    consensus = pivot.median(axis=1)

    for src in SOURCES:
        if src not in pivot.columns:
            continue
        vals     = pivot[src].dropna()
        dev      = (vals - consensus.reindex(vals.index)).abs().mean()
        var      = vals.var()
        avg_dev_all = (pivot - consensus.values.reshape(-1, 1)).abs().mean().mean()

        dev_label = "Low" if dev < avg_dev_all * 0.8 else ("High" if dev > avg_dev_all * 1.5 else "Moderate")
        var_label = "Stable" if var < vals.mean() * 0.001 else ("Volatile" if var > vals.mean() * 0.02 else "Moderate")

        rel_pct = int(100 * (df[df["source_id"] == src]["anomaly_score"].lt(0.5).mean())) \
                  if "anomaly_score" in df.columns else 75
        hist_label = "Excellent" if rel_pct > 80 else ("Moderate" if rel_pct > 55 else "Poor")
        color = COLORS[src]

        with st.expander(f"{src} — Trust Explanation", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("Deviation Level",     dev_label,   f"{dev:.4f} avg")
            c2.metric("Variance Level",      var_label,   f"{var:.4f}")
            c3.metric("Historical Consistency", hist_label, f"{rel_pct}%")

            notes = {
                "Low":      "✅ Source is closely aligned with consensus — low deviation penalty.",
                "Moderate": "⚠️ Source shows occasional drift — moderate deviation penalty applied.",
                "High":     "🚨 Source frequently diverges from consensus — heavy deviation penalty."
            }
            var_notes = {
                "Stable":   "✅ Source values are stable — minimal variance penalty.",
                "Moderate": "⚠️ Source values fluctuate moderately.",
                "Volatile": "🚨 Source values are highly volatile — variance penalty applied."
            }
            hist_notes = {
                "Excellent": "✅ Consistent performance gives a historical reliability bonus.",
                "Moderate":  "⚠️ Mixed history — neutral historical score.",
                "Poor":      "🚨 Frequent anomalies — reduced historical consistency score."
            }
            st.markdown(
                f'<div style="border-left:3px solid {color};padding-left:14px;margin-top:12px;font-size:0.9rem;">'
                f'<p>• {notes[dev_label]}</p>'
                f'<p>• {var_notes[var_label]}</p>'
                f'<p>• {hist_notes[hist_label]}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────────────────────
# PANEL 8 — Formula Visualization
# ──────────────────────────────────────────────────────────────
def _render_formula_viz(df: pd.DataFrame):
    _section("⚗️ Trust Score Formula Breakdown")

    st.markdown(
        '<div style="background:rgba(22,27,34,0.7);border:1px solid rgba(88,166,255,0.3);'
        'border-radius:16px;padding:22px;font-family:JetBrains Mono,monospace;font-size:0.9rem;'
        'line-height:2;">'
        '<b>Trust Score = 100</b><br/>'
        '&nbsp;&nbsp;&nbsp;− <span style="color:#f85149;">deviation_penalty</span> &nbsp;'
        '= |val − consensus| / threshold × 30<br/>'
        '&nbsp;&nbsp;&nbsp;− <span style="color:#d29922;">variance_penalty</span> &nbsp;&nbsp;'
        '= variance / max_variance × 20<br/>'
        '&nbsp;&nbsp;&nbsp;+ <span style="color:#3fb950;">consistency_bonus</span> &nbsp;'
        '= reliability% × 10<br/>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<p style='color:#8b949e;margin-top:1.5rem;'>Live Numbers (Latest Tick)</p>", unsafe_allow_html=True)
    pivot      = df.pivot_table(index="tick", columns="source_id", values="value", aggfunc="mean")
    consensus  = float(pivot.iloc[-1].median())
    threshold  = max(2 * pivot.std().mean(), 0.01)
    max_var    = float(pivot.var().max()) or 1.0

    rows = []
    for src in SOURCES:
        if src not in pivot.columns:
            continue
        val        = float(pivot[src].iloc[-1])
        dev        = abs(val - consensus)
        var        = float(pivot[src].var())
        rel_pct    = int(100 * (df[df["source_id"] == src]["anomaly_score"].lt(0.5).mean())) \
                     if "anomaly_score" in df.columns else 75
        dev_pen    = round(dev / threshold * 30, 2)
        var_pen    = round(var / max_var * 20, 2)
        hist_bonus = round(rel_pct / 100 * 10, 2)
        total      = round(100 - dev_pen - var_pen + hist_bonus, 2)
        rows.append({
            "Source": src, "Dev Penalty": dev_pen,
            "Var Penalty": var_pen, "Hist Bonus": hist_bonus,
            "Formula Score": total
        })
    st.dataframe(pd.DataFrame(rows).set_index("Source"), use_container_width=True)


# ──────────────────────────────────────────────────────────────
# PANEL 9 — Data Export
# ──────────────────────────────────────────────────────────────
def _render_export(df: pd.DataFrame):
    _section("📤 Data Export")

    pivot     = df.pivot_table(index="tick", columns="source_id", values="value", aggfunc="mean").reset_index()
    consensus = df.groupby("tick")["value"].median().reset_index().rename(columns={"value": "consensus"})
    export_df = pivot.merge(consensus, on="tick", how="left")

    # winner per tick
    if "smoothed" in df.columns:
        score_pivot = df.pivot_table(index="tick", columns="source_id", values="smoothed", aggfunc="mean")
        export_df["winner_source"] = score_pivot.idxmax(axis=1).values

    col1, col2 = st.columns(2)
    with col1:
        csv_bytes = export_df.to_csv(index=False).encode()
        st.download_button(
            label="⬇️ Export Trust Report (CSV)",
            data=csv_bytes,
            file_name="trustlayer_report.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        json_bytes = export_df.to_json(orient="records", indent=2).encode()
        st.download_button(
            label="⬇️ Export Analysis (JSON)",
            data=json_bytes,
            file_name="trustlayer_analysis.json",
            mime="application/json",
            use_container_width=True,
        )

    st.caption(f"Export contains {len(export_df)} ticks × {len(SOURCES)} sources.")


# ──────────────────────────────────────────────────────────────
# PANEL 10 — Architecture
# ──────────────────────────────────────────────────────────────
def _render_architecture():
    _section("🏗️ System Architecture")

    layers = [
        ("📥 Data Ingestion",      "#58a6ff",
         "Accepts CSV uploads or connects to the live Bitcoin price API. "
         "Raw values are streamed into per-source ring buffers."),
        ("⚙️ Simulation Engine",   "#bc8cff",
         "Derives Source_A (clean), Source_B (Gaussian noise), and Source_C (spikes + drift) "
         "from the single input stream to simulate a real multi-source feed."),
        ("🤝 Consensus Engine",    "#f0883e",
         "Computes median-based consensus per tick. Sources are scored by deviation "
         "from the median. Outliers are flagged using z-score and rolling std."),
        ("🛡️ Trust Scoring",       "#3fb950",
         "EMA-smoothed trust state per source. Combines anomaly score, consensus score, "
         "and historical performance into a composite [0–1] trust score."),
        ("📊 Visualization Layer", "#d29922",
         "Streamlit + Plotly frontend. Live placeholders update in-place using while loops "
         "and st.empty() — no page reloads, no ML, pure statistics."),
    ]

    for icon_title, color, desc in layers:
        _card(
            f'<div style="display:flex;align-items:flex-start;gap:16px;">'
            f'  <div style="width:4px;background:{color};border-radius:2px;align-self:stretch;"></div>'
            f'  <div>'
            f'    <p style="margin:0;font-size:1rem;font-weight:800;color:{color};">{icon_title}</p>'
            f'    <p style="margin:6px 0 0;font-size:0.9rem;color:#c9d1d9;">{desc}</p>'
            f'  </div>'
            f'</div>',
            border_color=color,
        )


# ──────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────
def render_research_lab():
    st.markdown(
        '<h1 class="dashboard-header">🔬 <span class="gradient-text">Research Lab</span></h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="dashboard-tagline">Deep statistical trust analysis and explainable reliability scoring.</p>',
        unsafe_allow_html=True,
    )

    records = _get_records()
    df      = _build_df(records)

    if df is None or df.empty:
        st.info(
            "📭 **No simulation data found.**  \n"
            "Run a **Live API** or **Stress Lab** simulation first, then return here to see the analysis."
        )
        _render_architecture()
        return

    tab_labels = [
        "📊 Reliability", "🤝 Consensus", "📡 Deviation",
        "🏆 Best Source", "🔍 Explainability",
        "⚗️ Formula", "📤 Export", "🏗️ Architecture"
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        _render_reliability_panel(df)
    with tabs[1]:
        _render_consensus_viz(df)
    with tabs[2]:
        _render_deviation_monitor(df)
    with tabs[3]:
        _render_best_source(df)
    with tabs[4]:
        _render_explanation(df)
    with tabs[5]:
        _render_formula_viz(df)
    with tabs[6]:
        _render_export(df)
    with tabs[7]:
        _render_architecture()
