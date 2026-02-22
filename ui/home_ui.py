import streamlit as st

def render_home_ui():
    """Renders the premium aesthetic Home Page."""
    
    # ── Hero Section ─────────────────────────────────────────────
    st.markdown("""
        <div style="text-align:center; padding: 3rem 0 4rem 0;">
            <h1 style="font-size:4.5rem; font-weight:800; line-height:1.1; margin-bottom:1.5rem;">
                <span class="gradient-text">TrustLayer</span><br/>
                <span style="font-size:2.5rem; font-weight:600; color:var(--text-main);">Data Trust & Verification Engine</span>
            </h1>
            <p style="font-size:1.2rem; color:var(--text-dim); max-width:700px; margin:0 auto; line-height:1.6;">
                A high-fidelity statistical engine designed to detect anomalies, mitigate collusion, and establish ground truth across divergent data streams in real-time.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ── Core Pillars (Glass Cards) ───────────────────────────────
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
            <div class="glass-card" style="height:280px; display:flex; flex-direction:column; justify-content:center; text-align:center;">
                <div style="font-size:3rem; margin-bottom:1rem;">🛡️</div>
                <h3 style="font-weight:700; margin-bottom:0.5rem;">Resilient Attribution</h3>
                <p style="font-size:0.9rem; color:var(--text-dim);">Multi-source cross-verification logic that isolates noise and malicious spikes before they pollute your data.</p>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
            <div class="glass-card" style="height:280px; display:flex; flex-direction:column; justify-content:center; text-align:center;">
                <div style="font-size:3rem; margin-bottom:1rem;">🧠</div>
                <h3 style="font-weight:700; margin-bottom:0.5rem;">Consensus Intelligence</h3>
                <p style="font-size:0.9rem; color:var(--text-dim);">Dynamic weighted averaging that adjusts in real-time based on historical stability and current behavior.</p>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
            <div class="glass-card" style="height:280px; display:flex; flex-direction:column; justify-content:center; text-align:center;">
                <div style="font-size:3rem; margin-bottom:1rem;">📡</div>
                <h3 style="font-weight:700; margin-bottom:0.5rem;">Live Verification</h3>
                <p style="font-size:0.9rem; color:var(--text-dim);">Seamlessly stream from APIs or replay historical datasets with millisecond-level trust assessments.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:4rem;'></div>", unsafe_allow_html=True)

    # ── Feature Breakdown ────────────────────────────────────────
    st.markdown("<h2 style='font-weight:800; text-align:center; margin-bottom:2.5rem;'>Engine Capabilities</h2>", unsafe_allow_html=True)
    
    f1, f2 = st.columns(2)
    
    with f1:
        st.markdown("""
            <div class="glass-card" style="margin-bottom:1.5rem; display:flex; align-items:center; gap:20px;">
                <div style="font-size:2rem; background:rgba(88,166,255,0.1); padding:15px; border-radius:12px; border:1px solid rgba(88,166,255,0.2);">📂</div>
                <div>
                    <h4 style="margin:0; font-weight:700;">CSV Mode Analysis</h4>
                    <p style="margin:5px 0 0 0; font-size:0.85rem; color:var(--text-dim);">Upload historical data to simulate source divergence and evaluate engine performance.</p>
                </div>
            </div>
            <div class="glass-card" style="display:flex; align-items:center; gap:20px;">
                <div style="font-size:2rem; background:rgba(188,140,255,0.1); padding:15px; border-radius:12px; border:1px solid rgba(188,140,255,0.2);">📡</div>
                <div>
                    <h4 style="margin:0; font-weight:700;">Live API Mode</h4>
                    <p style="margin:5px 0 0 0; font-size:0.85rem; color:var(--text-dim);">Connect to real-time streams and observe consensus logic under active network conditions.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown("""
            <div class="glass-card" style="margin-bottom:1.5rem; display:flex; align-items:center; gap:20px;">
                <div style="font-size:2rem; background:rgba(57,211,83,0.1); padding:15px; border-radius:12px; border:1px solid rgba(57,211,83,0.2);">🧪</div>
                <div>
                    <h4 style="margin:0; font-weight:700;">Stress Lab (Test Scenarios)</h4>
                    <p style="margin:5px 0 0 0; font-size:0.85rem; color:var(--text-dim);">Manually inject spikes, drift, and collusion to verify system resilience.</p>
                </div>
            </div>
            <div class="glass-card" style="display:flex; align-items:center; gap:20px;">
                <div style="font-size:2rem; background:rgba(231,76,60,0.1); padding:15px; border-radius:12px; border:1px solid rgba(231,76,60,0.2);">📉</div>
                <div>
                    <h4 style="margin:0; font-weight:700;">Historical Belief Charts</h4>
                    <p style="margin:5px 0 0 0; font-size:0.85rem; color:var(--text-dim);">Longitudinal tracking of confidence updates and isolation triggers.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:5rem;'></div>", unsafe_allow_html=True)

    # ── Call to Action ───────────────────────────────────────────
    st.markdown("""
        <div style="background:linear-gradient(135deg, rgba(88,166,255,0.05), rgba(188,140,255,0.05)); border:1px solid var(--border-color); border-radius:16px; padding:3rem; text-align:center;">
            <h2 style="font-weight:800; margin-bottom:1rem;">Ready to verify your data?</h2>
            <p style="color:var(--text-dim); margin-bottom:2rem;">Select a navigation mode from the sidebar to begin your analysis.</p>
            <div style="display:flex; justify-content:center; gap:15px;">
                <span style="background:var(--accent-blue); color:white; padding:10px 25px; border-radius:8px; font-weight:700; font-size:0.9rem;">Deployment Stable</span>
                <span style="background:rgba(255,255,255,0.05); color:var(--text-main); padding:10px 25px; border-radius:8px; font-weight:700; font-size:0.9rem; border:1px solid var(--border-color);">v1.4.2 Modular</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
