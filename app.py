import streamlit as st
import textwrap

def _html(s):
    import re
    return re.sub(r'^\s+', '', s, flags=re.MULTILINE)

import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, warnings
warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AccidentIQ | Severity Predictor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Load CSS ─────────────────────────────────────────────────────────────────

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─── Load Models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else None
        encoders = joblib.load("label_encoders.pkl") if os.path.exists("label_encoders.pkl") else {}
        return model, scaler, encoders, True
    except Exception as e:
        return None, None, {}, False

model, scaler, label_encoders, model_loaded = load_artifacts()

@st.cache_resource
def get_shap_explainer(_model):
    import shap
    # Using TreeExplainer
    return shap.TreeExplainer(_model)



# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(_html("""
    <div class="sidebar-logo">
        <span class="logo-icon">🚦</span>
        <span class="logo-text">AccidentIQ</span>
    </div>
    """), unsafe_allow_html=True)

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Home", "🔍  Predict Severity", "ℹ️  About"],
        label_visibility="collapsed"
    )

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown(_html("""
    <div class='sidebar-info'>
        <div class='team-label'>GROUP 3 · FR. CRIT</div>
        <div class='team-member'>Pranati Arun · 5023141</div>
        <div class='team-member'>Vaibhavi Rai · 5023143</div>
        <div class='team-member'>Ishwari Shinde · 5023155</div>
    </div>
    """), unsafe_allow_html=True)

    if not model_loaded:
        st.markdown("<div class='status-badge error'>⚠ Model not loaded</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='status-badge success'>✓ Model ready</div>", unsafe_allow_html=True)

# ─── Helper ───────────────────────────────────────────────────────────────────
SEVERITY_MAP = {0: "Slight Injury", 1: "Serious Injury", 2: "Fatal Injury"}
SEVERITY_COLOR = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444"}
SEVERITY_BG = {0: "#052e16", 1: "#1c1003", 2: "#1f0000"}
SEVERITY_ICON = {0: "🟢", 1: "🟠", 2: "🔴"}

# ─── HOME PAGE ────────────────────────────────────────────────────────────────
if "Home" in page:
    st.markdown(_html("""
    <div class="hero-section">
        <div class="hero-badge">MACHINE LEARNING PROJECT · GROUP 3</div>
        <h1 class="hero-title">Road Accident<br><span class="accent">Severity</span> Prediction</h1>
        <p class="hero-sub">A data-driven approach to classify accident severity into Slight, Serious, and Fatal categories using ensemble machine learning on real-world Ethiopian road accident data.</p>
    </div>
    """), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("12,316", "Accident Records", "🗂️"),
        ("22", "Features Used", "🔧"),
        ("5", "ML Models Compared", "🤖"),
        ("82.6%", "Best Accuracy", "🎯"),
    ]
    for col, (val, label, icon) in zip([c1, c2, c3, c4], stats):
        col.markdown(_html(f"""
        <div class="stat-card">
            <div class="stat-icon">{icon}</div>
            <div class="stat-val">{val}</div>
            <div class="stat-label">{label}</div>
        </div>
        """), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    cards = [
        ("🔍", "Predict Severity", "Enter accident conditions and get instant ML-powered severity classification with confidence scores."),
        ("📊", "Explore Dashboard", "Visual insights from EDA — class distributions, temporal patterns, and feature correlations."),
        ("📈", "Model Performance", "Compare all 5 trained models across accuracy, F1 weighted, F1 macro, and ROC-AUC metrics."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], cards):
        col.markdown(_html(f"""
        <div class="feature-card">
            <div class="fc-icon">{icon}</div>
            <div class="fc-title">{title}</div>
            <div class="fc-desc">{desc}</div>
        </div>
        """), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(_html("""
    <div class="pipeline-section">
        <div class="pipeline-title">ML Pipeline</div>
        <div class="pipeline-steps">
            <div class="step">📥<br><small>Data Collection</small></div>
            <div class="arrow">→</div>
            <div class="step">🧹<br><small>Preprocessing</small></div>
            <div class="arrow">→</div>
            <div class="step">🔬<br><small>EDA</small></div>
            <div class="arrow">→</div>
            <div class="step">⚖️<br><small>SMOTE Balancing</small></div>
            <div class="arrow">→</div>
            <div class="step">🌲<br><small>Random Forest</small></div>
            <div class="arrow">→</div>
            <div class="step">🎯<br><small>Prediction</small></div>
        </div>
    </div>
    """), unsafe_allow_html=True)

# ─── PREDICT PAGE ─────────────────────────────────────────────────────────────
elif "Predict" in page:
    st.markdown("<h2 class='page-title'>🔍 Predict Accident Severity</h2>", unsafe_allow_html=True)
    st.markdown("<p class='page-sub'>Fill in the accident conditions below to get an ML-powered severity prediction.</p>", unsafe_allow_html=True)

    from data_config import FEATURE_OPTIONS, FEATURE_LABELS, DEFAULT_FEATURE_ORDER

    def clean_dropdown_text(text):
        if not isinstance(text, str):
            return str(text)
        if text == "Not a Pedestrian":
            return text
        text = text.replace("statioNot a Pedestrianry", "stationary")
        text = text.replace("?", "-")
        if text.strip() == "unknown":
            return "Unknown"
        if text.strip() == "other":
            return "Other"
        return text.strip().capitalize() if text.islower() else text.strip()

    with st.form("prediction_form"):
        st.markdown("<div class='form-section-title'>🚗 Driver & Vehicle Details</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        inputs = {}

        field_cols = [c1, c2, c3, c1, c2, c3, c1, c2, c3, c1, c2, c3]
        for i, feat in enumerate(DEFAULT_FEATURE_ORDER):
            col = field_cols[i % 3]
            if i % 3 == 0 and i > 0 and i < 8:
                col.markdown("<div style='height:1px'></div>", unsafe_allow_html=True)
            if i == 8:
                st.markdown("<div class='form-section-title'>🛣️ Road & Environment</div>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                field_cols = [c1, c2, c3] * 10
            with field_cols[i % 3]:
                label = FEATURE_LABELS.get(feat, feat.replace("_", " ").title())
                options = FEATURE_OPTIONS.get(feat, ["Unknown"])
                if feat == "Number_of_vehicles_involved":
                    inputs[feat] = st.number_input(label, min_value=1, max_value=7, value=2)
                else:
                    inputs[feat] = st.selectbox(label, options, format_func=clean_dropdown_text)

        submitted = st.form_submit_button("🔍 Predict Severity", use_container_width=True)

    if submitted:
        if not model_loaded:
            st.error("Model not loaded. Please ensure best_model.pkl exists.")
        else:
            try:
                from predictor import run_prediction
                pred_class, proba = run_prediction(inputs, model, scaler, label_encoders, DEFAULT_FEATURE_ORDER)

                sev = SEVERITY_MAP[pred_class]
                color = SEVERITY_COLOR[pred_class]
                bg = SEVERITY_BG[pred_class]
                icon = SEVERITY_ICON[pred_class]

                st.markdown(_html(f"""
                <div class="result-card" style="border-color:{color}; background:{bg};">
                    <div class="result-icon">{icon}</div>
                    <div class="result-label">Predicted Severity</div>
                    <div class="result-severity" style="color:{color}">{sev}</div>
                    <div class="result-conf">Confidence: {proba[pred_class]*100:.1f}%</div>
                </div>
                """), unsafe_allow_html=True)

                if pred_class == 2:
                    st.markdown(_html("""
                    <div class="alert-fatal">
                        ⚠️ FATAL RISK DETECTED — Immediate intervention recommended.
                        Conditions indicate extremely high accident severity.
                    </div>
                    """), unsafe_allow_html=True)
                elif pred_class == 1:
                    st.markdown(_html("""
                    <div class="alert-serious">
                        ⚠️ SERIOUS INJURY RISK — Emergency response may be required.
                    </div>
                    """), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("<div class='chart-title'>Class Probability</div>", unsafe_allow_html=True)
                    classes = ["Slight Injury", "Serious Injury", "Fatal Injury"]
                    colors = ["#22c55e", "#f59e0b", "#ef4444"]
                    fig = go.Figure(go.Bar(
                        x=[f"{p*100:.1f}%" for p in proba],
                        y=classes,
                        orientation='h',
                        marker_color=colors,
                        text=[f"{p*100:.1f}%" for p in proba],
                        textposition='outside',
                    ))
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#cbd5e1', height=220,
                        margin=dict(l=10, r=40, t=10, b=10),
                        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 110]),
                        yaxis=dict(showgrid=False),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("<div class='chart-title'>Severity Gauge</div>", unsafe_allow_html=True)
                    gauge_val = proba[0] * 33 + proba[1] * 66 + proba[2] * 100
                    fig2 = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=round(gauge_val, 1),
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100], 'tickcolor': '#64748b'},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 33], 'color': '#052e16'},
                                {'range': [33, 66], 'color': '#1c1003'},
                                {'range': [66, 100], 'color': '#1f0000'}
                            ],
                            'threshold': {'line': {'color': color, 'width': 3}, 'value': gauge_val}
                        },
                        number={'font': {'color': color, 'size': 32}}
                    ))
                    fig2.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#cbd5e1', height=220,
                        margin=dict(l=10, r=10, t=10, b=10)
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                # ── SHAP Explanation (inline) ──────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<div class='chart-title'>🧠 Why This Prediction?</div>", unsafe_allow_html=True)

                try:
                    from predictor import get_preprocessed_df
                    import shap, matplotlib, matplotlib.pyplot as plt
                    matplotlib.use('Agg')  # non-interactive backend, thread-safe

                    df_shap = get_preprocessed_df(inputs, scaler, label_encoders, DEFAULT_FEATURE_ORDER)
                    explainer = get_shap_explainer(model)
                    shap_values = explainer.shap_values(df_shap)

                    # Handle list (per-class) vs 3D array output
                    if isinstance(shap_values, list):
                        sv = shap_values[pred_class][0]
                        base_val = explainer.expected_value[pred_class]
                    elif len(shap_values.shape) == 3:
                        sv = shap_values[0, :, pred_class]
                        base_val = explainer.expected_value[pred_class]
                    else:
                        sv = shap_values[0]
                        base_val = explainer.expected_value

                    # Build readable feature name dict
                    feat_labels = [f.replace('_', ' ').title() for f in DEFAULT_FEATURE_ORDER]
                    shap_pairs = list(zip(feat_labels, sv))
                    shap_sorted = sorted(shap_pairs, key=lambda x: x[1])

                    top_risk = [(k, v) for k, v in reversed(shap_sorted) if v > 0][:4]
                    top_safe = [(k, v) for k, v in shap_sorted if v < 0][:4]

                    # Human-readable summary
                    summary_parts = []
                    if top_risk:
                        summary_parts.append(f"**{top_risk[0][0]}** was the strongest factor pushing toward {sev}.")
                    if top_safe:
                        summary_parts.append(f"**{top_safe[0][0]}** helped reduce the severity risk.")
                    summary_text = ' '.join(summary_parts) if summary_parts else "All features contributed roughly equally to this prediction."

                    st.markdown(f"<p style='color:#94a3b8; font-size:0.95rem; line-height:1.7; padding:1rem; background:#0a0f1e; border-radius:10px; border:1px solid #1e293b'>{summary_text}</p>", unsafe_allow_html=True)

                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.markdown("<div class='chart-title'>🔴 Top Risk Factors</div>", unsafe_allow_html=True)
                        if top_risk:
                            for k, v in top_risk:
                                bar_pct = min(int(abs(v) * 400), 100)
                                st.markdown(f"<div style='background:#0a0f1e; border:1px solid #1e293b; border-left:4px solid #ef4444; padding:0.7rem 1rem; margin-bottom:0.5rem; border-radius:8px;'><span style='color:#f1f5f9; font-weight:600'>{k}</span><br><div style='background:#1f0000; border-radius:4px; height:6px; margin-top:6px;'><div style='background:#ef4444; width:{bar_pct}%; height:6px; border-radius:4px;'></div></div><span style='color:#ef4444; font-size:0.8rem; font-family:monospace'>+{v:.4f}</span></div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<p style='color:#64748b'>No significant risk factors detected.</p>", unsafe_allow_html=True)
                    with rc2:
                        st.markdown("<div class='chart-title'>🔵 Mitigating Factors</div>", unsafe_allow_html=True)
                        if top_safe:
                            for k, v in top_safe:
                                bar_pct = min(int(abs(v) * 400), 100)
                                st.markdown(f"<div style='background:#0a0f1e; border:1px solid #1e293b; border-left:4px solid #3b82f6; padding:0.7rem 1rem; margin-bottom:0.5rem; border-radius:8px;'><span style='color:#f1f5f9; font-weight:600'>{k}</span><br><div style='background:#0c1a3a; border-radius:4px; height:6px; margin-top:6px;'><div style='background:#3b82f6; width:{bar_pct}%; height:6px; border-radius:4px;'></div></div><span style='color:#3b82f6; font-size:0.8rem; font-family:monospace'>{v:.4f}</span></div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<p style='color:#64748b'>No strong mitigating factors found.</p>", unsafe_allow_html=True)

                    # SHAP Bar Chart via Plotly (no matplotlib threading issues)
                    all_feats = [k for k, v in shap_sorted]
                    all_vals  = [v for k, v in shap_sorted]
                    bar_colors = ["#ef4444" if v > 0 else "#3b82f6" for v in all_vals]

                    shap_fig = go.Figure(go.Bar(
                        x=all_vals,
                        y=all_feats,
                        orientation='h',
                        marker_color=bar_colors,
                        text=[f"{v:+.4f}" for v in all_vals],
                        textposition='outside',
                    ))
                    shap_fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#cbd5e1', height=500,
                        title=dict(text=f"SHAP Values — {sev} Class", font=dict(color='#f1f5f9', size=13)),
                        margin=dict(l=10, r=60, t=40, b=10),
                        xaxis=dict(showgrid=True, gridcolor='#1e293b', zeroline=True, zerolinecolor='#334155'),
                        yaxis=dict(showgrid=False),
                    )
                    st.plotly_chart(shap_fig, use_container_width=True)

                except Exception as shap_err:
                    st.warning(f"⚠️ Could not generate SHAP explanation: {shap_err}")

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("Ensure label_encoders.pkl contains encoders matching your training features.")


# ─── ABOUT PAGE ───────────────────────────────────────────────────────────────
elif "About" in page:
    st.markdown("<h2 class='page-title'>ℹ️ About This Project</h2>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        st.markdown(_html("""
        <div class="about-card">
            <div class="about-section-title">📌 Problem Statement</div>
            <p class="about-text">Road accidents are a major cause of injuries and fatalities, especially in developing countries. Traditional accident analysis is reactive, manual, and unable to scale. This project applies machine learning to predict accident severity from real-world data, enabling proactive road safety planning.</p>

            <div class="about-section-title">📦 Dataset</div>
            <ul class="about-list">
                <li><b>Source:</b> Kaggle — Ethiopian Road Accident Data (Addis Ababa, 2017–2020)</li>
                <li><b>Records:</b> 12,316 | <b>Features:</b> 32 (22 after preprocessing)</li>
                <li><b>Target:</b> Accident_severity (Slight / Serious / Fatal)</li>
                <li><b>Severe class imbalance:</b> 84.6% / 14.2% / 1.3%</li>
            </ul>

            <div class="about-section-title">⚙️ Tech Stack</div>
            <div class="tech-pills">
                <span class="tech-pill">Python 3.10</span>
                <span class="tech-pill">Scikit-learn</span>
                <span class="tech-pill">Random Forest</span>
                <span class="tech-pill">Imbalanced-learn</span>
                <span class="tech-pill">Pandas</span>
                <span class="tech-pill">Streamlit</span>
                <span class="tech-pill">Plotly</span>
                <span class="tech-pill">Joblib</span>
            </div>
        </div>
        """), unsafe_allow_html=True)

    with c2:
        st.markdown(_html("""
        <div class="team-card">
            <div class="team-title">👥 Team</div>
            <div class="team-dept">Department of Information Technology<br>Fr. C. Rodrigues Institute of Technology</div>

            <div class="member-block">
                <div class="member-name">Pranati Arun</div>
                <div class="member-id">Roll No: 5023141</div>
            </div>
            <div class="member-block">
                <div class="member-name">Vaibhavi Rai</div>
                <div class="member-id">Roll No: 5023143</div>
            </div>
            <div class="member-block">
                <div class="member-name">Ishwari Shinde</div>
                <div class="member-id">Roll No: 5023155</div>
            </div>

            <div class="team-group">Group 3 · Academic Year 2025–26</div>
        </div>
        """), unsafe_allow_html=True)
