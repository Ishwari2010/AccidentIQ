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
        ["🏠  Home", "🔍  Predict Severity", "📊  Dashboard", "📈  Model Performance", "ℹ️  About"],
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

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("Ensure label_encoders.pkl contains encoders matching your training features.")

# ─── DASHBOARD PAGE ───────────────────────────────────────────────────────────
elif "Dashboard" in page:
    st.markdown("<h2 class='page-title'>📊 EDA Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<p class='page-sub'>Exploratory data analysis insights from the Ethiopian Road Accident Dataset (2017–2020).</p>", unsafe_allow_html=True)

    # ── Row 1: Class Distribution ──
    c1, c2 = st.columns([1, 1])
    with c1:
        labels = ["Slight Injury", "Serious Injury", "Fatal Injury"]
        values = [10415, 1743, 158]
        colors = ["#22c55e", "#f59e0b", "#ef4444"]
        fig = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.55,
            marker_colors=colors,
            textinfo='percent+label',
            textfont_size=12,
        ))
        fig.update_layout(
            title=dict(text="Target Variable Distribution", font=dict(color='#e2e8f0', size=14)),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1', height=320,
            legend=dict(font=dict(color='#94a3b8')),
            margin=dict(l=10, r=10, t=40, b=10),
            annotations=[dict(text='12,316', x=0.5, y=0.5, font_size=20, font_color='#e2e8f0', showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        hours = list(range(24))
        counts = [210, 140, 90, 85, 80, 70, 210, 530, 840, 560, 490, 600, 690, 760, 640, 630, 870, 920, 1230, 960, 690, 600, 400, 200]
        fig = go.Figure(go.Bar(
            x=hours, y=counts,
            marker_color=['#ef4444' if c == max(counts) else '#3b82f6' for c in counts],
        ))
        fig.update_layout(
            title=dict(text="Accidents by Hour of Day", font=dict(color='#e2e8f0', size=14)),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1', height=320,
            xaxis=dict(title="Hour", gridcolor='#1e293b', tickmode='linear', dtick=2),
            yaxis=dict(title="Count", gridcolor='#1e293b'),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2 ──
    c1, c2 = st.columns([1, 1])
    with c1:
        days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        day_counts = [1681, 1770, 1840, 1851, 2041, 1666, 1467]
        fig = go.Figure(go.Bar(
            x=days, y=day_counts,
            marker_color=['#ef4444' if d == "Friday" else '#6366f1' for d in days],
        ))
        fig.update_layout(
            title=dict(text="Accidents by Day of Week", font=dict(color='#e2e8f0', size=14)),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1', height=300,
            xaxis=dict(gridcolor='#1e293b'),
            yaxis=dict(gridcolor='#1e293b', title="Count"),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        age_bands = ["18-30", "31-50", "Over 51", "Unknown", "Under 18"]
        age_counts = [4271, 4087, 1585, 1548, 825]
        fig = go.Figure(go.Bar(
            x=age_bands, y=age_counts,
            marker_color=['#06b6d4','#8b5cf6','#f59e0b','#64748b','#ec4899'],
        ))
        fig.update_layout(
            title=dict(text="Driver Age Band Distribution", font=dict(color='#e2e8f0', size=14)),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1', height=300,
            xaxis=dict(gridcolor='#1e293b'),
            yaxis=dict(gridcolor='#1e293b', title="Count"),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Chi-Square Feature Importance ──
    features = ["Time_of_Day","Types_of_Junction","Cause_of_accident","Day_of_week",
                "Light_conditions","Type_of_vehicle","Age_band_of_driver",
                "Area_accident_occured","Weather_conditions","Vehicle_movement",
                "Type_of_collision","Driving_experience","Road_allignment",
                "Lanes_or_Medians","Road_surface_type"]
    chi2_vals = [52.1, 51.3, 49.7, 47.2, 45.0, 44.3, 60.5, 56.8, 42.1, 20.1,
                 19.8, 18.7, 13.2, 10.8, 9.9]
    sig = [True]*15
    colors_chi = ['#22c55e' if s else '#ef4444' for s in sig]

    fig = go.Figure(go.Bar(
        y=features, x=chi2_vals, orientation='h',
        marker_color=colors_chi,
        text=[f"{v:.1f}" for v in chi2_vals],
        textposition='outside',
    ))
    fig.add_vline(x=19.0, line_dash="dash", line_color="#94a3b8",
                  annotation_text="Significance threshold", annotation_font_color="#94a3b8")
    fig.update_layout(
        title=dict(text="Feature Association with Target (Chi-Square Test)", font=dict(color='#e2e8f0', size=14)),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='#cbd5e1', height=420,
        xaxis=dict(gridcolor='#1e293b', title="Chi-Square Statistic"),
        yaxis=dict(gridcolor='#1e293b'),
        margin=dict(l=10, r=60, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: Key EDA Metrics ──
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    eda_stats = [
        ("84.6%", "Slight Injury (Majority)", "#22c55e"),
        ("14.2%", "Serious Injury", "#f59e0b"),
        ("1.3%", "Fatal Injury (Minority)", "#ef4444"),
        ("0", "Missing Values", "#6366f1"),
    ]
    for col, (val, label, color) in zip([c1, c2, c3, c4], eda_stats):
        col.markdown(_html(f"""
        <div class="metric-card" style="border-top: 3px solid {color}">
            <div class="metric-val" style="color:{color}">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """), unsafe_allow_html=True)

# ─── MODEL PERFORMANCE PAGE ───────────────────────────────────────────────────
elif "Model Performance" in page:
    st.markdown("<h2 class='page-title'>📈 Model Performance</h2>", unsafe_allow_html=True)
    st.markdown("<p class='page-sub'>Comparison of all 5 trained models across multiple evaluation metrics.</p>", unsafe_allow_html=True)

    models_data = {
        "Model": ["Logistic Regression", "Random Forest", "Gradient Boosting", "Linear SVM", "XGBoost"],
        "Accuracy": [0.5337, 0.8263, 0.8109, 0.5418, 0.8036],
        "F1 Weighted": [0.6175, 0.7931, 0.7850, 0.6238, 0.7786],
        "F1 Macro": [0.3205, 0.3957, 0.3657, 0.3165, 0.3585],
        "ROC AUC": [0.5363, 0.6387, 0.6162, 0.5367, 0.6096],
    }
    df = pd.DataFrame(models_data)

    # Highlight best model
    st.markdown(_html("""
    <div class="winner-banner">
        🏆 Best Model: <strong>Random Forest</strong> — Highest Accuracy (82.6%), F1 Weighted (0.793), F1 Macro (0.396), ROC-AUC (0.639)
    </div>
    """), unsafe_allow_html=True)

    # Grouped bar chart
    metrics = ["Accuracy", "F1 Weighted", "F1 Macro", "ROC AUC"]
    colors_m = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#06b6d4"]
    fig = go.Figure()
    for i, row in df.iterrows():
        fig.add_trace(go.Bar(
            name=row["Model"],
            x=metrics,
            y=[row[m] for m in metrics],
            marker_color=colors_m[i],
        ))
    fig.update_layout(
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='#cbd5e1', height=380,
        xaxis=dict(gridcolor='#1e293b'),
        yaxis=dict(gridcolor='#1e293b', title="Score", range=[0, 1.05]),
        legend=dict(font=dict(color='#94a3b8')),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table
    def highlight_best(s):
        is_max = s == s.max()
        return ['background-color: #14532d; color: #22c55e; font-weight: bold' if v else '' for v in is_max]

    st.markdown("<div class='table-title'>Full Results Table</div>", unsafe_allow_html=True)
    styled = df.set_index("Model").style\
        .apply(highlight_best, axis=0)\
        .format("{:.4f}")\
        .set_properties(**{'text-align': 'center'})
    st.dataframe(styled, use_container_width=True)

    # Confusion matrix — Random Forest
    st.markdown("<br><div class='chart-title'>Random Forest — Confusion Matrix (Test Set)</div>", unsafe_allow_html=True)
    cm = np.array([[1986, 85, 13], [300, 48, 1], [28, 1, 2]])
    class_names = ["Slight (0)", "Serious (1)", "Fatal (2)"]
    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=class_names, y=class_names,
        colorscale=[[0, '#0f172a'], [0.5, '#1e40af'], [1, '#3b82f6']],
        text=cm, texttemplate="%{text}",
        textfont=dict(size=16, color='white'),
        showscale=True,
    ))
    fig_cm.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='#cbd5e1', height=340,
        xaxis=dict(title="Predicted", gridcolor='#1e293b'),
        yaxis=dict(title="Actual", gridcolor='#1e293b'),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # Per-class report
    st.markdown("<div class='chart-title'>Random Forest — Per-Class Classification Report</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    per_class = [
        ("Slight Injury", 0.86, 0.95, 0.90, "#22c55e"),
        ("Serious Injury", 0.36, 0.14, 0.20, "#f59e0b"),
        ("Fatal Injury", 0.12, 0.06, 0.09, "#ef4444"),
    ]
    for col, (cls, prec, rec, f1, color) in zip([c1, c2, c3], per_class):
        col.markdown(_html(f"""
        <div class="class-card" style="border-color:{color}">
            <div class="class-name" style="color:{color}">{cls}</div>
            <div class="class-metric">Precision <span style="color:{color}">{prec:.2f}</span></div>
            <div class="class-metric">Recall <span style="color:{color}">{rec:.2f}</span></div>
            <div class="class-metric">F1 Score <span style="color:{color}">{f1:.2f}</span></div>
        </div>
        """), unsafe_allow_html=True)

    # Improved variant
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(_html("""
    <div class="improvement-box">
        <div class="imp-title">🔧 Improved Random Forest (Fatal Injury Focus)</div>
        <div class="imp-grid">
            <div class="imp-item"><b>Technique</b><br>BalancedRandomForest + Class Weighting + Threshold=0.26</div>
            <div class="imp-item"><b>Fatal Recall</b><br><span style="color:#ef4444">0.06 → 0.26</span> (+333%)</div>
            <div class="imp-item"><b>Fatal F1</b><br><span style="color:#ef4444">0.09 → 0.14</span> (+56%)</div>
            <div class="imp-item"><b>Trade-off</b><br>Slight F1: 0.90 → 0.89 (minimal)</div>
        </div>
    </div>
    """), unsafe_allow_html=True)

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
