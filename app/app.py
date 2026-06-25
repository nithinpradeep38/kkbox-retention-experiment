"""
KKBox Subscription Retention Experimentation Platform
Streamlit dashboard — reads pre-computed CSVs from app/data/
No Spark or Databricks connection required.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KKBox Retention Experiment",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ───────────────────────────────────────────────────────────────
BLUE    = "#2563EB"
TEAL    = "#0D9488"
AMBER   = "#D97706"
GREEN   = "#16A34A"
RED     = "#DC2626"
GRAY    = "#6B7280"
BG_CARD = "#F8FAFC"

st.markdown("""
<style>
/* Global */
[data-testid="stAppViewContainer"] { background: #F1F5F9; }
[data-testid="stSidebar"] { background: #0F172A; }
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] .stRadio label { color: #CBD5E1 !important; }
[data-testid="stSidebar"] hr { border-color: #1E293B; }

/* Metric cards */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    border: 1px solid #E2E8F0;
    text-align: center;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1.1;
    margin: 4px 0;
}
.metric-label {
    font-size: 0.78rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
}
.metric-sub {
    font-size: 0.82rem;
    color: #94A3B8;
    margin-top: 4px;
}

/* Section headers */
.section-header {
    font-size: 1.15rem;
    font-weight: 600;
    color: #0F172A;
    padding: 8px 0 4px;
    border-bottom: 2px solid #E2E8F0;
    margin-bottom: 16px;
}

/* Go/No-Go badge */
.go-badge {
    display: inline-block;
    background: #DCFCE7;
    color: #15803D;
    border: 1.5px solid #86EFAC;
    border-radius: 999px;
    padding: 6px 20px;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.04em;
}

/* Insight box */
.insight-box {
    background: #EFF6FF;
    border-left: 4px solid #2563EB;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 0.9rem;
    color: #1E3A8A;
}

/* Warning box */
.warning-box {
    background: #FFFBEB;
    border-left: 4px solid #D97706;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 0.9rem;
    color: #92400E;
}

/* Signal row */
.signal-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid #F1F5F9;
}

/* Step pill */
.step-pill {
    display: inline-block;
    background: #0F172A;
    color: #F8FAFC;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-right: 6px;
}

/* Guardrail item */
.guardrail-item {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.guardrail-title {
    font-weight: 600;
    color: #0F172A;
    font-size: 0.9rem;
}
.guardrail-detail {
    color: #64748B;
    font-size: 0.82rem;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Data loading ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"

@st.cache_data
def load_experiment_results():
    return pd.read_csv(DATA_DIR / "experiment_results.csv")

@st.cache_data
def load_segment_readout():
    return pd.read_csv(DATA_DIR / "segment_readout.csv")

@st.cache_data
def load_churn_signals():
    return pd.read_csv(DATA_DIR / "churn_signals.csv")

@st.cache_data
def load_feature_importance():
    return pd.read_csv(DATA_DIR / "feature_importance.csv")

df         = load_experiment_results()
seg_df     = load_segment_readout()
signals_df = load_churn_signals()
feat_df    = load_feature_importance()

# Pre-compute experiment stats
ctrl       = df[df["group"] == "control"]
treat      = df[df["group"] == "treatment"]
ctrl_n     = len(ctrl)
treat_n    = len(treat)
ctrl_ren   = ctrl["renewed"].sum()
treat_ren  = treat["renewed"].sum()
ctrl_rate  = ctrl_ren / ctrl_n
treat_rate = treat_ren / treat_n
abs_lift   = treat_rate - ctrl_rate
rel_lift   = abs_lift / ctrl_rate
ctrl_rpu   = ctrl["revenue"].mean()
treat_rpu  = treat["revenue"].mean()
DISCOUNT   = 0.10
avg_price  = ctrl["amount_paid_at_checkpoint"].mean()
expected   = treat_n * ctrl_rate
incr_ren   = treat_ren - expected
cannib     = expected * avg_price * DISCOUNT
incr_rev   = incr_ren * avg_price * (1 - DISCOUNT)
net_impact = incr_rev - cannib
CI_LOW     = 0.1156
CI_HIGH    = 0.1242

# ── Sidebar navigation ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### KKBox Retention Experiment")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "Overview",
            "EDA & Churn Signals",
            "Churn Model",
            "Experiment Design",
            "Results",
            "Business Recommendation",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#475569; line-height:1.6'>
    <b style='color:#94A3B8'>Stack</b><br>
    Databricks · PySpark · Delta Lake<br>
    scikit-learn · SciPy · statsmodels<br><br>
    <b style='color:#94A3B8'>Data</b><br>
    KKBox WSDM 2018 · Kaggle<br>
    970K labeled users · 1.2M scored<br><br>
    <b style='color:#94A3B8'>Author</b><br>
    Nithin Pradeep
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("## KKBox Subscription Retention Experimentation Platform")
    st.markdown(
        "An end-to-end product data science project — churn modeling, "
        "A/B test design with economic justification, statistical analysis, "
        "and a segment-targeted business recommendation."
    )

    # Headline metrics
    cols = st.columns(5)
    metrics = [
        (f"+{abs_lift*100:.2f}pp", "Renewal lift", f"95% CI [{CI_LOW*100:.1f}, {CI_HIGH*100:.1f}]pp", BLUE),
        (f"+{net_impact/1e6:.2f}M NTD", "Net revenue impact", "After cannibalization", GREEN),
        ("p < 0.0001", "Significance", "Two-proportion z-test", TEAL),
        ("205,883", "Eligible users", "High-risk · April expiry", AMBER),
        ("ROC-AUC 0.87", "Churn model", "Logistic regression", GRAY),
    ]
    for col, (val, label, sub, color) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Decision banner
    st.markdown(f"""
    <div style="background:white; border:1px solid #E2E8F0; border-radius:12px;
                padding:20px 28px; display:flex; align-items:center; gap:20px;">
        <span class="go-badge">✓ GO</span>
        <div>
            <div style="font-weight:600; color:#0F172A; font-size:1rem;">
                Offer a 10% renewal discount to High-risk subscribers expiring within 30 days
            </div>
            <div style="color:#64748B; font-size:0.85rem; margin-top:4px;">
                Lift of +{abs_lift*100:.2f}pp exceeds the 10pp MDE · net revenue positive ·
                all 14 segment tests significant and net positive
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline steps
    st.markdown('<div class="section-header">Project pipeline</div>', unsafe_allow_html=True)
    steps = [
        ("00", "Local preprocessing", "DuckDB trims 28 GB user_logs.csv to 1.8 GB — 60-day lookback, train_v2 users only"),
        ("01", "Data engineering", "PySpark ingests 6 files → 6 Bronze Delta tables with explicit schemas, date parsing, DQ report"),
        ("02", "EDA", "Derives 91.01% global renewal rate, identifies 3 strong churn signals, confirms engagement is flat"),
        ("03", "Feature engineering", "22 features per user — checkpoint-anchored reference logic prevents label leakage"),
        ("04", "Churn model", "Logistic regression (ROC-AUC 0.87) scores 1.2M March-active users → risk segments"),
        ("05", "Experiment design", "Lift × discount sensitivity matrix rejects 30% brief, locks 10% discount, power analysis"),
        ("06", "Experiment analysis", "SRM, z-test, CI, revenue breakdown, 14 segment tests — all net positive"),
        ("08", "Business recommendation", "Go/no-go, targeting criteria, launch guardrails"),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div style="display:flex; gap:14px; padding:10px 0; border-bottom:1px solid #F1F5F9; align-items:flex-start;">
            <span class="step-pill">{num}</span>
            <div>
                <span style="font-weight:600; color:#0F172A; font-size:0.9rem;">{title}</span>
                <span style="color:#64748B; font-size:0.85rem;"> — {desc}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA & CHURN SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "EDA & Churn Signals":
    st.markdown("## EDA & Churn Signals")
    st.markdown(
        "Exploratory analysis on 970,960 labeled users (Feb 2017 expiry cohort). "
        "The key finding: churn is driven by subscription mechanics, not engagement."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-header">Baseline renewal rate</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Global baseline (all labeled users)</div>
            <div class="metric-value" style="color:{GREEN}">91.01%</div>
            <div class="metric-sub">8.99% churn · n = 970,960</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">High-risk segment baseline</div>
            <div class="metric-value" style="color:{AMBER}">59.14%</div>
            <div class="metric-sub">40.86% churn · experiment population</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
        The 91% global rate reflects heavy auto-renew usage (90.4% of users).
        The High-risk segment is a fundamentally different population — 41% genuine
        churn risk — which is what makes the discount intervention viable.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Strong churn signals</div>', unsafe_allow_html=True)
        signal_data = {
            "Signal": ["No visible\ntransaction", "Zero-payment\nhistory", "Auto-renew\nOFF", "Ever\ncancelled"],
            "Churn rate": [84.29, 35.27, 29.48, 27.21],
            "Baseline": [8.99, 8.99, 8.99, 8.99],
            "Multiplier": ["9.4×", "3.9×", "3.3×", "3.0×"],
        }
        fig = go.Figure()
        colors = [RED, AMBER, BLUE, TEAL]
        fig.add_trace(go.Bar(
            x=signal_data["Signal"],
            y=signal_data["Churn rate"],
            marker_color=colors,
            text=[f"{v:.1f}%<br>{m}" for v, m in zip(signal_data["Churn rate"], signal_data["Multiplier"])],
            textposition="outside",
            textfont=dict(size=11),
        ))
        fig.add_hline(y=8.99, line_dash="dash", line_color=GRAY,
                      annotation_text="Baseline 8.99%", annotation_position="right")
        fig.update_layout(
            height=280, margin=dict(t=10, b=10, l=10, r=10),
            yaxis_title="Churn rate (%)", showlegend=False,
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#F1F5F9"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">Engagement vs churn — no relationship</div>',
                    unsafe_allow_html=True)
        eng_data = pd.DataFrame({
            "Bucket": ["Light\n(<20.6 streams/day)", "Medium\n(20.6-35.3)", "Heavy\n(>35.3)"],
            "Churn rate": [8.28, 9.57, 10.35],
        })
        fig2 = go.Figure(go.Bar(
            x=eng_data["Bucket"], y=eng_data["Churn rate"],
            marker_color=[GRAY, GRAY, GRAY],
            text=[f"{v:.2f}%" for v in eng_data["Churn rate"]],
            textposition="outside",
        ))
        fig2.add_hline(y=8.99, line_dash="dash", line_color=BLUE,
                       annotation_text="Baseline", annotation_position="right")
        fig2.update_layout(
            height=240, margin=dict(t=10, b=10, l=10, r=10),
            yaxis=dict(range=[0, 14], gridcolor="#F1F5F9"),
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("""
        <div class="insight-box">
        Active days, listening hours, completion rate — all flat. A user can listen
        heavily right up to the moment they cancel. Subscription mechanics dominate;
        satisfaction signals do not.
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="section-header">Tenure vs churn — also flat</div>',
                    unsafe_allow_html=True)
        tenure_data = pd.DataFrame({
            "Bucket": ["< 6 mo", "6-12 mo", "1-3 yr", "3-6 yr", "6+ yr"],
            "Churn rate": [9.19, 8.04, 9.63, 9.55, 9.76],
            "Users": [88237, 93727, 269829, 255391, 152726],
        })
        fig3 = go.Figure(go.Bar(
            x=tenure_data["Bucket"], y=tenure_data["Churn rate"],
            marker_color=[GRAY]*5,
            text=[f"{v:.2f}%" for v in tenure_data["Churn rate"]],
            textposition="outside",
        ))
        fig3.add_hline(y=8.99, line_dash="dash", line_color=BLUE,
                       annotation_text="Baseline", annotation_position="right")
        fig3.update_layout(
            height=240, margin=dict(t=10, b=10, l=10, r=10),
            yaxis=dict(range=[0, 14], gridcolor="#F1F5F9"),
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("""
        <div class="insight-box">
        Tenure spans range from 8 to 98 months; churn rate varies only 1.7pp.
        Retained for experiment segmentation, but not expected to drive the model.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CHURN MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Churn Model":
    st.markdown("## Churn Model")
    st.markdown(
        "Logistic regression trained on 679K users. "
        "Model quality is not the objective — the score identifies a high-risk "
        "experiment population. Report once, move on."
    )

    col1, col2, col3 = st.columns(3)
    for col, (val, label, sub, color) in zip(
        [col1, col2, col3],
        [
            ("0.8678", "ROC-AUC", "Test set", BLUE),
            ("0.7365", "Recall", "Catches 74% of churners", GREEN),
            ("0.4054", "Precision", "41% of flagged users churn", AMBER),
        ],
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.markdown('<div class="section-header">Feature importance (top 10 coefficients)</div>',
                    unsafe_allow_html=True)
        fi = feat_df.sort_values("coefficient", key=abs, ascending=True).tail(10)
        colors_bar = [RED if v > 0 else GREEN for v in fi["coefficient"]]
        fig4 = go.Figure(go.Bar(
            x=fi["coefficient"],
            y=fi["feature"],
            orientation="h",
            marker_color=colors_bar,
            text=[f"{v:+.3f}" for v in fi["coefficient"]],
            textposition="outside",
            textfont=dict(size=10),
        ))
        fig4.add_vline(x=0, line_color=GRAY, line_width=1)
        fig4.update_layout(
            height=340, margin=dict(t=10, b=10, l=10, r=60),
            xaxis_title="Standardised coefficient",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#F1F5F9"),
            showlegend=False,
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Model confirms EDA exactly</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        The four largest coefficients by magnitude match the four
        signals EDA identified as dominant:
        """)
        findings = [
            ("has_visible_transaction", "-0.98", "Strongest feature. No transaction → likely already gone."),
            ("auto_renew_at_checkpoint", "-0.65", "Auto-renew ON → dramatically lower churn risk."),
            ("ever_cancelled", "+0.64", "Cancellation history → higher churn risk."),
            ("transaction_count", "-0.47", "More renewals observed → lower risk."),
        ]
        for feat, coef, desc in findings:
            color = GREEN if coef.startswith("-") else RED
            st.markdown(f"""
            <div style="padding:8px 0; border-bottom:1px solid #F1F5F9;">
                <code style="font-size:0.78rem; background:#F1F5F9; padding:2px 6px;
                             border-radius:4px;">{feat}</code>
                <span style="font-weight:700; color:{color}; margin-left:8px;">{coef}</span>
                <div style="color:#64748B; font-size:0.8rem; margin-top:3px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warning-box" style="margin-top:14px;">
        <b>Model limitation:</b> within the High-risk segment, the churn probability
        is not well-calibrated — actual renewal rate is flat (~59%) across all
        probability deciles. Finer targeting within the segment adds no value.
        Model recalibration (Platt scaling) recommended before the next campaign.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EXPERIMENT DESIGN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Experiment Design":
    st.markdown("## Experiment Design")
    st.markdown(
        "Discount rate and MDE were derived analytically before randomisation — "
        "not chosen arbitrarily."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Lift × discount sensitivity matrix</div>',
                    unsafe_allow_html=True)

        BASELINE = 0.5914
        lifts     = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
        discounts = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        N         = 205883 / 2
        AVG_P     = 250.20

        matrix = []
        for l in lifts:
            row = []
            for d in discounts:
                net = (l * N * AVG_P * (1-d)) - (BASELINE * N * AVG_P * d)
                row.append(round(net / 1e6, 2))
            matrix.append(row)

        mat_df = pd.DataFrame(
            matrix,
            index=[f"{int(l*100)}pp" for l in lifts],
            columns=[f"{int(d*100)}%" for d in discounts],
        )

        fig5 = go.Figure(go.Heatmap(
            z=mat_df.values,
            x=mat_df.columns.tolist(),
            y=mat_df.index.tolist(),
            colorscale=[
                [0.0, "#FEE2E2"], [0.45, "#FEE2E2"],
                [0.5, "#F8FAFC"],
                [0.55, "#DCFCE7"], [1.0, "#15803D"],
            ],
            zmid=0,
            text=[[f"{v:+.1f}M" for v in row] for row in mat_df.values],
            texttemplate="%{text}",
            textfont=dict(size=10),
            showscale=False,
        ))
        fig5.add_shape(type="rect", x0=-0.5, x1=0.5, y0=3.5, y1=8.5,
                       line=dict(color=BLUE, width=2.5), fillcolor="rgba(0,0,0,0)")
        fig5.update_layout(
            height=320, margin=dict(t=10, b=10, l=10, r=10),
            xaxis_title="Discount rate",
            yaxis_title="Expected lift",
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.caption("Blue outline = chosen parameters (10% discount, ≥8pp lift). Green = net positive.")

    with col2:
        st.markdown('<div class="section-header">Why 10% discount?</div>', unsafe_allow_html=True)
        st.markdown("""
        The original brief specified **30% discount**. Sensitivity analysis showed:
        """)

        be_data = pd.DataFrame({
            "Expected lift": ["2pp", "4pp", "6pp", "8pp", "10pp", "12pp", "15pp", "20pp"],
            "Break-even discount": ["3.3%", "6.3%", "9.2%", "11.9%", "14.5%", "16.9%", "20.2%", "25.3%"],
            "Viable at 30%?": ["✗", "✗", "✗", "✗", "✗", "✗", "✗", "✗"],
        })
        st.dataframe(be_data, hide_index=True, use_container_width=True, height=200)

        st.markdown("""
        <div class="warning-box">
        <b>30% rejected:</b> net negative at every realistic lift level given the
        59.14% baseline renewal rate. Break-even would require >25pp lift.
        </div>
        <div class="insight-box">
        <b>10% chosen:</b> break-even at 14.5% for a 10pp lift gives a
        4.5pp buffer. 10% of ~250 NTD ≈ 25 NTD — minimum salient incentive.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:16px;">Power analysis</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">
            <div class="metric-card">
                <div class="metric-label">Baseline (high-risk proxy)</div>
                <div class="metric-value" style="color:{AMBER}; font-size:1.6rem;">59.14%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">MDE</div>
                <div class="metric-value" style="color:{BLUE}; font-size:1.6rem;">10pp</div>
                <div class="metric-sub">16.9% relative</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Required n (total)</div>
                <div class="metric-value" style="color:{GRAY}; font-size:1.6rem;">790</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Actual eligible</div>
                <div class="metric-value" style="color:{GREEN}; font-size:1.6rem;">205,883</div>
                <div class="metric-sub">260× overpowered</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Results":
    st.markdown("## Experiment Results")
    st.markdown(
        "Simulated experiment: 12pp lift injected into treatment group. "
        "The analysis recovers this known effect — the pipeline is the deliverable."
    )

    # Top-line metrics
    col1, col2, col3, col4 = st.columns(4)
    for col, (val, label, sub, color) in zip(
        [col1, col2, col3, col4],
        [
            (f"+{abs_lift*100:.2f}pp", "Absolute lift", f"Injected truth: 12pp", BLUE),
            (f"[{CI_LOW*100:.2f}, {CI_HIGH*100:.2f}]pp", "95% CI", "Entire CI above 10pp MDE", GREEN),
            ("p < 0.0001", "p-value", "z = 18.44", TEAL),
            (f"+{net_impact/1e6:.2f}M NTD", "Net impact", "After cannibalization", GREEN),
        ],
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color}; font-size:1.6rem;">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown('<div class="section-header">Renewal rate</div>', unsafe_allow_html=True)
        fig6 = go.Figure(go.Bar(
            x=["Control", "Treatment"],
            y=[ctrl_rate * 100, treat_rate * 100],
            marker_color=[GRAY, BLUE],
            text=[f"{ctrl_rate*100:.2f}%", f"{treat_rate*100:.2f}%"],
            textposition="outside", textfont=dict(size=13, color=["#374151", BLUE]),
        ))
        fig6.update_layout(
            height=280, margin=dict(t=10, b=10, l=10, r=10),
            yaxis=dict(range=[50, 80], gridcolor="#F1F5F9"),
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False,
        )
        st.plotly_chart(fig6, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Revenue per user (NTD)</div>', unsafe_allow_html=True)
        fig7 = go.Figure(go.Bar(
            x=["Control", "Treatment"],
            y=[ctrl_rpu, treat_rpu],
            marker_color=[GRAY, GREEN],
            text=[f"{ctrl_rpu:.1f}", f"{treat_rpu:.1f}"],
            textposition="outside", textfont=dict(size=13),
        ))
        fig7.update_layout(
            height=280, margin=dict(t=10, b=10, l=10, r=10),
            yaxis=dict(range=[0, 200], gridcolor="#F1F5F9"),
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False,
        )
        st.plotly_chart(fig7, use_container_width=True)

    with col_c:
        st.markdown('<div class="section-header">Business impact</div>', unsafe_allow_html=True)
        fig8 = go.Figure(go.Bar(
            x=["Incremental\nrevenue", "Cannibalization\ncost", "Net impact"],
            y=[incr_rev / 1e6, -cannib / 1e6, net_impact / 1e6],
            marker_color=[GREEN, RED, BLUE],
            text=[f"+{incr_rev/1e6:.2f}M", f"-{cannib/1e6:.2f}M", f"+{net_impact/1e6:.2f}M"],
            textposition="outside", textfont=dict(size=11),
        ))
        fig8.add_hline(y=0, line_color=GRAY, line_width=1)
        fig8.update_layout(
            height=280, margin=dict(t=10, b=10, l=10, r=10),
            yaxis=dict(gridcolor="#F1F5F9"),
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False,
        )
        st.plotly_chart(fig8, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Segment analysis — all 14 tests significant and net positive</div>',
                unsafe_allow_html=True)

    seg_display = seg_df[[
        "Segment", "Control Renewal %", "Treatment Renewal %",
        "Lift (pp)", "Net Impact (NTD)", "Significant", "Net Positive"
    ]].copy()
    seg_display["Net Impact (NTD)"] = seg_display["Net Impact (NTD)"].apply(lambda x: f"+{x:,.0f}")
    seg_display = seg_display.sort_values("Lift (pp)", ascending=False)

    st.dataframe(
        seg_display,
        hide_index=True,
        use_container_width=True,
        height=420,
        column_config={
            "Lift (pp)": st.column_config.NumberColumn(format="%.2f pp"),
            "Control Renewal %": st.column_config.NumberColumn(format="%.2f%%"),
            "Treatment Renewal %": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )

    st.markdown("""
    <div class="warning-box">
    <b>Multiple comparisons note:</b> 14 segment-level tests were pre-specified in the
    experiment design document. Lift magnitude differences between segments reflect noise
    as much as signal — no individual segment result overrides the primary decision.
    The uniform lift pattern (~11.5-13.6pp) confirms flat model calibration within
    the High-risk group.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — BUSINESS RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Business Recommendation":
    st.markdown("## Business Recommendation")

    # Go/No-Go banner
    st.markdown(f"""
    <div style="background:#F0FDF4; border:2px solid #86EFAC; border-radius:12px;
                padding:20px 28px; margin-bottom:20px;">
        <div style="display:flex; align-items:center; gap:16px;">
            <span class="go-badge" style="font-size:1.3rem; padding:8px 24px;">✓ GO</span>
            <div>
                <div style="font-weight:700; color:#0F172A; font-size:1.1rem;">
                    Launch the 10% renewal discount — targeted rollout only
                </div>
                <div style="color:#15803D; font-size:0.9rem; margin-top:4px;">
                    +{abs_lift*100:.2f}pp lift · 95% CI [{CI_LOW*100:.2f}pp, {CI_HIGH*100:.2f}pp] ·
                    p &lt; 0.0001 · net impact +{net_impact/1e6:.2f}M NTD
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Decision criteria — all passed</div>',
                    unsafe_allow_html=True)
        criteria = [
            ("Statistical significance", f"p < 0.0001 · z = 18.44", "✓"),
            ("Effect size exceeds MDE", f"+{abs_lift*100:.2f}pp > 10pp design MDE · entire CI above threshold", "✓"),
            ("Net revenue positive", f"+{net_impact/1e6:.2f}M NTD after {cannib/1e6:.2f}M NTD cannibalization", "✓"),
            ("All segments net positive", "14 / 14 pre-specified tests significant and net positive", "✓"),
        ]
        for criterion, detail, status in criteria:
            st.markdown(f"""
            <div style="display:flex; gap:12px; padding:10px 0; border-bottom:1px solid #F1F5F9;">
                <span style="color:{GREEN}; font-weight:700; font-size:1.1rem; flex-shrink:0;">{status}</span>
                <div>
                    <div style="font-weight:600; color:#0F172A; font-size:0.9rem;">{criterion}</div>
                    <div style="color:#64748B; font-size:0.82rem;">{detail}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:20px;">Targeting criteria</div>',
                    unsafe_allow_html=True)
        criteria2 = [
            ("risk_segment = High", "Top tercile of churn probability — driven by auto_renew OFF, cancellation history, zero-payment"),
            ("Expires within 30 days", "Pre-expiry window — discount must arrive before lapse, not as a win-back"),
            ("has_visible_transaction = 1", "Excludes users who are already gone — they need a win-back campaign, not a renewal offer"),
        ]
        for i, (rule, desc) in enumerate(criteria2, 1):
            st.markdown(f"""
            <div style="display:flex; gap:12px; padding:10px 0; border-bottom:1px solid #F1F5F9;">
                <span style="background:{BLUE}; color:white; border-radius:50%;
                             width:22px; height:22px; display:flex; align-items:center;
                             justify-content:center; font-size:0.75rem; font-weight:700;
                             flex-shrink:0;">{i}</span>
                <div>
                    <code style="font-size:0.8rem; background:#F1F5F9; padding:2px 6px;
                                 border-radius:4px;">{rule}</code>
                    <div style="color:#64748B; font-size:0.82rem; margin-top:3px;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box" style="margin-top:12px;">
        <b>Prioritise within eligible:</b> ever_cancelled=1 shows the highest lift (13.61pp)
        — these users are reconsidering and the discount tips them back. Tenure &lt; 6 months
        shows 12.70pp — the habit-formation window. Target these first if budget constrains
        the rollout below the full 205K.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Launch guardrails</div>', unsafe_allow_html=True)
        guardrails = [
            (
                "30-day renewal rate",
                "Primary success metric",
                f"Expected: ~{treat_rate*100:.0f}% treatment vs ~{ctrl_rate*100:.0f}% control. "
                "Pause if treatment rate < 65% after 14 days.",
            ),
            (
                "Net revenue impact",
                "Weekly check",
                "Expected positive from Week 1. Pause if net negative after 3 weeks.",
            ),
            (
                "Discount cost per incremental renewal",
                "Efficiency guardrail",
                f"Expected: ~148 NTD (< avg full price {avg_price:.0f} NTD). "
                "Flag if exceeds full price.",
            ),
            (
                "Plan downgrade rate",
                "Behavioural guardrail",
                "Flag if >5% of treatment renewers switch to a cheaper plan.",
            ),
            (
                "Discount expectation effect",
                "60-day post-renewal",
                "Monitor next-cycle renewal rate without discount — watch for users "
                "training themselves to wait for offers.",
            ),
            (
                "SRM check on launch day",
                "Assignment integrity",
                "Verify 50/50 split held in production. Any SRM (p < 0.05) → pause before "
                "reading outcome metrics.",
            ),
        ]
        for title, tag, detail in guardrails:
            st.markdown(f"""
            <div class="guardrail-item">
                <div class="guardrail-title">{title}
                    <span style="background:#EFF6FF; color:#1D4ED8; border-radius:4px;
                                 padding:1px 7px; font-size:0.72rem; font-weight:500;
                                 margin-left:8px;">{tag}</span>
                </div>
                <div class="guardrail-detail">{detail}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:8px;">What to improve next</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="warning-box">
        <b>Model calibration:</b> within the High-risk segment, actual renewal rate is flat
        (~59%) across all churn probability deciles. Calibrated scores (Platt scaling or
        isotonic regression) would enable finer targeting — potentially reducing cannibalization
        by identifying a genuinely low-baseline sub-population within High.
        </div>
        <div class="warning-box">
        <b>Richer features:</b> payment failures, device/platform changes, and customer
        service contacts may provide within-segment discrimination the current model lacks.
        </div>
        """, unsafe_allow_html=True)
