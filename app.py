import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Gold Price Forecaster",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — dark, glassmorphic, "futuristic" theme
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 20%, #1a1530 0%, #0b0b16 45%, #050508 100%);
        color: #eae6ff;
    }

    /* Hide default streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}

    /* ---- Hero header ---- */
    .hero {
        padding: 2.2rem 2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(255,215,120,0.12), rgba(120,90,255,0.10));
        border: 1px solid rgba(255,215,120,0.25);
        box-shadow: 0 0 40px rgba(255,215,120,0.08), inset 0 0 60px rgba(120,90,255,0.05);
        margin-bottom: 1.6rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: "";
        position: absolute;
        top: -50%; left: -20%;
        width: 60%; height: 200%;
        background: linear-gradient(120deg, transparent, rgba(255,215,120,0.08), transparent);
        transform: rotate(15deg);
        animation: shimmer 6s infinite linear;
    }
    @keyframes shimmer {
        0% { transform: translateX(-100%) rotate(15deg); }
        100% { transform: translateX(250%) rotate(15deg); }
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #ffd97a, #ffb3ff, #9ad1ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: #b8b3d9;
        margin-top: 0.5rem;
        font-size: 1.02rem;
        max-width: 720px;
        line-height: 1.5;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        background: rgba(120,255,180,0.12);
        border: 1px solid rgba(120,255,180,0.4);
        color: #9ffcc4;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.5px;
        margin-right: 8px;
    }

    /* ---- Glass cards for exog vars ---- */
    .glass-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 16px;
        padding: 1.1rem 1.2rem;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
        height: 100%;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(255,215,120,0.5);
    }
    .glass-card .label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #9a94c2;
        margin-bottom: 6px;
    }
    .glass-card .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #fdfaff;
    }
    .glass-card .unit {
        font-size: 0.85rem;
        color: #7d78a3;
        margin-left: 4px;
        font-weight: 400;
    }

    /* ---- Metric row ---- */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1rem 1.2rem;
    }
    div[data-testid="stMetricLabel"] { color: #9a94c2 !important; }
    div[data-testid="stMetricValue"] { color: #ffd97a !important; }

    /* ---- Buttons ---- */
    .stButton > button {
        background: linear-gradient(90deg, #ffb347, #ffd97a);
        color: #1a1200;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1.4rem;
        box-shadow: 0 4px 20px rgba(255,183,71,0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 6px 26px rgba(255,183,71,0.55);
        color: #1a1200;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0c1c, #06050c);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    .footer-note {
        color: #6b6690;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.06);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("src/model.pkl")
    exog_cols = joblib.load("src/exog_columns.pkl")
    last_known_exog = joblib.load("src/last_known_exog.pkl")
    return model, exog_cols, last_known_exog

model, exog_cols, last_known_exog = load_artifacts()

RETAIL_MARKUP = 1.13

display_info = {
    "USD_INR": ("USD / INR", "₹", "💵"),
    "Crude_Oil": ("Crude Oil (WTI)", "$/bbl", "🛢️"),
    "Silver": ("Silver", "$/oz", "🥈"),
    "Nifty50": ("Nifty 50", "pts", "📈"),
    "US10Y_Yield": ("US 10Y Yield", "%", "🏦"),
}

# ============================================================
# HERO HEADER
# ============================================================
st.markdown(f"""
<div class="hero">
    <span class="badge">● SARIMAX MODEL</span>
    <span class="badge">5 EXOG SIGNALS</span>
    <span class="badge">LIVE + HISTORICAL DATA</span>
    <h1>🪙 Indian Gold Price Forecaster</h1>
    <p>A time-series engine trained on gold's own price history plus five macro indicators —
    forecasting spot and estimated retail prices (₹ per 10g) for India's gold market.</p>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️  About this model & its assumptions", expanded=False):
    st.write(
        "This model blends a Kaggle historical dataset (2015–2022) with live data from "
        "Yahoo Finance (2022–present), and forecasts gold prices using gold's own past "
        "values alongside 5 exogenous variables. **Key assumption:** since future values "
        "of these variables are unknown, the forecast holds each one constant at its most "
        "recently observed level — accuracy may decline further out if these variables "
        "move significantly from current levels."
    )

# ============================================================
# EXOGENOUS VARIABLES — glass card grid
# ============================================================
st.markdown("### 📊 Live Exogenous Signals")
st.caption("Most recently observed values — held constant across the forecast horizon.")

cols = st.columns(len(exog_cols))
for c, col_name in zip(cols, exog_cols):
    label, unit, icon = display_info.get(col_name, (col_name, "", "📌"))
    val = last_known_exog[col_name]
    with c:
        st.markdown(f"""
        <div class="glass-card">
            <div class="label">{icon} {label}</div>
            <div class="value">{val:,.2f}<span class="unit">{unit}</span></div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Forecast Controls")
    days = st.slider("Forecast horizon (days)", 1, 90, 30)
    show_retail = st.toggle("Show estimated retail price", value=True)
    show_band = st.toggle("Show confidence band (approx.)", value=True)
    st.markdown("---")
    st.caption(f"Model run: {datetime.today().strftime('%d %b %Y')}")
    run_forecast = st.button("🚀 Run Forecast", use_container_width=True)

# ============================================================
# FORECAST + FUTURISTIC OUTPUT
# ============================================================
if run_forecast:
    with st.spinner("Running SARIMAX projection across exogenous signals..."):
        future_exog = pd.DataFrame([last_known_exog] * days, columns=exog_cols)
        forecast = np.array(model.predict(n_periods=days, X=future_exog))
        forecast_dates = pd.date_range(start=pd.Timestamp.today() + pd.Timedelta(days=1), periods=days)

        retail = forecast * RETAIL_MARKUP

        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Predicted Price (₹/10g, spot)": forecast,
            "Est. Retail Price (₹/10g)": retail
        })

    # ---- Summary metrics ----
    start_price = forecast[0]
    end_price = forecast[-1]
    pct_change = (end_price - start_price) / start_price * 100
    peak_price = forecast.max()
    trough_price = forecast.min()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Day 1 Forecast", f"₹{start_price:,.0f}")
    m2.metric(f"Day {days} Forecast", f"₹{end_price:,.0f}", f"{pct_change:+.2f}%")
    m3.metric("Projected Peak", f"₹{peak_price:,.0f}")
    m4.metric("Projected Trough", f"₹{trough_price:,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Futuristic Plotly chart ----
    fig = go.Figure()

    if show_band:
        band_width = np.linspace(0.01, 0.04, days) * forecast
        fig.add_trace(go.Scatter(
            x=np.concatenate([forecast_dates, forecast_dates[::-1]]),
            y=np.concatenate([forecast + band_width, (forecast - band_width)[::-1]]),
            fill="toself",
            fillcolor="rgba(255,215,120,0.08)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
            name="Uncertainty band"
        ))

    fig.add_trace(go.Scatter(
        x=forecast_df["Date"], y=forecast_df["Predicted Price (₹/10g, spot)"],
        mode="lines",
        name="Spot Forecast",
        line=dict(color="#ffd97a", width=3),
        fill="tozeroy",
        fillcolor="rgba(255,217,122,0.08)"
    ))

    if show_retail:
        fig.add_trace(go.Scatter(
            x=forecast_df["Date"], y=forecast_df["Est. Retail Price (₹/10g)"],
            mode="lines",
            name="Est. Retail Price",
            line=dict(color="#9ad1ff", width=2, dash="dot")
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk, sans-serif", color="#eae6ff"),
        title=dict(text="Gold Price Forecast — SARIMAX Projection", font=dict(size=18)),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="₹ per 10g"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(t=60, b=40, l=20, r=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---- Data table ----
    with st.expander("📋 View forecast data table", expanded=False):
        st.dataframe(
            forecast_df.style.format({
                "Predicted Price (₹/10g, spot)": "₹{:.2f}",
                "Est. Retail Price (₹/10g)": "₹{:.2f}"
            }),
            use_container_width=True
        )
        st.download_button(
            "⬇️ Download forecast as CSV",
            forecast_df.to_csv(index=False).encode("utf-8"),
            file_name="gold_price_forecast.csv",
            mime="text/csv"
        )
else:
    st.info("Set your forecast horizon in the sidebar and click **🚀 Run Forecast** to generate projections.")

st.markdown(
    '<div class="footer-note">Built with SARIMAX · Streamlit · Plotly — for educational/demo purposes only, not financial advice.</div>',
    unsafe_allow_html=True
)