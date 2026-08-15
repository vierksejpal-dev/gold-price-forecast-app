import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

model = joblib.load("src/model.pkl")

st.set_page_config(page_title="Gold Price Forecaster", layout="centered")
st.title("🪙 Indian Gold Price Forecaster")
st.write("Forecast future gold prices (₹ per 10g) using a Holt-Winters model.")

st.info(
    "This model is trained on a Kaggle historical dataset (Sept 2015 – July 2022, "
    "in USD/oz) combined with live daily gold futures data from Yahoo Finance "
    "(July 2022 – present), both converted to ₹ per 10g using the daily USD/INR rate. "
    "The 'Predicted Price' reflects the international spot price converted to ₹. "
    "The 'Est. Retail Price' adds an approximate adjustment for import duty, GST, "
    "and dealer premium, which typically add 10–15% over spot in the Indian market."
)

# Approximate combined markup for import duty (~6%) + GST (~3%) + dealer premium (~4%)
# Based on the observed gap between our spot-converted price and real Indian retail rates
RETAIL_MARKUP = 1.13

days = st.slider("How many days ahead do you want to forecast?", 1, 90, 30)

if st.button("Forecast"):
    forecast = model.forecast(days)
    forecast_dates = pd.date_range(start=pd.Timestamp.today() + pd.Timedelta(days=1), periods=days)

    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Predicted Price (₹/10g, spot)": forecast.values,
        "Est. Retail Price (₹/10g)": forecast.values * RETAIL_MARKUP
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"], y=forecast_df["Predicted Price (₹/10g, spot)"],
        name="Predicted (Spot)", line=dict(color="#1f77b4")
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"], y=forecast_df["Est. Retail Price (₹/10g)"],
        name="Est. Retail (with duty/GST)", line=dict(color="#ff7f0e", dash="dash")
    ))
    fig.update_layout(title="Gold Price Forecast", yaxis_title="Price (₹/10g)")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        forecast_df.style.format({
            "Predicted Price (₹/10g, spot)": "₹{:.2f}",
            "Est. Retail Price (₹/10g)": "₹{:.2f}"
        })
    )