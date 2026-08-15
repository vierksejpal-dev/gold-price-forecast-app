import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

model = joblib.load("src/model.pkl")

st.set_page_config(page_title="Gold Price Forecaster", layout="centered")
st.title("🪙 Indian Gold Price Forecaster")
st.write("Forecast future gold prices (₹ per 10g) using a Holt-Winters model.")

st.info(
    "This model is trained on a Kaggle historical dataset (Sept 2015 – July 2022, "
    "in USD/oz) combined with live daily gold futures data from Yahoo Finance "
    "(July 2022 – present), both converted to ₹ per 10g using the daily USD/INR rate. "
    "Note: forecasts reflect the international spot price converted to ₹, and will run "
    "roughly 10–15% below Indian retail/bullion market rates, which additionally include "
    "import duty, GST, and dealer premiums not captured in this model."
)

days = st.slider("How many days ahead do you want to forecast?", 1, 90, 30)

if st.button("Forecast"):
    forecast = model.forecast(days)
    forecast_dates = pd.date_range(start=pd.Timestamp.today() + pd.Timedelta(days=1), periods=days)

    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Forecasted Price (₹/10g)": forecast.values
    })

    fig = px.line(forecast_df, x="Date", y="Forecasted Price (₹/10g)", title="Gold Price Forecast")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(forecast_df)