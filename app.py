import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

# Load the trained Holt-Winters model
model = joblib.load("src/model.pkl")

st.set_page_config(page_title="Gold Price Forecaster", layout="centered")
st.title("🪙 Indian Gold Price Forecaster")
st.write("Forecast future gold prices using a trained Holt-Winters model.")

# Slider lets the user choose how many days ahead to forecast
days = st.slider("How many days ahead do you want to forecast?", 1, 90, 30)

if st.button("Forecast"):
    forecast = model.forecast(days)

    forecast_dates = pd.date_range(start=pd.Timestamp.today(), periods=days)
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Forecasted Price": forecast.values
    })

    # Interactive Plotly chart, zoomed into the actual price range
    fig = px.line(forecast_df, x="Date", y="Forecasted Price", title="Gold Price Forecast")
    fig.update_yaxes(range=[
        forecast_df["Forecasted Price"].min() - 20,
        forecast_df["Forecasted Price"].max() + 20
    ])
    st.plotly_chart(fig, use_container_width=True)

    # Table of forecasted values below the chart
    st.dataframe(forecast_df)