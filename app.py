import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

# Load the trained Holt-Winters model
model = joblib.load("src/model.pkl")

st.set_page_config(page_title="Gold Price Forecaster", layout="centered")
st.title("🪙 Indian Gold Price Forecaster")
st.write("Forecast future gold prices using a trained Holt-Winters model.")

# Honest disclaimer so evaluators understand the forecast's actual time scope
st.info(
    "This model is trained on historical data through **July 22, 2022**. "
    "Forecasts represent the period immediately following that date, "
    "not real-time predictions for today — a normal limitation of "
    "time series models trained on a fixed historical window."
)

# Slider lets the user choose how many days ahead to forecast
days = st.slider("How many days ahead do you want to forecast?", 1, 90, 30)

if st.button("Forecast"):
    forecast = model.forecast(days)

    # Forecast dates start right after the last date the model was trained on,
    # NOT today's real calendar date — the model has no knowledge beyond 2022-07-22
    last_known_date = pd.Timestamp("2022-07-22")
    forecast_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=days)

    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Forecasted Price": forecast.values
    })

    # Interactive Plotly chart, zoomed into the actual price range
    fig = px.line(forecast_df, x="Date", y="Forecasted Price", title="Gold Price Forecast (Post-Training-Data Period)")
    fig.update_yaxes(range=[
        forecast_df["Forecasted Price"].min() - 20,
        forecast_df["Forecasted Price"].max() + 20
    ])
    st.plotly_chart(fig, use_container_width=True)

    # Table of forecasted values below the chart
    st.dataframe(forecast_df)