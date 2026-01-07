import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="NIFTY 50 Forecast", layout="centered")

st.title("NIFTY 50 Stock Price Forecast")
st.write("Auto-ARIMA + SARIMAX forecasting using historical NSE data")

@st.cache_data
def load_data():
    df = pd.read_csv("nse_indexes.csv")
    df = df[df["Index"] == "NIFTY 50"]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.asfreq("B")
    df["Close"] = df["Close"].ffill()
    return df

@st.cache_resource
def train_model(series):
    arima_model = auto_arima(
        series,
        seasonal=False,
        m=1,
        stepwise=True,
        suppress_warnings=True
    )

    model = SARIMAX(
        series,
        order=arima_model.order,
        seasonal_order=arima_model.seasonal_order
    )

    return model.fit()

df = load_data()
model = train_model(df["Close"])

days = st.number_input(
    "Enter number of days to forecast (max 60)",
    min_value=1,
    max_value=60,
    value=30
)

if st.button("Predict"):
    forecast = model.forecast(steps=int(days))

    st.subheader(f"Forecast for next {int(days)} business days")
    st.dataframe(forecast.rename("Predicted Close Price"))

    fig, ax = plt.subplots()
    df["Close"].tail(200).plot(ax=ax, label="Historical")
    forecast.plot(ax=ax, label="Forecast", color="red")
    ax.set_xlabel("Time")
    ax.set_ylabel("NIFTY 50 Closing Price")
    ax.legend()
    st.pyplot(fig)