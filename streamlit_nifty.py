import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from pmdarima import auto_arima

st.set_page_config(
    page_title="NSE Index Forecast",
    layout="centered"
)

st.title("NSE Index Stock Price Forecast")
st.write("Auto-ARIMA forecasting using historical NSE index data")

@st.cache_data
def load_data():
    df = pd.read_csv("nse_indexes.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_resource
def train_model(series):
    model = auto_arima(
        series,
        seasonal=False,
        m=1,
        stepwise=True,
        suppress_warnings=True
    )
    return model

df_raw = load_data()

index_list = sorted(df_raw["Index"].unique())

selected_index = st.selectbox(
    "Select Index",
    index_list,
    index_list.index("NIFTY 50") if "NIFTY 50" in index_list else 0
)

df = df_raw[df_raw["Index"] == selected_index].copy()

df = df.set_index("Date").sort_index()
df = df.asfreq("B")
df["Close"] = df["Close"].ffill()

model = train_model(df["Close"])

days = st.number_input(
    "Enter number of days to forecast (max 60)",
    min_value=1,
    max_value=60,
    value=30
)

if st.button("Predict"):
    forecast = model.predict(n_periods=int(days))

    forecast_index = pd.date_range(
        start=df.index[-1] + pd.offsets.BDay(),
        periods=int(days),
        freq="B"
    )

    forecast = pd.Series(forecast, index=forecast_index)

    fig, ax = plt.subplots(figsize=(10, 5))
    df["Close"].tail(200).plot(ax=ax, label="Historical")
    forecast.plot(ax=ax, label="Forecast", color="red")

    ax.set_title(f"{selected_index} Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price")
    ax.legend()

    st.pyplot(fig)

    st.subheader(f"Forecast for next {int(days)} business days")
    st.dataframe(
        forecast.rename("Predicted Close Price").to_frame()
    )
