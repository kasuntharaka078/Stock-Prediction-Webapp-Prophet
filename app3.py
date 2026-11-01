import streamlit as st
import pandas as pd
from prophet import Prophet
import yfinance as yf
from prophet.plot import plot_plotly, plot_components_plotly

st.title("Stock Price Forecasting with Prophet")

symbol = st.text_input("Enter stock symbol (e.g., BTC-USD):", "BTC-USD")
period = st.text_input("Enter stock period (e.g., 10y):", "10y")

if symbol:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)

    if df.empty:
        st.error(f"No historical data found for '{symbol}'. Please use a valid ticker symbol and a different period.")
        st.stop()

    df = df.reset_index()

    if "Date" in df.columns:
        date_col = "Date"
    elif "Datetime" in df.columns:
        date_col = "Datetime"
    else:
        st.error(f"Unexpected data format returned for '{symbol}'.")
        st.stop()

    if "Close" not in df.columns:
        st.error(f"Ticker '{symbol}' does not contain a valid closing price column.")
        st.stop()

    df = df[[date_col, "Close"]].copy()
    df.columns = ["ds", "y"]

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce", utc=True)
    df["ds"] = df["ds"].dt.tz_localize(None)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    df = df.dropna().sort_values("ds").drop_duplicates(subset="ds")

    if df.empty:
        st.error(f"No valid price data is available for '{symbol}'.")
        st.stop()

    if len(df) < 2:
        st.error(f"Not enough valid data points for '{symbol}' to train Prophet.")
        st.stop()

    st.header("Data")
    st.write(df)

    st.header("Forecast")
    m = Prophet()
    m.fit(df)

    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)

    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.header("Components")
    fig2 = plot_components_plotly(m, forecast)
    st.plotly_chart(fig2)
