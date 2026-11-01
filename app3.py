# Import the libraries
import streamlit as st
import pandas as pd
from prophet import Prophet
import yfinance as yf
from prophet.plot import plot_plotly, plot_components_plotly

# Set the title of the webapp
st.title("Stock Price Forecasting with Prophet")

# Get the symbol and period from the user
symbol = st.text_input("Enter stock symbol (e.g., BTC-USD):", "BTC-USD")
period = st.text_input("Enter stock period (e.g., 10y):", "10y")

# Get the data from yfinance
if symbol:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)

    if df.empty:
        st.error(f"No historical data found for '{symbol}'. Please use a valid ticker symbol and period.")
        st.stop()

    df = df.reset_index()

    # Support either 'Date' or 'Datetime' depending on yfinance
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

    # Clean and convert to Prophet format
    df = df[[date_col, "Close"]].copy()
    df.columns = ["ds", "y"]

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce", utc=True)
    df["ds"] = df["ds"].dt.tz_localize(None)   # remove timezone to satisfy Prophet
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    df = df.dropna().sort_values("ds").drop_duplicates(subset="ds")

    if df.empty:
        st.error(f"No valid price data is available for '{symbol}'.")
        st.stop()

    if len(df) < 2:
        st.error(f"Not enough valid data points for '{symbol}' to train Prophet.")
        st.stop()

    # Display the data
    st.header("Data")
    st.write(df)

    # Fit the prophet model
    st.header("Forecast")
    m = Prophet()
    m.fit(df)

    # Make future predictions
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)

    # Plot the forecast
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    # Plot the components
    st.header("Components")
    fig2 = plot_components_plotly(m, forecast)
    st.plotly_chart(fig2)
