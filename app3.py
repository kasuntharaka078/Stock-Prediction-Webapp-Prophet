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
        st.error(f"No historical data found for '{symbol}'. Please enter a valid ticker symbol or a different period.")
        st.stop()

    df = df.reset_index()

    if "Date" not in df.columns or "Close" not in df.columns:
        st.error(f"The ticker '{symbol}' did not return the expected data format.")
        st.stop()

    df = df[["Date", "Close"]].dropna().copy()

    if df.empty:
        st.error(f"No valid price data is available for '{symbol}'.")
        st.stop()

    # Convert to Prophet format
    df["ds"] = pd.to_datetime(df["Date"])
    df["y"] = df["Close"]
    df = df[["ds", "y"]]

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
