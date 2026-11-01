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
ticker = yf.Ticker(symbol)
df = ticker.history(period=period)

if df.empty:
    st.error(f"No historical data found for '{symbol}'. Please try a valid ticker and different period.")
    st.stop()

df = df.reset_index()

# Validate the expected columns
required = {"Date", "Close"}
if not required.issubset(df.columns):
    st.error(f"Ticker '{symbol}' did not return the expected data shape for forecasting.")
    st.stop()

# Keep only needed columns and clean invalid rows
df = df[["Date", "Close"]].copy()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df = df.dropna().sort_values("Date")

if df.empty:
    st.error(f"No valid daily price data is available for '{symbol}'.")
    st.stop()

if len(df) < 2:
    st.error(f"Not enough valid data points for '{symbol}' to train Prophet.")
    st.stop()

# Rename for Prophet
df = df.rename(columns={"Date": "ds", "Close": "y"})
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
