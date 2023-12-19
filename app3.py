!pip install prophet
# Import the libraries
import streamlit as st
import pandas as pd
from prophet import Prophet
import yfinance as yf
from prophet.plot import plot_plotly, plot_components_plotly

# Set the title of the webapp
st.title("Stock Price Forecasting with Prophet")

# Get the symbol and period from the user
symbol = st.sidebar.text_input("Enter the symbol", "BTC-USD")
period = st.sidebar.selectbox("Select the period", ["1y", "2y", "3y", "4y", "5y"])

# Get the data from yfinance
ticker = yf.Ticker(symbol)
df = ticker.history(period)
df = df.reset_index()
df = df.drop(['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'], axis = 1)
df['Date'] = pd.to_datetime(df['Date']) #convert to datetime
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d') #format as YYYY-MM-DD
df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

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
