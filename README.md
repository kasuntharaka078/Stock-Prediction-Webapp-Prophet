# Stock Price Forecasting with Prophet

This Streamlit web application utilizes the Prophet library to forecast stock prices based on historical data fetched from Yahoo Finance.

## Description

This app allows users to input a stock symbol and a period to retrieve historical stock data using Yahoo Finance API. It then uses the Prophet library to forecast future stock prices and displays the forecasted data along with its components.

## Installation

1. Ensure you have Python installed.
2. Clone this repository:

    ```bash
    git clone https://github.com/your-username/stock-price-forecasting.git
    ```

3. Navigate to the project directory and install the required dependencies:

    ```bash
    cd stock-price-forecasting
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

2. Access the application through your web browser at [http://localhost:8501](http://localhost:8501).

## How to Use

- **Enter Stock Symbol and Period**: Input the stock symbol (e.g., AAPL, MSFT) and the period for historical data retrieval (e.g., 10y for 10 years).
- **Data Display**: Visualizes the retrieved historical stock data in a table format.
- **Forecast**: Generates a forecast using the Prophet library and displays the predicted stock price.
- **Components**: Displays the individual forecasted components (trend, seasonality) using interactive plots.

## Technologies Used

- Streamlit
- Pandas
- Prophet
- yfinance

## Credits

This application utilizes the following libraries:
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Prophet](https://facebook.github.io/prophet/)
- [yfinance](https://pypi.org/project/yfinance/)

## Webapp

[stockpredictionfb](https://stockpredictionfb.streamlit.app/)
