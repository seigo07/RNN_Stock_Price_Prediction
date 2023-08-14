# Project

RNN Stock Price Prediction.

## Description

This program is a time-series forecasting model that predicts future stock price movements from stock price trading information for a specific period of time for selected companies.
Traditional and new regression algorithms LSTM, GRU, and ARIMA were selected for comparison and evaluation of model and profitability performance.
Specify the stock price of your favourite stock, train the model using about three years of data, and output trading results for one year.

## Getting Started

### Dependencies

* Python Version: 3.9.16

### Executing program

* Please run the following command

```
cd RNN_Stock_Price_Prediction
python3 src/lstm.py <ticker>
python3 src/gru.py <ticker>
python3 src/arima.py <ticker>
 
<ticker> : Specify one ticker name you like,
eg. AAPL MSFT AMZN NVDA TSLA
```
