# Project

RNN Stock Price Prediction.

## Description

This is the regression system that can accurately predict the future stock prices using several models: LSTM, GRU, and ARIMA.
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
