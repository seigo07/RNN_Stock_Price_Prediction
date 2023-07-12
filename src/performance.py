#!/usr/bin/env python
# coding: utf-8

# In[202]:


import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yfinance as yf
from keras.layers import LSTM, GRU, Dense
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

warnings.simplefilter("ignore")

tf.random.set_seed(455)
np.random.seed(455)

ticker = "AAPL"
end = datetime.now()
start = datetime(2016, end.month, end.day)
dataset = yf.download(ticker, start, end)
dataset


# In[203]:


# Preprocess the data
data = dataset['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Split the data into train and test sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]


# Function to create sequences for LSTM and GRU models
def create_dataset(dataset, time_steps=1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - time_steps):
        a = dataset[i:(i + time_steps), 0]
        data_X.append(a)
        data_Y.append(dataset[i + time_steps, 0])
    return np.array(data_X), np.array(data_Y)


# Define sequence length
sequence_length = 20

# Create sequences for LSTM and GRU models
X_train, y_train = create_dataset(train_data, sequence_length)
X_test, y_test = create_dataset(test_data, sequence_length)


# In[204]:


# Function for get Bid and Ask
def get_prices(prices, predictions, threshold):
    buy_prices = []
    sell_prices = []

    for i in range(len(predictions)):
        if predictions[i] > prices[i] + threshold:
            sell_prices.append(prices[i])
        elif predictions[i] < prices[i] - threshold:
            buy_prices.append(prices[i])

    return buy_prices, sell_prices


# Define the trading strategy
def trading_strategy(actual_prices, predicted_prices):
    signal = np.zeros(len(actual_prices))
    for i in range(1, len(actual_prices)):
        if predicted_prices[i] > actual_prices[i - 1]:
            signal[i] = 1
        elif predicted_prices[i] < actual_prices[i - 1]:
            signal[i] = -1
    return signal


# In[205]:


# Build the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model
lstm_model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

# Generate predictions for the LSTM model
lstm_predictions = lstm_model.predict(X_test)
lstm_predictions = scaler.inverse_transform(lstm_predictions)

# Calculate indexes
lstm_rmse = np.sqrt(mean_squared_error(data[train_size + sequence_length:], lstm_predictions))
lstm_mae = mean_absolute_error(data[train_size + sequence_length:], lstm_predictions)
lstm_r2 = r2_score(data[train_size + sequence_length:], lstm_predictions)
print(f"RMSE: {lstm_rmse}")
print(f"MAE: {lstm_mae}")
print(f"R2: {lstm_r2}")

# Plot the predictions
plt.figure(figsize=(10, 6))
plt.plot(dataset.index[train_size + sequence_length:], data[train_size + sequence_length:], label='Actual')
plt.plot(dataset.index[train_size + sequence_length:], lstm_predictions, label='LSTM Prediction')
plt.xlabel('Date')
plt.ylabel('Index Value')
plt.title(f'{ticker} Index Predictions')
plt.legend()
plt.grid(True)
plt.show()


# In[206]:


# Apply simple trading strategy
threshold = 10  # Adjust the threshold as needed
buy_prices, sell_prices = get_prices(dataset['Close'][train_size + 1:], lstm_predictions, threshold)

# Plot strategy
lstm_signal = trading_strategy(data[train_size + sequence_length:], lstm_predictions)
# plt.plot(np.where(lstm_signal == 1, data[train_size + sequence_length:], None), 'ms', markersize=8, label='LSTM Buy')
# plt.plot(np.where(lstm_signal == -1, data[train_size + sequence_length:], None), 'ys', markersize=8, label='LSTM Sell')

# Print buy and sell prices
print("Buy Prices:")
print(buy_prices)
print("Sell Prices:")
print(sell_prices)


# In[216]:


# Define the trading strategy
def test(predicted_prices):
    signals = np.zeros_like(predicted_prices)
    signals[predicted_prices > np.roll(predicted_prices, 1)] = 1  # Generate buy signals
    signals[predicted_prices < np.roll(predicted_prices, 1)] = -1  # Generate sell signals
    return signals


lstm_signals = test(lstm_predictions)
actual_prices = dataset['Close'].values[train_size:]
lstm_strategy_returns = np.sum(actual_prices[:-1] * lstm_signals[1:])
print('LSTM Strategy Returns:', lstm_strategy_returns)


# In[207]:


# Build the GRU model
gru_model = Sequential()
gru_model.add(GRU(50, activation='relu', input_shape=(sequence_length, 1)))
gru_model.add(Dense(1))
gru_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the GRU model
gru_model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

# Generate predictions for the GRU model
gru_predictions = gru_model.predict(X_test)
gru_predictions = scaler.inverse_transform(gru_predictions)

# Calculate indexes
gru_rmse = np.sqrt(mean_squared_error(data[train_size + sequence_length:], gru_predictions))
gru_mae = mean_absolute_error(data[train_size + sequence_length:], gru_predictions)
gru_r2 = r2_score(data[train_size + sequence_length:], gru_predictions)
print(f"RMSE: {gru_rmse}")
print(f"MAE: {gru_mae}")
print(f"R2: {gru_r2}")

# Plot the predictions
plt.figure(figsize=(10, 6))
plt.plot(dataset.index[train_size + sequence_length:], data[train_size + sequence_length:], label='Actual')
plt.plot(dataset.index[train_size + sequence_length:], gru_predictions, label='GRU Prediction')
plt.xlabel('Date')
plt.ylabel('Index Value')
plt.title(f'{ticker} Index Predictions')
plt.legend()
plt.grid(True)
plt.show()


# In[208]:


# Apply simple trading strategy
threshold = 10  # Adjust the threshold as needed
buy_prices, sell_prices = get_prices(dataset['Close'][train_size + 1:], gru_predictions, threshold)

# Plot strategy
gru_signal = trading_strategy(data[train_size + sequence_length:], gru_predictions)
# plt.plot(np.where(gru_signal == 1, data[train_size + sequence_length:], None), 'ms', markersize=8, label='GRU Buy')
# plt.plot(np.where(gru_signal == -1, data[train_size + sequence_length:], None), 'ys', markersize=8, label='GRU Sell')

# Print buy and sell prices
print("Buy Prices:")
print(buy_prices)
print("Sell Prices:")
print(sell_prices)


# In[209]:


history = [x for x in train_data]
arima_predictions = []
for t in range(len(test_data)):
    model = ARIMA(history, order=(1, 2, 0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    arima_predictions.append(yhat)
    obs = test_data[t]
    history.append(obs)

# Calculate RMSE
arima_rmse = np.sqrt(mean_squared_error(test_data, arima_predictions))
arima_mae = mean_absolute_error(test_data, arima_predictions)
arima_r2 = r2_score(test_data, arima_predictions)
print(f"RMSE: {arima_rmse}")
print(f"MAE: {arima_mae}")
print(f"R2: {arima_r2}")

# Plot the actual and predicted values
plt.figure(figsize=(12, 6))
plt.plot(dataset.index[train_size:], test_data, label='Actual')
plt.plot(dataset.index[train_size:], arima_predictions, label='ARIMA Prediction')
plt.title('ARIMA Model - Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()


# In[210]:


# Apply simple trading strategy
threshold = 10  # Adjust the threshold as needed
buy_prices, sell_prices = get_prices(dataset['Close'][train_size:], arima_predictions, threshold)

# Plot strategy
arima_signal = trading_strategy(data[train_size + sequence_length:], arima_predictions)
# plt.plot(np.where(arima_signal == 1, data[train_size + sequence_length:], None), 'ms', markersize=8, label='ARIMA Buy')
# plt.plot(np.where(arima_signal == -1, data[train_size + sequence_length:], None), 'ys', markersize=8, label='ARIMA Sell')

# Print buy and sell prices
print("Buy Prices:")
print(buy_prices)
print("Sell Prices:")
print(sell_prices)


# In[210]:




