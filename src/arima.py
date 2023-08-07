#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import TimeSeriesSplit
from itertools import product
import random
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

warnings.simplefilter("ignore")

# Initialize GPU config
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    try:
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    except RuntimeError:
        # Memory growth must be set before GPUs have been initialized
        pass

SEED_VALUE = 42
np.random.seed(SEED_VALUE)
random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

ticker = "AAPL"
end = datetime.now()
start = datetime(2020, end.month, end.day)
dataset = yf.download(ticker, start, end)

# Data Cleaning
# Handle Missing Data
dataset = dataset.dropna()  # Remove rows with missing data

# Remove Duplicates
dataset = dataset[~dataset.index.duplicated(keep='first')]

# Handle Outliers (Clipping values)
lower_bound = 0  # Define lower bound for clipping
upper_bound = np.percentile(dataset['Close'], 99)  # Define upper bound for clipping (99th percentile)
dataset['Close'] = np.clip(dataset['Close'], lower_bound, upper_bound)

# Preprocess the data
data = dataset['Close'].values.reshape(-1, 1)

# Split the data into train and test sets
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

scaler = MinMaxScaler(feature_range=(0, 1))
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Reshape train_data and test_data to 2D arrays
train_data_2d = train_data.reshape(-1, 1)
test_data_2d = test_data.reshape(-1, 1)

# Function to calculate Theil U statistic
def theil_u_statistic(actual, predicted, naive):
    mse_actual = mean_squared_error(actual, naive)
    mse_predicted = mean_squared_error(actual, predicted)
    theil_u = np.sqrt(mse_predicted / mse_actual)
    return theil_u

# Grid search for best ARIMA hyperparameters
param_grid = {
    'p': range(0, 3),  # Narrow the range for p
    'd': range(1, 3),
    'q': range(0, 2)  # Narrow the range for q
}

# Function to perform time series cross-validation
def time_series_cross_validation(data, n_splits, model_order):
    tscv = TimeSeriesSplit(n_splits=n_splits)

    theil_u_scores = []
    for train_index, test_index in tscv.split(data):
        train_data = data[train_index]
        test_data = data[test_index]

        # Preprocess the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)

        # Reshape train_data and test_data to 2D arrays
        train_data_2d = train_data.reshape(-1, 1)
        test_data_2d = test_data.reshape(-1, 1)

        history = [x for x in train_data_2d]
        predictions = []
        for t in range(len(test_data_2d)):
            model = ARIMA(history, order=model_order)
            model_fit = model.fit()
            output = model_fit.forecast(steps=1)
            yhat = output[0]
            predictions.append(yhat)
            obs = test_data_2d[t]
            history.append(obs)

        # Calculate Theil U
        arima_naive_predictions = np.full_like(test_data_2d, train_data_2d[-1])
        arima_theil_u = theil_u_statistic(test_data_2d, predictions, arima_naive_predictions)
        theil_u_scores.append(arima_theil_u)

    return np.mean(theil_u_scores)

# Perform grid search with time series cross-validation
best_theil_u = float('inf')
best_arima_predictions = None
for p, d, q in product(param_grid['p'], param_grid['d'], param_grid['q']):
    model_order = (p, d, q)
    theil_u_score = time_series_cross_validation(data, n_splits=5, model_order=model_order)

    # Update best hyperparameters if Theil U improves
    if theil_u_score < best_theil_u:
        best_theil_u = theil_u_score
        best_p, best_d, best_q = p, d, q

# Train the final model using the best hyperparameters
history = [x for x in train_data_2d]
best_arima_predictions = []
for t in range(len(test_data_2d)):
    model = ARIMA(history, order=(best_p, best_d, best_q))
    model_fit = model.fit()
    output = model_fit.forecast(steps=1)
    yhat = output[0]
    best_arima_predictions.append(yhat)
    obs = test_data_2d[t]
    history.append(obs)

# # Generate predictions for the ARIMA model
arima_predictions = best_arima_predictions
arima_predictions = np.array(arima_predictions).flatten()
arima_predictions = scaler.inverse_transform(arima_predictions.reshape(-1, 1)).flatten()

# Generate naive predictions (using the last value in the training set)
naive_predictions = np.full_like(test_data, train_data[-1])

# Calculate RMSE
arima_rmse = np.sqrt(mean_squared_error(data[train_size:], arima_predictions))
arima_mae = mean_absolute_error(data[train_size:], arima_predictions)
arima_r2 = r2_score(data[train_size:], arima_predictions)
arima_mape = mean_absolute_percentage_error(data[train_size:], arima_predictions)

print(f"Best ARIMA Parameters: p={best_p}, d={best_d}, q={best_q}")
print(f"RMSE: {arima_rmse}")
print(f"MAE: {arima_mae}")
print(f"R2: {arima_r2}")
print(f"MAPE: {arima_mape:.2f}%")
print(f"Theil U statistic : {best_theil_u:.2f}")

# Get data for the last one year
one_year_ago = datetime.now() - timedelta(days=365)
one_year_data = dataset[dataset.index >= one_year_ago]

# Rescale the one-year data for plotting
one_year_data_scaled = scaler.transform(one_year_data['Close'].values.reshape(-1, 1))

# Reshape one_year_data_scaled to 2D array
one_year_data_2d = one_year_data_scaled.reshape(-1, 1)
one_year_2d = one_year_data['Close'].values.reshape(-1, 1)

# Reshape history to include one_year_data
history = train_data_2d.tolist() + one_year_data_2d.tolist()

# Predict using ARIMA model for the one-year period
one_year_arima_predictions = []
for t in range(len(one_year_data_2d)):
    model = ARIMA(history, order=(1, 2, 0))
    model_fit = model.fit()
    output = model_fit.forecast(steps=1)
    yhat = output[0]
    one_year_arima_predictions.append(yhat)
    obs = one_year_data_2d[t]
    history.append(obs)

# Convert one_year_arima_predictions list to a 1D numpy array
one_year_arima_predictions = np.array(one_year_arima_predictions).flatten()
one_year_arima_predictions = scaler.inverse_transform(one_year_arima_predictions.reshape(-1, 1)).flatten()

# Initialize variables for the trading strategy for the one-year period
initial_balance = 10000  # Initial balance (USD)
balance = initial_balance
stocks = 0
N = len(one_year_arima_predictions)  # Use the one-year price direction data

# Implement the trading strategy for the one-year period
for i in range(N):
    if one_year_arima_predictions[i] > one_year_2d[i]:  # Predicted price will rise
        stocks_to_buy = int(balance / one_year_data['Close'].iloc[i])
        stocks += stocks_to_buy
        balance -= stocks_to_buy * one_year_data['Close'].iloc[i]
    else:  # Predicted price will fall
        balance += stocks * one_year_data['Close'].iloc[i]
        stocks = 0

# Calculate profit or loss at the end of the one-year period
final_balance = balance + stocks * one_year_data['Close'].iloc[-1]
profit_or_loss = final_balance - initial_balance

print(f"Initial Balance: ${initial_balance}")
print(f"Final Balance: ${final_balance:.2f}")
print(f"Profit or Loss: ${profit_or_loss:.2f}")

# Plot the predictions with buy/sell points for the one-year period
plt.figure(figsize=(10, 6))

plt.plot(one_year_data.index, one_year_data['Close'].values, label='Actual Stock Price')
plt.plot(one_year_data.index, one_year_arima_predictions, label='ARIMA Predicted Stock Price')
plt.scatter(one_year_data.index, one_year_arima_predictions, marker='o', color='g',
            label='Buy' if one_year_arima_predictions[-1] > one_year_2d[-1] else 'Sell')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.title(f'{ticker} Stock Price Predictions with Buy/Sell Points (Last One Year)')
plt.legend()
plt.grid(True)

# Format x-axis ticks to show one month intervals
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))

plt.xticks(rotation=45)
plt.tight_layout()

# Add more descriptive labels to the X and Y axes
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')

plt.show()


# In[ ]:




