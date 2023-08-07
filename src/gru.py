#!/usr/bin/env python
# coding: utf-8

# In[13]:


import random
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yfinance as yf
from keras.layers import GRU, Dense
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter("ignore")

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


# Function to create sequences for GRU models
def create_dataset(dataset, time_steps=1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - time_steps):
        a = dataset[i:(i + time_steps), 0]
        data_X.append(a)
        data_Y.append(dataset[i + time_steps, 0])
    return np.array(data_X), np.array(data_Y)


# Define sequence length
sequence_length = 20

# Create sequences for GRU models
X_train, y_train = create_dataset(train_data, sequence_length)
X_test, y_test = create_dataset(test_data, sequence_length)


# Build the GRU model
def build_gru_model(lr, initializer='he_normal', loss_function='mse'):
    gru_model = Sequential()
    gru_model.add(GRU(50, activation='tanh', recurrent_dropout=0, unroll=False, input_shape=(sequence_length, 1), kernel_initializer=initializer))
    gru_model.add(Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    gru_model.compile(optimizer=optimizer, loss=loss_function)
    return gru_model


# Define hyperparameter grid for grid search
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'epochs': [10, 20, 30],
    'batch_size': [16, 32, 64],
    'initializer': ['he_normal', 'glorot_uniform'],
    'loss_function': ['mse', 'mae']
}

best_params = None
best_rmse = float('inf')

# Grid search for best hyperparameters using TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for params in ParameterGrid(param_grid):
    rmse_sum = 0.0
    for train_index, val_index in tscv.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        gru_model = build_gru_model(lr=params['learning_rate'], initializer=params['initializer'])
        gru_model.compile(optimizer='adam', loss=params['loss_function'])
        gru_model.fit(X_train_fold, y_train_fold, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
        gru_predictions = gru_model.predict(X_val_fold)
        gru_predictions = scaler.inverse_transform(gru_predictions)
        rmse_fold = np.sqrt(mean_squared_error(y_val_fold, gru_predictions))
        rmse_sum += rmse_fold

    avg_rmse = rmse_sum / tscv.n_splits
    if avg_rmse < best_rmse:
        best_rmse = avg_rmse
        best_params = params

print("Best hyperparameters:")
print(best_params)

# Build and train the GRU model with best hyperparameters using the entire training data
gru_model = build_gru_model(lr=best_params['learning_rate'], initializer=best_params['initializer'], loss_function=best_params['loss_function'])
gru_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=1)

# Generate predictions for the GRU model
gru_predictions = gru_model.predict(X_test)


# Function to calculate Theil U statistic
def theil_u_statistic(actual, predicted, naive):
    mse_actual = mean_squared_error(actual, naive)
    mse_predicted = mean_squared_error(actual, predicted)
    theil_u = np.sqrt(mse_predicted / mse_actual)
    return theil_u


# Generate naive predictions (using the last value in the training set)
naive_predictions = np.full_like(y_test, y_train[-1])

# Calculate indexes
gru_rmse = np.sqrt(mean_squared_error(y_test, gru_predictions))
gru_mae = mean_absolute_error(y_test, gru_predictions)
gru_r2 = r2_score(y_test, gru_predictions)
gru_mape = mean_absolute_percentage_error(y_test, gru_predictions)
gru_theil_u = theil_u_statistic(y_test, gru_predictions, naive_predictions)

print(f"RMSE: {gru_rmse}")
print(f"MAE: {gru_mae}")
print(f"R2: {gru_r2}")
print(f"MAPE: {gru_mape:.2f}%")
print(f"Theil U statistic : {gru_theil_u:.2f}")

gru_predictions_list = gru_predictions.flatten().tolist()

# Get data for the last one year
one_year_ago = datetime.now() - timedelta(days=365)
one_year_data = dataset[dataset.index >= one_year_ago]

# Rescale the one-year data for plotting
one_year_data_2d = one_year_data['Close'].values.reshape(-1, 1)
one_year_data_scaled = scaler.transform(one_year_data['Close'].values.reshape(-1, 1))

# Create sequences for GRU models for the one-year data
X_one_year, y_one_year = create_dataset(one_year_data_scaled, sequence_length)

# Generate predictions for the GRU model on the one-year data
gru_predictions_one_year = gru_model.predict(X_one_year)
gru_predictions_one_year = scaler.inverse_transform(gru_predictions_one_year)

# Initialize variables for the trading strategy for the one-year period
initial_balance = 10000  # Initial balance (USD)
balance = initial_balance
stocks = 0
N = len(gru_predictions_one_year)  # Use the one-year price direction data

# Implement the trading strategy for the one-year period
for i in range(N):
    if gru_predictions_one_year[i] > one_year_data_2d[i]:  # Predicted price will rise
        stocks_to_buy = int(balance / one_year_data['Close'][i + sequence_length])
        stocks += stocks_to_buy
        balance -= stocks_to_buy * one_year_data['Close'][i + sequence_length]
    else:  # Predicted price will fall
        balance += stocks * one_year_data['Close'][i + sequence_length]
        stocks = 0

# Calculate profit or loss at the end of the one-year period
final_balance = balance + stocks * one_year_data['Close'][-1]
profit_or_loss = final_balance - initial_balance

print(f"Initial Balance: ${initial_balance}")
print(f"Final Balance: ${final_balance:.2f}")
print(f"Profit or Loss: ${profit_or_loss:.2f}")

# Plot the predictions with buy/sell points for the one-year period
plt.figure(figsize=(10, 6))

plt.plot(one_year_data.index[sequence_length:], one_year_data['Close'][sequence_length:], label='Actual Stock Price')
plt.plot(one_year_data.index[sequence_length:], gru_predictions_one_year, label='GRU Predicted Stock Price')
plt.scatter(one_year_data.index[sequence_length:], gru_predictions_one_year, marker='o', color='g',
            label='Buy' if gru_predictions_one_year[-1] > one_year_data_2d[-1] else 'Sell')

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


# In[9]:




