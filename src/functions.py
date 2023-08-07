import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, GRU, Dense
from keras.models import Sequential
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


# Data Loading
def load_data(ticker):
    end = datetime.now()
    start = datetime(2020, end.month, end.day)
    dataset = yf.download(ticker, start, end)
    return dataset


# Data Cleaning
def clean_data(dataset):
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
    return dataset, data


# Split the data into train and test sets
def split_data(data):
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    return train_data, test_data, scaler


# Function to create sequences
def create_dataset(dataset, time_steps=1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - time_steps):
        a = dataset[i:(i + time_steps), 0]
        data_X.append(a)
        data_Y.append(dataset[i + time_steps, 0])
    return np.array(data_X), np.array(data_Y)


# Build the model
def build_model(algorithm, sequence_length, tf, lr, initializer='he_normal', loss_function='mse'):
    model = Sequential()
    if algorithm == "LSTM":
        model.add(LSTM(50, activation='tanh', recurrent_dropout=0, unroll=False, input_shape=(sequence_length, 1), kernel_initializer=initializer))
    elif algorithm == "GRU":
        model.add(GRU(50, activation='tanh', recurrent_dropout=0, unroll=False, input_shape=(sequence_length, 1), kernel_initializer=initializer))
    else:
        model.add(LSTM(50, activation='tanh', recurrent_dropout=0, unroll=False, input_shape=(sequence_length, 1), kernel_initializer=initializer))
    model.add(Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss_function)
    return model


# Grid search for best hyperparameters using TimeSeriesSplit
def get_best_params(algorithm, tf, sequence_length, scaler, X_train, y_train):
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

            model = build_model(algorithm, sequence_length, tf, lr=params['learning_rate'],
                                                    initializer=params['initializer'])
            model.compile(optimizer='adam', loss=params['loss_function'])
            model.fit(X_train_fold, y_train_fold, epochs=params['epochs'], batch_size=params['batch_size'],
                           verbose=0)
            predictions = model.predict(X_val_fold)
            predictions = scaler.inverse_transform(predictions)
            rmse_fold = np.sqrt(mean_squared_error(y_val_fold, predictions))
            rmse_sum += rmse_fold

        avg_rmse = rmse_sum / tscv.n_splits
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_params = params

    print("Best hyperparameters:")
    print(best_params)
    return best_params


# Function to calculate Theil U statistic
def theil_u_statistic(actual, predicted, naive):
    mse_actual = mean_squared_error(actual, naive)
    mse_predicted = mean_squared_error(actual, predicted)
    theil_u = np.sqrt(mse_predicted / mse_actual)
    return theil_u


# Calculate each metrics
def print_metrics(y_test, predictions, naive_predictions):
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    theil_u = theil_u_statistic(y_test, predictions, naive_predictions)

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Theil U statistic : {theil_u:.2f}")


def get_one_year_data(dataset, sequence_length, scaler, model):
    # Get data for the last one year
    one_year_ago = datetime.now() - timedelta(days=365)
    one_year_data = dataset[dataset.index >= one_year_ago]

    # Rescale the one-year data for plotting
    one_year_data_2d = one_year_data['Close'].values.reshape(-1, 1)
    one_year_data_scaled = scaler.transform(one_year_data['Close'].values.reshape(-1, 1))

    # Create sequences for the model for the one-year data
    X_one_year, _ = create_dataset(one_year_data_scaled, sequence_length)

    # Generate predictions for the model on the one-year data
    one_year_predictions = model.predict(X_one_year)
    one_year_predictions = scaler.inverse_transform(one_year_predictions)

    return one_year_data, one_year_data_2d, one_year_predictions


def print_trading_result(one_year_data, one_year_data_2d, one_year_predictions, sequence_length):
    # Initialize variables for the trading strategy for the one-year period
    initial_balance = 10000  # Initial balance (USD)
    balance = initial_balance
    stocks = 0
    N = len(one_year_predictions)  # Use the one-year price direction data

    # Implement the trading strategy for the one-year period
    for i in range(N):
        if one_year_predictions[i] > one_year_data_2d[i]:  # Predicted price will rise
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


def plot_trading_result(algorithm, ticker, sequence_length, one_year_data, one_year_data_2d, one_year_predictions):
    # Plot the predictions with buy/sell points for the one-year period
    plt.figure(figsize=(10, 6))

    plt.plot(one_year_data.index[sequence_length:], one_year_data['Close'][sequence_length:],
             label='Actual Stock Price')
    plt.plot(one_year_data.index[sequence_length:], one_year_predictions, label=f'{algorithm} Predicted Stock Price')
    plt.scatter(one_year_data.index[sequence_length:], one_year_predictions, marker='o', color='g',
                label='Buy' if one_year_predictions[-1] > one_year_data_2d[-1] else 'Sell')

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