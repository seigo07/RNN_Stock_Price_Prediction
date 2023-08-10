import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, GRU, Dense
from keras.models import Sequential
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from itertools import product

# Define error messages for invalid ticker
INVALID_TICKER_ERROR = "Invalid ticker. Please enter a valid ticker"


def load_data(ticker):
    """
    Load historical stock data for a given ticker.

    Parameters:
    - ticker (str): Stock ticker symbol.

    Returns:
    - DataFrame: Historical stock data for the ticker.
    """

    # Get the current date, which will serve as the end date for our data collection
    end = datetime.now()

    # Set the start date for data collection as the same day and month but in the year 2020
    start = datetime(2020, end.month, end.day)

    # Download the stock data for the provided ticker symbol for a specified time frame
    dataset = yf.download(ticker, start, end)

    # Validate if data for the provided ticker is available
    if len(dataset) == 0:
        exit(INVALID_TICKER_ERROR)
    return dataset


def clean_data(dataset):
    """
    Clean and preprocess the dataset.

    Parameters:
    - dataset (DataFrame): Raw stock data.

    Returns:
    - tuple: Cleaned dataset and close prices reshaped as a 2D array.
    """

    # Remove rows containing missing or NaN values
    dataset = dataset.dropna()

    # Remove rows with duplicate index values (dates), keeping only the first occurrence
    dataset = dataset[~dataset.index.duplicated(keep='first')]

    # Define the lower bound for clipping close prices; setting it to 0 means no lower bound clipping
    lower_bound = 0

    # Define the upper bound for clipping close prices as the 99th percentile value
    upper_bound = np.percentile(dataset['Close'], 99)

    # Clip 'Close' column values to fall within the specified lower and upper bounds
    dataset['Close'] = np.clip(dataset['Close'], lower_bound, upper_bound)

    # Extract the 'Close' column values and reshape them into a 2D array
    data = dataset['Close'].values.reshape(-1, 1)

    return dataset, data


def split_data(data):
    """
    Split data into training and testing sets and normalize values.

    Parameters:
    - data (array): 2D array of close prices.

    Returns:
    - tuple: Normalized training data, normalized testing data, and the scaler object.
    """

    # Define the size of the training set as 80% of the total data size
    train_size = int(len(data) * 0.8)

    # Split data into training and testing sets
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Initialize a MinMaxScaler to scale values between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit the scaler on the training data and transform the training data
    train_data = scaler.fit_transform(train_data)

    # Transform the testing data using the same scaler
    test_data = scaler.transform(test_data)

    return train_data, test_data, scaler


def create_dataset(dataset, time_steps=1):
    """
    Convert the dataset into a format suitable for time series forecasting.

    Parameters:
    - dataset (array): 2D array of time series data.
    - time_steps (int): Number of past observations to consider for predicting the next observation.

    Returns:
    - tuple: Arrays of sequences (X) and corresponding next observations (Y).
    """

    data_X, data_Y = [], []

    # Loop through the dataset and extract sequences of length 'time_steps' and the next observation
    for i in range(len(dataset) - time_steps):
        # Extract a sequence of data of length 'time_steps'
        a = dataset[i:(i + time_steps), 0]

        # Append the sequence to the data_X list
        data_X.append(a)

        # Append the next observation to the data_Y list
        data_Y.append(dataset[i + time_steps, 0])

    # Convert lists to numpy arrays and return
    return np.array(data_X), np.array(data_Y)


def build_model(algorithm, sequence_length, tf, lr, initializer='he_normal', loss_function='mse'):
    """
    Build a recurrent neural network model based on the specified algorithm.

    Parameters:
    - algorithm (str): Algorithm to use ("LSTM" or "GRU").
    - sequence_length (int): Length of input sequences.
    - tf (module): TensorFlow module.
    - lr (float): Learning rate for optimizer.
    - initializer (str): Weight initializer method (default is 'he_normal').
    - loss_function (str): Loss function to use for model training (default is 'mse' or mean squared error).

    Returns:
    - model (tf.keras.Model): Compiled recurrent neural network model.
    """

    model = Sequential()

    # If the specified algorithm is LSTM
    if algorithm == "LSTM":
        model.add(LSTM(50, activation='tanh', recurrent_dropout=0, unroll=False, input_shape=(sequence_length, 1),
                       kernel_initializer=initializer))

    # If the specified algorithm is GRU
    elif algorithm == "GRU":
        model.add(GRU(50, activation='tanh', recurrent_dropout=0, unroll=False, input_shape=(sequence_length, 1),
                      kernel_initializer=initializer))

    # Default to LSTM if no valid algorithm is specified
    else:
        model.add(LSTM(50, activation='tanh', recurrent_dropout=0, unroll=False, input_shape=(sequence_length, 1),
                       kernel_initializer=initializer))

    # Add a dense layer with a single neuron for prediction
    model.add(Dense(1))

    # Define the optimizer with the specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Compile the model with the specified optimizer and loss function
    model.compile(optimizer=optimizer, loss=loss_function)

    return model


def time_series_cross_validation(data, n_splits, model_order):
    """
    Performs time series cross-validation on the data using the ARIMA model and calculates the average Theil U statistic.

    Parameters:
    - data (array): The time series data to be processed.
    - n_splits (int): The number of train-test splits for the time series cross-validation.
    - model_order (tuple): The parameters (p,d,q) of the ARIMA model.

    Returns:
    - float: The average Theil U statistic across all splits.
    """

    # Initialize TimeSeriesSplit object with the given number of splits.
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # List to store Theil U statistic scores for each split.
    theil_u_scores = []

    # Iterate over each train/test split.
    for train_index, test_index in tscv.split(data):

        # Segment the data into training and testing sets using the provided indices.
        train_data = data[train_index]
        test_data = data[test_index]

        # Initialize and fit the MinMaxScaler to the training data and apply it to both train and test data.
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)

        # Reshape train and test data into two-dimensional arrays.
        train_data_2d = train_data.reshape(-1, 1)
        test_data_2d = test_data.reshape(-1, 1)

        # Use the training data to create a history list for the ARIMA model.
        history = [x for x in train_data_2d]
        predictions = []

        # For each data point in the test set, fit the ARIMA model using the history and make a prediction.
        for t in range(len(test_data_2d)):
            model = ARIMA(history, order=model_order)
            model_fit = model.fit()
            output = model_fit.forecast(steps=1)
            yhat = output[0]

            # Add the prediction to the predictions list.
            predictions.append(yhat)

            # Add the actual observed value to the history for the next iteration.
            obs = test_data_2d[t]
            history.append(obs)

        # Create naive forecasts (using the last value of the training data) for Theil U statistic calculation.
        naive_predictions = np.full_like(test_data_2d, train_data_2d[-1])

        # Calculate Theil U statistic for the predictions and add to the list of scores.
        theil_u = theil_u_statistic(test_data_2d, predictions, naive_predictions)
        theil_u_scores.append(theil_u)

    # Return the average of all Theil U statistic scores.
    return np.mean(theil_u_scores)


def get_best_params(algorithm, tf, sequence_length, scaler, X_train, y_train):
    """
    Search for the best hyperparameters for a given algorithm using time series cross-validation.

    Parameters:
    - algorithm (str): The type of deep learning model (e.g., LSTM, GRU).
    - tf: TensorFlow module.
    - sequence_length (int): Number of time steps to consider in the input sequence.
    - scaler (object): The scaler object used to inverse transform predictions.
    - X_train (array): Training data features.
    - y_train (array): Training data targets.

    Returns:
    - dict: The best hyperparameters found.
    """

    # Define hyperparameter grid for searching.
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'epochs': [10, 20, 30],
        'batch_size': [16, 32, 64],
        'initializer': ['he_normal', 'glorot_uniform'],
        'loss_function': ['mse', 'mae']
    }

    # Initialize best hyperparameters and RMSE.
    best_params = None
    best_rmse = float('inf')

    # Time series cross-validation with 5 splits.
    tscv = TimeSeriesSplit(n_splits=5)

    # Iterate over all combinations of hyperparameters.
    for params in ParameterGrid(param_grid):
        rmse_sum = 0.0
        # For each split, validate the model's performance with the given hyperparameters.
        for train_index, val_index in tscv.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            # Build and compile the model using current hyperparameters.
            model = build_model(algorithm, sequence_length, tf, lr=params['learning_rate'],
                                initializer=params['initializer'])
            model.compile(optimizer='adam', loss=params['loss_function'])

            # Train the model using current fold.
            model.fit(X_train_fold, y_train_fold, epochs=params['epochs'], batch_size=params['batch_size'],
                      verbose=0)

            # Make predictions using the trained model.
            predictions = model.predict(X_val_fold)

            # Convert normalized predictions back to original scale.
            predictions = scaler.inverse_transform(predictions)

            # Calculate RMSE for the current fold and sum up.
            rmse_fold = np.sqrt(mean_squared_error(y_val_fold, predictions))
            rmse_sum += rmse_fold

        # Average RMSE across all folds.
        avg_rmse = rmse_sum / tscv.n_splits

        # Update best RMSE and hyperparameters if current combination is better.
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_params = params

    # Print the best hyperparameters.
    print("Best hyperparameters:")
    print(best_params)

    return best_params


def get_best_params_arima(data):
    """
    Search for the best ARIMA parameters for a given dataset using time series cross-validation.

    Parameters:
    - data (array): Time series data for ARIMA modeling.

    Returns:
    - float: Best Theil's U statistic score.
    - int: Best order parameter p.
    - int: Best order parameter d.
    - int: Best order parameter q.
    """

    # Define hyperparameter grid for ARIMA's p, d, q.
    param_grid = {
        'p': range(0, 3),  # Possible values for AR order.
        'd': range(1, 3),  # Possible values for differencing order.
        'q': range(0, 2)  # Possible values for MA order.
    }

    # Initialize the best Theil's U statistic as infinite (to be minimized).
    best_theil_u = float('inf')

    # Iterate over all combinations of p, d, and q.
    for p, d, q in product(param_grid['p'], param_grid['d'], param_grid['q']):
        model_order = (p, d, q)
        # Evaluate the ARIMA model with the current parameters using cross-validation.
        theil_u_score = time_series_cross_validation(data, n_splits=5, model_order=model_order)

        # Update the best parameters if current combination is better.
        if theil_u_score < best_theil_u:
            best_theil_u = theil_u_score
            best_p, best_d, best_q = p, d, q

    # Print the best hyperparameters.
    print(f"Best Parameters: p={best_p}, d={best_d}, q={best_q}")

    return best_theil_u, best_p, best_d, best_q


def get_arima_predictions(scaler, train_data_2d, test_data_2d, best_p, best_d, best_q):
    """
    Generate ARIMA model predictions for the test data using the best parameters.

    Parameters:
    - scaler (Scaler object): Scaler used for data normalization.
    - train_data_2d (array): Training data in 2D format.
    - test_data_2d (array): Test data in 2D format.
    - best_p (int): Optimal AR order.
    - best_d (int): Optimal differencing order.
    - best_q (int): Optimal MA order.

    Returns:
    - array: Predictions for the test set.
    """

    # Create a history list from the training data to be updated during forecasting.
    history = [x for x in train_data_2d]
    best_predictions = []

    # Iteratively make a forecast for each time step in the test data.
    for t in range(len(test_data_2d)):
        # Define and fit the ARIMA model with the best parameters.
        model = ARIMA(history, order=(best_p, best_d, best_q))
        model_fit = model.fit()

        # Generate a forecast for the next time step.
        output = model_fit.forecast(steps=1)
        yhat = output[0]

        # Append the forecast to the predictions list.
        best_predictions.append(yhat)

        # Add the actual observation to the history for the next iteration.
        obs = test_data_2d[t]
        history.append(obs)

    # Convert predictions list to a NumPy array and flatten.
    predictions = np.array(best_predictions).flatten()

    # Inverse transform the normalized predictions to the original scale.
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    return predictions


def theil_u_statistic(actual, predicted, naive):
    """
    Compute the Theil U statistic for model evaluation.

    Parameters:
    - actual (array): True values.
    - predicted (array): Model's predictions.
    - naive (array): Baseline or naive predictions.

    Returns:
    - float: Theil U statistic value.
    """

    # Compute the mean squared error of actual values against the naive forecast.
    mse_actual = mean_squared_error(actual, naive)

    # Compute the mean squared error of actual values against the model's predictions.
    mse_predicted = mean_squared_error(actual, predicted)

    # Calculate the Theil U statistic.
    theil_u = np.sqrt(mse_predicted / mse_actual)

    return theil_u


def print_metrics(y_test, predictions, naive_predictions):
    """
    Print various evaluation metrics for model performance.

    Parameters:
    - y_test (array): True values.
    - predictions (array): Model's predictions.
    - naive_predictions (array): Baseline or naive predictions.
    """

    # Calculate different metrics.
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    theil_u = theil_u_statistic(y_test, predictions, naive_predictions)

    # Display the metrics.
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Theil U statistic : {theil_u:.2f}")


def print_metrics_arima(data, train_size, predictions, best_theil_u):
    """
    Print evaluation metrics specifically for ARIMA model performance.

    Parameters:
    - data (array): Entire dataset (including training and test).
    - train_size (int): Number of observations in the training set.
    - predictions (array): ARIMA model's predictions.
    - best_theil_u (float): The best Theil U statistic value for the model.
    """

    # Calculate metrics using the test portion of the data.
    rmse = np.sqrt(mean_squared_error(data[train_size:], predictions))
    mae = mean_absolute_error(data[train_size:], predictions)
    r2 = r2_score(data[train_size:], predictions)
    mape = mean_absolute_percentage_error(data[train_size:], predictions)

    # Display the metrics.
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Theil U statistic : {best_theil_u:.2f}")


def get_one_year_data(dataset, sequence_length, scaler, model):
    """
    Extracts data for the last year from the provided dataset and generates predictions using a given model.

    Parameters:
    - dataset (DataFrame): The input data with a datetime index.
    - sequence_length (int): The length of sequences for training/testing.
    - scaler (object): The scaler object used for data normalization.
    - model (object): Trained model for prediction.

    Returns:
    - DataFrame: One year's worth of actual data.
    - array: One year's worth of actual data in 2D format.
    - array: Model's predictions for the last year.
    """

    # Calculate the date from one year ago.
    one_year_ago = datetime.now() - timedelta(days=365)

    # Filter data for the last year.
    one_year_data = dataset[dataset.index >= one_year_ago]

    # Reshape and scale the data.
    one_year_data_2d = one_year_data['Close'].values.reshape(-1, 1)
    one_year_data_scaled = scaler.transform(one_year_data_2d)

    # Generate sequences from the scaled data.
    X_one_year, _ = create_dataset(one_year_data_scaled, sequence_length)

    # Predict using the model.
    one_year_predictions = model.predict(X_one_year)
    one_year_predictions = scaler.inverse_transform(one_year_predictions)

    return one_year_data, one_year_data_2d, one_year_predictions


def get_one_year_data_arima(dataset, scaler, train_data_2d):
    """
    Extracts data for the last year from the provided dataset and generates ARIMA predictions.

    Parameters:
    - dataset (DataFrame): The input data with a datetime index.
    - scaler (object): The scaler object used for data normalization.
    - train_data_2d (array): 2D array of training data.

    Returns:
    - DataFrame: One year's worth of actual data.
    - array: One year's worth of scaled data in 2D format.
    - array: One year's worth of actual data in 2D format.
    - array: ARIMA predictions for the last year.
    """

    # Calculate the date from one year ago.
    one_year_ago = datetime.now() - timedelta(days=365)

    # Filter data for the last year.
    one_year_data = dataset[dataset.index >= one_year_ago]

    # Scale and reshape the data.
    one_year_data_scaled = scaler.transform(one_year_data['Close'].values.reshape(-1, 1))
    one_year_data_2d = one_year_data_scaled.reshape(-1, 1)
    one_year_2d = one_year_data['Close'].values.reshape(-1, 1)

    # Create a history list combining the training data with the one year data.
    history = train_data_2d.tolist() + one_year_data_2d.tolist()

    # Predict using ARIMA model.
    one_year_predictions = []
    for t in range(len(one_year_data_2d)):
        model = ARIMA(history, order=(1, 2, 0))
        model_fit = model.fit()
        output = model_fit.forecast(steps=1)
        yhat = output[0]
        one_year_predictions.append(yhat)
        obs = one_year_data_2d[t]
        history.append(obs)

    # Convert predictions to the original scale.
    one_year_predictions = np.array(one_year_predictions).flatten()
    one_year_predictions = scaler.inverse_transform(one_year_predictions.reshape(-1, 1)).flatten()

    return one_year_data, one_year_data_2d, one_year_2d, one_year_predictions


def print_trading_result(one_year_data, one_year_data_2d, one_year_predictions, sequence_length):
    """
    Simulates a trading strategy using predicted and actual data for one year and
    calculates the profit or loss made during the period.

    Parameters:
    - one_year_data (DataFrame): One year's worth of actual data with datetime index.
    - one_year_data_2d (array): Actual data in 2D format.
    - one_year_predictions (array): Predicted data for one year.
    - sequence_length (int): The length of sequences used in the prediction model.

    Returns:
    None
    """

    # Set initial trading parameters.
    initial_balance = 10000
    balance = initial_balance
    stocks = 0
    N = len(one_year_predictions)

    # Trading strategy: Buy if predicted price is higher than current and sell otherwise.
    for i in range(N):
        if one_year_predictions[i] > one_year_data_2d[i]:
            stocks_to_buy = int(balance / one_year_data['Close'][i + sequence_length])
            stocks += stocks_to_buy
            balance -= stocks_to_buy * one_year_data['Close'][i + sequence_length]
        else:
            balance += stocks * one_year_data['Close'][i + sequence_length]
            stocks = 0

    # Calculate final balance.
    final_balance = balance + stocks * one_year_data['Close'][-1]
    profit_or_loss = final_balance - initial_balance

    # Print results.
    print(f"Initial Balance: ${initial_balance}")
    print(f"Final Balance: ${final_balance:.2f}")
    print(f"Profit or Loss: ${profit_or_loss:.2f}")


def print_trading_result_arima(one_year_data, one_year_2d, one_year_predictions):
    """
    Simulates a trading strategy using ARIMA predicted and actual data for one year
    and calculates the profit or loss made during the period.

    Parameters:
    - one_year_data (DataFrame): One year's worth of actual data with datetime index.
    - one_year_2d (array): Actual data in 2D format.
    - one_year_predictions (array): ARIMA predicted data for one year.

    Returns:
    None
    """

    # Set initial trading parameters.
    initial_balance = 10000
    balance = initial_balance
    stocks = 0
    N = len(one_year_predictions)

    # Trading strategy: Buy if predicted price is higher than current and sell otherwise.
    for i in range(N):
        if one_year_predictions[i] > one_year_2d[i]:
            stocks_to_buy = int(balance / one_year_data['Close'].iloc[i])
            stocks += stocks_to_buy
            balance -= stocks_to_buy * one_year_data['Close'].iloc[i]
        else:
            balance += stocks * one_year_data['Close'].iloc[i]
            stocks = 0

    # Calculate final balance.
    final_balance = balance + stocks * one_year_data['Close'].iloc[-1]
    profit_or_loss = final_balance - initial_balance

    # Print results.
    print(f"Initial Balance: ${initial_balance}")
    print(f"Final Balance: ${final_balance:.2f}")
    print(f"Profit or Loss: ${profit_or_loss:.2f}")


def plot_trading_result(algorithm, ticker, sequence_length, one_year_data, one_year_data_2d, one_year_predictions):
    """
    Plot the trading results for a given stock using actual and predicted stock prices.

    Parameters:
    - algorithm (str): Name of the prediction algorithm used (e.g., "LSTM", "CNN").
    - ticker (str): The stock ticker symbol.
    - sequence_length (int): The length of sequences used in the prediction model.
    - one_year_data (DataFrame): One year's worth of actual stock data with datetime index.
    - one_year_data_2d (array): Actual stock data in 2D format.
    - one_year_predictions (array): Predicted stock data for one year.

    Returns:
    None
    """

    plt.figure(figsize=(10, 6))

    # Plot actual and predicted stock prices.
    plt.plot(one_year_data.index[sequence_length:], one_year_data['Close'][sequence_length:],
             label='Actual Stock Price')
    plt.plot(one_year_data.index[sequence_length:], one_year_predictions,
             label=f'{algorithm} Predicted Stock Price')

    # Add buy or sell indication based on the predicted price.
    plt.scatter(one_year_data.index[sequence_length:], one_year_predictions, marker='o', color='g',
                label='Buy' if one_year_predictions[-1] > one_year_data_2d[-1] else 'Sell')

    # Set axis labels and title.
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.title(f'{ticker} Stock Price Predictions with Buy/Sell Points (Last One Year)')
    plt.legend()
    plt.grid(True)

    # Adjust x-axis to show months.
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def plot_trading_result_arima(algorithm, ticker, one_year_data, one_year_2d, one_year_predictions):
    """
    Plot the trading results for a given stock using ARIMA actual and predicted stock prices.

    Parameters:
    - algorithm (str): Name of the prediction algorithm used (e.g., "ARIMA").
    - ticker (str): The stock ticker symbol.
    - one_year_data (DataFrame): One year's worth of actual stock data with datetime index.
    - one_year_2d (array): Actual stock data in 2D format.
    - one_year_predictions (array): ARIMA predicted stock data for one year.

    Returns:
    None
    """

    plt.figure(figsize=(10, 6))

    # Plot actual and ARIMA predicted stock prices.
    plt.plot(one_year_data.index, one_year_data['Close'].values, label='Actual Stock Price')
    plt.plot(one_year_data.index, one_year_predictions, label=f'{algorithm} Predicted Stock Price')

    # Add buy or sell indication based on the ARIMA predicted price.
    plt.scatter(one_year_data.index, one_year_predictions, marker='o', color='g',
                label='Buy' if one_year_predictions[-1] > one_year_2d[-1] else 'Sell')

    # Set axis labels and title.
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.title(f'{ticker} Stock Price Predictions with Buy/Sell Points (Last One Year)')
    plt.legend()
    plt.grid(True)

    # Adjust x-axis to show months.
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
