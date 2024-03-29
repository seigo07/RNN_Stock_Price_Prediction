#!/usr/bin/env python
# coding: utf-8

# In[74]:


import sys
import random
import warnings
import tensorflow as tf
from functions import load_data, clean_data, split_data, create_dataset, build_model, get_best_params, print_metrics, get_one_year_data, print_trading_result, plot_trading_result
import numpy as np


def main():

    # Define error messages for invalid arguments
    INVALID_ARGS_NUMBER_ERROR = "Usage: python src/gru.py <TICKER>"

    # Define the number of expected arguments (script name + 2 arguments)
    ARGV_NUMBER = 2

    # Set the default algorithm for stock prediction
    ALGORITHM = "GRU"

    # Setting a consistent seed value ensures reproducibility across different runs
    SEED_VALUE = 42

    # Specify the length of input sequences for the neural network
    SEQUENCE_LENGTH = 20

    # Check if the number of arguments provided is correct
    if len(sys.argv) != ARGV_NUMBER:
        exit(INVALID_ARGS_NUMBER_ERROR)

    # Get the ticker symbol from command-line arguments
    ticker = sys.argv[1]

    # Suppress all warnings for cleaner output
    warnings.simplefilter("ignore")

    np.random.seed(SEED_VALUE)        # Set seed for numpy operations
    random.seed(SEED_VALUE)           # Set seed for Python's built-in random module
    tf.random.set_seed(SEED_VALUE)    # Set seed for TensorFlow operations

    # Load the dataset corresponding to the specified stock ticker
    dataset = load_data(ticker)

    # Pre-process and clean the data to make it suitable for training
    dataset, data = clean_data(dataset)

    # Split the cleaned data into training and test sets, also retrieve a scaler for normalization
    train_data, test_data, scaler = split_data(data)

    # Convert the training data into sequences of the specified length
    X_train, y_train = create_dataset(train_data, SEQUENCE_LENGTH)

    # Convert the test data into sequences of the specified length
    X_test, y_test = create_dataset(test_data, SEQUENCE_LENGTH)

    # Retrieve the best hyperparameters for the chosen algorithm using a function
    best_params = get_best_params(ALGORITHM, tf, SEQUENCE_LENGTH, scaler, X_train, y_train)

    # Construct the neural network model based on the chosen architecture and best hyperparameters
    model = build_model(ALGORITHM, SEQUENCE_LENGTH, tf, lr=best_params['learning_rate'], initializer=best_params['initializer'], loss_function=best_params['loss_function'])

    # Train the model on the training data
    model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=1)

    # Use the trained model to predict on the test set
    predictions = model.predict(X_test)

    # For benchmarking, generate naive predictions using the last value from the training set
    naive_predictions = np.full_like(y_test, y_train[-1])

    # Evaluate and print the model's performance metrics compared to naive predictions
    print_metrics(y_test, predictions, naive_predictions, best_params)

    # Get data and predictions for a duration of one year using the trained model
    one_year_data, one_year_data_2d, one_year_predictions = get_one_year_data(dataset, SEQUENCE_LENGTH, scaler, model)

    # Display the trading results derived from one-year predictions
    print_trading_result(one_year_data, one_year_data_2d, one_year_predictions, SEQUENCE_LENGTH)

    # Visualize the trading results for a comprehensive analysis
    plot_trading_result(ALGORITHM, ticker, SEQUENCE_LENGTH, one_year_data, one_year_data_2d, one_year_predictions)


if __name__ == "__main__":
    main()

# In[75]:
