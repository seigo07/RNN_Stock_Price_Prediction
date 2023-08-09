#!/usr/bin/env python
# coding: utf-8

# In[74]:


import sys
import random
import warnings
import tensorflow as tf
from functions import *

# Define error messages for invalid arguments
INVALID_ARGS_NUMBER_ERROR = "Usage: python src/lstm.py <TICKER> <INITIAL_BALANCE>"
INVALID_ARGS_INITIAL_BALANCE_ERROR = "Please enter an amount greater than or equal to 0 yen"
# Define the number of expected arguments (script name + 2 arguments)
ARGV_NUMBER = 3

# Check if the number of arguments provided is correct
if len(sys.argv) != ARGV_NUMBER:
    exit(INVALID_ARGS_NUMBER_ERROR)

# Set the default algorithm for stock prediction
algorithm = "LSTM"

# Get the ticker symbol from command-line arguments
ticker = sys.argv[1]

# Validate and get the initial balance from command-line arguments
if int(sys.argv[2]) <= 0:
    exit(INVALID_ARGS_INITIAL_BALANCE_ERROR)
initial_balance = sys.argv[2]

# Suppress all warnings to ensure cleaner output
warnings.simplefilter("ignore")

# Setting a consistent seed value to ensure reproducibility across various random operations
SEED_VALUE = 42
np.random.seed(SEED_VALUE)        # Seed for numpy-based operations
random.seed(SEED_VALUE)           # Seed for native Python's random module
tf.random.set_seed(SEED_VALUE)    # Seed for TensorFlow's random operations

# Load the dataset corresponding to the provided stock ticker
dataset = load_data(ticker)

# Process and clean the dataset to make it suitable for analysis
dataset, data = clean_data(dataset)

# Divide the processed data into training and testing subsets, and also retrieve a scaler for data normalization
train_data, test_data, scaler = split_data(data)

# Specify the length of sequences to be used in time-series forecasting
sequence_length = 20

# Convert the training data into sequences suitable for LSTM's input
X_train, y_train = create_dataset(train_data, sequence_length)

# Convert the testing data into sequences suitable for LSTM's input
X_test, y_test = create_dataset(test_data, sequence_length)

# Determine the optimal hyperparameters for the specified algorithm
best_params = get_best_params(algorithm, tf, sequence_length, scaler, X_train, y_train)

# Construct the LSTM neural network model using the optimal hyperparameters
model = build_model(algorithm, sequence_length, tf, lr=best_params['learning_rate'], initializer=best_params['initializer'], loss_function=best_params['loss_function'])

# Train the constructed model using the training dataset
model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=1)

# Use the trained model to make predictions on the testing dataset
predictions = model.predict(X_test)

# For comparison, generate naive predictions by simply using the last observed value from the training dataset
naive_predictions = np.full_like(y_test, y_train[-1])

# Evaluate the model's performance by comparing its predictions with both actual and naive predictions
print_metrics(y_test, predictions, naive_predictions)

# Retrieve data and its corresponding predictions for a duration of one year
one_year_data, one_year_data_2d, one_year_predictions = get_one_year_data(dataset, sequence_length, scaler, model)

# Analyze and display trading results derived from the one-year predictions
print_trading_result(one_year_data, one_year_data_2d, one_year_predictions, sequence_length)

# For better visual understanding, plot the trading outcomes
plot_trading_result(algorithm, ticker, sequence_length, one_year_data, one_year_data_2d, one_year_predictions)


# In[75]:




