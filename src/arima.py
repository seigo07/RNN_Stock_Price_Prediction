#!/usr/bin/env python
# coding: utf-8

# In[13]:


import sys
import random
import warnings
import tensorflow as tf
from functions import *

# Define error messages for invalid arguments
INVALID_ARGS_NUMBER_ERROR = "Usage: python src/arima.py <TICKER>"

# Define the number of expected arguments (script name + 2 arguments)
ARGV_NUMBER = 2

# Check if the number of arguments provided is correct
if len(sys.argv) != ARGV_NUMBER:
    exit(INVALID_ARGS_NUMBER_ERROR)

# Set the default algorithm for stock prediction
algorithm = "ARIMA"

# Get the ticker symbol from command-line arguments
ticker = sys.argv[1]

# Suppress all warnings for cleaner output
warnings.simplefilter("ignore")

# Set a consistent seed value to ensure reproducibility across different runs
SEED_VALUE = 42
np.random.seed(SEED_VALUE)        # Set seed for numpy operations
random.seed(SEED_VALUE)           # Set seed for Python's built-in random module
tf.random.set_seed(SEED_VALUE)    # Set seed for TensorFlow operations

# Load the dataset corresponding to the specified stock ticker
dataset = load_data(ticker)

# Process and clean the dataset to prepare it for analysis
dataset, data = clean_data(dataset)

# Determine the size of the training dataset, assuming it's 80% of the total data
train_size = int(len(data) * 0.8)

# Split the data into training and test sets, and retrieve a scaler for data normalization
train_data, test_data, scaler = split_data(data)

# Reshape the training and test data to be 2-dimensional, as required by some libraries or functions
train_data_2d = train_data.reshape(-1, 1)
test_data_2d = test_data.reshape(-1, 1)

# Find the best ARIMA parameters (p, d, q) for the dataset
best_theil_u, best_p, best_d, best_q = get_best_params_arima(data)

# Use the best ARIMA parameters to forecast on the test set
predictions = get_arima_predictions(scaler, train_data_2d, test_data_2d, best_p, best_d, best_q)

# Evaluate the model's performance using various metrics
print_metrics_arima(data, train_size, predictions, best_theil_u)

# Retrieve data and predictions for a duration of one year
one_year_data, one_year_data_2d, one_year_2d, one_year_predictions = get_one_year_data_arima(dataset, scaler, train_data_2d)

# Calculate and display the trading outcomes based on the one-year predictions
print_trading_result_arima(one_year_data, one_year_2d, one_year_predictions)

# Visualize the trading results for a more intuitive understanding of performance
plot_trading_result_arima(algorithm, ticker, one_year_data, one_year_2d, one_year_predictions)


# In[ ]:




