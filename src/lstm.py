#!/usr/bin/env python
# coding: utf-8

# In[74]:


import random
import warnings
import tensorflow as tf
from functions import *

warnings.simplefilter("ignore")

SEED_VALUE = 42
np.random.seed(SEED_VALUE)
random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

algorithm = "LSTM"
ticker = "AAPL"
dataset = load_data(ticker)
dataset, data = clean_data(dataset)
train_data, test_data, scaler = split_data(data)

sequence_length = 20
X_train, y_train = create_dataset(train_data, sequence_length)
X_test, y_test = create_dataset(test_data, sequence_length)

# Build and train the model with the best hyperparameters using the entire training data
best_params = get_best_params(algorithm, tf, sequence_length, scaler, X_train, y_train)
model = build_model(algorithm, sequence_length, tf, lr=best_params['learning_rate'], initializer=best_params['initializer'], loss_function=best_params['loss_function'])
model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=1)

# Generate predictions
predictions = model.predict(X_test)

# Generate naive predictions (using the last value in the training set)
naive_predictions = np.full_like(y_test, y_train[-1])

print_metrics(y_test, predictions, naive_predictions)

one_year_data, one_year_data_2d, one_year_predictions = get_one_year_data(dataset, sequence_length, scaler, model)

print_trading_result(one_year_data, one_year_data_2d, one_year_predictions, sequence_length)

plot_trading_result(algorithm, ticker, sequence_length, one_year_data, one_year_data_2d, one_year_predictions)


# In[75]:




