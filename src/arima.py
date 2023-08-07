#!/usr/bin/env python
# coding: utf-8

# In[13]:


import random
import warnings
import tensorflow as tf
from functions import *

warnings.simplefilter("ignore")

SEED_VALUE = 42
np.random.seed(SEED_VALUE)
random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

algorithm = "ARIMA"
ticker = "AAPL"
dataset = load_data(ticker)
dataset, data = clean_data(dataset)
train_size = int(len(data) * 0.8)
train_data, test_data, scaler = split_data(data)
train_data_2d = train_data.reshape(-1, 1)
test_data_2d = test_data.reshape(-1, 1)

best_theil_u, best_p, best_d, best_q = get_best_params(data)
predictions = get_arima_predictions(scaler, train_data_2d, test_data_2d, best_p, best_d, best_q)
print_metrics(data, train_size, predictions, best_theil_u)

one_year_data, one_year_data_2d, one_year_2d, one_year_predictions = get_one_year_data(dataset, scaler, train_data_2d)
print_trading_result(one_year_data, one_year_2d, one_year_predictions)
plot_trading_result(algorithm, ticker, one_year_data, one_year_2d, one_year_predictions)


# In[ ]:




