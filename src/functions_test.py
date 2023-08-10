import contextlib
import unittest
import pandas as pd
import numpy as np
import tensorflow as tf
import warnings
import sys
from io import StringIO
from datetime import datetime
from unittest.mock import patch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from src.functions import load_data, clean_data, split_data, create_dataset, build_model, time_series_cross_validation, get_arima_predictions, theil_u_statistic, print_metrics, print_metrics_arima, get_one_year_data, get_one_year_data_arima, print_trading_result
from keras.layers import LSTM, GRU, Dense


INVALID_TICKER_ERROR = "Invalid ticker. Please enter a valid ticker"  # Assuming you have a constant for this


class MockModel:
    """
    A mock model class to simulate predictions.
    This will just return the input for simplicity.
    """
    def predict(self, data):
        return data


class TestLoadData(unittest.TestCase):

    # Test for a valid ticker
    @patch('yfinance.download')
    def test_load_data_valid_ticker(self, mock_download):
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'Close': [101, 102, 103],
        })
        mock_download.return_value = mock_data

        result = load_data("AAPL")
        self.assertEqual(len(result), 3)  # Check if we got the mocked data

    # Test for an invalid ticker
    @patch('yfinance.download')
    def test_load_data_invalid_ticker(self, mock_download):
        mock_download.return_value = pd.DataFrame()

        with self.assertRaises(SystemExit) as cm:  # Since you are exiting for invalid tickers
            load_data("INVALID")

        self.assertEqual(cm.exception.code, INVALID_TICKER_ERROR)


class TestCleanData(unittest.TestCase):

    def setUp(self):
        self.dataset = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'Close': [101, 102, np.nan, 105, 110],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }, index=pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-03', '2021-01-05']))

    def test_remove_nan_values(self):
        cleaned_data, _ = clean_data(self.dataset)
        self.assertFalse(cleaned_data['Close'].isnull().any())

    def test_remove_duplicate_dates(self):
        cleaned_data, _ = clean_data(self.dataset)
        self.assertEqual(cleaned_data.index.duplicated().sum(), 0)

    def test_clip_close_values(self):
        cleaned_data, _ = clean_data(self.dataset)
        self.assertTrue((cleaned_data['Close'] <= np.percentile(self.dataset['Close'].dropna(), 99)).all())
        self.assertTrue((cleaned_data['Close'] >= 0).all())

    def test_reshape_close_values(self):
        _, data = clean_data(self.dataset)
        self.assertEqual(data.shape, (4, 1))


class TestSplitData(unittest.TestCase):

    def setUp(self):
        self.data = np.array([[100], [101], [102], [103], [104], [105], [106], [107], [108], [109]])

    def test_split_ratio(self):
        train_data, test_data, _ = split_data(self.data)
        self.assertEqual(len(train_data), 8)  # 80% of 10
        self.assertEqual(len(test_data), 2)   # 20% of 10

    def test_normalization(self):
        train_data, test_data, _ = split_data(self.data)
        self.assertTrue((0 <= train_data).all() and (train_data <= 1).all())

        # Instead of enforcing that test_data values should be between 0 and 1,
        # check if they are correctly scaled based on the training data's range.
        min_train = np.min(self.data[:8])
        max_train = np.max(self.data[:8])
        min_test = np.min(self.data[8:])
        max_test = np.max(self.data[8:])

        self.assertAlmostEqual(train_data[0], 0, delta=1e-10)  # Train data minimum should be approximately 0
        self.assertAlmostEqual(train_data[-1], (max_train - min_train) / (max_train - min_train),
                               delta=1e-10)  # Last train data point scaled
        self.assertAlmostEqual(test_data[0], (min_test - min_train) / (max_train - min_train),
                               delta=1e-10)  # First test data point scaled
        self.assertAlmostEqual(test_data[-1], (max_test - min_train) / (max_train - min_train),
                               delta=1e-10)  # Last test data point scaled

    def test_using_same_scaler(self):
        _, _, scaler = split_data(self.data)
        manual_test_data = scaler.transform(self.data[-2:])  # manually transform the last two data points
        _, test_data, _ = split_data(self.data)
        self.assertTrue(np.array_equal(manual_test_data, test_data))

    def test_return_scaler_type(self):
        _, _, scaler = split_data(self.data)
        self.assertIsInstance(scaler, MinMaxScaler)


class TestCreateDataset(unittest.TestCase):

    def setUp(self):
        self.data = np.array([[100], [101], [102], [103], [104], [105], [106], [107], [108], [109]])

    def test_dataset_length(self):
        time_steps = 3
        data_X, data_Y = create_dataset(self.data, time_steps)

        expected_length = len(self.data) - time_steps
        self.assertEqual(len(data_X), expected_length)
        self.assertEqual(len(data_Y), expected_length)

    def test_sequence_length(self):
        time_steps = 3
        data_X, _ = create_dataset(self.data, time_steps)

        for sequence in data_X:
            self.assertEqual(len(sequence), time_steps)

    def test_correct_sequence_values(self):
        time_steps = 3
        data_X, data_Y = create_dataset(self.data, time_steps)

        for i in range(len(data_X)):
            np.testing.assert_array_equal(data_X[i], self.data[i:i + time_steps].flatten())
            self.assertEqual(data_Y[i], self.data[i + time_steps][0])


class TestBuildModel(unittest.TestCase):

    def setUp(self):
        self.tf = tf
        self.sequence_length = 5
        self.lr = 0.001

    def test_lstm_model(self):
        model = build_model("LSTM", self.sequence_length, self.tf, self.lr)

        # Check if the first layer is LSTM
        self.assertIsInstance(model.layers[0], LSTM)
        # Check if input shape is correct
        self.assertEqual(model.layers[0].input_shape, (None, self.sequence_length, 1))
        # Check learning rate
        self.assertAlmostEqual(model.optimizer.learning_rate.numpy(), self.lr, places=10)
        # Check the dense layer
        self.assertIsInstance(model.layers[1], Dense)
        self.assertEqual(model.layers[1].units, 1)

    def test_gru_model(self):
        model = build_model("GRU", self.sequence_length, self.tf, self.lr)

        # Check if the first layer is GRU
        self.assertIsInstance(model.layers[0], GRU)
        # Check if input shape is correct
        self.assertEqual(model.layers[0].input_shape, (None, self.sequence_length, 1))
        # Check learning rate
        self.assertAlmostEqual(model.optimizer.learning_rate.numpy(), self.lr, places=10)
        # Check the dense layer
        self.assertIsInstance(model.layers[1], Dense)
        self.assertEqual(model.layers[1].units, 1)

    def test_invalid_algorithm(self):
        model = build_model("INVALID", self.sequence_length, self.tf, self.lr)

        # Check if the default layer is LSTM
        self.assertIsInstance(model.layers[0], LSTM)


class TestTimeSeriesCrossValidation(unittest.TestCase):

    def setUp(self):
        # Suppress all warnings to ensure cleaner output
        warnings.filterwarnings("ignore")

        # Reshape the data array so it becomes 2D
        self.data = np.arange(100).reshape(-1, 1)
        self.n_splits = 5
        self.model_order = (1, 0, 1)

    def test_cross_validation_output(self):
        result = time_series_cross_validation(self.data, self.n_splits, self.model_order)

        # Check if the returned result is a float value for the Theil U statistic
        self.assertIsInstance(result, float)

        # Optionally, you can check the range of the result, since Theil U statistic is >= 0
        self.assertGreaterEqual(result, 0)


class TestArimaPredictions(unittest.TestCase):

    def setUp(self):
        # Suppress all warnings to ensure cleaner output
        warnings.filterwarnings("ignore")

        # Create a mock dataset
        self.train_data = np.array([x for x in range(10)])
        self.test_data = np.array([x for x in range(10, 15)])

        # Scale the data using MinMaxScaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.train_data_2d = self.scaler.fit_transform(self.train_data.reshape(-1, 1))
        self.test_data_2d = self.scaler.transform(self.test_data.reshape(-1, 1))

        # Set mock ARIMA parameters. These may not be optimal for the mock data but serve demonstration purposes.
        self.best_p = 1
        self.best_d = 1
        self.best_q = 1

    def test_get_arima_predictions(self):
        # Call the function to get predictions
        predictions = get_arima_predictions(self.scaler, self.train_data_2d, self.test_data_2d, self.best_p, self.best_d, self.best_q)

        # Ensure the predictions are of the right shape
        self.assertEqual(predictions.shape, (len(self.test_data),))

        # Note: In a real-world scenario, you'd compare the values of `predictions` with some expected results.
        # However, given that ARIMA models involve a certain degree of randomness and approximation,
        # it's challenging to define an "expected" result for this mock data.
        # A common approach is to ensure that the model's predictions follow the general trend or pattern of the data.
        # Alternatively, the RMSE or other metrics can be used to ensure the model's performance is within acceptable bounds.


class TestTheilUStatistic(unittest.TestCase):

    def test_theil_u_statistic(self):
        # Mock data
        actual = np.array([3, 2, 4, 5, 6])
        predicted = np.array([2.8, 2.1, 3.9, 5.2, 6.1])
        naive = np.array([3, 3, 2, 4, 5])

        # Expected Theil U statistic for mock data
        mse_actual = mean_squared_error(actual, naive)
        mse_predicted = mean_squared_error(actual, predicted)
        expected_theil_u = np.sqrt(mse_predicted / mse_actual)

        # Use the function to compute the Theil U statistic
        computed_theil_u = theil_u_statistic(actual, predicted, naive)

        # Compare the computed Theil U statistic to the expected value
        self.assertAlmostEqual(computed_theil_u, expected_theil_u)


class TestPrintMetrics(unittest.TestCase):

    def setUp(self):
        self.original_stdout = sys.stdout
        sys.stdout = StringIO()  # Redirect stdout

    def tearDown(self):
        sys.stdout = self.original_stdout  # Restore stdout

    def test_print_metrics(self):
        y_test = np.array([3, 2, 4, 5, 6])
        predictions = np.array([2.8, 2.1, 3.9, 5.2, 6.1])
        naive_predictions = np.array([3, 3, 2, 4, 5])

        print_metrics(y_test, predictions, naive_predictions)

        output = sys.stdout.getvalue().strip().split("\n")

        # Check each of the printed outputs except MAPE
        self.assertEqual(output[0], f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions))}")
        self.assertEqual(output[1], f"MAE: {mean_absolute_error(y_test, predictions)}")
        self.assertEqual(output[2], f"R2: {r2_score(y_test, predictions)}")
        mape = mean_absolute_percentage_error(y_test, predictions)
        self.assertEqual(output[3], f"MAPE: {mape:.2f}%")
        self.assertEqual(output[4], f"Theil U statistic : {theil_u_statistic(y_test, predictions, naive_predictions):.2f}")


class TestPrintMetricsArima(unittest.TestCase):

    def setUp(self):
        self.original_stdout = sys.stdout
        sys.stdout = StringIO()  # Redirect stdout

    def tearDown(self):
        sys.stdout = self.original_stdout  # Restore stdout

    def test_print_metrics_arima(self):
        # Sample data, predictions, and Theil U value
        data = np.array([2, 3, 5, 8, 12, 18, 27, 39])
        train_size = 6
        predictions = np.array([25, 36])
        best_theil_u = 1.2  # Assume this value for the purpose of the test

        print_metrics_arima(data, train_size, predictions, best_theil_u)

        output = sys.stdout.getvalue().strip().split("\n")

        # Check each of the printed outputs except MAPE
        self.assertEqual(output[0], f"RMSE: {np.sqrt(mean_squared_error(data[train_size:], predictions))}")
        self.assertEqual(output[1], f"MAE: {mean_absolute_error(data[train_size:], predictions)}")
        self.assertEqual(output[2], f"R2: {r2_score(data[train_size:], predictions)}")
        mape = mean_absolute_percentage_error(data[train_size:], predictions)
        self.assertEqual(output[3], f"MAPE: {mape:.2f}%")
        self.assertEqual(output[4], f"Theil U statistic : {best_theil_u:.2f}")


class TestGetOneYearData(unittest.TestCase):

    def setUp(self):
        # Generate a dummy dataset
        date_rng = pd.date_range(start='2020-01-01', end='2023-01-01', freq='D')
        df = pd.DataFrame(date_rng, columns=['date'])
        df['Close'] = np.random.randn(df.shape[0])
        df.set_index('date', inplace=True)
        self.dataset = df

        # Define other parameters for testing
        self.sequence_length = 5
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.dataset['Close'].values.reshape(-1, 1))
        self.model = MockModel()

    def test_get_one_year_data(self):
        one_year_data, one_year_data_2d, one_year_predictions = get_one_year_data(self.dataset, self.sequence_length,
                                                                                  self.scaler, self.model)

        # Assert that the returned data frame is no more than 1 year in length
        self.assertTrue((datetime.now() - one_year_data.index[-1]).days <= 365)

        # Assert the returned 2D data matches the original data's shape
        self.assertEqual(one_year_data_2d.shape[0], one_year_data.shape[0])

        # Assert that the predictions' shape matches the expected shape
        expected_rows = one_year_data_2d.shape[0] - self.sequence_length
        self.assertEqual(one_year_predictions.shape[0], expected_rows)


class TestGetOneYearDataArima(unittest.TestCase):

    def setUp(self):
        # Generate a dummy dataset
        date_rng = pd.date_range(start='2020-01-01', end='2023-01-01', freq='D')
        df = pd.DataFrame(date_rng, columns=['date'])
        df['Close'] = np.random.randn(df.shape[0])
        df.set_index('date', inplace=True)
        self.dataset = df

        # Define other parameters for testing
        self.scaler = MinMaxScaler()
        train_data = self.dataset[self.dataset.index < '2022-01-01']
        self.train_data_2d = train_data['Close'].values.reshape(-1, 1)
        self.scaler.fit(self.train_data_2d)

    def test_get_one_year_data_arima(self):
        one_year_data, one_year_data_scaled_2d, one_year_data_2d, one_year_predictions = get_one_year_data_arima(
            self.dataset, self.scaler, self.train_data_2d)

        # Assert that the returned data frame is no more than 1 year in length
        self.assertTrue((datetime.now() - one_year_data.index[-1]).days <= 365)

        # Assert the returned 2D data matches the original data's shape
        self.assertEqual(one_year_data_scaled_2d.shape, one_year_data_2d.shape)

        # Assert that the predictions' shape matches the expected shape
        self.assertEqual(one_year_predictions.shape[0], one_year_data_2d.shape[0])


class TestPrintTradingResult(unittest.TestCase):

    def setUp(self):
        # Generate a dummy dataset
        date_rng = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        df = pd.DataFrame(date_rng, columns=['date'])
        df['Close'] = np.linspace(10, 50, num=df.shape[0])  # Linearly increasing prices
        df.set_index('date', inplace=True)
        self.one_year_data = df

        self.sequence_length = 10

        # Actual data in 2D
        self.one_year_data_2d = df['Close'].values[:-self.sequence_length].reshape(-1, 1)

        # Simulate predictions that always expect an increase in prices
        self.one_year_predictions = df['Close'].values[self.sequence_length - 1:-1] + 1

    @contextlib.contextmanager
    def capture_stdout(self):
        new_stdout = StringIO()
        old_stdout = sys.stdout
        sys.stdout = new_stdout
        yield new_stdout
        sys.stdout = old_stdout

    def test_print_trading_result(self):
        with self.capture_stdout() as capturedOutput:
            # Call the function
            print_trading_result(self.one_year_data, self.one_year_data_2d, self.one_year_predictions,
                                 self.sequence_length)

        # Validate the captured output
        output = capturedOutput.getvalue().split("\n")
        self.assertTrue("Initial Balance: $10000" in output)

        # The exact profit or loss can vary depending on the trading strategy and predictions.
        # Thus, checking for a positive balance here since our predictions always expect an increase.
        final_balance = float(output[1].split(": $")[1])
        profit_or_loss = float(output[2].split(": $")[1])
        self.assertTrue(final_balance > 10000)
        self.assertTrue(profit_or_loss > 0)


if __name__ == '__main__':
    unittest.main()
