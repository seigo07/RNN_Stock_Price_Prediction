import contextlib
import unittest
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
import sys
from io import StringIO
from datetime import datetime
from unittest.mock import patch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from src.functions import load_data, clean_data, split_data, create_dataset, build_model, time_series_cross_validation, get_arima_predictions, theil_u_statistic, print_metrics, print_metrics_arima, get_one_year_data, get_one_year_data_arima, print_trading_result, print_trading_result_arima, plot_trading_result, plot_trading_result_arima
from keras.layers import LSTM, GRU, Dense

# Define error messages for invalid ticker
INVALID_TICKER_ERROR = "Invalid ticker. Please enter a valid ticker"


class MockModel:
    """
    A mock model class to simulate predictions.
    This will just return the input for simplicity.
    """
    def predict(self, data):
        return data


class TestLoadData(unittest.TestCase):

    # Mocking the `yfinance.download` function allows us to test the `load_data` function
    # without actually making a request to the Yahoo Finance API.

    # Test to ensure that the function behaves correctly when provided with a valid ticker.
    @patch('yfinance.download')  # We use patch to mock the yfinance.download function
    def test_load_data_valid_ticker(self, mock_download):
        # Mock data to simulate the response we would get from Yahoo Finance for a valid ticker.
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'Close': [101, 102, 103],
        })
        # Here we specify the return value of the mocked function when it is called.
        mock_download.return_value = mock_data

        # Now we call the actual `load_data` function we want to test.
        result = load_data("AAPL")

        # We then check if the returned data from our `load_data` function matches the mocked data.
        self.assertEqual(len(result), 3)  # Check if we got the mocked data

    # Test to ensure that the function behaves correctly when provided with an invalid ticker.
    @patch('yfinance.download')  # Again, we mock the yfinance.download function
    def test_load_data_invalid_ticker(self, mock_download):
        # This time, we simulate the scenario where an invalid ticker returns an empty DataFrame.
        mock_download.return_value = pd.DataFrame()

        # We expect our function to raise a SystemExit exception for invalid tickers.
        # Hence, we use `assertRaises` to check if this exception is indeed raised.
        with self.assertRaises(SystemExit) as cm:  # Since you are exiting for invalid tickers
            load_data("INVALID")

        # Finally, we check if the exception code matches the expected error code for invalid tickers.
        self.assertEqual(cm.exception.code, INVALID_TICKER_ERROR)


class TestCleanData(unittest.TestCase):

    # Setup method to create a mock dataset that will be used in the tests.
    def setUp(self):
        # Create a mock dataset with some missing data and duplicate dates for testing.
        self.dataset = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'Close': [101, 102, np.nan, 105, 110],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }, index=pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-03', '2021-01-05']))

    # Test to check if the NaN values in the 'Close' column have been removed after cleaning.
    def test_remove_nan_values(self):
        cleaned_data, _ = clean_data(self.dataset)
        # Ensure that there are no NaN values in the 'Close' column after cleaning.
        self.assertFalse(cleaned_data['Close'].isnull().any())

    # Test to check if duplicate dates are removed after cleaning.
    def test_remove_duplicate_dates(self):
        cleaned_data, _ = clean_data(self.dataset)
        # Assert that no dates are duplicated in the index after cleaning.
        self.assertEqual(cleaned_data.index.duplicated().sum(), 0)

    # Test to ensure that 'Close' values are within a valid range after cleaning.
    # Specifically, this checks if values are below the 99th percentile and non-negative.
    def test_clip_close_values(self):
        cleaned_data, _ = clean_data(self.dataset)
        # Check if all 'Close' values are below or equal to the 99th percentile of the original 'Close' values.
        self.assertTrue((cleaned_data['Close'] <= np.percentile(self.dataset['Close'].dropna(), 99)).all())
        # Check if all 'Close' values are non-negative.
        self.assertTrue((cleaned_data['Close'] >= 0).all())

    # Test to check the shape of the reshaped 'Close' values to ensure they've been converted to 2D.
    def test_reshape_close_values(self):
        _, data = clean_data(self.dataset)
        # Check if the shape of the cleaned 'Close' data is as expected (4 rows and 1 column).
        self.assertEqual(data.shape, (4, 1))


class TestSplitData(unittest.TestCase):

    # Setup method to initialize a mock dataset that will be used in the tests.
    def setUp(self):
        # Creating a simple dataset with 10 consecutive numbers for testing purposes.
        self.data = np.array([[100], [101], [102], [103], [104], [105], [106], [107], [108], [109]])

    # Test to ensure the split ratio of the dataset into training and testing sets.
    def test_split_ratio(self):
        train_data, test_data, _ = split_data(self.data)
        # Ensure that 80% of the data is used for training.
        self.assertEqual(len(train_data), 8)
        # Ensure that 20% of the data is used for testing.
        self.assertEqual(len(test_data), 2)

    # Test to check if the training and testing datasets are normalized properly.
    def test_normalization(self):
        train_data, test_data, _ = split_data(self.data)
        # Ensure all training data values lie between 0 and 1 (inclusive).
        self.assertTrue((0 <= train_data).all() and (train_data <= 1).all())

        # Obtain min and max values of training and testing subsets for scaling checks.
        min_train = np.min(self.data[:8])
        max_train = np.max(self.data[:8])
        min_test = np.min(self.data[8:])
        max_test = np.max(self.data[8:])

        # Check if train data is correctly scaled.
        self.assertAlmostEqual(train_data[0], 0, delta=1e-10)  # Scaled minimum value in training set.
        self.assertAlmostEqual(train_data[-1], (max_train - min_train) / (max_train - min_train),
                               delta=1e-10)  # Scaled last value in training set.

        # Check if test data is correctly scaled using the same scale as the training data.
        self.assertAlmostEqual(test_data[0], (min_test - min_train) / (max_train - min_train),
                               delta=1e-10)  # Scaled first value in testing set.
        self.assertAlmostEqual(test_data[-1], (max_test - min_train) / (max_train - min_train),
                               delta=1e-10)  # Scaled last value in testing set.

    # Test to ensure the same scaler is used for both training and testing data.
    def test_using_same_scaler(self):
        _, _, scaler = split_data(self.data)
        # Manually transform the last two data points using the returned scaler.
        manual_test_data = scaler.transform(self.data[-2:])
        _, test_data, _ = split_data(self.data)
        # Ensure the manual scaling result matches with the test data.
        self.assertTrue(np.array_equal(manual_test_data, test_data))

    # Test to confirm the type of scaler returned by the function is MinMaxScaler.
    def test_return_scaler_type(self):
        _, _, scaler = split_data(self.data)
        self.assertIsInstance(scaler, MinMaxScaler)


class TestCreateDataset(unittest.TestCase):

    # Setup method to initialize a mock dataset that will be used in the tests.
    def setUp(self):
        # Creating a simple dataset with 10 consecutive numbers for testing purposes.
        self.data = np.array([[100], [101], [102], [103], [104], [105], [106], [107], [108], [109]])

    # Test to ensure the created dataset (X and Y) have the correct lengths.
    def test_dataset_length(self):
        time_steps = 3
        data_X, data_Y = create_dataset(self.data, time_steps)

        # After creating sequences of 'time_steps' length, the expected remaining data points will be:
        # total_data_points - time_steps
        expected_length = len(self.data) - time_steps
        self.assertEqual(len(data_X), expected_length)
        self.assertEqual(len(data_Y), expected_length)

    # Test to verify if each sequence in the dataset X has the specified 'time_steps' length.
    def test_sequence_length(self):
        time_steps = 3
        data_X, _ = create_dataset(self.data, time_steps)

        # Ensure each sequence in the dataset X has the correct length.
        for sequence in data_X:
            self.assertEqual(len(sequence), time_steps)

    # Test to ensure that the sequences in dataset X and the corresponding target values in Y are correct.
    def test_correct_sequence_values(self):
        time_steps = 3
        data_X, data_Y = create_dataset(self.data, time_steps)

        # Verify each sequence in dataset X and the corresponding target value in Y.
        for i in range(len(data_X)):
            # Ensure the current sequence in dataset X matches the corresponding values from the original data.
            np.testing.assert_array_equal(data_X[i], self.data[i:i + time_steps].flatten())
            # Verify that the target value for the current sequence matches the expected value from the original data.
            self.assertEqual(data_Y[i], self.data[i + time_steps][0])


class TestBuildModel(unittest.TestCase):

    # Setup method for initializing the parameters that will be used in the tests.
    def setUp(self):
        # TensorFlow module, which will be used for testing the model's structure.
        self.tf = tf
        # Specified length of the input sequence.
        self.sequence_length = 5
        # Specified learning rate for the optimizer.
        self.lr = 0.001

    # Test to ensure that when "LSTM" is passed as the algorithm type, the correct model structure is built.
    def test_lstm_model(self):
        model = build_model("LSTM", self.sequence_length, self.tf, self.lr)

        # Ensure the first layer is an LSTM layer.
        self.assertIsInstance(model.layers[0], LSTM)
        # Ensure the input shape to the LSTM layer is correct.
        self.assertEqual(model.layers[0].input_shape, (None, self.sequence_length, 1))
        # Validate the learning rate of the model's optimizer.
        self.assertAlmostEqual(model.optimizer.learning_rate.numpy(), self.lr, places=10)
        # Ensure the next layer after LSTM is a dense layer and verify its number of units.
        self.assertIsInstance(model.layers[1], Dense)
        self.assertEqual(model.layers[1].units, 1)

    # Test to ensure that when "GRU" is passed as the algorithm type, the correct model structure is built.
    def test_gru_model(self):
        model = build_model("GRU", self.sequence_length, self.tf, self.lr)

        # Ensure the first layer is a GRU layer.
        self.assertIsInstance(model.layers[0], GRU)
        # Ensure the input shape to the GRU layer is correct.
        self.assertEqual(model.layers[0].input_shape, (None, self.sequence_length, 1))
        # Validate the learning rate of the model's optimizer.
        self.assertAlmostEqual(model.optimizer.learning_rate.numpy(), self.lr, places=10)
        # Ensure the next layer after GRU is a dense layer and verify its number of units.
        self.assertIsInstance(model.layers[1], Dense)
        self.assertEqual(model.layers[1].units, 1)

    # Test the case where an invalid algorithm type is passed. The default should be "LSTM".
    def test_invalid_algorithm(self):
        model = build_model("INVALID", self.sequence_length, self.tf, self.lr)

        # Even if the algorithm type is invalid, the default layer should be LSTM.
        self.assertIsInstance(model.layers[0], LSTM)


class TestTimeSeriesCrossValidation(unittest.TestCase):

    def setUp(self):
        # To make the output cleaner, we suppress all warnings that might arise during the testing process.
        warnings.filterwarnings("ignore")

        # Create a 2D data array with sequential numbers ranging from 0 to 99.
        self.data = np.arange(100).reshape(-1, 1)
        # Number of splits for the cross-validation.
        self.n_splits = 5
        # Define the order of the ARIMA model for the time series cross-validation.
        self.model_order = (1, 0, 1)

    def test_cross_validation_output(self):
        # Call the time_series_cross_validation function with given data, number of splits, and model order.
        result = time_series_cross_validation(self.data, self.n_splits, self.model_order)

        # Ensure that the output of the cross-validation function (i.e., the Theil U statistic) is a float.
        self.assertIsInstance(result, float)

        # The Theil U statistic is a measure of forecasting accuracy and is always non-negative.
        # Here, we check if the computed value is indeed greater than or equal to 0.
        self.assertGreaterEqual(result, 0)


class TestArimaPredictions(unittest.TestCase):

    def setUp(self):
        # To ensure the output remains clean during tests, suppress any potential warnings.
        warnings.filterwarnings("ignore")

        # Generate mock datasets for training and testing. The train dataset consists of the integers from 0 to 9,
        # while the test dataset consists of the integers from 10 to 14.
        self.train_data = np.array([x for x in range(10)])
        self.test_data = np.array([x for x in range(10, 15)])

        # Utilize the MinMaxScaler to scale the data between 0 and 1. This helps in normalizing the data which can
        # potentially improve the performance of the ARIMA model.
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.train_data_2d = self.scaler.fit_transform(self.train_data.reshape(-1, 1))
        self.test_data_2d = self.scaler.transform(self.test_data.reshape(-1, 1))

        # For demonstration purposes, mock ARIMA parameters are defined.
        # Note that these may not be optimal for the mock data.
        self.best_p = 1
        self.best_d = 1
        self.best_q = 1

    def test_get_arima_predictions(self):
        # Use the provided ARIMA function to generate predictions on the test data using the trained ARIMA model.
        predictions = get_arima_predictions(self.scaler, self.train_data_2d, self.test_data_2d, self.best_p, self.best_d, self.best_q)

        # Confirm that the shape of the predictions matches the test data length.
        self.assertEqual(predictions.shape, (len(self.test_data),))

        # In actual tests, the values of `predictions` would ideally be compared to known expected results.
        # Given ARIMA's inherent variability and approximation, direct comparisons for mock data can be challenging.
        # It's often more practical to ensure that predictions follow data trends or to check prediction performance metrics.


class TestTheilUStatistic(unittest.TestCase):

    def test_theil_u_statistic(self):
        # Define mock data for the test. This includes the actual values, predicted values, and naive forecast.
        actual = np.array([3, 2, 4, 5, 6])
        predicted = np.array([2.8, 2.1, 3.9, 5.2, 6.1])
        naive = np.array([3, 3, 2, 4, 5])

        # Calculate the expected Theil U statistic for the mock data:
        # First, compute the Mean Squared Error (MSE) of the actual values and the naive forecast.
        # Then, compute the MSE of the actual values and the predicted values.
        # The Theil U statistic is the square root of the ratio of these two MSE values.
        mse_actual = mean_squared_error(actual, naive)
        mse_predicted = mean_squared_error(actual, predicted)
        expected_theil_u = np.sqrt(mse_predicted / mse_actual)

        # Use the provided function to compute the Theil U statistic.
        computed_theil_u = theil_u_statistic(actual, predicted, naive)

        # Compare the computed Theil U statistic to the expected value.
        # Since this is a float comparison, we use `assertAlmostEqual` to account for potential minor discrepancies.
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


class TestPrintTradingResultARIMA(unittest.TestCase):

    def setUp(self):
        # Generate a dummy dataset
        date_rng = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        df = pd.DataFrame(date_rng, columns=['date'])
        df['Close'] = np.linspace(10, 50, num=df.shape[0])  # Linearly increasing prices
        df.set_index('date', inplace=True)
        self.one_year_data = df

        # Actual data in 2D
        self.one_year_2d = df['Close'].values.reshape(-1, 1)

        # Simulate ARIMA predictions that always expect an increase in prices
        self.one_year_predictions = df['Close'].values[:-1] + 1

    @contextlib.contextmanager
    def capture_stdout(self):
        new_stdout = StringIO()
        old_stdout = sys.stdout
        sys.stdout = new_stdout
        yield new_stdout
        sys.stdout = old_stdout

    def test_print_trading_result_arima(self):
        with self.capture_stdout() as capturedOutput:
            # Call the function
            print_trading_result_arima(self.one_year_data, self.one_year_2d, self.one_year_predictions)

        # Validate the captured output
        output = capturedOutput.getvalue().split("\n")
        self.assertTrue("Initial Balance: $10000" in output)

        # The exact profit or loss can vary depending on the trading strategy and predictions.
        # Thus, checking for a positive balance here since our predictions always expect an increase.
        final_balance = float(output[1].split(": $")[1])
        profit_or_loss = float(output[2].split(": $")[1])
        self.assertTrue(final_balance > 10000)
        self.assertTrue(profit_or_loss > 0)


class TestPlotTradingResult(unittest.TestCase):

    def setUp(self):
        # Generate a dummy dataset
        date_rng = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        df = pd.DataFrame(date_rng, columns=['date'])
        df['Close'] = np.linspace(10, 50, num=df.shape[0])  # Linearly increasing prices
        df.set_index('date', inplace=True)
        self.one_year_data = df

        # Actual data in 2D
        self.one_year_data_2d = df['Close'].values.reshape(-1, 1)

        # Simulate ARIMA predictions that always expect an increase in prices
        # Deduct sequence_length from the predictions length to match the x and y dimensions when plotting
        sequence_length = 5
        self.one_year_predictions = (df['Close'].values + 1)[sequence_length:]

    @patch("matplotlib.pyplot.show")
    def test_plot_trading_result(self, mock_show):
        # Call the function
        plot_trading_result('LSTM', 'AAPL', 5, self.one_year_data, self.one_year_data_2d, self.one_year_predictions)

        # Check that show was called once (plot was generated)
        mock_show.assert_called_once()

        # Check if the title of the plot is correct
        title = plt.gca().get_title()
        self.assertEqual(title, 'AAPL Stock Price Predictions with Buy/Sell Points (Last One Year)')

        # Check if the plot contains the right labels
        labels = [t.get_text() for t in plt.gca().get_legend().get_texts()]
        self.assertIn('Actual Stock Price', labels)
        self.assertIn('LSTM Predicted Stock Price', labels)
        self.assertIn('Buy', labels)  # because our predicted price is always higher


class TestPlotTradingResultARIMA(unittest.TestCase):

    def setUp(self):
        # Generate a dummy dataset
        date_rng = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        df = pd.DataFrame(date_rng, columns=['date'])
        df['Close'] = np.linspace(10, 50, num=df.shape[0])  # Linearly increasing prices
        df.set_index('date', inplace=True)
        self.one_year_data = df

        # Actual data in 2D
        self.one_year_data_2d = df['Close'].values.reshape(-1, 1)

        # Simulate ARIMA predictions that always expect an increase in prices
        self.one_year_predictions = df['Close'].values + 1

    def test_plot_trading_result_arima(self):
        # Suppress the actual plot during testing
        plt.ioff()

        # Run the plotting function with mock data
        try:
            plot_trading_result_arima("ARIMA", "AAPL", self.one_year_data, self.one_year_data_2d, self.one_year_predictions)
            result = True
        except Exception as e:
            print(e)  # Print exception for debugging
            result = False

        # Ensure that no errors were encountered during execution
        self.assertTrue(result)

        # Resume normal plotting after test
        plt.ion()


if __name__ == '__main__':
    unittest.main()
