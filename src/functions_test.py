import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
from sklearn.preprocessing import MinMaxScaler

from src.functions import load_data, clean_data, split_data, create_dataset

INVALID_TICKER_ERROR = "Invalid ticker. Please enter a valid ticker"  # Assuming you have a constant for this


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


if __name__ == '__main__':
    unittest.main()
