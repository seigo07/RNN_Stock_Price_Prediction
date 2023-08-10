import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np

# Assuming your function is in a module named 'data_loader'
from src.functions import load_data, clean_data

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


if __name__ == '__main__':
    unittest.main()
