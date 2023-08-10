import unittest
from unittest.mock import patch
import pandas as pd

# Assuming your function is in a module named 'data_loader'
from functions import load_data

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


if __name__ == '__main__':
    unittest.main()
