"""Regression tests for feature engineering used by training and inference."""

import unittest

import pandas as pd

from src.preprocessing import DataPreprocessor


class TimeFeatureEngineeringTests(unittest.TestCase):
    def test_engineers_time_features_from_transaction_timestamp(self):
        dataframe = pd.DataFrame({"TransactionDT": [0, 23 * 3600, 24 * 3600]})

        result = DataPreprocessor().engineer_time_features(dataframe)

        self.assertEqual(result["hour_of_day"].tolist(), [0, 23, 0])
        self.assertEqual(result["day_of_week"].tolist(), [0, 0, 1])
        self.assertEqual(result["is_night_transaction"].tolist(), [1, 1, 1])

    def test_preserves_dashboard_time_features(self):
        dataframe = pd.DataFrame(
            {
                "TransactionDT": [0],
                "hour_of_day": [14],
                "day_of_week": [3],
                "is_night_transaction": [0],
            }
        )

        result = DataPreprocessor().engineer_time_features(dataframe)

        self.assertEqual(result.loc[0, "hour_of_day"], 14)
        self.assertEqual(result.loc[0, "day_of_week"], 3)
        self.assertEqual(result.loc[0, "is_night_transaction"], 0)
