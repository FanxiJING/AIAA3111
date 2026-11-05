import unittest

import pandas as pd

from data_preprocessing.label_matching import match_labels


class LabelMatchingTests(unittest.TestCase):
    def test_closed_boundaries_gaps_nested_and_conflicting_intervals(self):
        origin = pd.Timestamp("2008-01-01")
        labels = pd.DataFrame({
            "start": [origin + pd.Timedelta(seconds=n) for n in [0, 2, 10]],
            "end": [origin + pd.Timedelta(seconds=n) for n in [10, 4, 20]],
            "mode": ["walk", "walk", "bus"],
        })
        timestamps = pd.Series([origin + pd.Timedelta(seconds=n) for n in [21, 0, 3, 5, 10, 11, 20]] + [pd.NaT])
        actual = match_labels(timestamps, labels)
        self.assertEqual(actual.label_status.tolist(), ["unlabeled", "matched", "matched", "matched", "conflict", "matched", "matched", "invalid_timestamp"])
        self.assertEqual(actual.label_match_count.tolist(), [0, 1, 2, 1, 2, 1, 1, 0])
        self.assertEqual(actual["mode"].fillna("NONE").tolist(), ["NONE", "walk", "walk", "walk", "NONE", "bus", "bus", "NONE"])


if __name__ == "__main__":
    unittest.main()
