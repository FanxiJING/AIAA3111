import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/model_training"))
from window_controlled import smooth_points


class TestPhysicalSmoothing(unittest.TestCase):
    def test_matches_brute_force_in_actual_time_and_respects_blocks(self):
        seconds = np.array([0, 1, 30, 200, 201, 202, 0, 2])
        members = pd.DataFrame({"user_id": ["001"] * 8, "feature_block_id": ["a"] * 6 + ["b"] * 2,
                                "timestamp": pd.Timestamp("2020-01-01") + pd.to_timedelta(seconds, unit="s"),
                                "window_index": [0, 0, 1, 2, 2, 3, 4, 4]})
        scores = np.random.default_rng(42).normal(size=(5, 5))
        actual = smooth_points(members, scores, 60)
        expected = []
        for i in range(len(members)):
            same = members.feature_block_id.eq(members.feature_block_id.iloc[i]).to_numpy()
            inside = same & (np.abs(seconds - seconds[i]) <= 30)
            expected.append(scores[members.window_index.to_numpy()[inside]].mean(axis=0).argmax())
        np.testing.assert_array_equal(actual, expected)

    def test_duplicate_timestamps_and_negative_spans_rejected(self):
        d = pd.DataFrame({"user_id": ["001"] * 2, "feature_block_id": ["a"] * 2,
                          "timestamp": pd.to_datetime(["2020-01-01", "2020-01-01"]), "window_index": [0, 0]})
        with self.assertRaises(ValueError):
            smooth_points(d, np.ones((1, 5)), 1200)
        with self.assertRaises(ValueError):
            smooth_points(d, np.ones((1, 5)), -1)


if __name__ == "__main__":
    unittest.main()
