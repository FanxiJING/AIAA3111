import unittest

import numpy as np
import pandas as pd

from report import replay_confusion
from run import COUNT_COLS, KEEP_COLS, dense_evaluation, match_events, prefix_soft, thin_indices, time_windows


class ExperimentTests(unittest.TestCase):
    def test_thinning_keeps_first_and_respects_elapsed_time(self):
        np.testing.assert_array_equal(thin_indices(np.array([0, 1, 16, 17, 29, 31]), 15), [0, 2, 5])

    def test_real_time_windows_do_not_cross_blocks(self):
        times = pd.Timestamp("2008-01-01") + pd.to_timedelta([0, 1, 120, 121, 180], unit="s")
        data = pd.DataFrame({"timestamp": times, "feature_block_id": ["a", "a", "a", "a", "b"],
                             "mode": ["walk", "bike", "bike", "bike", "bike"]})
        np.testing.assert_array_equal(time_windows(data, 120, False), [0, 0, 1, 1, 2])
        np.testing.assert_array_equal(time_windows(data, 120, True), [0, 1, 1, 2, 3])

    def test_dense_denominator_is_the_original_labeled_stream(self):
        times = pd.Timestamp("2008-01-01") + pd.to_timedelta([0, 10, 150, 160, 170], unit="s")
        data = pd.DataFrame({"timestamp": times, "user_id": "007", "feature_block_id": "a",
                             "mode": ["walk", "bike", "car", "car", "train"]})
        retained = data.iloc[[0, 2]].reset_index(drop=True)
        windows = pd.DataFrame({"start_time": retained.timestamp, "end_time": retained.timestamp,
                                **{name: 0 for name in COUNT_COLS}})
        dense = dense_evaluation(data, retained, windows, np.array([0, 1]), "t120")
        np.testing.assert_array_equal(dense[COUNT_COLS], [[1, 1, 0, 0, 0], [0, 0, 0, 2, 1]])
        self.assertEqual(dense[KEEP_COLS].to_numpy().sum(), 0)
        self.assertEqual(dense.release_s[0] - times[0].timestamp(), 120)

    def test_trailing_soft_is_prefix_invariant(self):
        scores = np.array([[2, 0], [0, 1], [0, 10], [0, 100]])
        ready = np.array([0, 50, 50, 171])
        for policy in ("raw", "past3", "past120s"):
            full = prefix_soft(scores, ready, policy)
            for n in range(1, len(ready) + 1):
                np.testing.assert_array_equal(full[:n], prefix_soft(scores[:n], ready[:n], policy))
        np.testing.assert_array_equal(prefix_soft(scores[:3], ready[:3], "past120s"), [0, 0, 1])

    def test_boundary_matching_is_one_to_one_and_inclusive(self):
        self.assertEqual(match_events([0, 10], [-4, 5, 6, 30], tolerance=5), [-4, -5])
        self.assertEqual(match_events([0], [6], tolerance=5), [])

    def test_replay_abstains_until_first_release_and_uses_latest_at_ties(self):
        matrix = replay_confusion(
            np.array([0, 10, 19, 20, 21]), np.array([0, 0, 1, 2, 2]),
            np.array([10, 20, 20]), np.array([0, 1, 2]),
        )
        self.assertEqual(matrix.sum(), 4)
        self.assertEqual(matrix[0, 0], 1)
        self.assertEqual(matrix[1, 0], 1)
        self.assertEqual(matrix[2, 2], 2)


if __name__ == "__main__":
    unittest.main()
