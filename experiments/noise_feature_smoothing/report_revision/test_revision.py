"""Known answers for reporting transformations with mixed window truth."""
import unittest

import numpy as np
import pandas as pd

from build_revision import boundary_support, summarize, unique_join
from verify_revision import point_scores


class ReportingTests(unittest.TestCase):
    def make_windows(self, labels, seconds, block="a"):
        return pd.DataFrame({
            "user_id": ["052"] * len(labels), "feature_block_id": [block] * len(labels),
            "label": labels,
            "start_time": pd.Timestamp("2020-01-01") + pd.to_timedelta(seconds, unit="s")})

    def test_irregular_boundary_tolerance(self):
        out = boundary_support(self.make_windows(["walk", "bike", "bike", "walk"],
                                                [0, 10, 100, 140]))
        np.testing.assert_array_equal(out.early_tolerance_s, [0, 40])
        np.testing.assert_array_equal(out.late_tolerance_s, [90, 0])
        np.testing.assert_array_equal(out.full_candidate_span_s, [90, 40])

    def test_no_boundary_across_blocks(self):
        left = self.make_windows(["walk"], [0], "a")
        right = self.make_windows(["bike"], [100], "b")
        self.assertEqual(len(boundary_support(pd.concat([left, right], ignore_index=True))), 0)

    def test_timestamp_resolution_does_not_change_seconds(self):
        for unit in ("us", "ns"):
            with self.subTest(unit=unit):
                windows = self.make_windows(["walk", "walk", "bike", "bike"],
                                            [0, 10, 112, 212])
                windows["start_time"] = windows.start_time.dt.as_unit(unit)
                self.assertEqual(str(windows.start_time.dtype), f"datetime64[{unit}]")
                out = boundary_support(windows)
                np.testing.assert_array_equal(out.early_tolerance_s, [102])
                np.testing.assert_array_equal(out.late_tolerance_s, [100])
                np.testing.assert_array_equal(out.full_candidate_span_s, [202])

    def test_two_window_block_has_exact_position_only(self):
        out = boundary_support(self.make_windows(["walk", "bike"], [0, 123]))
        self.assertEqual(out.eligible_boundary_positions.iloc[0], 1)
        self.assertEqual(out.full_candidate_span_s.iloc[0], 0)

    def test_mixed_truth_is_point_weighted(self):
        f1, accuracy, classes = point_scores([[3, 2, 0, 0, 0], [0, 1, 0, 0, 0]],
                                             ["walk", "bike"])
        self.assertAlmostEqual(accuracy, 4 / 6)
        np.testing.assert_allclose(classes, [.75, .5, 0, 0, 0])
        self.assertAlmostEqual(f1, .25)

    def test_unknown_predictions_rejected(self):
        with self.assertRaises(ValueError):
            point_scores([[1, 0, 0, 0, 0]], ["taxi"])

    def test_duplicate_or_missing_join_rejected(self):
        with self.assertRaises(ValueError):
            unique_join(pd.DataFrame({"id": [1, 1]}), pd.DataFrame({"id": [1]}), ["id"])
        with self.assertRaises(ValueError):
            unique_join(pd.DataFrame({"id": [1]}), pd.DataFrame({"id": [2]}), ["id"])

    def make_metric_rows(self):
        rows = []
        metrics = ["macro_f1", "accuracy", "transition_precision", "transition_recall",
                   "transition_f1", "wrong_isolated_flips_per100_windows",
                   "prediction_change_from_clean_point_fraction",
                   "f1_walk", "f1_bike", "f1_bus", "f1_car", "f1_train"]
        for sigma, seed, value in [(0, 0, .5), *[(20, i, i / 10) for i in range(1, 6)]]:
            rows.append({"group": "G1", "smoother": "S0_raw_k1", "sigma": sigma,
                         "condition": f"{sigma}_{seed}", "seed": seed,
                         **{metric: value for metric in metrics}})
        return pd.DataFrame(rows)

    def test_sd_is_sample_sd_and_clean_n_one(self):
        out = summarize(self.make_metric_rows()).set_index("sigma")
        self.assertEqual(out.loc[0, "seed_count"], 1)
        self.assertEqual(out.loc[0, "macro_f1_seed_sd"], 0)
        self.assertIn("not_applicable", out.loc[0, "seed_sd_interpretation"])
        self.assertAlmostEqual(out.loc[20, "macro_f1_seed_mean"], .3)
        self.assertAlmostEqual(out.loc[20, "macro_f1_seed_sd"], np.sqrt(.025))

    def test_incomplete_seed_conditions_rejected(self):
        with self.assertRaises(ValueError):
            summarize(self.make_metric_rows().iloc[:-1])


if __name__ == "__main__":
    unittest.main()
