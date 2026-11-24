"""Targeted independent tests for post-hoc diagnostics, not model experiments."""
import itertools
import unittest

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import analyze as a


class DiagnosticTests(unittest.TestCase):
    def test_sampling_bins(self):
        np.testing.assert_array_equal(
            a.sampling_bins([np.nan, 0, -1, .5, 1, 1.01, 2, 2.01, 5, 5.01, np.inf]),
            ["invalid_dt", "invalid_dt", "invalid_dt", "<=1s", "<=1s",
             "1-2s", "1-2s", "2-5s", "2-5s", ">5s", "invalid_dt"])

    def test_boundary_cardinality_exhaustively(self):
        for pmask, tmask in itertools.product(range(16), repeat=2):
            pred = np.r_[0, np.cumsum([pmask >> i & 1 for i in range(4)])]
            truth = np.r_[0, np.cumsum([tmask >> i & 1 for i in range(4)])]
            pi, ti, pairs = a.boundary_matches(pred, truth)
            edges = np.abs(pi[:, None] - ti[None, :]) <= 1
            r, c = linear_sum_assignment(-edges.astype(int))
            self.assertEqual(len(pairs), int(edges[r, c].sum()))
            self.assertEqual(len({p for p, _ in pairs}), len(pairs))
            self.assertEqual(len({t for _, t in pairs}), len(pairs))
            self.assertTrue(all(abs(p - t) <= 1 for p, t in pairs))

    def test_support_edges_and_causality(self):
        for n in (1, 2, 5, 11, 19):
            for smoother in a.SMOOTHERS:
                lo, hi = a.support_indices(n, smoother)
                self.assertTrue(np.all(lo >= 0) and np.all(hi < n))
                self.assertTrue(np.all(lo <= np.arange(n)))
                self.assertTrue(np.all(hi >= np.arange(n)))
                self.assertTrue(np.all(hi - lo + 1 <= 11))
                if smoother == a.SMOOTHERS[2]:
                    np.testing.assert_array_equal(hi, np.arange(n))

    def test_fixed_five_class_bootstrap(self):
        cm = np.zeros((2, 5, 5), dtype=int)
        cm[0, 0, 0] = 100
        cm[1, 1, 2] = 100
        macro, acc = a.bootstrap_metrics(cm)
        np.testing.assert_allclose(macro, [.2, 0])
        np.testing.assert_allclose(acc, [1, 0])

    def test_mixed_windows_are_not_majority_weighted(self):
        meta = pd.DataFrame({"cnt_walk": [30], "cnt_bike": [20],
                             "cnt_bus": [0], "cnt_car": [0], "cnt_train": [0]})
        cm, metrics = a.class_metrics(meta, ["walk"])
        self.assertEqual(cm[0, 0], 30)
        self.assertEqual(cm[1, 0], 20)
        self.assertAlmostEqual(metrics["accuracy"], .6)

    def test_run_slices_and_bins(self):
        self.assertEqual(a.run_slices(["a", "a", "b", "a"]), [(0, 2), (2, 3), (3, 4)])
        self.assertEqual([a.length_bin(n) for n in [1, 2, 5, 6, 10, 11]],
                         ["1", "2-5", "2-5", "6-10", "6-10", ">=11"])

    def test_absence_denominators(self):
        rows = []
        for raw_present, raw_all, absent in [(True, True, True), (True, False, False),
                                             (False, False, True)]:
            rows.append(dict(sigma=0, seed=0, condition="clean", smoother="s",
                             length_bin="1", duration_bin="<=60s", mode="walk",
                             linked_group="g", raw_present=raw_present,
                             raw_all_correct=raw_all, absent=absent,
                             newly_absent_from_raw=raw_present and absent,
                             all_raw_correct_then_absent=raw_all and absent,
                             window_majority_agreement=float(not absent)))
        got = a.aggregate_runs(pd.DataFrame(rows))
        row = got[got.stratum_type.eq("all")].iloc[0]
        self.assertEqual(row.runs, 3)
        self.assertAlmostEqual(row.absent_run_fraction, 2 / 3)
        self.assertAlmostEqual(row.newly_absent_given_raw_present_fraction, .5)
        self.assertAlmostEqual(row.absent_given_raw_all_correct_fraction, 1)


if __name__ == "__main__":
    unittest.main()
