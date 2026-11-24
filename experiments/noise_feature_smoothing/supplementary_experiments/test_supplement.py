"""Known-answer and invariance tests, independent of the new model outcomes."""
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from run_supplement import (
    MODES, SEEDS, confusion, displace, ou_field, ou_from_innovations,
    point_metrics, trailing_soft,
)


class SupplementalTests(unittest.TestCase):
    def test_exact_irregular_recurrence_and_block_reset(self):
        z = np.array([[1., -1.], [2., 3.], [-2., 4.], [5., 2.], [9., 8.]])
        t = np.array([0., 2., 100., 0., 3.])
        block = np.array(["a", "a", "a", "b", "b"])
        q = ou_from_innovations(z, t, block)
        np.testing.assert_array_equal(q[[0, 3]], z[[0, 3]])
        for i, dt in [(1, 2.), (2, 98.), (4, 3.)]:
            a = np.exp(-dt / 30)
            np.testing.assert_allclose(q[i], a * q[i - 1] + np.sqrt(1 - a*a) * z[i])
        changed = z.copy()
        changed[:3] *= 1000
        np.testing.assert_array_equal(
            ou_from_innovations(changed, t, block)[3:], q[3:])

    def test_stationary_marginal_and_covariance(self):
        # Independent synthetic blocks provide independent Monte Carlo replications.
        n = 30000
        z = np.random.default_rng(123).standard_normal((n * 3, 2))
        t = np.tile([0., 2., 30.], n)
        q = ou_from_innovations(z, t, np.repeat(np.arange(n), 3)).reshape(n, 3, 2)
        np.testing.assert_allclose(q.mean(axis=0), 0, atol=.025)
        np.testing.assert_allclose(q.var(axis=0), 1, atol=.04)
        for a, b, dt in [(0, 1, 2), (0, 2, 30), (1, 2, 28)]:
            np.testing.assert_allclose((q[:, a] * q[:, b]).mean(axis=0),
                                       np.exp(-dt / 30), atol=.03)
        self.assertLess(abs(np.mean(q[:, :, 0] * q[:, :, 1])), .03)

    def test_actual_seconds_and_seed_pairing(self):
        p = pd.DataFrame({
            "user_id": ["052"] * 4, "feature_block_id": ["a"] * 3 + ["b"],
            "timestamp": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 00:00:02",
                                         "2020-01-01 00:00:08", "2020-02-01 00:00:00"]),
            "latitude": [30.] * 4, "longitude": [114.] * 4,
        })
        q = ou_field(p, SEEDS[0])
        p_micro = p.copy()
        p_micro.timestamp = p.timestamp.astype("datetime64[us]")
        np.testing.assert_array_equal(ou_field(p_micro, SEEDS[0]), q)
        p_nano = p.copy()
        p_nano.timestamp = p.timestamp.astype("datetime64[ns]")
        np.testing.assert_array_equal(ou_field(p_nano, SEEDS[0]), q)
        rng = np.random.default_rng(np.random.SeedSequence([SEEDS[0], 52]))
        z = rng.standard_normal((4, 2))
        np.testing.assert_array_equal(q[[0, 3]], z[[0, 3]])
        np.testing.assert_array_equal(q * 20, (q * 5) * 4)
        pd.testing.assert_frame_equal(displace(p, q, 0), p)
        self.assertFalse(np.array_equal(ou_field(p, SEEDS[1]), q))

    def test_duplicate_and_invalid_time(self):
        z = np.array([[1., 2.], [9., 8.]])
        q = ou_from_innovations(z, [0, 0], ["a", "a"])
        np.testing.assert_array_equal(q[0], q[1])
        with self.assertRaises(AssertionError):
            ou_from_innovations(z, [0, -1], ["a", "a"])

    def test_mixed_window_point_scoring(self):
        counts = np.array([[3, 1, 0, 0, 0], [0, 0, 2, 1, 0], [0, 0, 0, 0, 2]])
        pred = np.array(["bike", "bus", "walk"])
        y = np.concatenate([np.repeat(MODES, c) for c in counts])
        p = np.concatenate([np.repeat(label, row.sum()) for row, label in zip(counts, pred)])
        cm = confusion(counts, pred)
        np.testing.assert_array_equal(cm, confusion_matrix(y, p, labels=MODES))
        self.assertAlmostEqual(float(point_metrics(cm)["macro_f1"]),
                               f1_score(y, p, labels=MODES, average="macro", zero_division=0))
        # A class absent in truth/prediction contributes zero, not a changed denominator.
        pure = np.diag([4, 4, 0, 0, 0])
        self.assertAlmostEqual(float(point_metrics(pure)["macro_f1"]), .4)

    def test_loo_confusion_additivity(self):
        a = np.array([[2, 1, 0, 0, 0], [0, 1, 1, 0, 0]])
        b = np.array([[0, 0, 0, 4, 0], [0, 0, 0, 1, 2]])
        pa, pb = ["walk", "bike"], ["car", "train"]
        full = confusion(np.vstack([a, b]), pa + pb)
        np.testing.assert_array_equal(full - confusion(a, pa), confusion(b, pb))

    def test_trailing_has_no_future_or_cross_block_information(self):
        rng = np.random.default_rng(321)
        scores = rng.standard_normal((30, 5))
        blocks = np.repeat(["a", "b"], 15)
        times = pd.date_range("2020-01-01", periods=30, freq="min")
        pred = trailing_soft(scores, blocks, times)
        changed = scores.copy()
        changed[10:] *= -1000
        np.testing.assert_array_equal(trailing_soft(changed, blocks, times)[:10], pred[:10])
        changed = scores.copy()
        changed[:15] += 1000
        np.testing.assert_array_equal(trailing_soft(changed, blocks, times)[15:], pred[15:])
        np.testing.assert_array_equal(pred[[0, 15]],
                                     np.asarray(MODES)[scores[[0, 15]].argmax(axis=1)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
