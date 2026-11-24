"""Known-answer checks independent of the frozen experiment outputs."""
import unittest

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import build_analysis as a


class PairedAnalysisTests(unittest.TestCase):
    def test_fixed_five_classes(self):
        cm = np.zeros((5, 5), int)
        cm[0, 0] = 20
        got = a.point_metrics(cm)
        self.assertEqual(got["macro_f1"], .2)
        self.assertEqual(got["accuracy"], 1)

    def test_mixed_window_confusion(self):
        cm = a.confusion([[30, 20, 0, 0, 0]], ["walk"])
        self.assertEqual(cm[0, 0], 30)
        self.assertEqual(cm[1, 0], 20)
        self.assertAlmostEqual(a.point_metrics(cm)["accuracy"], .6)

    def test_expanded_point_oracle(self):
        counts = np.array([[3, 2, 0, 0, 0], [0, 0, 4, 1, 0]])
        predictions = ["bike", "bus"]
        # Expand each window separately so mixed labels remain associated correctly.
        truth = np.concatenate([np.repeat(a.MODES, c) for c in counts])
        pred = np.repeat(predictions, counts.sum(axis=1))
        got = a.point_metrics(a.confusion(counts, predictions))
        self.assertAlmostEqual(got["accuracy"], accuracy_score(truth, pred))
        self.assertAlmostEqual(got["macro_f1"],
                               f1_score(truth, pred, labels=a.MODES, average="macro", zero_division=0))

    def test_flow_can_rescue_and_damage_in_same_window(self):
        got = a.point_flow([[30, 20, 0, 0, 0]], ["walk"], ["bike"])
        self.assertEqual(got["rescued_points"], 20)
        self.assertEqual(got["damaged_points"], 30)
        self.assertEqual(got["net_correct_points"], -10)
        self.assertEqual(got["changed_still_wrong_points"], 0)
        self.assertAlmostEqual(got["accuracy_delta_from_raw"], -.2)

    def test_unchanged_flow_has_no_rescue_or_damage(self):
        got = a.point_flow([[30, 20, 0, 0, 0]], ["walk"], ["walk"])
        self.assertEqual(got["rescued_points"], 0)
        self.assertEqual(got["damaged_points"], 0)
        self.assertEqual(got["changed_prediction_points"], 0)

    def test_changed_but_still_wrong(self):
        got = a.point_flow([[0, 0, 50, 0, 0]], ["walk"], ["bike"])
        self.assertEqual(got["changed_still_wrong_points"], 50)
        self.assertEqual(got["net_correct_points"], 0)

    def test_unknown_labels_rejected(self):
        with self.assertRaises(ValueError):
            a.confusion([[1, 0, 0, 0, 0]], ["other"])
        with self.assertRaises(ValueError):
            a.point_flow([[1, 0, 0, 0, 0]], ["walk"], ["other"])

    def test_seed_average_is_after_nonlinear_f1(self):
        first, second = np.zeros((5, 5), int), np.zeros((5, 5), int)
        first[0, 0], second[0, 1] = 1, 9
        correct = a.seed_mean([a.point_metrics(first)["macro_f1"],
                               a.point_metrics(second)["macro_f1"]])
        wrong = a.point_metrics((first + second) / 2)["macro_f1"]
        self.assertAlmostEqual(float(correct), .1)
        self.assertNotAlmostEqual(float(correct), float(wrong))

    def test_paired_deltas_keep_draw_alignment(self):
        np.testing.assert_allclose(a.paired_delta([1, 10, 100], [0, 9, 99]), [1, 1, 1])
        with self.assertRaises(ValueError):
            a.paired_delta([1, 2], [1])

    def test_boundary_micro_not_mean_user_ratios(self):
        got = a.boundary_metrics(101, 101, 100)
        self.assertAlmostEqual(float(got["transition_recall"]), 100 / 101)
        self.assertNotAlmostEqual(float(got["transition_recall"]), .5)

    def test_zero_boundary_denominators(self):
        got = a.boundary_metrics(np.array([0, 10]), [0, 20], [0, 5])
        np.testing.assert_allclose(got["transition_precision"], [0, .25])
        np.testing.assert_allclose(got["transition_recall"], [0, .5])
        np.testing.assert_allclose(got["transition_f1"], [0, 1 / 3])

    def test_sigma_zero_is_single_condition(self):
        self.assertEqual(a.seeds(0), (0,))
        self.assertEqual(len(a.seeds(20)), 5)


if __name__ == "__main__":
    unittest.main()
