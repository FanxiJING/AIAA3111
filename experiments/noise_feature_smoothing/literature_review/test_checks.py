"""Known-answer checks for the new descriptive calculations."""
import unittest

import numpy as np

from build_checks import constant_references, metrics, point_confusion, stationary_reference


class CheckTests(unittest.TestCase):
    def test_stationary_scale(self):
        mean, rms, _ = stationary_reference(5, np.array([1., 2., 5.]), .5)
        np.testing.assert_allclose(mean, 5 * np.sqrt(np.pi) / np.array([1, 2, 5]))
        np.testing.assert_allclose(rms, [10, 5, 2])

    def test_threshold_probability(self):
        _, _, exceed = stationary_reference(5, 2, 2)
        self.assertAlmostEqual(float(exceed), np.exp(-.16))

    def test_zero_noise_limit(self):
        for value in stationary_reference(0, [1, 2], .5):
            np.testing.assert_array_equal(value, [0, 0])

    def test_invalid_support(self):
        for args in ((5, 0, .5), (-1, 1, .5), (5, 1, -1)):
            with self.assertRaises(ValueError):
                stationary_reference(*args)

    def test_mixed_label_points(self):
        cm = point_confusion([[3, 2, 0, 0, 0], [0, 0, 4, 0, 0]], ["bike", "bus"])
        np.testing.assert_array_equal(cm[:, 1], [3, 2, 0, 0, 0])
        np.testing.assert_array_equal(cm[:, 2], [0, 0, 4, 0, 0])
        self.assertAlmostEqual(metrics(cm)[1], 6 / 9)

    def test_constant_reference_fixed_five_classes(self):
        rows = constant_references([[3, 1, 0, 0, 0]])
        self.assertAlmostEqual(rows[0]["accuracy"], .75)
        self.assertAlmostEqual(rows[0]["macro_f1"], (6 / 7) / 5)
        self.assertEqual(rows[-1]["macro_f1"], 0)

    def test_differenced_iid_errors_are_dependent(self):
        a, b = np.array([-1, 1, 0]), np.array([0, -1, 1])
        self.assertEqual(a @ a, 2)
        self.assertEqual(a @ b, -1)
        self.assertAlmostEqual((a @ b) / np.sqrt((a @ a) * (b @ b)), -.5)

    def test_coordinate_stencil_variances(self):
        self.assertEqual(np.array([1, -2, 1]) @ np.array([1, -2, 1]), 6)
        self.assertEqual(np.array([-1, 3, -3, 1]) @ np.array([-1, 3, -3, 1]), 20)

    def test_unknown_labels_rejected(self):
        with self.assertRaises(ValueError):
            point_confusion([[1, 0, 0, 0, 0]], ["subway"])


if __name__ == "__main__":
    unittest.main()
