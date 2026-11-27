"""Closed-form checks for the weighted two-node height graph."""
import contextlib
import io
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.rh2ah_converter import RH2AHConverter


class GraphWeightTests(unittest.TestCase):
    def solve(self, overlap_lambda, common=1.0):
        images = [np.ones((1, 1)), np.ones((1, 1))]
        constraints = [(0, -1, 1.0, 1.0), (1, -1, 1.0, 3.0), (0, 1, 1.0, 1.0)]
        with contextlib.redirect_stdout(io.StringIO()):
            result = RH2AHConverter.joint_RHCH_scale(
                images, constraints, np.asarray([1.0, 1.0, overlap_lambda]) * common)
        return np.asarray([image.item() for image in result])

    def test_squared_error_coefficient_matches_analytic_minimum(self):
        for coefficient in (0.0, 0.25, 1.0, 4.0):
            expected = np.asarray([1 + 4 * coefficient, 3 + 4 * coefficient]) / (1 + 2 * coefficient)
            np.testing.assert_allclose(self.solve(coefficient), expected, rtol=1e-6, atol=1e-6)

    def test_common_scaling_preserves_the_solution(self):
        np.testing.assert_allclose(self.solve(0.25), self.solve(0.25, common=7), rtol=1e-6, atol=1e-6)

    def test_invalid_weights_are_rejected(self):
        for weights in ([1.0], [-1.0, 1.0], [float('nan'), 1.0], [0.0, 0.0]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                RH2AHConverter.joint_RHCH_scale([np.ones((1, 1))], [(0, -1, 1, 1)] * 2, weights)

    def test_nonground_mask_uses_common_positions(self):
        images = [np.asarray([[0., 0., 1., 10.]] * 2), np.asarray([[2., 20., 0., 0.]] * 2)]
        positions = [(0, 0), (2, 0)]
        all_pixels = RH2AHConverter.construct_overlaps(images, positions, 4, 0.5)
        nonground = RH2AHConverter.construct_overlaps(images, positions, 4, 0.5, valid_mask_mode="nonground")
        self.assertEqual(all_pixels[0][2:], (5.5, 11.0))
        self.assertEqual(nonground[0][2:], (10.0, 20.0))


if __name__ == '__main__':
    unittest.main()
