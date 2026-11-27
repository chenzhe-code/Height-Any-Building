"""Exact regression against the frozen pre-optimization photon implementation."""
from pathlib import Path
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.neighbors import KDTree

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tests/fixtures/photon_reference/ground_building.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = load_module("photon_region_growth_before", BASELINE)
current = load_module("photon_region_growth_current", ROOT / "PLiDAR_processing/plidar_toolbox/ground_building.py")


def data(x, h):
    x, h = np.asarray(x), np.asarray(h)
    return dict(x_act=x, h=h, photon_id=np.arange(x.size),
                lon=x * 0.01, extra=np.column_stack((x, h)), metadata="ignored")


class PhotonRegionGrowthTests(unittest.TestCase):
    def compare(self, function, photons, seeds, **kwargs):
        traversals = []
        results = []
        for module in (legacy, current):
            traversal = []

            class TracedTree:
                def __init__(self, *args, **kw):
                    self.tree = KDTree(*args, **kw)

                def query_radius(self, point, radius):
                    traversal.append(np.asarray(point).copy())
                    return self.tree.query_radius(point, radius)

            with patch.object(module, "KDTree", TracedTree):
                results.append(getattr(module, function)(photons, seeds, **kwargs))
            traversals.append(traversal)
        self.assertEqual(results[0].keys(), results[1].keys())
        for key in results[0]:
            np.testing.assert_array_equal(results[0][key], results[1][key], err_msg=key)
        self.assertEqual(len(traversals[0]), len(traversals[1]))
        for before, after in zip(*traversals):
            np.testing.assert_array_equal(before, after, err_msg="FIFO traversal changed")

    def test_dense_sparse_and_repeated_positions(self):
        for dtype in (np.float32, np.float64):
            for spacing in (0.02, 0.5, 10.0):
                rng = np.random.RandomState(412)
                x = np.repeat(np.arange(80) * spacing, 3).astype(dtype)
                h = (rng.normal(0, 0.02, x.size) + np.sin(x / 20)).astype(dtype)
                photons = data(x, h)
                for indices in (np.array([], dtype=int), np.arange(0, x.size, 17)):
                    with self.subTest(dtype=dtype, spacing=spacing, count=indices.size):
                        self.compare("grow_region_from_seed", photons,
                                     data(x[indices], h[indices]),
                                     max_height_diff=0.2, max_slope=0.1, search_radius=3)

    def test_height_slope_distance_boundaries(self):
        x = np.array([0., 1e-7, 1e-6, 1., 2., 3., 4., 5.] * 6)
        h = np.array([0., 0., 0., .1, .2, .3, .4, .6] * 6)
        photons = data(x, h)
        for height, slope, radius in ((.1, .1, 3), (0., 0., 5),
                                      (1., np.nan, 5), (np.nan, .1, 3)):
            with self.subTest(height=height, slope=slope):
                self.compare("grow_region_from_seed", photons, data(x[:1], h[:1]),
                             max_height_diff=height, max_slope=slope, search_radius=radius)

    def test_randomized_ties_and_multiple_components(self):
        for seed in range(15):
            rng = np.random.RandomState(seed)
            x = rng.randint(0, 60, 150).astype(float)
            h = rng.randint(0, 10, 150).astype(float) / 10
            indices = rng.choice(len(x), 7, replace=False)
            with self.subTest(seed=seed):
                self.compare("grow_region_from_seed", data(x, h), data(x[indices], h[indices]),
                             max_height_diff=.3, max_slope=.1, search_radius=15)

    def test_refine_ground_and_building_keeps_duplicates(self):
        rng = np.random.RandomState(1234)
        x = np.repeat(np.arange(120) * .2, 2)
        h = .001 * x + rng.normal(0, .005, x.size)
        h[120:] += 8
        chosen = np.arange(0, len(x), 6)
        self.compare("refine_ground_and_building_candidates", data(x, h),
                     data(x[chosen], h[chosen]), search_radius=10)

    def test_nan_and_infinite_seed_semantics(self):
        photons = data(np.arange(80, dtype=float), np.zeros(80))
        for seed_x in ([3., np.nan], [3., np.inf], [np.inf], [-np.inf, 3.], []):
            with self.subTest(seed_x=seed_x):
                with np.errstate(invalid="ignore"):
                    self.compare("grow_region_from_seed", photons,
                                 data(np.asarray(seed_x), np.zeros(len(seed_x))))

    def test_invalid_point_coordinates_still_fail(self):
        for invalid in (np.nan, np.inf):
            photons = data(np.array([0., invalid]), np.zeros(2))
            for module in (legacy, current):
                with self.subTest(invalid=invalid, module=module.__name__):
                    with self.assertRaises(ValueError):
                        module.grow_region_from_seed(photons, data(np.array([0.]), np.array([0.])))

    def test_integer_input_retains_subtraction_behavior(self):
        photons = data(np.arange(50, dtype=np.int64), np.zeros(50, dtype=np.int64))
        self.compare("grow_region_from_seed", photons, data(np.array([0, 20]), np.array([0, 0])))

    def test_mixed_float_precision_retains_scalar_promotion(self):
        x = np.array([1. + 1e-9, 2., 3.], dtype=np.float64)
        self.compare("grow_region_from_seed", data(x, np.zeros(3)),
                     data(np.array([1.], dtype=np.float32), np.zeros(1)), search_radius=.5)

    def test_float32_rejection_thresholds_keep_scalar_precision(self):
        cases = ((1., .1, .1, 1.), (1e-6, 0., 1., 1.), (1., .1, 1., .1))
        for dx, dz, max_height, max_slope in cases:
            x = np.array([0.] + [dx] * 79, dtype=np.float32)
            h = np.array([0.] + [dz] * 79, dtype=np.float32)
            with self.subTest(dx=dx, dz=dz, max_height=max_height, max_slope=max_slope):
                self.compare("grow_region_from_seed", data(x, h), data(x[:1], h[:1]),
                             max_height_diff=max_height, max_slope=max_slope, search_radius=100)


if __name__ == "__main__":
    unittest.main()
