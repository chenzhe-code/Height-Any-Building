"""Bin normalization must preserve frozen DBSCAN's clusters and input order."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = ROOT / "tests/fixtures/photon_reference"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = load_module("photon_bins_before", OLD_ROOT / "photon_features.py")
current = load_module("photon_bins_current", ROOT / "PLiDAR_processing/plidar_toolbox/photon_features.py")


class PhotonBinTests(unittest.TestCase):
    def compare_labels(self, h):
        expected = DBSCAN(eps=.5, min_samples=2).fit(h.reshape(-1, 1)).labels_
        np.testing.assert_array_equal(expected, current._bin_elevation_labels(h))

    def compare_bins(self, x, h, determine_num, bin_size=1.):
        data = dict(x_act=x, h=h)
        expected = legacy.bin_and_normalize(data, bin_size, determine_num)
        actual = current.bin_and_normalize(data, bin_size, determine_num)
        self.assertEqual(list(expected), list(actual), "bin insertion order changed")
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key], err_msg=str(key))
            self.assertEqual([type(item) for item in expected[key]],
                             [type(item) for item in actual[key]])

    def test_random_components_and_bin_insertion_order(self):
        for dtype in (np.float32, np.float64):
            for seed in range(10):
                rng = np.random.RandomState(seed)
                h = (rng.randint(0, 12, 160) * 2 + rng.normal(0, .07, 160)).astype(dtype)
                x = rng.randint(0, 15, len(h)).astype(float)
                with self.subTest(dtype=dtype, seed=seed):
                    self.compare_labels(h)
                    self.compare_bins(x, h, .02)
                    self.compare_bins(x, h, .005)

    def test_component_order_matches_first_encounter(self):
        h = np.array([8., 1., 5., 8.1, 1.1, 5.1, 20., 8.2, 5.2])
        self.compare_labels(h)
        self.compare_bins(np.zeros(len(h)), h, .02)
        self.compare_bins(np.zeros(len(h)), h, .005)

    def test_singletons_duplicates_and_center_selection(self):
        for n in range(1, 13):
            h = np.r_[np.zeros(n), np.full(n + 1, 3.), [19.]]
            with self.subTest(n=n):
                self.compare_labels(h)
                self.compare_bins(np.zeros(len(h)), h, .005)

    def test_eps_nextafter_and_large_offsets(self):
        for dtype in (np.float32, np.float64):
            for offset in (0., 100., 10000., 1e8):
                middle = dtype(offset + .5)
                for edge in (np.nextafter(middle, dtype(-np.inf)), middle,
                             np.nextafter(middle, dtype(np.inf))):
                    h = np.array([offset, edge, offset + 3, offset + 3.1], dtype=dtype)
                    with self.subTest(dtype=dtype, offset=offset, edge=edge):
                        self.compare_labels(h)
                        self.compare_bins(np.zeros(len(h)), h, .005)

    def test_dense_bridging_and_float32_threshold(self):
        for dtype in (np.float32, np.float64):
            for n in (5, 30, 100):
                h = (100 + np.arange(n) * .49).astype(dtype)[::-1]
                with self.subTest(dtype=dtype, n=n):
                    self.compare_labels(h)
                    self.compare_bins(np.zeros(len(h)), h, .005)

    def test_borderline_unsupported_and_nonfinite_fall_back(self):
        for h in (np.array([100., 100.5]), np.array([1., np.nan]),
                  np.array([1., np.inf]), np.array([1, 2], dtype=np.int64),
                  np.array([1e8, 1e8 + .7])):
            with self.subTest(h=h):
                with patch.object(current, "DBSCAN", wraps=DBSCAN) as constructor:
                    try:
                        expected = DBSCAN(eps=.5, min_samples=2).fit(h.reshape(-1, 1)).labels_
                    except ValueError:
                        with self.assertRaises(ValueError):
                            current._bin_elevation_labels(h)
                    else:
                        np.testing.assert_array_equal(expected, current._bin_elevation_labels(h))
                    constructor.assert_called_once_with(eps=.5, min_samples=2)

    def test_empty_input_still_raises(self):
        for module in (legacy, current):
            with self.assertRaises(ValueError):
                module.bin_and_normalize(dict(x_act=np.array([]), h=np.array([])), 1., .1)


if __name__ == "__main__":
    unittest.main()
