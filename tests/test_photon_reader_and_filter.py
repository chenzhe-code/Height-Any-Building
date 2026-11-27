"""Exact I/O and frequency-filter regression against the frozen ATL03 code.

The old filter indexes its keep mask in the filtered subset's coordinate system.
These tests intentionally freeze that behavior; changing the scientific selection
is outside the performance-only change being tested here.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "tests/fixtures/photon_reference"
SOURCE = ROOT / "PLiDAR_processing/plidar_toolbox"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy_reader = load_module("photon_reader_before", FROZEN / "readers.py")
reader = load_module("photon_reader_current", SOURCE / "readers.py")
legacy_filter = load_module("photon_filter_before", FROZEN / "denoise.py")
frequency_filter = load_module("photon_filter_current", SOURCE / "denoise.py")


def beam_data(counts, begins=None, dtype=np.float32, photon_count=None):
    counts = np.asarray(counts, dtype=np.int32)
    if begins is None:
        begins = np.r_[0, np.cumsum(counts)[:-1]] + 1 if counts.size else np.array([], dtype=int)
    if photon_count is None:
        photon_count = int(counts.sum())
    index = np.arange(photon_count)
    confidence = np.full((photon_count, 5), 3, dtype=np.int8)
    confidence[::3, 0] = 1
    return {
        "heights": {
            "h_ph": (index * .25 + 60).astype(dtype),
            "dist_ph_along": (index * .125).astype(dtype),
            "dist_ph_across": (index * -.5).astype(dtype),
            "lat_ph": (30 + index * .001).astype(np.float64),
            "lon_ph": (114 + index * .001).astype(np.float64),
            "signal_conf_ph": confidence,
        },
        "geolocation": {
            "ph_index_beg": np.asarray(begins, dtype=np.int64),
            "segment_ph_cnt": counts,
            "segment_dist_x": (np.arange(counts.size) * 20).astype(dtype),
            "ref_elev": (np.arange(counts.size) * .01 + .5).astype(dtype),
        },
        "bckgrd_atlas": {},
    }


def photon_data(x, h):
    x, h = np.asarray(x), np.asarray(h)
    return {"x_act": x, "h": h, "photon_id": np.arange(x.size),
            "extra": np.column_stack((x, h)), "confidence": np.ones(x.size, dtype=np.int8)}


class ArrayComparisonMixin:
    def assert_mapping_equal(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for key in actual:
            if isinstance(actual[key], dict):
                self.assert_mapping_equal(actual[key], expected[key])
            else:
                self.assertEqual(actual[key].dtype, expected[key].dtype, key)
                # assert_array_equal also checks NaNs at matching positions.
                np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


class AlongTrackCoordinatesTests(ArrayComparisonMixin, unittest.TestCase):
    def compare(self, data):
        before, after = deepcopy(data), deepcopy(data)
        legacy_reader.add_along_track_coordinates(before)
        reader.add_along_track_coordinates(after)
        self.assert_mapping_equal(after, before)
        return after

    def test_regular_segments_and_dtypes(self):
        for dtype in (np.float32, np.float64, np.int32):
            for counts in ([2, 3, 1, 4], [2, 0, 3, 1], [0, 0, 0], [0, 3, 0]):
                with self.subTest(dtype=dtype, counts=counts):
                    result = self.compare(beam_data(counts, dtype=dtype))
                    self.assertEqual(result["heights"]["x_atc"].dtype, np.dtype("float64"))
                    self.assertEqual(result["heights"]["y_atc"].dtype, np.dtype("float64"))
                    self.assertEqual(result["geolocation"]["ref_elev_all"].dtype, np.dtype("float64"))

    def test_invalid_segments_and_last_reference_elevation(self):
        for begins in ([1, 0, 5, 7], [1, 0, 0, 7], [1, 3, 0, 7], [0, 3, 5, 7]):
            with self.subTest(begins=begins):
                self.compare(beam_data([2, 2, 2, 2], begins))
        result = self.compare(beam_data([2, 2, 2]))
        np.testing.assert_array_equal(result["geolocation"]["ref_elev_all"][-2:], [0., 0.])

    def test_no_or_one_segment(self):
        for counts in ([], [0], [3]):
            with self.subTest(counts=counts):
                result = self.compare(beam_data(counts))
                self.assertTrue(np.all(np.isnan(result["heights"]["x_atc"])))
                self.assertTrue(np.all(result["geolocation"]["ref_elev_all"] == 0))

    def test_noncontiguous_and_overlapping_segments_use_legacy_semantics(self):
        for begins in ([1, 3, 7], [1, 2, 3], [5, 1, 3]):
            with self.subTest(begins=begins):
                self.compare(beam_data([2, 2, 2], begins, photon_count=9))

    def test_nonfinite_distances_and_reference_values(self):
        data = beam_data([2, 3, 2])
        data["heights"]["dist_ph_along"][2] = np.nan
        data["geolocation"]["segment_dist_x"][1] = np.inf
        data["geolocation"]["ref_elev"][1] = np.nan
        self.compare(data)


class PhotonReaderTests(ArrayComparisonMixin, unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "ATL03_tiny.h5"
        self.bounds = {"min_lat": 30.002, "max_lat": 30.009,
                       "min_lon": 114.003, "max_lon": 114.010}
        self.write(beam_data([3, 3, 3, 3]))

    def write(self, data):
        with h5py.File(self.path, "w") as handle:
            handle.create_group("METADATA")
            for group, fields in data.items():
                target = handle.create_group("gt1l/" + group)
                for key, value in fields.items():
                    target.create_dataset(key, data=value)

    def compare_read(self, bounds=None, beam="gt1l", path=None):
        args = (str(path or self.path), beam, bounds or self.bounds)
        with redirect_stdout(io.StringIO()):
            expected = legacy_reader.read_plidar(*args)
            actual = reader.read_plidar(*args)
        self.assert_mapping_equal(actual, expected)
        return actual

    def test_geographic_confidence_and_valid_coordinate_filters(self):
        result = self.compare_read()
        np.testing.assert_array_equal(result["h"], [61., 61.25, 61.75, 62.])
        for min_lon, max_lon in ((114.008, 114.003), (None, 114.008), (114.003, None)):
            bounds = dict(self.bounds, min_lon=min_lon, max_lon=max_lon,
                          min_lat=30.009, max_lat=30.002)
            with self.subTest(bounds=bounds):
                self.compare_read(bounds)
        data = beam_data([3, 3, 3, 3], begins=[1, 0, 7, 10])
        self.write(data)
        self.compare_read()

    def test_empty_geographic_selection(self):
        self.compare_read(dict(self.bounds, min_lat=50., max_lat=51.))

    def test_missing_file_beam_groups_and_required_datasets(self):
        self.compare_read(path=Path(self.directory.name) / "absent*.h5")
        self.compare_read(beam="gt3r")
        for missing in ("heights", "geolocation", "heights/h_ph", "geolocation/ref_elev"):
            self.write(beam_data([3, 3, 3, 3]))
            with h5py.File(self.path, "a") as handle:
                del handle["gt1l/" + missing]
            with self.subTest(missing=missing):
                self.compare_read()

    def test_default_beam_read_preserves_extra_datasets(self):
        data = beam_data([3, 3, 3])
        data["heights"]["unused_extra"] = np.arange(18).reshape(9, 2)
        data["geolocation"]["unused_extra"] = np.arange(3)
        data["bckgrd_atlas"]["unused_extra"] = np.arange(23)
        self.write(data)
        self.assert_mapping_equal(reader.read_atl03_beam(str(self.path), "gt1l"),
                                  legacy_reader.read_atl03_beam(str(self.path), "gt1l"))
        reduced = reader.read_atl03_beam(str(self.path), "gt1l", required_only=True)
        for group in reduced.values():
            self.assertNotIn("unused_extra", group)
        self.assertEqual(set(reduced["heights"]), set(beam_data([3, 3, 3])["heights"]))
        self.assertEqual(set(reduced["geolocation"]), set(beam_data([3, 3, 3])["geolocation"]))
        self.compare_read()

    def test_optional_background_group_absent(self):
        with h5py.File(self.path, "a") as handle:
            del handle["gt1l/bckgrd_atlas"]
        self.assert_mapping_equal(reader.read_atl03_beam(str(self.path), "gt1l"),
                                  legacy_reader.read_atl03_beam(str(self.path), "gt1l"))
        self.compare_read()


class ElevationFrequencyTests(ArrayComparisonMixin, unittest.TestCase):
    def compare(self, photons, start, end, bins=10):
        original = deepcopy(photons)
        with redirect_stdout(io.StringIO()):
            expected = legacy_filter.reduce_noise_by_elevation_frequency(photons, start, end, bins)
            actual = frequency_filter.reduce_noise_by_elevation_frequency(photons, start, end, bins)
        self.assert_mapping_equal(actual, expected)
        self.assert_mapping_equal(photons, original)
        return actual

    def test_zero_and_one_photon_passthrough(self):
        for x in ([], [20.]):
            result = self.compare(photon_data(np.asarray(x), np.zeros(len(x))), 0., 10.)
            self.assertEqual(result["x_act"].size, len(x))

    def test_duplicates_boundaries_partial_windows_and_subset_indexing(self):
        for dtype in (np.float32, np.float64, np.int32):
            x = np.asarray([-4, 0, 0, 1, 2, 2, 3, 4, 5, 6, 6, 7, 9], dtype=dtype)
            h = np.asarray([50, 0, 0, 0, 1, 4, 3, 5, 5, 6, 7, 6, 80], dtype=dtype)
            for start, end in ((0, 6), (.5, 6.5), (-5, 10), (20, 25), (-10, -5)):
                for order in (np.arange(len(x)), np.random.RandomState(12).permutation(len(x))):
                    with self.subTest(dtype=dtype, bounds=(start, end), order=order):
                        self.compare(photon_data(x[order], h[order]), start, end)
        result = self.compare(photon_data(np.array([-2., 0., 1., 2., 4.]), np.zeros(5)), 0., 4.)
        # Freeze the pre-existing subset-relative keep-mask indexing explicitly.
        np.testing.assert_array_equal(result["x_act"], [0., 1.])

    def test_randomized_bin_ties_and_shuffled_photons(self):
        for seed in range(30):
            rng = np.random.RandomState(seed)
            x = rng.randint(-8, 30, 80).astype(np.float64) / 2
            h = rng.randint(0, 12, len(x)).astype(np.float32) / 3
            with self.subTest(seed=seed):
                self.compare(photon_data(x, h), .25, 11.75, bins=(seed % 5 + 1))


if __name__ == "__main__":
    unittest.main()
