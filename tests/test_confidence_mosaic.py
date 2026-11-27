"""Numerical and GeoTIFF regression checks for confidence-weighted mosaicking."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import io_utils


def legacy_reference(patches, uncertainties, positions, shape, use_conf=True):
    """Small independent reference: visit pixels and contributors in input order."""
    result = np.zeros(shape, dtype=np.float64)
    for row in range(shape[0]):
        for col in range(shape[1]):
            values, uncertainty = [], []
            for index, ((x, y), patch) in enumerate(zip(positions, patches)):
                if y <= row < y + patch.shape[0] and x <= col < x + patch.shape[1]:
                    values.append(patch[row - y, col - x])
                    if use_conf:
                        uncertainty.append(uncertainties[index][row - y, col - x])
            if values:
                if use_conf:
                    uncertainty = np.stack(uncertainty)
                    weights = softmax(1 - uncertainty / (np.sum(uncertainty) + 1e-5))
                    result[row, col] = np.sum(weights * np.stack(values), axis=0)
                else:
                    result[row, col] = sum(values) / len(values)
    return result


class ConfidenceMosaicTests(unittest.TestCase):
    def make_inputs(self, root, uncertainty_dtype, include_uncertainty=True):
        # Irregular final overlaps create up to nine contributors, then a detached
        # short rectangular patch leaves uncovered pixels in the full extent.
        positions = [(6, 3), (0, 0), (4, 0), (6, 0), (0, 3),
                     (4, 3), (0, 4), (4, 4), (6, 4), (19, 13)]
        shapes = [(5, 7)] * 9 + [(3, 4)]
        rng = np.random.RandomState(812)
        patches = [rng.uniform(-5.2, 50.7, size=shape).astype(np.float32)
                   for shape in shapes]
        uncertainties = []
        for index, shape in enumerate(shapes):
            if np.dtype(uncertainty_dtype).kind in 'ui':
                uncertainty = rng.randint(0, 256, size=shape).astype(uncertainty_dtype)
            else:
                uncertainty = rng.uniform(0.01, 8.5, size=shape).astype(uncertainty_dtype)
            uncertainty[0, 0] = 0
            if index == 1:
                uncertainty[:] = 0
            uncertainties.append(uncertainty)
        paths, metadata = [], []
        (root / 'depth').mkdir()
        if include_uncertainty:
            (root / 'uncertainty').mkdir()
        for index, (patch, uncertainty, (x, y)) in enumerate(zip(patches, uncertainties, positions)):
            # Filename origins deliberately differ from zero; relative positions
            # above are normalized only for the independent reference.
            path = root / 'depth' / f'window_{index}_x{x + 100}_y{y + 200}_depth.tif'
            meta = dict(driver='GTiff', width=patch.shape[1], height=patch.shape[0],
                        count=1, dtype='float32', crs=rasterio.crs.CRS.from_epsg(32650),
                        transform=from_origin(600000 + x * 2, 3400000 - y * 3, 2, 3))
            with rasterio.open(path, 'w', **meta) as target:
                target.write(patch, 1)
            if include_uncertainty:
                conf_path = root / 'uncertainty' / path.name.replace('depth', 'uncertainty')
                with rasterio.open(conf_path, 'w', **dict(meta, dtype=np.dtype(uncertainty_dtype).name)) as target:
                    target.write(uncertainty, 1)
            paths.append(str(path))
            metadata.append(meta)
        return patches, uncertainties, positions, paths, metadata, (18, 26)

    def check_public_mosaic(self, uncertainty_dtype, save_tif=True, use_conf=True):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patches, uncertainty, positions, paths, metadata, shape = self.make_inputs(
                root, uncertainty_dtype, include_uncertainty=use_conf)
            metadata_before = copy.deepcopy(metadata)
            expected = legacy_reference(patches, uncertainty, positions, shape, use_conf)
            actual, full_meta = io_utils.mosaic_patches(
                patches, metadata, paths, (5, 7), str(root / 'results'), 'mosaic',
                show_grayscale=True, use_conf=use_conf, save_tif=save_tif)
            tolerance = 1e-6 if np.dtype(uncertainty_dtype) == np.dtype('float32') else 1e-12
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
            self.assertEqual(metadata, metadata_before, 'Input metadata must not be modified')
            self.assertEqual(actual.dtype, np.float64)
            self.assertEqual(actual.shape, shape)
            self.assertEqual(actual[11, 15], 0, 'Uncovered pixels retain zero')
            self.assertEqual(full_meta['count'], 3)
            self.assertEqual(full_meta['dtype'], 'int16')
            self.assertEqual(full_meta['crs'], rasterio.crs.CRS.from_epsg(32650))
            self.assertEqual(full_meta['transform'], from_origin(600000, 3400000, 2, 3))
            self.assertEqual((full_meta['height'], full_meta['width']), shape)
            output_dir = root / 'results' / 'mosaic'
            if save_tif:
                for path, patch, meta in zip(paths, patches, metadata):
                    with rasterio.open(output_dir / Path(path).name) as saved:
                        self.assertEqual(saved.count, 3)
                        self.assertEqual(saved.dtypes, ('int16',) * 3)
                        self.assertEqual(saved.crs, meta['crs'])
                        self.assertEqual(saved.transform, meta['transform'])
                        np.testing.assert_array_equal(saved.read(), np.repeat(patch.astype(np.int16)[None], 3, axis=0))
                with rasterio.open(output_dir / 'output.tif') as saved:
                    self.assertEqual(saved.count, 3)
                    self.assertEqual(saved.dtypes, ('int16',) * 3)
                    self.assertEqual(saved.crs, full_meta['crs'])
                    self.assertEqual(saved.transform, full_meta['transform'])
                    np.testing.assert_array_equal(saved.read(), np.repeat(expected.astype(np.int16)[None], 3, axis=0))
            else:
                self.assertEqual(list(output_dir.iterdir()), [])
            return patches, uncertainty, positions, shape, expected

    def test_uint8_irregular_overlaps_zero_uncertainty_and_saved_geotiffs(self):
        self.check_public_mosaic(np.uint8)

    def test_float32_uncertainty(self):
        self.check_public_mosaic(np.float32)

    def test_float64_uncertainty(self):
        self.check_public_mosaic(np.float64)

    def test_save_tif_false(self):
        self.check_public_mosaic(np.uint8, save_tif=False)

    def test_mean_fusion_does_not_require_uncertainty_files(self):
        self.check_public_mosaic(np.float64, use_conf=False)

    def test_confidence_favors_lower_uncertainty(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patches, uncertainty, positions, paths, metadata, shape = self.make_inputs(root, np.uint8)
            # All patches contribute equal-size, constant-valued pixels here;
            # uncertainty is the only reason the result differs from the mean.
            for index in range(len(patches)):
                patches[index][:] = index * 10 + 0.25
                uncertainty[index][:] = 0 if index == 1 else 200
                conf_path = root / 'uncertainty' / Path(paths[index]).name.replace('depth', 'uncertainty')
                with rasterio.open(conf_path, 'r+') as target:
                    target.write(uncertainty[index], 1)
            actual, _ = io_utils.mosaic_patches(patches, metadata, paths, (5, 7),
                str(root / 'results'), 'mosaic', save_tif=False)
            expected = legacy_reference(patches, uncertainty, positions, shape)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
            ordinary_mean = legacy_reference(patches, uncertainty, positions, shape, False)
            self.assertLess(actual[3, 6], ordinary_mean[3, 6])

    def test_chunk_boundaries_match_pixel_reference(self):
        helper = getattr(io_utils, '_confidence_mosaic', None)
        if helper is None:
            self.skipTest('Private chunked implementation not available')
        with tempfile.TemporaryDirectory() as temporary:
            patches, uncertainty, positions, _, _, shape = self.make_inputs(Path(temporary), np.float64)
            expected = legacy_reference(patches, uncertainty, positions, shape)
            for block_size in (1, 2, 5, 256):
                with self.subTest(block_size=block_size):
                    actual = helper(patches, uncertainty, positions, shape, block_size=block_size)
                    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
