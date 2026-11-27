from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))


from tools.assign_building_heights import extract_height_from_geotiff, match_polygons_with_height
from merge_city_height_results import (
    UINT16_NODATA,
    merge_buildings,
    merge_height_raster,
    validate_city_outputs,
    validate_height_tiles,
)
from validate_patch_grid import validate_patch_grid
from utils.rh2ah_converter import RH2AHConverter
from utils.io_utils import read_geotiff as read_height
from utils.raster_io import save_geotiff
from utils.split_image import read_geotiff as read_image, split_image
from utils.window_grid import axis_window_starts, window_positions


class WindowGridTests(unittest.TestCase):
    def test_geotiff_readers_preserve_image_and_height_band_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raster = Path(directory) / "input.tif"
            pixels = np.arange(18, dtype=np.uint8).reshape(3, 2, 3)
            profile = dict(driver="GTiff", width=3, height=2, count=3, dtype="uint8",
                           crs="EPSG:32650", transform=from_origin(400000, 3400000, 1, 1))
            save_geotiff(raster, pixels, profile)

            image, image_metadata = read_image(raster)
            height, height_metadata = read_height(raster)
            np.testing.assert_array_equal(image, pixels)
            np.testing.assert_array_equal(height, pixels[0])
            self.assertEqual(image_metadata, height_metadata)
            self.assertEqual(image_metadata["transform"], profile["transform"])

    def test_expected_axis_starts_and_complete_coverage(self) -> None:
        expected = {
            1024: [0],
            1200: [0, 176],
            1536: [0, 512],
            5000: [0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 3976],
        }
        for size, starts in expected.items():
            with self.subTest(size=size):
                actual = axis_window_starts(size, 1024, 0.5)
                self.assertEqual(actual, starts)
                covered = np.zeros(size, dtype=bool)
                for start in actual:
                    covered[start:start + 1024] = True
                self.assertTrue(covered.all())
                self.assertEqual(len(actual), len(set(actual)))

    def test_5000_grid_is_9_by_9(self) -> None:
        positions = window_positions(5000, 5000, 1024, 0.5)
        self.assertEqual(len(positions), 81)
        self.assertEqual(positions[-1], (3976, 3976))
        self.assertIn((3584, 3976), positions)
        self.assertEqual(3976 - 3584, 392)
        self.assertEqual(1024 - (3976 - 3584), 632)

    def test_irregular_last_overlap_remains_connected(self) -> None:
        images = [
            np.full((8, 8), 1.0),
            np.full((8, 8), 2.0),
            np.full((8, 8), 3.0),
        ]
        overlaps = RH2AHConverter.construct_overlaps(
            images, [(0, 0), (4, 0), (7, 0)], 8, 0.5
        )
        self.assertEqual([(item[0], item[1]) for item in overlaps], [(0, 1), (1, 2)])
        self.assertEqual((overlaps[-1][2], overlaps[-1][3]), (2.0, 3.0))


class RasterRegressionTests(unittest.TestCase):
    def test_split_and_validate_full_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = root / "source.tif"
            patch_dir = root / "patches"
            data = np.zeros((3, 1200, 1200), dtype=np.uint8)
            with rasterio.open(
                source_path,
                "w",
                driver="GTiff",
                width=1200,
                height=1200,
                count=3,
                dtype="uint8",
                crs="EPSG:32640",
                transform=from_origin(300000, 2800000, 1, 1),
            ) as target:
                target.write(data)

            split_image(str(source_path), str(patch_dir), 1024, 0.5, 1)
            summary = validate_patch_grid(source_path, patch_dir, 1024, 0.5, 1)
            self.assertEqual(summary["patch_count"], 4)
            self.assertEqual(summary["covered_width"], 1200)
            self.assertEqual(summary["covered_height"], 1200)

    def test_bottom_right_building_receives_height(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raster_path = Path(tmp) / "height.tif"
            values = np.arange(100, dtype=np.float32).reshape(10, 10)
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                width=10,
                height=10,
                count=1,
                dtype="float32",
                crs="EPSG:32640",
                transform=from_origin(0, 10, 1, 1),
            ) as target:
                target.write(values, 1)
            height = extract_height_from_geotiff(raster_path, box(8, 0, 10, 2))
            self.assertIsNotNone(height)
            self.assertEqual(height, 99.0)

    def test_assign_heights_reuses_raster_and_preserves_column_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, raster, result = (root / name for name in ("input.geojson", "height.tif", "result.geojson"))
            values = np.array([[[5, 8, 12]], [[105, 108, 112]]], dtype="uint16")
            profile = dict(driver="GTiff", width=3, height=1, count=2, dtype="uint16",
                           crs="EPSG:32640", transform=from_origin(300000, 2800000, 1, 1))
            save_geotiff(raster, values, profile)
            polygons = [box(300000, 2799999, 300001, 2800000), box(300001, 2799999, 300002, 2800000)]
            gpd.GeoDataFrame({"Offset_X": [1, 1]}, geometry=polygons, crs=profile["crs"]).to_file(source)
            for apply_offset, custom, column, expected in (
                (True, None, "Extracted_Height", [8, 12]),
                (False, None, "LiDAR_Height", [5, 8]),
                (False, "Extracted_Height", "Extracted_Height", [5, 8]),
            ):
                with self.subTest(apply_offset=apply_offset, custom=custom):
                    with patch("tools.assign_building_heights.rasterio.open", wraps=rasterio.open) as opened:
                        match_polygons_with_height(str(source), str(raster), str(result),
                                                   apply_offset, False, height_column=custom)
                    self.assertEqual(opened.call_count, 1)
                    self.assertEqual(gpd.read_file(result)[column].tolist(), expected)

    def test_city_with_no_buildings_writes_an_empty_geojson(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tile_geometry, raster, city_geometry = (root / name for name in
                                                     ("tile.geojson", "city.tif", "city.geojson"))
            crs = rasterio.crs.CRS.from_epsg(32640)
            gpd.GeoDataFrame(geometry=[], crs=crs).to_file(tile_geometry)
            grid = dict(width=2, height=2, crs=crs)
            profile = dict(grid, driver="GTiff", count=1, dtype="uint16", nodata=UINT16_NODATA,
                           transform=from_origin(300000, 2800000, 1, 1))
            save_geotiff(raster, np.ones((1, 2, 2), dtype="uint16"), profile)
            self.assertEqual(merge_buildings([("tile", raster, tile_geometry)], city_geometry, crs), 0)
            validate_city_outputs(raster, city_geometry, grid, 0)
            self.assertTrue(gpd.read_file(city_geometry).empty)

    def test_ground_filter_can_be_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            depth_dir = root / "depth"
            depth_dir.mkdir()
            values = np.arange(64 * 64, dtype=np.float32).reshape(64, 64) / 10 + 5
            patch_path = depth_dir / "window_0_x0_y0_depth.tif"
            with rasterio.open(
                patch_path,
                "w",
                driver="GTiff",
                width=64,
                height=64,
                count=1,
                dtype="float32",
                crs="EPSG:32640",
                transform=from_origin(0, 64, 1, 1),
            ) as target:
                target.write(values, 1)

            unfiltered = RH2AHConverter(
                str(depth_dir), str(root), None, window_size=64,
                apply_ground_filter=False,
            )
            unfiltered._preprocess_and_filter()
            np.testing.assert_allclose(unfiltered.imgs[0], values)

            filtered = RH2AHConverter(
                str(depth_dir), str(root), None, window_size=64,
                apply_ground_filter=True,
            )
            filtered._preprocess_and_filter()
            self.assertFalse(np.array_equal(filtered.imgs[0], values))
            self.assertTrue(np.all(filtered.imgs[0] >= 0))

    def test_single_relative_patch_does_not_require_an_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            depth_dir = root / "depth"
            depth_dir.mkdir()
            values = np.arange(64, dtype=np.float32).reshape(8, 8)
            with rasterio.open(
                depth_dir / "window_0_x0_y0_depth.tif", "w", driver="GTiff",
                width=8, height=8, count=1, dtype="float32", crs="EPSG:32640",
                transform=from_origin(300000, 2800000, 1, 1),
            ) as target:
                target.write(values, 1)
            converter = RH2AHConverter(str(depth_dir), str(root), None,
                                        window_size=8, apply_ground_filter=False)
            converter._preprocess_and_filter()
            converter._align_relative_heights()
            np.testing.assert_array_equal(converter.norm_rh[0], values)

    def test_metric_input_preserves_every_patch_scale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            depth_dir = root / "depth"
            depth_dir.mkdir()
            for index, (x, value) in enumerate(((0, 10.0), (2, 20.0))):
                with rasterio.open(
                    depth_dir / f"window_{index}_x{x}_y0_depth.tif",
                    "w",
                    driver="GTiff",
                    width=4,
                    height=4,
                    count=1,
                    dtype="float32",
                    crs="EPSG:32640",
                    transform=from_origin(x, 4, 1, 1),
                ) as target:
                    target.write(np.full((4, 4), value, np.float32), 1)

            converter = RH2AHConverter(
                str(depth_dir), str(root), None, window_size=4, overlap=0.5,
                use_conf=False, metric_input=True, apply_ground_filter=False,
            )
            converter.run_metric()

            with rasterio.open(root / "AbsoluteHeight" / "output.tif") as source:
                actual = source.read(1)
            expected = np.array([
                [10, 10, 15, 15, 20, 20],
                [10, 10, 15, 15, 20, 20],
                [10, 10, 15, 15, 20, 20],
                [10, 10, 15, 15, 20, 20],
            ], dtype=np.int16)
            np.testing.assert_array_equal(actual, expected)
            self.assertIs(converter.rh_norm_imgs, converter.imgs)


    def test_city_raster_uint16_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = []
            source_arrays = (
                np.array([[-2.0, 1.4], [1.6, 70000.0]], dtype=np.float32),
                np.full((2, 2), 10.0, dtype=np.float32),
            )
            for index, values in enumerate(source_arrays):
                stem = f"tile{index}"
                path = root / f"{stem}.tif"
                with rasterio.open(
                    path,
                    "w",
                    driver="GTiff",
                    width=2,
                    height=2,
                    count=1,
                    dtype="float32",
                    crs="EPSG:32640",
                    transform=from_origin(index * 2, 2, 1, 1),
                ) as target:
                    target.write(values, 1)
                outputs.append((stem, path, root / f"{stem}.geojson"))

            grid = validate_height_tiles(outputs)
            merged = root / "merged.tif"
            merge_height_raster(outputs, merged, grid)
            with rasterio.open(merged) as source:
                actual = source.read(1)
                self.assertEqual(source.count, 1)
                self.assertEqual(source.dtypes[0], "uint16")
                self.assertEqual(source.nodata, UINT16_NODATA)
                self.assertGreater(len(source.overviews(1)), 0)
            np.testing.assert_array_equal(
                actual,
                np.array([[0, 1, 10, 10], [2, 65534, 10, 10]], dtype=np.uint16),
            )


if __name__ == "__main__":
    unittest.main()
