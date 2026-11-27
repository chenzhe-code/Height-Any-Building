from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tools"))

from global_optimize_height import (
    PATCH_SIZE, Tile, axis_starts, build_equations, discover, fit_global,
    inner_window, quality_report, read_rgb, recompute_buildings, side_weight,
    write_uint16_tiles,
)


class GlobalSeamTests(unittest.TestCase):
    def test_axis_starts_cover_lagos_edge(self):
        starts = axis_starts(5000)
        self.assertEqual(len(starts), 9)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 3978)
        covered = np.zeros(5000, dtype=bool)
        for start in starts:
            covered[start:start + PATCH_SIZE] = True
        self.assertTrue(covered.all())

    def test_horizontal_seam_and_four_tile_junction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            for name, x, y, value in (
                ("nw", 0, 2 * PATCH_SIZE, 20),
                ("ne", PATCH_SIZE, 2 * PATCH_SIZE, 40),
                ("sw", 0, PATCH_SIZE, 60),
                ("se", PATCH_SIZE, PATCH_SIZE, 80),
            ):
                with rasterio.open(
                    images / f"{name}.tif", "w", driver="GTiff",
                    width=PATCH_SIZE, height=PATCH_SIZE, count=3, dtype="uint8",
                    crs="EPSG:32631", transform=from_origin(x, y, 1, 1),
                ) as target:
                    target.write(np.full((3, PATCH_SIZE, PATCH_SIZE), value,
                                         dtype=np.uint8))
            tiles, seams, pairs = discover(images, root / "unused", root / "unused", False)
            self.assertEqual((len(tiles), pairs, len(seams)), (4, 4, 4))
            horizontal = next(seam for seam in seams
                              if seam.a == "nw" and seam.b == "sw")
            rgb = read_rgb(horizontal, tiles)
            self.assertTrue(np.all(rgb[:511] == 20))
            self.assertTrue(np.all(rgb[511:] == 60))
            self.assertEqual(side_weight(horizontal, "nw").shape,
                             (256, PATCH_SIZE))
            self.assertEqual(int(inner_window(tiles["nw"], horizontal).row_off),
                             PATCH_SIZE - 256)

    def test_building_height_recomputed_with_pixel_offset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.geojson"
            target = root / "refined.geojson"
            raster = root / "refined.tif"
            values = np.zeros((10, 10), dtype=np.uint16)
            values[4, 5] = 12
            with rasterio.open(
                raster, "w", driver="GTiff", width=10, height=10,
                count=1, dtype="uint16", nodata=65535, crs="EPSG:32631",
                transform=from_origin(500000, 100, 1, 1),
            ) as output:
                output.write(values, 1)
            frame = gpd.GeoDataFrame(
                {"Offset_X": [1.0], "Offset_Y": [0.0],
                 "Extracted_Height": [0.0]},
                geometry=[box(500004, 95, 500005, 96)], crs="EPSG:32631")
            frame.to_file(source, driver="GeoJSON")
            tile = Tile("tile", "", "", str(source), "", "", 500000,
                        500010, 100, 90, 10, 10)
            count = recompute_buildings([("tile", raster, target)], {"tile": tile})
            self.assertEqual(count, 1)
            refined = gpd.read_file(target)
            self.assertEqual(float(refined.loc[0, "Extracted_Height"]), 12)

    def test_crossing_window_and_uint16_blend(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            outputs = root / "hab"
            controls = root / "controls"
            images.mkdir()
            tiles = [("left", 0, 10), ("right", PATCH_SIZE, 12)]
            for name, x, height in tiles:
                image = np.empty((3, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
                image[:] = 30 if name == "left" else 200
                with rasterio.open(
                    images / f"{name}.tif", "w", driver="GTiff",
                    width=PATCH_SIZE, height=PATCH_SIZE, count=3, dtype="uint8",
                    crs="EPSG:32631", transform=from_origin(x, PATCH_SIZE, 1, 1),
                ) as target:
                    target.write(image)
                directory = outputs / name
                directory.mkdir(parents=True)
                with rasterio.open(
                    directory / f"{name}_absolute_height.tif", "w", driver="GTiff",
                    width=PATCH_SIZE, height=PATCH_SIZE, count=3, dtype="int16",
                    crs="EPSG:32631", transform=from_origin(x, PATCH_SIZE, 1, 1),
                ) as target:
                    target.write(np.full((3, PATCH_SIZE, PATCH_SIZE), height, dtype=np.int16))
                (directory / f"{name}_building_heights.geojson").write_text(
                    '{"type":"FeatureCollection","features":[]}', encoding="utf-8")
                (directory / "status.json").write_text(
                    json.dumps({"status": "success", "model_name": "SynRS3D"}),
                    encoding="utf-8")
                control_dir = controls / name
                control_dir.mkdir(parents=True)
                sparse = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
                sparse[200, 200] = 10
                with rasterio.open(
                    control_dir / "sparse_control_height.tif", "w", driver="GTiff",
                    width=PATCH_SIZE, height=PATCH_SIZE, count=1, dtype="float32",
                    nodata=0, crs="EPSG:32631",
                    transform=from_origin(x, PATCH_SIZE, 1, 1),
                ) as target:
                    target.write(sparse, 1)
            discovered, seams, pairs = discover(images, outputs, controls, True)
            self.assertEqual((len(discovered), pairs, len(seams)), (2, 1, 1))
            seam = seams[0]
            rgb = read_rgb(seam, discovered)
            self.assertEqual(rgb.shape, (PATCH_SIZE, PATCH_SIZE, 3))
            self.assertTrue(np.all(rgb[:, :511] == 30))
            self.assertTrue(np.all(rgb[:, 511:] == 200))
            self.assertEqual(int(inner_window(discovered["left"], seam).col_off),
                             PATCH_SIZE - 256)
            self.assertGreater(float(side_weight(seam, "left")[:, -1].min()), 0.9)
            self.assertGreater(float(side_weight(seam, "right")[:, 0].min()), 0.9)
            cache = root / "seam.npz"
            np.savez_compressed(cache, prediction=np.full(
                (PATCH_SIZE, PATCH_SIZE), 10, dtype=np.float32))
            cache_paths = {seam.name: cache}
            rows, diagnostic = build_equations(discovered, seams, cache_paths)
            solution = fit_global(rows, 2, 1)
            report = quality_report(solution, 2, diagnostic, seams)
            self.assertLessEqual(report["seam_mae_after"], report["seam_mae_before"] + 0.5)
            refined = write_uint16_tiles(root / "refined", discovered, seams,
                                         cache_paths, solution, diagnostic["tile_names"])
            for _, path, _ in refined:
                with rasterio.open(path) as source:
                    self.assertEqual(source.dtypes, ("uint16",))
                    self.assertEqual(source.nodata, 65535)
                    self.assertEqual(source.count, 1)
            with rasterio.open(refined[0][1]) as left:
                self.assertNotEqual(left.read(1)[400, -1], 65535)


if __name__ == "__main__":
    unittest.main()
