"""Small georeferenced regression tests for the patch-level city workflow."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import global_patch_height as gp
from merge_city_height_results import merge_height_raster, validate_height_tiles


class RobustControlTests(unittest.TestCase):
    def solve(self, controls, prior=1.5):
        return gp.solve_scales(
            1, np.empty((0, 2), dtype=int), np.empty((0, 2)), controls,
            prior_scales=np.array([prior]), prior_weight=5.,
        )[0] * 10.

    def test_large_control_error_has_less_influence_on_consistent_heights(self):
        from unittest import mock
        controls = [(0, 10., 15.)] * 5 + [(0, 10., 100.)]
        with mock.patch.object(gp, "CONTROL_ROBUST_POWER", 1.):
            previous_height = self.solve(controls)
        height = self.solve(controls)
        self.assertAlmostEqual(height, 15., delta=.1)
        self.assertLess(abs(height - 15.), abs(previous_height - 15.) / 5.)

    def test_consistent_controls_still_correct_an_inaccurate_prior(self):
        controls = [(0, 10., 15.)] * 10
        height = self.solve(controls, prior=1.)
        self.assertAlmostEqual(height, 15., delta=.1)


class PatchCityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input_dir = self.root / "images"
        self.results = self.root / "results"
        self.input_dir.mkdir()
        self.results.mkdir()
        for name, x in (("left", 0), ("right", 1024)):
            image = self.input_dir / f"{name}.tif"
            with rasterio.open(image, "w", driver="GTiff", width=1024, height=1024,
                               count=3, dtype="uint8", crs="EPSG:32630",
                               transform=from_origin(x, 1024, 1, 1)) as target:
                target.write(np.zeros((3, 1024, 1024), np.uint8))
            tile_dir = self.results / name
            tile_dir.mkdir()
            control = np.zeros((1024, 1024), np.float32)
            control[100:1000:100, 100:1000:100] = 15
            with rasterio.open(tile_dir / "sparse_control_height.tif", "w",
                               driver="GTiff", width=1024, height=1024,
                               count=1, dtype="float32", crs="EPSG:32630",
                               transform=from_origin(x, 1024, 1, 1), nodata=0) as target:
                target.write(control, 1)
            with rasterio.open(tile_dir / f"{name}_absolute_height.tif", "w",
                               driver="GTiff", width=1024, height=1024,
                               count=1, dtype="uint16", crs="EPSG:32630",
                               transform=from_origin(x, 1024, 1, 1), nodata=65535) as target:
                target.write(np.full((1024, 1024), 15, np.uint16), 1)
            (tile_dir / "status.json").write_text(json.dumps({"status": "success"}))
        self.cache = self.results / "_global_patch_cache"
        (self.cache / "seams").mkdir(parents=True)
        self.tiles, self.edges = gp.discover(self.input_dir, self.results, True)
        self.seams = gp.seam_records(self.tiles, self.edges)
        self.patches = []
        for name, x, value in (("left", 0, 10), ("right", 1024, 20), ("cross", 512, 15)):
            cache = self.cache / f"{name}.npz"
            np.savez_compressed(cache, prediction=np.full((1024, 1024), value, np.float32))
            self.patches.append(gp.Patch(name, str(cache), x, 1024, 1024,
                                         "seam" if name == "cross" else "interior", name))

    def tearDown(self):
        gp.prediction.cache_clear()
        self.temp.cleanup()

    def test_controls_keep_finite_predictions_outside_previous_range(self):
        patch = self.patches[0]
        values = np.full((1024, 1024), 10., np.float32)
        expected = (0., 1., 2., 300., 301., 500., -1.)
        locations = [(r, c) for r in range(100, 1000, 100)
                     for c in range(100, 1000, 100)]
        for index, (r, c) in enumerate(locations):
            values[r, c] = expected[index % len(expected)]
        np.savez_compressed(patch.cache, prediction=values)
        controls, holdout, _ = gp.control_equations(
            {"left": self.tiles["left"]}, [patch],
        )
        self.assertEqual(len(controls) + len(holdout), len(locations))
        self.assertEqual({pred for _, pred, _ in controls}, set(expected))
        self.assertTrue({0., 1., 301.}.issubset(
            {pred for _, pred, _ in holdout}
        ))

    def test_previous_control_filter_cache_is_not_reused(self):
        from unittest import mock
        np.savez_compressed(
            self.patches[0].cache,
            prediction=np.ones((1024, 1024), np.float32),
        )
        with mock.patch.object(gp, "SOLVER_INPUT_VERSION", b"london-graph-inputs-v1"):
            previous_cache = gp.solver_input_cache_path(
                self.tiles, self.patches, self.cache,
            )
            with mock.patch.object(gp, "control_equations", return_value=(
                [(1, 20., 15.)], [(1, 20., 15.)], [0.],
            )):
                gp.solver_inputs(self.tiles, self.patches, self.cache)
        self.assertTrue(previous_cache.is_file())
        current_cache = gp.solver_input_cache_path(self.tiles, self.patches, self.cache)
        self.assertNotEqual(current_cache, previous_cache)
        *_, controls, holdout, _ = gp.solver_inputs(self.tiles, self.patches, self.cache)
        self.assertTrue(current_cache.is_file())
        self.assertIn((0, 1., 15.), controls)
        self.assertIn((0, 1., 15.), holdout)

    def test_crossing_rgb_uses_original_both_sides_and_world_grid(self):
        import infer_global_seam_windows as worker
        for name, value in (("left", 25), ("right", 200)):
            with rasterio.open(self.input_dir / f"{name}.tif", "r+") as target:
                target.write(np.full((3, 1024, 1024), value, np.uint8))
        seam, left, top = self.seams[0]
        rgb = worker.read_crossing_rgb(seam, {
            name: {"image": tile.image, "width": tile.width,
                   "height_px": tile.height_px}
            for name, tile in self.tiles.items()
        }, 1024)
        self.assertEqual((left, top), (512, 1024))
        self.assertEqual(rgb.shape, (1024, 1024, 3))
        self.assertTrue(np.all(rgb[:, :512] == 25))
        self.assertTrue(np.all(rgb[:, 512:] == 200))

    def test_crossing_patch_joins_both_tiles_and_scales(self):
        self.assertEqual(len(self.edges), 1)
        self.assertEqual(len(self.seams), 1)
        edges, coefficients, baseline = gp.overlap_equations(self.patches)
        self.assertEqual(len(edges), 2)
        self.assertEqual(gp.component_count(3, edges), 1)
        controls, holdout, hab_errors = gp.control_equations(self.tiles, self.patches)
        scales = gp.solve_scales(3, edges, coefficients, controls)
        hab_warm_start = gp.solve_scales(
            3, edges, coefficients, controls, initial_scales=np.array([1.5, 0.75, 1.0])
        )
        self.assertLess(float(np.max(np.abs(scales - hab_warm_start))), 0.01)
        report = gp.evaluate(edges, coefficients, baseline, holdout, scales, hab_errors)
        self.assertLess(report["overlap_median_abs_after_m"],
                        report["overlap_median_abs_before_m"])
        self.assertLess(report["holdout_control_mae_global_m"],
                        report["holdout_control_mae_raw_m"])
        self.assertAlmostEqual(scales[0] * 10, scales[1] * 20, delta=1.5)
        self.assertAlmostEqual(scales[0] * 10, scales[2] * 15, delta=1.5)

    def test_calibrated_windows_optimize_and_reject_a_one_sided_seam(self):
        from utils.rh2ah_converter import RH2AHConverter

        size, overlap = 768, 0.25
        work_directory = "calibrated-depth"
        for tile, value in zip(self.tiles.values(), (12.25, 17.75)):
            output = self.results / tile.name / work_directory
            converter = RH2AHConverter(None, output, None, window_size=size,
                                       overlap=overlap, export_calibrated=True)
            converter.position = sorted(gp.window_positions(tile.width, tile.height_px, size, overlap))
            with rasterio.open(tile.image) as source:
                converter.metas = [
                    {"crs": source.crs, "transform": source.transform * rasterio.Affine.translation(x, y)}
                    for x, y in converter.position
                ]
            converter._save_calibrated_windows([
                np.full((size, size), value) for _ in converter.position
            ])
        interiors = gp.load_calibrated(self.tiles, self.results, work_directory, size, overlap)
        self.assertEqual(len(interiors), 8)
        self.assertEqual(gp.prediction(interiors[0].cache, size)[0, 0], 12.25)
        patches, scales, report, retained = gp.optimize_calibrated(
            self.tiles, self.edges, interiors, self.patches[2:], self.cache
        )
        self.assertEqual(report["optimized_components"], 1)
        self.assertFalse(retained)
        component = report["components"][0]
        self.assertLess(component["seam_overlap_median_abs_after_m"],
                        component["seam_overlap_median_abs_before_m"])
        for tile in self.tiles.values():
            output = self.root / f"{tile.name}.tif"
            gp.render_tile(tile, patches, scales, output, calibrated=True)
            with rasterio.open(output) as source:
                self.assertEqual(source.read(1)[512, 512], 15)

        crossing = self.patches[2]
        prediction = np.full((1024, 1024), 15, np.float32)
        prediction[:, :512] = 0
        np.savez_compressed(crossing.cache, prediction=prediction)
        gp.prediction.cache_clear()
        _, _, report, retained = gp.optimize_calibrated(
            self.tiles, self.edges, interiors, self.patches[2:], self.cache
        )
        self.assertEqual(report["optimized_components"], 0)
        self.assertEqual(retained, set(self.tiles))
        self.assertEqual(report["components"][0]["fallback_reason"], "insufficient_seam_observations")

    def test_hab_patch_priors_center_every_patch_node(self):
        edges, coefficients, baseline = gp.overlap_equations(self.patches)
        priors, summary = gp.hab_patch_priors(
            self.tiles, self.patches, sample_step=32, min_samples=4,
        )
        np.testing.assert_allclose(priors, [1.5, .75, 1.], atol=1e-6)
        self.assertEqual(summary["hab_prior_direct_patches"], 3)
        self.assertEqual(summary["hab_prior_fallback_patches"], 0)
        scales = gp.solve_scales(
            3, edges, coefficients, [],
            initial_scales=priors, prior_scales=priors,
        )
        np.testing.assert_allclose(scales, priors, atol=1e-3)
        report = gp.evaluate(
            edges, coefficients, baseline, [(0, 10., 15.)], scales, [],
            seam_mask=np.ones(len(edges), dtype=bool),
            reference_scales=priors, quality_reference="tile_hab",
        )
        self.assertEqual(report["quality_reference"], "tile_hab")
        self.assertAlmostEqual(report["holdout_control_mae_prior_m"], 0.)

    def test_quality_gate_excludes_artificial_zero_weak_edges(self):
        edges = np.array([[0, 1], [1, 2], [2, 3]])
        coefficients = np.array([[2., 2.], [2., 2.], [10., 20.]])
        baseline = np.array([0., 0., 10.])
        scales = np.array([1., 1.2, .6, .3])
        holdout = [(0, 10., 10.)]
        report = gp.evaluate(
            edges, coefficients, baseline, holdout, scales, [],
            seam_mask=np.ones(3, dtype=bool),
        )
        self.assertEqual(report["observed_overlap_edges"], 1)
        self.assertEqual(report["weak_overlap_edges"], 2)
        self.assertEqual(report["overlap_median_abs_before_m"], 10.)
        self.assertEqual(report["overlap_median_abs_after_m"], 0.)
        self.assertEqual(report["observed_seam_overlap_edges"], 1)
        self.assertEqual(report["weak_seam_overlap_edges"], 2)

    def test_hab_prior_weight_adapts_until_quality_gate_passes(self):
        edges = np.array([[0, 1]])
        coefficients = np.array([[10., 10.]])
        baseline = np.array([1.])
        controls = [(0, 10., 20.), (1, 10., 20.)]
        holdout = [(0, 10., 10.), (1, 10., 10.)]
        scales, report = gp.solve_hab_prior_scales(
            2, edges, coefficients, controls, baseline, holdout, [],
            np.ones(1, dtype=bool), np.ones(2), prior_weights=(1., 100.),
        )
        self.assertEqual(report["scale_prior_weight"], 100.)
        self.assertEqual(report["scale_prior_weight_selection"],
                         "minimum_passing_quality_gate")
        self.assertEqual([item["passed"] for item in report["scale_prior_weight_attempts"]],
                         [False, True])
        self.assertIn("held_out_photon",
                      report["scale_prior_weight_attempts"][0]["quality_failures"])
        self.assertLess(float(np.max(np.abs(scales - 1.))), 0.02)

    def test_optimized_geojson_recomputes_height_with_offset(self):
        import geopandas as gpd
        from shapely.geometry import box
        from assign_city_building_heights import write_city_buildings
        for name, polygon, offset in (
            ("left", box(100, 100, 101, 101), 1),
            ("right", box(1124, 100, 1125, 101), 0),
        ):
            frame = gpd.GeoDataFrame(
                {"BuildingId": [1], "Offset_X": [offset], "Offset_Y": [0],
                 "Extracted_Height": [3], "geometry": [polygon]},
                crs="EPSG:32630",
            )
            frame.to_file(self.results / name / f"{name}_building_heights.geojson",
                          driver="GeoJSON")
        city = self.results / "optimized.tif"
        pixels = np.full((1024, 2048), 10, np.uint16)
        pixels[:, 1024:] = 20
        pixels[923, 101] = 99
        with rasterio.open(city, "w", driver="GTiff", width=2048, height=1024,
                           count=1, dtype="uint16", crs="EPSG:32630",
                           transform=from_origin(0, 1024, 1, 1), nodata=65535) as target:
            target.write(pixels, 1)
        output = self.results / "city_global_patch_building_heights.geojson"
        result = write_city_buildings(self.input_dir, self.results, city, output,
                                      expected_tiles=2)
        self.assertEqual(result["features"], 2)
        merged = gpd.read_file(output).set_index("source_tile")
        self.assertEqual(int(merged.loc["left", "Extracted_Height"]), 99)
        self.assertEqual(int(merged.loc["right", "Extracted_Height"]), 20)
        self.assertEqual(merged.loc["left", "geometry"].bounds, (100., 100., 101., 101.))
        self.assertEqual(len(set(merged["GlobalBuildingId"])), 2)

    def test_raw_fusion_is_arithmetic_mean(self):
        right = gp.Patch("overlap", self.patches[1].cache, 512, 1024,
                         1024, "interior", "left")
        tile = gp.Tile("wide", "", 0, 1024, 1536, 1024, "EPSG:32630", "", "", "")
        output = self.root / "mean.tif"
        gp.render_tile(tile, [self.patches[0], right], np.ones(2), output,
                       feathered=False)
        with rasterio.open(output) as source:
            values = source.read(1, window=rasterio.windows.Window(0, 0, 1536, 1))[0]
        self.assertEqual(int(values[100]), 10)
        self.assertEqual(int(values[700]), 15)
        self.assertEqual(int(values[1400]), 20)

    def test_raw_and_global_city_uint16(self):
        with tempfile.TemporaryDirectory(dir=self.root) as temp:
            raw = self.results / "raw.tif"
            opt = self.results / "optimized.tif"
            for patches, scales, output in (
                (self.patches[:2], np.ones(2), raw),
                (self.patches, np.array([1.5, .75, 1.]), opt),
            ):
                staged = gp.merge_rendered(self.tiles, patches, scales, output, Path(temp))
                staged.replace(output)
                with rasterio.open(output) as source:
                    self.assertEqual(source.count, 1)
                    self.assertEqual(source.dtypes, ("uint16",))
                    self.assertEqual(source.nodata, 65535)
                    self.assertEqual(source.crs.to_epsg(), 32630)
                    self.assertEqual((source.width, source.height), (2048, 1024))
                    self.assertTrue(np.all(source.read(1) < 65535))
                for path in Path(temp).glob("*.tif"):
                    path.unlink()
        with rasterio.open(raw) as source:
            self.assertEqual(int(source.read(1, window=rasterio.windows.Window(0, 0, 1, 1))[0, 0]), 10)
        with rasterio.open(opt) as source:
            self.assertEqual(int(source.read(1, window=rasterio.windows.Window(0, 0, 1, 1))[0, 0]), 15)
            self.assertEqual(int(source.read(1, window=rasterio.windows.Window(1024, 0, 1, 1))[0, 0]), 15)

    def test_end_to_end_publication_uses_raw_patch_nodes(self):
        from unittest import mock
        import geopandas as gpd
        from shapely.geometry import box
        for i, name in enumerate(("left", "right")):
            x = 0 if name == "left" else 1024
            footprint = gpd.GeoDataFrame(
                {"BuildingId": [1], "Offset_X": [0], "Offset_Y": [0],
                 "Extracted_Height": [3],
                 "geometry": [box(x + 100, 100, x + 101, 101)]},
                crs="EPSG:32630",
            )
            footprint.to_file(
                self.results / name / f"{name}_building_heights.geojson",
                driver="GeoJSON",
            )
            folder = self.cache / "interior" / name
            folder.mkdir(parents=True)
            manifest = {
                "version": gp.VERSION,
                "input": str((self.input_dir / f"{name}.tif").resolve()),
                "patches": [{"x": 0, "y": 0, "cache": self.patches[i].cache}],
            }
            (folder / "manifest.json").write_text(json.dumps(manifest))
        checkpoint = self.root / "checkpoint.pth"
        checkpoint.write_bytes(b"synthetic")
        argv = [
            str(Path(gp.__file__)), "--input-dir", str(self.input_dir),
            "--results-root", str(self.results), "--cache-root", str(self.cache),
            "--checkpoint", str(checkpoint), "--export-raw", "--global-optimize",
            "--expected-tiles", "2",
        ]
        with mock.patch.object(gp, "infer_seams", return_value=self.patches[2:]):
            with mock.patch.object(sys, "argv", argv):
                gp.main()
        for name in ("city_synrs3d_raw_1m_uint16.tif",
                     "city_global_patch_height_1m_uint16.tif"):
            with rasterio.open(self.results / name) as source:
                self.assertEqual(source.dtypes, ("uint16",))
                self.assertEqual(source.nodata, 65535)
        report = json.loads((self.results / "global_patch_diagnostics.json").read_text())
        optimized_buildings = gpd.read_file(
            self.results / "city_global_patch_building_heights.geojson"
        )
        raw_buildings = gpd.read_file(
            self.results / "city_synrs3d_raw_building_heights.geojson"
        )
        self.assertEqual(len(raw_buildings), 2)
        self.assertEqual(set(raw_buildings["Extracted_Height"]), {10, 20})
        self.assertEqual(set(raw_buildings["GlobalBuildingId"]),
                         set(optimized_buildings["GlobalBuildingId"]))
        self.assertEqual(len(optimized_buildings), 2)
        self.assertEqual(set(optimized_buildings["Extracted_Height"]), {15})
        self.assertEqual(report["optimized_building_count"], 2)
        with mock.patch.object(gp, "infer_seams", return_value=self.patches[2:]):
            with mock.patch.object(sys, "argv", [*argv, "--hab-prior"]):
                gp.main()
        for name in ("city_global_patch_hab_height_1m_uint16.tif",
                     "city_global_patch_hab_building_heights.geojson",
                     "global_patch_hab_diagnostics.json"):
            self.assertTrue((self.results / name).is_file())
        hab_report = json.loads(
            (self.results / "global_patch_hab_diagnostics.json").read_text()
        )
        self.assertEqual(hab_report["scale_prior"], "tile_hab")
        self.assertEqual(hab_report["hab_prior_direct_patches"], 3)
        raw_output = self.results / "city_synrs3d_raw_building_heights.geojson"
        raw_output.unlink()
        with mock.patch.object(sys, "argv", [*argv[:argv.index("--global-optimize")],
                                            "--expected-tiles", "2"]):
            gp.main()
        self.assertTrue(raw_output.is_file())
        with mock.patch.object(gp, "overlap_equations", side_effect=AssertionError("cache missed")):
            cached = gp.solver_inputs(self.tiles, self.patches, self.cache)
        self.assertIsInstance(cached[3][0][0], int)
        self.assertEqual(report["interior_patches"], 2)
        self.assertEqual(report["crossing_patches"], 1)
        self.assertEqual(report["graph_components"], 1)

    def test_archive_resume_and_dry_run_without_writes(self):
        depth = self.root / "depth"
        (depth / "depth").mkdir(parents=True)
        raw = depth / "depth" / "window_0_x0_y0_depth.tif"
        with rasterio.open(raw, "w", driver="GTiff", width=1024, height=1024,
                           count=1, dtype="float32", crs="EPSG:32630",
                           transform=from_origin(0, 1024, 1, 1)) as target:
            target.write(np.full((1024, 1024), 10, np.float32), 1)
        class Args:
            input_tif = self.input_dir / "left.tif"
            depth_dir = depth
            cache_root = self.cache
            tile_name = "left"
        gp.archive_tile(Args)
        cache = self.cache / "interior" / "left" / "x0_y0.npz"
        stamp = cache.stat().st_mtime_ns
        gp.archive_tile(Args)
        self.assertEqual(stamp, cache.stat().st_mtime_ns)
        cache.write_bytes(b"interrupted cache")
        gp.archive_tile(Args)
        with np.load(cache) as restored:
            self.assertEqual(restored["prediction"].shape, (1024, 1024))
        dry_root = self.root / "dry_outputs"
        command = [sys.executable, str(Path(gp.__file__)), "--input-dir",
                   str(self.input_dir), "--results-root", str(dry_root),
                   "--export-raw", "--global-optimize", "--dry-run",
                   "--expected-tiles", "2"]
        import subprocess
        subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertFalse(dry_root.exists())


if __name__ == "__main__":
    unittest.main()
