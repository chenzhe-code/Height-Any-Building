"""Georeferenced regression coverage for independently calibrated city components."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import global_patch_height as gp


class ComponentPatchCityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inputs = self.root / "images"
        self.results = self.root / "results"
        self.cache = self.results / "_global_patch_cache"
        self.inputs.mkdir()
        self.results.mkdir()
        self.cache.mkdir()
        self.checkpoint = self.root / "checkpoint.pth"
        self.checkpoint.write_bytes(b"synthetic")

    def tearDown(self):
        gp.prediction.cache_clear()
        self.temp.cleanup()

    def add_tile(self, name, x, top=1024, raw=10, hab=15, control=15,
                 buildings=False, width=1024):
        transform = from_origin(x, top, 1, 1)
        base = dict(driver="GTiff", width=width, height=1024,
                    crs="EPSG:32630", transform=transform)
        with rasterio.open(self.inputs / f"{name}.tif", "w", count=3,
                           dtype="uint8", **base) as target:
            target.write(np.zeros((3, 1024, width), np.uint8))
        folder = self.results / name
        folder.mkdir()
        pixels = np.zeros((1024, width), np.float32)
        if control:
            pixels[100:1000:100, 100:1000:100] = control
        with rasterio.open(folder / "sparse_control_height.tif", "w", count=1,
                           dtype="float32", nodata=0, **base) as target:
            target.write(pixels, 1)
        with rasterio.open(folder / f"{name}_absolute_height.tif", "w", count=1,
                           dtype="uint16", nodata=65535, **base) as target:
            target.write(np.full((1024, width), hab, np.uint16), 1)
        (folder / "status.json").write_text(json.dumps({"status": "success"}))
        archive = self.cache / "interior" / name
        archive.mkdir(parents=True)
        records = []
        for px, py in sorted(gp.expected_positions(width, 1024)):
            prediction = archive / f"x{px}_y{py}.npz"
            np.savez_compressed(prediction,
                                prediction=np.full((1024, 1024), raw, np.float32))
            records.append({"x": px, "y": py, "cache": str(prediction)})
        (archive / "manifest.json").write_text(json.dumps({
            "version": gp.VERSION,
            "input": str((self.inputs / f"{name}.tif").resolve()),
            "patches": records,
        }))
        if buildings:
            import geopandas as gpd
            from shapely.geometry import box
            gpd.GeoDataFrame({
                "BuildingId": [1], "Offset_X": [0], "Offset_Y": [0],
                "Extracted_Height": [3],
                "geometry": [box(x + 100, top - 101, x + 101, top - 100)],
            }, crs="EPSG:32630").to_file(
                folder / f"{name}_building_heights.geojson", driver="GeoJSON",
            )

    def argv(self, *extra, results=None):
        return [str(Path(gp.__file__)), "--input-dir", str(self.inputs),
                "--results-root", str(results or self.results),
                "--cache-root", str(self.cache),
                "--checkpoint", str(self.checkpoint), *extra]

    def crossing_predictions(self, tiles, seams, *unused):
        """Replace GPU inference only; spatial grouping and calibration remain real."""
        records = []
        folder = self.cache / "seams"
        folder.mkdir(exist_ok=True)
        for seam, left, top in seams:
            path = folder / f"{seam['name']}.npz"
            np.savez_compressed(path,
                                prediction=np.full((1024, 1024), 10, np.float32))
            records.append(gp.Patch(seam["name"], str(path), left, top,
                                    1024, "seam", ""))
        return records

    def test_full_edges_connect_but_corner_contacts_and_gaps_do_not(self):
        for name, x, top in (("a", 0, 1024), ("b", 1024, 1024),
                             ("corner", 2048, 0), ("gap", 4096, 1024)):
            self.add_tile(name, x, top)
        with self.assertRaisesRegex(ValueError, "disconnected"):
            gp.discover(self.inputs, self.results, True)
        tiles, edges = gp.discover(self.inputs, self.results, True,
                                   allow_disconnected=True)
        self.assertEqual(edges, [("a", "b", "vertical")])
        self.assertEqual([list(group) for group in gp.tile_components(tiles, edges)],
                         [["a", "b"], ["corner"], ["gap"]])
        # File discovery and graph traversal order must not change group identity.
        self.assertEqual([list(group) for group in gp.tile_components(
            dict(reversed(list(tiles.items()))), list(reversed(edges)))],
            [["a", "b"], ["corner"], ["gap"]])

    def test_single_tile_has_one_component_and_no_crossing_patch(self):
        self.add_tile("island", 0)
        tiles, edges = gp.discover(self.inputs, self.results, True)
        self.assertEqual(edges, [])
        self.assertEqual([list(group) for group in gp.tile_components(tiles, edges)],
                         [["island"]])
        self.assertEqual(gp.seam_records(tiles, edges), [])
        self.assertEqual(gp.component_count(1, np.empty((0, 2), dtype=int)), 1)

    def test_allow_disconnected_still_rejects_overlapping_images(self):
        self.add_tile("a", 0)
        self.add_tile("b", 512)
        with self.assertRaisesRegex(ValueError, "overlapping input tiles"):
            gp.discover(self.inputs, self.results, True, allow_disconnected=True)

    def test_disconnected_dry_run_creates_no_outputs_or_cache(self):
        self.add_tile("a", 0)
        self.add_tile("b", 4096)
        output = self.root / "dry_results"
        dry_cache = self.root / "dry_cache"
        args = self.argv("--export-raw", "--global-optimize", "--hab-prior",
                         "--by-component", "--dry-run", "--expected-tiles", "2",
                         results=output)
        args[args.index("--cache-root") + 1] = str(dry_cache)
        before = {str(p.relative_to(self.root)): p.stat().st_mtime_ns
                  for p in self.root.rglob("*")}
        with mock.patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
            gp.main()
        after = {str(p.relative_to(self.root)): p.stat().st_mtime_ns
                 for p in self.root.rglob("*")}
        self.assertEqual(before, after)
        self.assertFalse(output.exists())
        self.assertFalse(dry_cache.exists())

    def test_separate_groups_publish_city_and_retain_uncontrolled_island(self):
        import geopandas as gpd
        for name, x, hab, control in (("a", 0, 15, 15), ("b", 1024, 15, 15),
                                      ("c", 4096, 8, 8), ("d", 5120, 8, 8),
                                      ("island", 8192, 37, 0)):
            self.add_tile(name, x, hab=hab, control=control, buildings=True)
        args = self.argv("--export-raw", "--global-optimize", "--hab-prior",
                         "--by-component", "--expected-tiles", "5")
        with mock.patch.object(gp, "infer_seams", side_effect=self.crossing_predictions), \
                mock.patch.object(gp, "hab_patch_priors", wraps=gp.hab_patch_priors) as priors, \
                mock.patch.object(sys, "argv", args):
            gp.main()
        # A component may use its own neighbours for fallback, never another group.
        self.assertEqual({tuple(sorted(call.args[0])) for call in priors.call_args_list},
                         {("a", "b"), ("c", "d")})
        output = self.results / "city_global_patch_hab_height_1m_uint16.tif"
        with rasterio.open(output) as source:
            self.assertEqual((source.width, source.height), (9216, 1024))
            self.assertEqual(source.dtypes, ("uint16",))
            self.assertEqual(source.nodata, 65535)
            row = source.read(1, window=rasterio.windows.Window(0, 0, 9216, 1))[0]
            self.assertEqual([int(row[x]) for x in (100, 1124, 4196, 5220, 8292)],
                             [15, 15, 8, 8, 37])
            self.assertTrue(np.all(row[2048:4096] == 65535))
            self.assertTrue(np.all(row[6144:8192] == 65535))
        raw = gpd.read_file(self.results / "city_synrs3d_raw_building_heights.geojson")
        optimized = gpd.read_file(self.results / "city_global_patch_hab_building_heights.geojson")
        self.assertEqual(len(raw), 5)
        self.assertEqual(set(raw["Extracted_Height"]), {10})
        self.assertEqual(dict(zip(optimized["source_tile"], optimized["Extracted_Height"])),
                         {"a": 15, "b": 15, "c": 8, "d": 8, "island": 37})
        self.assertEqual(set(raw["GlobalBuildingId"]), set(optimized["GlobalBuildingId"]))
        report = json.loads((self.results / "global_patch_hab_diagnostics.json").read_text())
        self.assertEqual(report["graph_components"], 3)
        self.assertEqual(report["optimized_building_count"], 5)
        components = {tuple(item["tile_names"]): item for item in report["components"]}
        self.assertEqual(set(components), {("a", "b"), ("c", "d"), ("island",)})
        for names in (("a", "b"), ("c", "d")):
            self.assertEqual(components[names]["graph_components"], 1)
            self.assertNotEqual(components[names]["method"], "retained_hab")
        island = components[("island",)]
        self.assertEqual(island["method"], "retained_hab")
        self.assertTrue(island["fallback_reason"])
        self.assertFalse(island.get("photon_validation_passed", False))
        self.assertEqual(island["photon_validation"], "unavailable")
        self.assertNotEqual(island.get("scale_prior_weight_selection"),
                            "minimum_passing_quality_gate")

    def test_singleton_without_controls_uses_real_hab_pixels(self):
        self.add_tile("island", 0, hab=37, control=0, buildings=True)
        retained = self.results / "island" / "island_absolute_height.tif"
        with rasterio.open(retained, "r+") as target:
            pixels = target.read(1)
            pixels[:, 512:] = 53
            target.write(pixels, 1)
        args = self.argv("--global-optimize", "--hab-prior", "--by-component")
        with mock.patch.object(gp, "infer_seams", side_effect=self.crossing_predictions), \
                mock.patch.object(sys, "argv", args):
            gp.main()
        with rasterio.open(self.results / "city_global_patch_hab_height_1m_uint16.tif") as source, \
                rasterio.open(retained) as original:
            np.testing.assert_array_equal(source.read(1), original.read(1))
        report = json.loads((self.results / "global_patch_hab_diagnostics.json").read_text())
        self.assertEqual(report["graph_components"], 1)
        component = report["components"][0]
        self.assertEqual(component["method"], "retained_hab")
        self.assertTrue(component["fallback_reason"])
        self.assertFalse(component.get("photon_validation_passed", False))
        self.assertEqual(component["photon_validation"], "unavailable")

    def test_singleton_with_controls_calibrates_its_interior_patch_graph(self):
        self.add_tile("island", 0, raw=10, hab=15, control=15,
                      buildings=True, width=1536)
        args = self.argv("--global-optimize", "--hab-prior", "--by-component")
        with mock.patch.object(gp, "infer_seams", side_effect=self.crossing_predictions), \
                mock.patch.object(sys, "argv", args):
            gp.main()
        report = json.loads((self.results / "global_patch_hab_diagnostics.json").read_text())
        component = report["components"][0]
        self.assertEqual(component["method"], "patch_optimization")
        self.assertEqual(component["graph_components"], 1)
        self.assertEqual(component["crossing_patches"], 0)
        self.assertEqual(component["interior_patches"], 2)
        with rasterio.open(self.results / "city_global_patch_hab_height_1m_uint16.tif") as source:
            self.assertTrue(np.all(source.read(1) == 15))

    def test_raw_only_disconnected_city_needs_no_component_flag(self):
        self.add_tile("a", 0, raw=10, buildings=True)
        self.add_tile("b", 4096, raw=20, buildings=True)
        with mock.patch.object(sys, "argv", self.argv("--export-raw")):
            gp.main()
        with rasterio.open(self.results / "city_synrs3d_raw_1m_uint16.tif") as source:
            row = source.read(1, window=rasterio.windows.Window(0, 0, 5120, 1))[0]
            self.assertEqual(int(row[100]), 10)
            self.assertEqual(int(row[4196]), 20)
            self.assertTrue(np.all(row[1024:4096] == 65535))
        self.assertFalse((self.results / "global_patch_diagnostics.json").exists())

    def test_hab_prior_fallback_uses_only_current_component(self):
        self.add_tile("a", 0, raw=10, hab=15)
        self.add_tile("b", 1024, raw=1, hab=15)
        self.add_tile("other", 4096, raw=10, hab=8)
        tiles, edges = gp.discover(self.inputs, self.results, True,
                                   allow_disconnected=True)
        groups = gp.tile_components(tiles, edges)
        first = groups[0]
        prior, summary = gp.hab_patch_priors(first, gp.load_interior(first, self.cache))
        np.testing.assert_allclose(prior, [1.5, 1.5])
        self.assertEqual(summary["hab_prior_fallback_patches"], 1)
        other_prior, _ = gp.hab_patch_priors(groups[1], gp.load_interior(groups[1], self.cache))
        np.testing.assert_allclose(other_prior, [.8])

    def test_later_component_quality_failure_preserves_all_published_outputs(self):
        for name, x, hab in (("a", 0, 15), ("b", 1024, 15),
                              ("c", 4096, 8), ("d", 5120, 8)):
            self.add_tile(name, x, hab=hab, control=hab, buildings=True)
        outputs = (
            "city_synrs3d_raw_1m_uint16.tif",
            "city_synrs3d_raw_building_heights.geojson",
            "city_global_patch_hab_height_1m_uint16.tif",
            "city_global_patch_hab_building_heights.geojson",
            "global_patch_hab_diagnostics.json",
        )
        args = self.argv("--export-raw", "--global-optimize", "--hab-prior",
                         "--by-component", "--expected-tiles", "4")
        real_solve = gp.solve_hab_prior_scales
        for previous_publication in (False, True):
            with self.subTest(previous_publication=previous_publication):
                if previous_publication:
                    for name in outputs:
                        (self.results / name).write_bytes(f"previous {name}".encode())
                before = {p.name: p.read_bytes() for p in self.results.iterdir()
                          if p.is_file()}
                attempts = []

                def fail_second_component(*solver_args, **solver_kwargs):
                    attempts.append(solver_args)
                    if len(attempts) == 2:
                        raise ValueError("held-out photon error worsened: component 2")
                    return real_solve(*solver_args, **solver_kwargs)

                with mock.patch.object(gp, "infer_seams", side_effect=self.crossing_predictions), \
                        mock.patch.object(gp, "solve_hab_prior_scales",
                                          side_effect=fail_second_component), \
                        mock.patch.object(sys, "argv", args):
                    with self.assertRaisesRegex(ValueError, "held-out photon error worsened"):
                        gp.main()
                self.assertEqual(len(attempts), 2)
                after = {p.name: p.read_bytes() for p in self.results.iterdir()
                         if p.is_file()}
                self.assertEqual(before, after)
                self.assertEqual(list(self.cache.glob("global_patch_render_*")), [])


if __name__ == "__main__":
    unittest.main()
