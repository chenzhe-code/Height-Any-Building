"""Regression tests for recovering only raw model caches from retained HAB."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

from tools.maintenance import recover_synrs3d_patch_cache as recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.images = self.root / "images"
        self.results = self.root / "results"
        self.images.mkdir()
        self.results.mkdir()
        self.image = self.images / "isolated.tif"
        self.profile = dict(driver="GTiff", width=1024, height=1024, count=3,
                            dtype="uint8", crs="EPSG:32650", transform=from_origin(0, 1024, 1, 1))
        with rasterio.open(self.image, "w", **self.profile):
            pass
        stable = self.results / "isolated"
        stable.mkdir()
        (stable / "status.json").write_text(json.dumps({"status": "success", "model_name": "SynRS3D",
                                                       "input_tif": str(self.image)}))
        for name in ("isolated_building_heights.geojson", "footprint_direct_nms.geojson"):
            (stable / name).write_text('{"type":"FeatureCollection","features":[]}')
        for name in ("sparse_control_height.tif", "isolated_absolute_height.tif"):
            with rasterio.open(stable / name, "w", **dict(self.profile, count=1, dtype="float32")):
                pass
        self.stable_snapshot = {p: p.read_bytes() for p in stable.iterdir()}
        self.cache = self.results / "_global_patch_cache"
        checkpoint = self.root / "checkpoint.pth"
        checkpoint.write_bytes(b"test model")
        self.args = SimpleNamespace(input_dir=self.images, results_root=self.results, cache_root=self.cache,
                                    checkpoint=checkpoint, conda_bin=Path(sys.executable),
                                    gpus=["0", "1"], expected_tiles=1, dry_run=False)

    def tearDown(self):
        self.temporary.cleanup()

    def write_archive(self):
        directory = self.cache / "interior" / "isolated"
        directory.mkdir(parents=True)
        prediction = directory / "x0_y0.npz"
        np.savez_compressed(prediction, prediction=np.full((1024, 1024), 12, np.float32))
        manifest = dict(version=recovery.gp.VERSION, input=str(self.image.resolve()), tile="isolated",
                        width=1024, height=1024, patches=[dict(x=0, y=0, cache=str(prediction))])
        (directory / "manifest.json").write_text(json.dumps(manifest))
        return directory, manifest

    def assert_stable_unchanged(self):
        for path, content in self.stable_snapshot.items():
            self.assertEqual(path.read_bytes(), content)

    def test_dry_run_accepts_single_isolated_tile_and_writes_nothing(self):
        self.args.dry_run = True
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(recovery.recover(self.args), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["recover_tiles"], 1)
        self.assertEqual(plan["workers"], {"0": ["isolated"], "1": []})
        self.assertFalse(self.cache.exists())
        self.assert_stable_unchanged()

    def test_reuses_complete_archive_without_launching_any_process(self):
        self.write_archive()
        with patch.object(recovery, "run_command", side_effect=AssertionError("must reuse")):
            self.assertEqual(recovery.recover(self.args), 0)
        self.assert_stable_unchanged()

    def test_cache_validation_checks_coordinates_and_prediction_values(self):
        directory, manifest = self.write_archive()
        self.assertTrue(recovery.cache_complete(self.image, self.cache))
        manifest["patches"][0]["x"] = 512
        (directory / "manifest.json").write_text(json.dumps(manifest))
        self.assertFalse(recovery.cache_complete(self.image, self.cache))
        manifest["patches"][0]["x"] = 0
        (directory / "manifest.json").write_text(json.dumps(manifest))
        np.savez_compressed(directory / "x0_y0.npz", prediction=np.full((1024, 1024), np.nan))
        self.assertFalse(recovery.cache_complete(self.image, self.cache))

    def test_cache_validation_rejects_truncated_archive(self):
        directory, _ = self.write_archive()
        (directory / "x0_y0.npz").write_bytes(b"PK\x03\x04truncated")
        self.assertFalse(recovery.cache_complete(self.image, self.cache))

    def test_recovery_cleans_temporary_files_and_uses_only_three_existing_stages(self):
        commands = []

        def fake_command(command, log, env):
            commands.append(command)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
            if "--img_path" in command:
                depth = Path(command[command.index("--output_dir") + 1])
                (depth / "depth").mkdir(parents=True)
                with rasterio.open(depth / "depth/window_0_x0_y0_depth.tif", "w",
                                   **dict(self.profile, count=1, dtype="float32")) as target:
                    target.write(np.full((1024, 1024), 12, np.float32), 1)
            elif "--archive-tile" in command:
                recovery.gp.archive_tile(SimpleNamespace(input_tif=self.image,
                    tile_name=self.image.stem, cache_root=self.cache,
                    depth_dir=Path(command[command.index("--depth-dir") + 1])))

        with patch.object(recovery, "run_command", side_effect=fake_command):
            self.assertEqual(recovery.recover(self.args), 0)
        self.assertEqual(len(commands), 3)
        self.assertTrue(recovery.cache_complete(self.image, self.cache))
        self.assertEqual(list((self.cache / "_recovery").glob("*/temporary/*")), [])
        self.assertEqual(len(list((self.cache / "_recovery").glob("*/*.log"))), 1)
        self.assert_stable_unchanged()

    def test_failed_inference_keeps_temporary_directory_and_log(self):
        with patch.object(recovery, "run_command", side_effect=RuntimeError("inference failed")):
            self.assertEqual(recovery.recover(self.args), 1)
        self.assertEqual(len(list((self.cache / "_recovery").glob("*/temporary/isolated"))), 1)
        log = next((self.cache / "_recovery").glob("*/*.log"))
        self.assertIn("inference failed", log.read_text())
        self.assertFalse(recovery.cache_complete(self.image, self.cache))
        self.assert_stable_unchanged()

    def test_concurrent_recovery_is_rejected(self):
        with recovery.recovery_lock(self.cache):
            with self.assertRaisesRegex(RuntimeError, "another raw patch recovery"):
                with recovery.recovery_lock(self.cache):
                    self.fail("should not acquire the held lock")

    def test_failed_retained_hab_stops_before_any_cache_write(self):
        status = self.results / "isolated/status.json"
        status.write_text('{"status":"failed","model_name":"SynRS3D"}')
        with self.assertRaisesRegex(ValueError, "successful retained"):
            recovery.recover(self.args)
        self.assertFalse(self.cache.exists())


if __name__ == "__main__":
    unittest.main()
