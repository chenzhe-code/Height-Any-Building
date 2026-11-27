"""Check the UI command adapter and worker lifecycle using local fixtures."""

from base64 import b64decode
from hashlib import sha256
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from scripts.runtime import ui_jobs, ui_lod1, ui_preview


class BrowserJobTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        self.inputs = self.root / "data"
        self.inputs.mkdir()
        self.image = self.inputs / "wuhan image.tif"
        self.image.touch()
        self.photons = self.inputs / "atl03"
        self.photons.mkdir()
        (self.photons / "test.h5").touch()
        self.output = self.root / "output"

    def test_image_command_uses_the_existing_cli_and_local_atl03(self):
        command = ui_jobs.build_command({
            "mode": "image", "action": "preview", "input": str(self.image),
            "model": "Distill-Any-Depth", "gpus": "0", "global_mode": "patch",
            "atl03_source": "local", "atl03_dir": str(self.photons),
        }, [self.inputs], self.output)
        self.assertEqual(command[2:6], ["--image", str(self.image), "--work-dir", str(self.output)])
        self.assertEqual(command[command.index("--global-mode") + 1], "patch")
        self.assertEqual(command[-3:], ["--local-atl03-dir", str(self.photons), "--dry-run"])
        self.assertFalse(self.output.exists())

    def test_batch_command_uses_shared_photons_and_opt_in_patch_optimization(self):
        command = ui_jobs.build_command({
            "mode": "batch", "action": "check", "input": str(self.inputs), "global_optimize": True,
        }, [self.inputs], self.output)
        self.assertEqual(command[command.index("--model") + 1], "SynRS3D")
        self.assertEqual(command[command.index("--atl03-scope") + 1], "shared")
        self.assertIn("--merge-city", command)
        self.assertIn("--global-patch-optimize", command)
        self.assertEqual(command[-1], "--check")

    def test_building_model_selectors_use_the_shared_mls_backend(self):
        values = {"mode": "image", "input": str(self.image)}
        default = ui_jobs.build_command(values, [self.inputs], self.output)
        explicit = ui_jobs.build_command(
            {**values, "offset_model": "MLS-BRN", "footprint_model": "MLS-BRN"},
            [self.inputs], self.output,
        )
        self.assertEqual(default, explicit)
        for field in ui_jobs.BUILDING_MODELS:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "MLS-BRN"):
                ui_jobs.build_command({**values, field: "unsupported"}, [self.inputs], self.output)

    def test_demo_outputs_are_discoverable_and_paired_for_preview(self):
        job = {"action": "run", "mode": "demo", "status": "succeeded"}
        layouts = (
            ("building_height_raster.tif", "per_building_heights.geojson"),
            ("AbsoluteHeight/output.tif", "building_heights.geojson"),
        )
        for index, (raster, vector) in enumerate(layouts):
            with self.subTest(raster=raster):
                folder = self.root / f"demo-{index}"
                for name in (raster, vector):
                    path = folder / "output" / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"result")
                outputs = ui_jobs.collect_outputs(folder, job)
                self.assertEqual([output["id"] for output in outputs], [raster, vector])
                self.assertEqual(ui_jobs.footprint_artifact(raster), vector)

    def test_result_preview_preserves_the_grid_and_reprojects_footprints(self):
        import cv2
        import geopandas as gpd
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        from shapely.geometry import box

        height = self.root / "height.tif"
        vector = self.root / "buildings.geojson"
        values = np.full((8, 16), 22, dtype="uint16")
        values[:, :4] = 65535
        with rasterio.open(height, "w", driver="GTiff", height=8, width=16, count=1,
                           dtype=values.dtype, crs="EPSG:32650",
                           transform=from_origin(350000, 3400000, 1, 1), nodata=65535) as dest:
            dest.write(values, 1)
        gpd.GeoSeries([box(350010, 3399994, 350014, 3399998)], crs=32650).to_crs(4326).to_file(vector)
        with patch.object(ui_preview, "MAX_EDGE", 8):
            result = ui_preview.render(height, vector)
        self.assertEqual((result["width"], result["height"], result["buildings"]), (16, 8, 1))
        self.assertEqual(result["height_max"], 22)

        def alpha(png):
            data = np.frombuffer(b64decode(png.split(",", 1)[1]), dtype="uint8")
            return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)[:, :, 3]

        raster = alpha(result["height_png"])
        self.assertEqual(raster.shape, (4, 8))
        self.assertFalse(raster[:, :2].any())
        self.assertTrue(raster[:, 2:].all())
        outlines = alpha(result["footprint_png"])
        self.assertEqual(outlines.shape, raster.shape)
        self.assertTrue(outlines[:, 4:].any())
        self.assertFalse(outlines[:, :4].any())

    def test_lod1_mesh_uses_extracted_height_and_preserves_courtyards(self):
        import geopandas as gpd
        import numpy as np
        from shapely.geometry import Polygon, box
        from shapely.ops import unary_union

        vector = self.root / "lod1.geojson"
        outer = [(0, 0), (20, 0), (20, 20), (12, 20), (12, 10), (7, 10), (7, 20), (0, 20)]
        courtyard = [(2, 2), (5, 2), (5, 5), (2, 5)]
        footprint = Polygon(outer, [courtyard])
        gpd.GeoDataFrame({"Extracted_Height": [18, 0, None], "Height": [180, 200, 250]},
                         geometry=[footprint, box(30, 0, 32, 2), box(40, 0, 42, 2)],
                         crs="EPSG:32650").to_file(vector)
        result = ui_lod1.render(vector)
        self.assertEqual((result["buildings"], result["skipped"], result["height_max"]), (1, 2, 18))
        vertices = np.frombuffer(b64decode(result["mesh"]), dtype="<f4").reshape(-1, 3, 5)
        self.assertEqual(set(vertices[:, :, 2].flatten()), {0, 18})
        self.assertTrue((vertices[:, :, 3] == 18).all())
        roof = [Polygon(face[:, :2] + 10) for face in vertices[(vertices[:, :, 2] == 18).all(axis=1)]]
        self.assertAlmostEqual(unary_union(roof).symmetric_difference(footprint).area, 0)
        self.assertEqual(sum((face[:, 2].min() == 0 and face[:, 2].max() == 18) for face in vertices),
                         2 * (len(outer) + len(courtyard)))

    def test_file_picker_and_inputs_reject_links_outside_the_data_root(self):
        outside = self.root / "private.tif"
        outside.touch()
        link = self.inputs / outside.name
        link.symlink_to(outside)
        entries = ui_jobs.list_directory(str(self.inputs), (self.inputs,), "image")["entries"]
        self.assertNotIn(outside.name, [item["name"] for item in entries])
        with self.assertRaisesRegex(ValueError, "路径不在"):
            ui_jobs.build_command({"mode": "image", "input": str(link)}, [self.inputs], self.output)

    def test_advanced_values_map_to_both_commands(self):
        for mode, source in (("image", self.image), ("batch", self.inputs)):
            with self.subTest(mode=mode):
                command = ui_jobs.build_command({
                    "mode": mode, "input": str(source), "action": "preview", "window_size": "512",
                    "overlap": "0.25", "min_photon_controls": "6", "ground_filter": "on",
                }, [self.inputs], self.output)
                for flag, expected in (("--window-size", "512"), ("--overlap", "0.25"),
                                       ("--min-photon-controls", "6")):
                    self.assertEqual(command[command.index(flag) + 1], expected)
                self.assertIn("--ground-filter", command)
        base = {"mode": "batch", "input": str(self.inputs), "global_optimize": True,
                "model": "Distill-Any-Depth"}
        command = ui_jobs.build_command({**base, "window_size": "512", "overlap": "0.25",
                                         "ground_filter": "on"}, [self.inputs], self.output)
        self.assertIn("--global-patch-optimize", command)
        self.assertIn("--ground-filter", command)
        for value in ("1", "-0.1", "nan", "0.99999"):
            with self.subTest(overlap=value), self.assertRaisesRegex(ValueError, "重叠率"):
                ui_jobs.build_command({**base, "global_optimize": False, "overlap": value},
                                      [self.inputs], self.output)

    def test_browser_reuses_and_clears_earthdata_credentials(self):
        import json
        from threading import Thread
        from urllib.error import HTTPError
        from urllib.request import HTTPCookieProcessor, Request, build_opener

        from scripts.run_ui import Server

        jobs = ui_jobs.Jobs(self.output, [self.inputs])
        login, password = "browser-session-user", "browser-session-password"
        digest = sha256(f"{login}:{password}".encode()).hexdigest()
        code = (
            "from hashlib import sha256; import os; "
            "auth = os.environ['EARTHDATA_USERNAME'] + ':' + os.environ['EARTHDATA_PASSWORD']; "
            f"print(sha256(auth.encode()).hexdigest() == {digest!r}); raise SystemExit(1)"
        )
        first = build_opener(HTTPCookieProcessor())
        separate = build_opener(HTTPCookieProcessor())
        payload = {"mode": "image"}

        with Server(0, jobs) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"

            def request(path, body=None, opener=first):
                data = json.dumps(body).encode() if body is not None else None
                headers = {"Content-Type": "application/json"} if data is not None else {}
                with opener.open(Request(base + path, data=data, headers=headers), timeout=5) as reply:
                    return json.load(reply), reply.headers

            def wait_failed(job_id):
                for _ in range(100):
                    record, _ = request("/api/jobs/" + job_id)
                    if record["status"] == "failed":
                        self.assertEqual(record["log"].strip(), "True")
                        self.assertNotIn(login, json.dumps(record))
                        self.assertNotIn(password, json.dumps(record))
                        return
                    time.sleep(0.05)
                self.fail("The test worker did not finish")

            try:
                with patch.object(ui_jobs, "build_command", return_value=[sys.executable, "-c", code]):
                    saved, headers = request("/api/jobs", {**payload, "earthdata_username": login,
                                                         "earthdata_password": password})
                    self.assertTrue(saved["earthdata_saved"])
                    for token in ("HttpOnly", "SameSite=Strict"):
                        self.assertIn(token, headers["Set-Cookie"])
                    self.assertNotIn(password, headers["Set-Cookie"])
                    wait_failed(saved["id"])
                    request("/api/jobs/" + saved["id"] + "/resume", {})
                    wait_failed(saved["id"])
                    again, _ = request("/api/jobs", payload)
                    wait_failed(again["id"])
                    self.assertFalse(request("/api/earthdata", opener=separate)[0]["saved"])
                    with self.assertRaises(HTTPError) as denied:
                        request("/api/jobs", payload, opener=separate)
                    self.assertEqual(denied.exception.code, 400)
                    request("/api/earthdata/clear", {})
                    self.assertFalse(request("/api/earthdata")[0]["saved"])
                    with self.assertRaises(HTTPError) as cleared:
                        request("/api/jobs", payload)
                    self.assertEqual(cleared.exception.code, 400)
            finally:
                server.shutdown()
                thread.join(timeout=5)

    def test_online_workers_require_and_receive_fresh_credentials(self):
        jobs = ui_jobs.Jobs(self.output, [self.inputs])
        login = "synthetic-user-edl"
        password = "synthetic secret @edl"
        digest = sha256(f"{login}:{password}".encode()).hexdigest()
        code = (
            "from hashlib import sha256; import os; "
            "auth = os.environ['EARTHDATA_USERNAME'] + ':' + os.environ['EARTHDATA_PASSWORD']; "
            f"ok = sha256(auth.encode()).hexdigest() == {digest!r}; "
            "print('credentials accepted' if ok else 'credentials rejected'); raise SystemExit(0 if ok else 1)"
        )
        with patch.object(ui_jobs, "build_command", return_value=[sys.executable, "-c", code]):
            with self.assertRaisesRegex(ValueError, "登录名和密码"):
                jobs.start({"mode": "image"})
            self.assertEqual(jobs.list(), [])
            job, child = jobs.start({"mode": "image", "earthdata_username": login,
                                     "earthdata_password": "synthetic incorrect password"})
        child.wait(timeout=10)
        self.assertEqual(jobs.detail(job["id"])["status"], "failed")
        with self.assertRaisesRegex(ValueError, "登录名和密码"):
            jobs.resume(job["id"])
        _, child = jobs.resume(job["id"], {"earthdata_username": login, "earthdata_password": password})
        child.wait(timeout=10)
        result = jobs.detail(job["id"])
        self.assertTrue(result["requires_earthdata"])
        self.assertIn("credentials accepted", result["log"])
        for path in (self.output / job["id"]).iterdir():
            if path.is_file():
                self.assertNotIn(login, path.read_text())
                self.assertNotIn(password, path.read_text())
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["attempt"], 2)
        for mode, action, argv in (("image", "preview", []), ("image", "check", []),
                                   ("image", "run", ["--local-atl03-dir"]), ("demo", "run", [])):
            with self.subTest(mode=mode, action=action):
                self.assertEqual(ui_jobs.earthdata_environment(
                    {"mode": mode, "action": action, "argv": argv}, {}), {})

    def test_cancel_and_resume_keep_the_same_work_directory(self):
        jobs = ui_jobs.Jobs(self.output, [self.inputs])
        marker = self.root / "started"
        code = (
            "from pathlib import Path; import time; "
            f"marker = Path({str(marker)!r}); "
            "seen = marker.exists(); marker.touch(); "
            "print('worker running', flush=True); time.sleep(0 if seen else 30)"
        )
        with patch.object(ui_jobs, "build_command",
                          return_value=[sys.executable, "-c", code, "--local-atl03-dir", str(self.photons)]):
            job, worker = jobs.start({"mode": "image"})
        try:
            deadline = time.monotonic() + 10
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(marker.exists())
            other = ui_jobs.Jobs(self.output, [self.inputs])
            self.assertEqual(other.detail(job["id"])["status"], "running")
            with self.assertRaises(ui_jobs.BusyError):
                other.resume(job["id"])
        finally:
            jobs.cancel(job["id"])
            worker.wait(timeout=10)
        cancelled = jobs.detail(job["id"])
        self.assertEqual(cancelled["status"], "cancelled")
        _, worker = jobs.resume(job["id"])
        worker.wait(timeout=10)
        result = jobs.detail(job["id"])
        self.assertEqual((result["status"], result["attempt"]), ("success", 2))
        self.assertEqual(result["output"], cancelled["output"])
        self.assertEqual(result["argv"], cancelled["argv"])
