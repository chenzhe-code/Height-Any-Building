"""Regression checks for stage reuse and ATL03 request isolation."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts.runtime import pipeline_cache as cache
from utils.io_utils import list_files_in_directory

ROOT = Path(__file__).resolve().parents[1]

# Stand in for the heavy stage executables, keeping the real shell, cache logic,
# artifact names and stage validation contracts.
FAKE_CONDA = """
import os
from pathlib import Path
import sys

args = sys.argv[sys.argv.index("python") + 1:]
script = Path(args[0]).name
job = Path(os.environ["JOB_DIR"])
patches = job / "windowsize-1024-overlap-0.5-downratio-1-patches"
depth = job / "Distill-Any-Depth-windowsize-1024-overlap-0.5-downratio-1-depth"

def write(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("generated")

def arg(name):
    return args[args.index(name) + 1]

if script == "split_image.py":
    write(patches / "window_0_x0_y0.tif")
elif script == "prepare_footprint_inputs.py":
    write(Path(arg("--output-folder")) / "window_0_x0_y0.png")
    write(arg("--json-output"))
elif script == "-m":
    write(arg("--out"))
elif script == "bonai_evaluation.py":
    write(Path(args[2]) / "footprint_direct.csv")
elif script == "convert_footprints.py":
    write(arg("--output-path"))
elif script == "PLiDAR_HeightExtract.py":
    write(arg("--save-path"))
elif script == "build_control_height.py":
    write(arg("--output-tif-path"))
elif script == "run_depth_backend.py":
    if os.environ.get("HAB_TEST_FAIL_DEPTH") == "1":
        sys.exit(4)
    write(depth / "depth/window_0_x0_y0_depth.tif")
elif script == "calibrate_heights.py":
    write(depth / "AbsoluteHeight/output.tif")
    write(depth / "coarse_AH/CH_final_ransac.tif")
elif script == "assign_building_heights.py":
    write(arg("--output-path"))
elif script == "-c" and "count_nonzero" in args[1]:
    print(10)
"""


class PipelineCacheTests(unittest.TestCase):
    def test_stage_reuse_tracks_parameters_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stub = root / "conda"
            stub.write_text(f"#!{sys.executable}\n" + textwrap.dedent(FAKE_CONDA))
            stub.chmod(0o755)
            image = root / "image.tif"
            image.write_text("input")
            atl03 = root / "atl03"
            atl03.mkdir()
            photon = atl03 / "track.h5"
            photon.write_text("photons")
            environment = os.environ | {
                "PATH": str(root) + os.pathsep + os.environ["PATH"],
                "JOB_DIR": str(root / "job"), "INPUT_TIF": str(image),
                "ICESAT_PATH": str(atl03), "MODEL_NAME": "Distill-Any-Depth",
                "COMPUTE_UNCERTAINTY": "0", "PR": "700", "RESUME": "1",
                "WS": "1024", "OL": "0.5", "DR": "1", "BS": "-1,-1",
                "FUSION_MODE": "mean", "CONTROL_SOURCE": "photon", "PHOTON_TABLE": "",
                "PHOTON_TABLE_SHARED": "0", "HEIGHT_SOURCE": "absolute",
            }

            def launch(script="run_full_pipeline.sh", args=(), **changes):
                return subprocess.run(
                    ["bash", str(ROOT / "scripts" / script), *args],
                    env=environment | changes, capture_output=True, text=True,
                )

            def runs(result):
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                return [int(stage) for stage in re.findall(r"\[RUN\] \[(\d+)/10\]", result.stdout)]

            self.assertEqual(runs(launch()), list(range(1, 11)))
            self.assertEqual(runs(launch()), [])
            status_path = root / "job/stage_status.json"
            self.assertEqual(json.loads(status_path.read_text())["stages"]["8"]["status"], "reused")
            self.assertEqual(runs(launch(ICESAT_PATH="")), [])
            self.assertEqual(runs(launch(PR="560")), [8, 9, 10])
            photon.write_text("new photon source")
            self.assertEqual(runs(launch(PR="560")), [6, 7, 9, 10])
            failed = launch(PR="561", HAB_TEST_FAIL_DEPTH="1")
            self.assertEqual(failed.returncode, 4, failed.stderr + failed.stdout)
            status = json.loads(status_path.read_text())
            self.assertEqual((status["status"], status["exit_code"]), ("failed", 4))
            stage = status["stages"]["8"]
            self.assertEqual((stage["status"], stage["exit_code"]), ("failed", 4))
            self.assertGreaterEqual(stage["elapsed_seconds"], 0)
            self.assertFalse((root / "job/.pipeline-cache/stage-8.json").exists())
            self.assertEqual(runs(launch(PR="561")), [8, 9, 10])
            self.assertEqual(runs(launch(PR="561", GRAPH_OVERLAP_LAMBDA="1.5")), [9, 10])
            self.assertEqual(runs(launch(PR="561", GRAPH_OVERLAP_LAMBDA="1.5",
                                         OVERLAP_MASK="nonground")), [9, 10])
            (root / "job/.pipeline-cache/stage-8.json").unlink()
            self.assertEqual(runs(launch(PR="561")), [8, 9, 10])

            modular_job = root / "modular"
            self.assertEqual(runs(launch("split_image.sh", JOB_DIR=str(modular_job),
                                         MODEL_NAME="DEPTH-ANYTHING-V1", COMPUTE_UNCERTAINTY="1")), [1])
            self.assertEqual(runs(launch("depth_estimation.sh", JOB_DIR=str(modular_job),
                                         PHOTON_TABLE_SHARED="1",
                                         PHOTON_TABLE=str(root / "unavailable.csv.gz"))), [8])
            self.assertFalse((modular_job / "mls_results").exists())
            self.assertFalse((modular_job / "sparse_control_height.tif").exists())
            for entry, steps in (("building_extraction.sh", [2, 3, 4, 5]),
                                 ("sparse_height_extraction.sh", [6, 7]),
                                 ("PRD2WAH.sh", [9])):
                self.assertEqual(runs(launch(entry, JOB_DIR=str(modular_job))), steps)
            self.assertEqual(runs(launch(args=("--steps", "10"), JOB_DIR=str(modular_job))), [10])
            self.assertEqual(runs(launch(JOB_DIR=str(modular_job))), [])
            self.assertEqual(launch("depth_estimation.sh", args=("--steps", "9")).returncode, 2)

    def test_batch_plan_does_not_require_a_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "tile.tif").touch()
            output = root / "output"
            env = os.environ | {"PYTHON_BIN": sys.executable, "CONDA_BIN": str(root / "missing-conda")}
            command = [
                "bash", str(ROOT / "scripts/run_batch_work_dir.sh"),
                "--input-dir", str(inputs), "--output-root", str(output),
            ]
            plan = subprocess.run(command + ["--dry-run"], env=env, capture_output=True, text=True)
            self.assertEqual(plan.returncode, 0, plan.stderr)
            self.assertIn("Dry-run tile plan", plan.stdout)
            check = subprocess.run(command + ["--check"], env=env, capture_output=True, text=True)
            self.assertNotEqual(check.returncode, 0)
            self.assertIn("conda executable not found", check.stderr)
            self.assertFalse(output.exists())

    def test_atl03_reuse_is_scoped_to_dates_and_complete_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.tif"
            image.write_text("extent")
            args = SimpleNamespace(tif=image, tif_dir=None, cache_root=root / "downloads")
            downloads = []

            def download(command, env, **kwargs):
                active = Path(env["ATL03_OUTPUT_DIR"])
                downloads.append(active)
                track = active / "track.h5"
                track.write_text(env["START_DATE"])
                cache.write_json(active / "atl03_download_complete.json", {
                    "status": "complete", "files": [{"name": track.name, "size_bytes": track.stat().st_size}],
                })

            def select(start):
                stream = io.StringIO()
                with redirect_stdout(stream), mock.patch.dict(
                        os.environ, START_DATE=start, END_DATE="2025-12-31", DOWNLOAD="true"):
                    cache.atlas_download(args)
                return Path(stream.getvalue().strip())

            with mock.patch.object(cache.subprocess, "run", side_effect=download):
                first = select("2022-01-01")
                marker = first / "atl03_download_complete.json"
                legacy = json.loads(marker.read_text())
                del legacy["status"]
                cache.write_json(marker, legacy)
                cache.write_json(first / "atl03_harmony_job.json", {"status": "successful"})
                self.assertEqual(select("2022-01-01"), first)
                self.assertEqual(downloads, [first])
                second = select("2025-01-01")
                self.assertNotEqual(first, second)
                self.assertEqual(downloads, [first, second])
                (second / "track.h5").unlink()
                self.assertEqual(select("2025-01-01"), second)
                self.assertEqual(downloads, [first, second, second])
                self.assertEqual((first / "track.h5").read_text(), "2022-01-01")

    def test_ambiguous_atl03_cache_requires_a_selection_then_remembers_it(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            root = job / "ICESat-2"
            for name in ("request-old", "request-new"):
                request = root / name
                request.mkdir(parents=True)
                (request / "track.h5").write_text(name)
            with self.assertRaisesRegex(ValueError, "Set ICESAT_PATH"):
                cache.select_atl03(job, root)
            chosen = root / "request-new"
            self.assertEqual(cache.select_atl03(job, root, chosen), chosen)
            self.assertEqual(cache.select_atl03(job, root), chosen)
            self.assertEqual(json.loads((job / "atl03_selection.json").read_text())["directory"], str(chosen))

    def test_partial_atl03_is_opt_in_and_never_committed_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.tif"
            image.write_text("extent")
            args = SimpleNamespace(tif=image, tif_dir=None, cache_root=root / "downloads")

            def download(command, env, **kwargs):
                active = Path(env["ATL03_OUTPUT_DIR"])
                (active / "track.h5").write_text("subset")
                cache.write_json(active / "atl03_download_partial.json", {
                    "status": "partial", "files": [{"name": "track.h5", "size_bytes": 6}],
                })

            with mock.patch.object(cache.subprocess, "run", side_effect=download) as launch:
                with mock.patch.dict(os.environ, HARMONY_IGNORE_ERRORS="false", DOWNLOAD="true"):
                    with self.assertRaisesRegex(ValueError, "No complete ATL03 result"):
                        cache.atlas_download(args)
                with mock.patch.dict(os.environ, HARMONY_IGNORE_ERRORS="true", DOWNLOAD="true"):
                    stream = io.StringIO()
                    with redirect_stdout(stream):
                        cache.atlas_download(args)
                    selected = Path(stream.getvalue().strip())
            self.assertEqual(launch.call_count, 2)
            self.assertFalse((selected / ".complete.json").exists())
            with self.assertRaisesRegex(ValueError, "Partial ATL03"):
                cache.select_atl03(root, args.cache_root, selected)
            self.assertEqual(cache.select_atl03(root, args.cache_root, selected, allow_partial=True), selected)
            self.assertEqual(cache.select_atl03(root, args.cache_root), selected)

    def test_existing_unmanaged_successful_harmony_download_is_usable_as_local_data(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            data = job / "ICESat-2"
            data.mkdir()
            (data / "track.h5").write_text("subset")
            report = data / "atl03_harmony_job.json"
            report.write_text('{"status": "successful"}')
            self.assertEqual(cache.select_atl03(job, data), data)
            report.write_text('{"status": "successful with errors"}')
            with self.assertRaisesRegex(ValueError, "No verified Harmony result"):
                cache.select_atl03(job, data)

    def test_rh2ah_window_listing_has_stable_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["window_2_x1_y0_depth.tif", "window_0_x0_y0_depth.tif", "window_1_x0_y1_depth.tif"]:
                (root / name).touch()
            names = [Path(path).name for path in list_files_in_directory(root)]
            self.assertEqual(names, sorted(names))


if __name__ == "__main__":
    unittest.main()
