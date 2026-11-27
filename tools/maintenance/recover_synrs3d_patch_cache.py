#!/usr/bin/env python3
"""Recover raw SynRS3D patch archives without rerunning retained HAB stages.

Run in the hab-core environment. Only _global_patch_cache is written. Each GPU
processes one tile at a time using the existing split, inference and archive
entry points; successful temporary RGB/depth files are removed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from zipfile import BadZipFile

import numpy as np
import rasterio

PROJECT = Path(__file__).resolve().parents[2]
if not __package__:
    sys.path.insert(0, str(PROJECT))

from scripts.runtime.model_config import checkpoint_path
from tools import global_patch_height as gp


def gpu_ids(value):
    ids = value.split(",")
    if not ids or any(not item.isdigit() for item in ids) or len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError("--gpus requires distinct nonnegative GPU IDs")
    return ids


def retained_images(input_dir, results_root):
    images = sorted([*input_dir.glob("*.tif"), *input_dir.glob("*.tiff")])
    if not images:
        raise ValueError(f"no input tiles: {input_dir}")
    if len({image.stem for image in images}) != len(images):
        raise ValueError("duplicate input tile stems")
    for image in images:
        directory = results_root / image.stem
        status = json.loads((directory / "status.json").read_text())
        if status.get("status") != "success" or status.get("model_name") != "SynRS3D":
            raise ValueError(f"expected successful retained SynRS3D HAB tile: {image.stem}")
        if status.get("input_tif") and Path(status["input_tif"]).resolve() != image.resolve():
            raise ValueError(f"retained tile input mismatch: {image.stem}")
        for name in (f"{image.stem}_building_heights.geojson", "footprint_direct_nms.geojson",
                     "sparse_control_height.tif", f"{image.stem}_absolute_height.tif"):
            path = directory / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing retained artifact: {path}")
        with rasterio.open(image) as source:
            if (source.count != 3 or source.dtypes != ("uint8",) * 3 or source.crs is None
                    or not source.crs.is_projected or source.res != (1.0, 1.0)
                    or source.transform.b or source.transform.d
                    or source.transform.a != 1 or source.transform.e != -1):
                raise ValueError(f"expected north-up 1 m projected UInt8 RGB: {image}")
            gp.expected_positions(source.width, source.height)
            with rasterio.open(directory / f"{image.stem}_absolute_height.tif") as hab:
                if (hab.count < 1 or hab.crs != source.crs or hab.transform != source.transform
                        or hab.shape != source.shape):
                    raise ValueError(f"retained HAB grid mismatch: {image.stem}")
    return images


def cache_complete(image, cache_root):
    """Validate the whole expected coordinate set and all archived predictions."""
    directory = cache_root / "interior" / image.stem
    manifest = directory / "manifest.json"
    try:
        data = json.loads(manifest.read_text())
        with rasterio.open(image) as source:
            positions = gp.expected_positions(source.width, source.height)
            if (data.get("width"), data.get("height")) != (source.width, source.height):
                return False
        if (data.get("version") != gp.VERSION or data.get("input") != str(image.resolve())
                or data.get("tile") != image.stem):
            return False
        records = data["patches"]
        if len(records) != len(positions) or {(r["x"], r["y"]) for r in records} != positions:
            return False
        for record in records:
            path = Path(record["cache"])
            expected = directory / f"x{record['x']}_y{record['y']}.npz"
            if path.resolve() != expected.resolve():
                return False
            with path.open("rb") as archive:
                with np.load(archive, allow_pickle=False) as saved:
                    prediction = saved["prediction"]
                    if prediction.shape != (gp.PATCH_SIZE, gp.PATCH_SIZE) or not np.isfinite(prediction).all():
                        return False
        return True
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
        return False


@contextmanager
def recovery_lock(cache_root):
    cache_root.mkdir(parents=True, exist_ok=True)
    # Keep the lock file: unlinking it could let another process lock a new inode.
    with (cache_root / ".recover_synrs3d.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another raw patch recovery is running for this cache") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run_command(command, log, env):
    print("RUN " + " ".join(str(item) for item in command), file=log, flush=True)
    subprocess.run([str(item) for item in command], cwd=PROJECT, env=env,
                   stdout=log, stderr=subprocess.STDOUT, check=True)


def recover_tile(image, gpu, args, run_root):
    temporary = run_root / "temporary" / image.stem
    temporary.mkdir(parents=True)
    rgb, depth = temporary / "rgb", temporary / "prediction"
    log_path = run_root / f"{image.stem}.log"
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1", NO_ALBUMENTATIONS_UPDATE="1")
    env.pop("PROJ_DATA", None)
    print(f"[RUN GPU {gpu}] {image.stem}; log: {log_path}", flush=True)
    try:
        with log_path.open("w") as log:
            run_command([sys.executable, PROJECT / "utils/split_image.py", "--input_path", image,
                         "--output_dir", rgb, "--window_size", "1024", "--overlap", "0.5",
                         "--down_ratio", "1"], log, env)
            run_command([args.conda_bin, "run", "--no-capture-output", "-n", "synrs3d", "python",
                         PROJECT / "MDE/SynRS3D/run_tif.py", "--img_path", rgb,
                         "--checkpoint", args.checkpoint, "--output_dir", depth,
                         "--seed", "1234", "--processing_res", "1022",
                         "--arch_name", "depthanything-large", "--grayscale", "--save_tif"], log, env)
            run_command([sys.executable, PROJECT / "tools/global_patch_height.py", "--archive-tile",
                         "--input-tif", image, "--tile-name", image.stem,
                         "--depth-dir", depth, "--cache-root", args.cache_root], log, env)
            if not cache_complete(image, args.cache_root):
                raise ValueError(f"archive validation failed: {image.stem}")
        shutil.rmtree(temporary)
        print(f"[OK GPU {gpu}] {image.stem}", flush=True)
        return {"tile": image.stem, "gpu": gpu, "status": "success", "log": str(log_path)}
    except Exception as error:
        with log_path.open("a") as log:
            print(f"FAILED: {type(error).__name__}: {error}", file=log)
        print(f"[FAIL GPU {gpu}] {image.stem}: {error}; retained: {temporary}", file=sys.stderr, flush=True)
        return {"tile": image.stem, "gpu": gpu, "status": "failed", "error": str(error),
                "log": str(log_path), "temporary": str(temporary)}


def worker(gpu, images, args, run_root):
    return [recover_tile(image, gpu, args, run_root) for image in images]


def recover(args):
    images = retained_images(args.input_dir, args.results_root)
    if args.expected_tiles and len(images) != args.expected_tiles:
        raise ValueError(f"expected {args.expected_tiles} tiles, found {len(images)}")
    if not args.checkpoint.is_file() or not args.conda_bin.is_file():
        raise ValueError("checkpoint and conda executable must exist")
    if args.dry_run:
        pending = [image for image in images if not cache_complete(image, args.cache_root)]
        print(json.dumps({"dry_run": True, "input_tiles": len(images),
                          "reused_tiles": len(images) - len(pending), "recover_tiles": len(pending),
                          "cache_root": str(args.cache_root),
                          "workers": {gpu: [p.stem for p in pending[i::len(args.gpus)]]
                                      for i, gpu in enumerate(args.gpus)}}, indent=2))
        return 0
    with recovery_lock(args.cache_root):
        # Check after acquiring the lock so an earlier completed run can be reused.
        pending = [image for image in images if not cache_complete(image, args.cache_root)]
        if not pending:
            print(f"All {len(images)} raw patch archives are complete; nothing to recover.")
            return 0
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        run_root = args.cache_root / "_recovery" / run_id
        run_root.mkdir(parents=True)
        print(f"Recovering {len(pending)} tiles; reusing {len(images) - len(pending)}; logs: {run_root}", flush=True)
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
            futures = [executor.submit(worker, gpu, pending[i::len(args.gpus)], args, run_root)
                       for i, gpu in enumerate(args.gpus)]
            outcomes = [record for future in futures for record in future.result()]
        failed = sum(record["status"] == "failed" for record in outcomes)
        (run_root / "summary.json").write_text(json.dumps({"checkpoint": str(args.checkpoint),
            "reused_tiles": len(images) - len(pending), "failed_tiles": failed, "tiles": outcomes}, indent=2) + "\n")
        print(f"Recovery complete: {len(outcomes) - failed} recovered, {failed} failed; {run_root / 'summary.json'}", flush=True)
        return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--checkpoint", type=Path,
                        default=checkpoint_path("SynRS3D"))
    parser.add_argument("--gpus", type=gpu_ids, default="0")
    parser.add_argument("--conda-bin", type=Path, default=Path(shutil.which("conda") or "conda"))
    parser.add_argument("--expected-tiles", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in ("input_dir", "results_root", "checkpoint", "conda_bin"):
        setattr(args, name, getattr(args, name).resolve())
    args.cache_root = (args.cache_root or args.results_root / "_global_patch_cache").resolve()
    try:
        return recover(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
