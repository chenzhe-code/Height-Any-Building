"""Check the runtimes and checkpoints needed by the full HAB workflow."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess

if __package__:
    from .model_config import (
        DEFAULT_GEO_ENV, DEFAULT_MLSBRN_ENV, DEPTH_MODELS, MLS_CONFIG, ROOT,
        checkpoint_path, depth_directory, depth_environment, depth_runner,
    )
else:
    from model_config import (
        DEFAULT_GEO_ENV, DEFAULT_MLSBRN_ENV, DEPTH_MODELS, MLS_CONFIG, ROOT,
        checkpoint_path, depth_directory, depth_environment, depth_runner,
    )


def _probe(conda, environment, command, env, *, cwd=ROOT):
    result = subprocess.run([conda, "run", "-n", environment, "python", *command],
                            cwd=cwd, env=env, capture_output=True, text=True)
    if result.returncode:
        details = "\n".join(part.strip() for part in (result.stderr, result.stdout) if part.strip())
        raise ValueError(f"Runtime check failed in Conda environment {environment}. "
                         f"See environments/README.md.\n{details}")


def check_runtime(model, gpus, env=None, *, seam=False):
    env = dict(os.environ if env is None else env)
    runner = depth_runner(model)
    mls = ROOT / "MLS-BRN"
    seam_runner = ROOT / "tools/infer_global_seam_windows.py"
    assets = {
        "depth checkpoint": checkpoint_path(model),
        "MLS-BRN checkpoint": checkpoint_path("MLS-BRN"),
        "depth entry point": runner,
        "MLS-BRN config": MLS_CONFIG,
        "MLS-BRN backend": mls / "mmdet/__init__.py",
        "bstool backend": mls / "bstool/bstool/__init__.py",
        "wwtool backend": mls / "wwtool/wwtool/__init__.py",
    }
    if seam:
        assets["SynRS3D seam checkpoint"] = checkpoint_path("SynRS3D")
        assets["SynRS3D seam entry point"] = seam_runner
    missing = [f"{label}: {path}" for label, path in assets.items()
               if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError("Required runtime files are missing or empty. See the README model "
                         "download table and environments/README.md.\n" + "\n".join(missing))
    conda = shutil.which(env.get("CONDA_BIN") or "conda", path=env.get("PATH"))
    if conda is None:
        raise ValueError("Conda executable not found. Set CONDA_BIN or activate Conda.")
    geo_env = env.get("GEO_ENV") or DEFAULT_GEO_ENV
    mls_env = env.get("MLSBRN_ENV") or DEFAULT_MLSBRN_ENV
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    env["CUDA_VISIBLE_DEVICES"] = gpus
    gpu_check = (
        "import torch; "
        f"assert torch.cuda.device_count() == {len(gpus.split(','))}, 'requested CUDA devices unavailable'; "
        "torch.ones(1, device='cuda').cpu()"
    )
    _probe(conda, geo_env, ["-c",
        "import CSF, geopandas, h5py, rasterio, requests, scipy, shapely; "
        "from osgeo import gdal; from rasterio.crs import CRS; CRS.from_epsg(4326); "
        "assert hasattr(CSF, 'VecInt'); CSF.CSF()"], env)
    primary_env = depth_environment(model, env)
    seam_env = DEPTH_MODELS["SynRS3D"][1]
    backends = [(model, primary_env, runner)]
    if seam and (model != "SynRS3D" or primary_env != seam_env):
        backends.append(("SynRS3D", seam_env, seam_runner))
    for depth_model, backend_env, entry in backends:
        depth_env_vars = env | {"PYTHONPATH": os.pathsep.join(
            path for path in (str(depth_directory(depth_model)), env.get("PYTHONPATH")) if path)}
        depth_check = (
            gpu_check + "; import cv2, numpy, os, sys; torch.from_numpy(numpy.zeros(1, dtype=numpy.float32)); "
        )
        if depth_model == "SynRS3D":
            depth_check += "from models.dpt import _load_dinov2_backbone; _load_dinov2_backbone('vits', False); "
        _probe(conda, backend_env, ["-c", depth_check +
            "os.execv(sys.executable, [sys.executable, sys.argv[1], '--help'])",
            str(entry)], depth_env_vars, cwd=depth_directory(depth_model))
    if seam and len(backends) == 1:
        _probe(conda, seam_env, [str(seam_runner), "--help"], env)
    mls_env_vars = env | {"PYTHONPATH": os.pathsep.join(
        path for path in (str(mls), str(mls / "bstool"), env.get("PYTHONPATH")) if path)}
    _probe(conda, mls_env, ["-c",
        gpu_check + "; import bstool.datasets.dump, geojson, geopandas, mmdet.models, mmcv.ops, "
        "networkx, rasterio, seaborn, skimage, wwtool"], mls_env_vars)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=DEPTH_MODELS)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--seam", action="store_true", help="Also check the independent SynRS3D seam model.")
    args = parser.parse_args()
    try:
        check_runtime(args.model, args.gpus, seam=args.seam)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
