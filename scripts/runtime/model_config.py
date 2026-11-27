"""Model entry points and checkpoint locations shared by the release runners."""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GEO_ENV = "hab-core"
DEFAULT_MLSBRN_ENV = "mlsbrn"
DEPTH_MODELS = {
    # Runner relative to model directory, Conda environment, legacy checkpoint.
    "Distill-Any-Depth": ("tools/testers/run_tif.py", "distill-any-depth", "checkpoint/Dav2-large.safetensors"),
    "DEPTH-ANYTHING-V1": ("run_tif.py", "depth-anything-v1", "checkpoints/depth_anything_vitl14.pth"),
    "DEPTH-ANYTHING-V2": ("run_tif.py", "depth-anything-v2", "checkpoints/depth_anything_v2_vitl.pth"),
    "SynRS3D": ("run_tif.py", "synrs3d", "pretrain/RS3DAda_vitl_DPT_height.pth"),
}
MLS_MISSION = ROOT / "MLS-BRN/work_dirs/loft_foahfm_r50_fpn_2x_bonai_ssl[test]"
MLS_CONFIG = MLS_MISSION / "loft_foahfm_r50_fpn_2x_bonai_ssl.py"


def depth_directory(model):
    return ROOT / "MDE" / model


def depth_environment(model, environ=None):
    """Resolve the primary backend environment; seam workers use their own environment."""
    environ = os.environ if environ is None else environ
    return environ.get("HAB_DEPTH_ENV") or DEPTH_MODELS[model][1]


def checkpoint_path(model):
    if model == "MLS-BRN":
        preferred = ROOT / "MLS-BRN/checkpoints/mlsbrn-inference.pth"
        legacy_paths = (
            ROOT / "checkpoints/MLS-BRN/latest.pth",
            MLS_MISSION / "latest.pth",
            MLS_MISSION / "epoch_24.pth",
        )
    else:
        legacy = Path(DEPTH_MODELS[model][2])
        preferred = ROOT / "checkpoints" / model / legacy.name
        legacy_paths = (depth_directory(model) / legacy,)
    return next((path for path in (preferred, *legacy_paths) if path.is_file()), preferred)


def depth_runner(model):
    return depth_directory(model) / DEPTH_MODELS[model][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=DEPTH_MODELS)
    args = parser.parse_args()
    # One value per line for Bash mapfile. Do not generate executable shell text.
    print(depth_runner(args.model))
    print(depth_environment(args.model))
    print(checkpoint_path(args.model))
    print(MLS_CONFIG)
    print(checkpoint_path("MLS-BRN"))


if __name__ == "__main__":
    main()
