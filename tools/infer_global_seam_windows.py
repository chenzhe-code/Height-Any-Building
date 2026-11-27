#!/usr/bin/env python3
"""SynRS3D worker for crossing RGB windows; writes only compressed NumPy caches."""

import argparse
import json
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    os.environ.pop("PROJ_DATA", None)

import numpy as np
import rasterio
from rasterio.windows import Window


def read_crossing_rgb(seam: dict, tiles: dict, patch_size: int = 1022) -> np.ndarray:
    half = patch_size // 2
    a = tiles[seam["a"]]
    b = tiles[seam["b"]]
    start = seam["start"]
    if seam["axis"] == "vertical":
        aw = Window(a["width"] - half, start, half, patch_size)
        bw = Window(0, start, half, patch_size)
        axis = 2
    else:
        aw = Window(start, a["height_px"] - half, patch_size, half)
        bw = Window(start, 0, patch_size, half)
        axis = 1
    with rasterio.open(a["image"]) as left, rasterio.open(b["image"]) as right:
        pixels = np.concatenate((left.read(window=aw), right.read(window=bw)), axis=axis)
    if pixels.shape != (3, patch_size, patch_size):
        raise ValueError(f"wrong crossing window shape: {seam['name']}: {pixels.shape}")
    return np.moveaxis(pixels, 0, -1).copy()


def main() -> None:
    import torch
    from albumentations import Compose, Normalize
    from albumentations.pytorch import ToTensorV2

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "MDE" / "SynRS3D"))
    from models.dpt import DPT_DINOv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    jobs = json.loads(args.task_file.read_text(encoding="utf-8"))
    tiles = jobs["tiles"]
    patch_size = int(jobs.get("patch_size", 1022))
    if patch_size <= 0 or patch_size % 2:
        raise ValueError("patch_size must be a positive even integer")
    model = DPT_DINOv2(
        encoder="vitl",
        head_configs=[{"name": "regression", "nclass": 1},
                      {"name": "segmentation", "nclass": 8}],
        pretrained=False,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.cuda().eval()
    transform = Compose([
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        ToTensorV2(),
    ])
    for number, item in enumerate(jobs["seams"], 1):
        seam = item["seam"]
        target = Path(item["cache"])
        if target.is_file():
            continue
        rgb = read_crossing_rgb(seam, tiles, patch_size)
        pad = (14 - patch_size % 14) % 14
        if pad:
            rgb = np.pad(rgb, ((0, pad), (0, pad), (0, 0)), mode="reflect")
        tensor = transform(image=rgb)["image"].unsqueeze(0).float().cuda()
        with torch.no_grad():
            prediction = model(tensor)["regression"].data.cpu().numpy().squeeze()
        prediction = prediction[:patch_size, :patch_size].astype(np.float32)
        if prediction.shape != (patch_size, patch_size) or not np.isfinite(prediction).all():
            raise ValueError(f"invalid SynRS3D output: {seam['name']}")
        temporary = target.with_suffix(".tmp")
        with temporary.open("wb") as output:
            np.savez_compressed(output, prediction=prediction)
        os.replace(temporary, target)
        if number % 25 == 0 or number == len(jobs["seams"]):
            print(f"[GPU worker] {number}/{len(jobs['seams'])} crossing windows", flush=True)


if __name__ == "__main__":
    main()
