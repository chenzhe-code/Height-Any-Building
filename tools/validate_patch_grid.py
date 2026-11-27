#!/usr/bin/env python3
"""Validate that a patch directory exactly covers its source GeoTIFF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import rasterio
from rasterio.windows import Window

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.window_grid import window_positions


def expected_patch_records(
    input_tif: Path,
    window_size: int,
    overlap: float,
    down_ratio: float,
) -> list[dict]:
    if down_ratio <= 0:
        raise ValueError("down_ratio must be positive")
    with rasterio.open(input_tif) as source:
        positions = window_positions(source.width, source.height, window_size, overlap)
    records = []
    for index, (x, y) in enumerate(positions):
        records.append(
            {
                "filename": (
                    f"window_{index}_x{int(x // down_ratio)}_"
                    f"y{int(y // down_ratio)}.tif"
                ),
                "x": x,
                "y": y,
            }
        )
    return records


def validate_patch_grid(
    input_tif: Path,
    patch_dir: Path,
    window_size: int,
    overlap: float,
    down_ratio: float,
) -> dict:
    records = expected_patch_records(input_tif, window_size, overlap, down_ratio)
    expected_names = {record["filename"] for record in records}
    actual_names = {
        path.name
        for path in patch_dir.glob("*.tif")
        if path.is_file()
    } if patch_dir.is_dir() else set()
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        raise ValueError(
            f"patch grid mismatch: missing={missing[:5]} ({len(missing)} total), "
            f"unexpected={unexpected[:5]} ({len(unexpected)} total)"
        )

    output_size = int(window_size // down_ratio)
    with rasterio.open(input_tif) as source:
        for record in records:
            patch_path = patch_dir / record["filename"]
            with rasterio.open(patch_path) as patch:
                expected_transform = rasterio.windows.transform(
                    Window(record["x"], record["y"], window_size, window_size),
                    source.transform,
                ) * source.transform.scale(down_ratio, down_ratio)
                if patch.width != output_size or patch.height != output_size:
                    raise ValueError(
                        f"invalid patch size for {patch_path}: "
                        f"{patch.width}x{patch.height}, expected {output_size}x{output_size}"
                    )
                if patch.crs != source.crs or not patch.transform.almost_equals(expected_transform):
                    raise ValueError(f"invalid georeferencing for patch: {patch_path}")

    return {
        "patch_count": len(records),
        "x_starts": len({record["x"] for record in records}),
        "y_starts": len({record["y"] for record in records}),
        "covered_width": max(record["x"] for record in records) + window_size,
        "covered_height": max(record["y"] for record in records) + window_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tif", required=True)
    parser.add_argument("--patch-dir", required=True)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--down-ratio", type=float, default=1)
    args = parser.parse_args()
    result = validate_patch_grid(
        Path(args.input_tif),
        Path(args.patch_dir),
        args.window_size,
        args.overlap,
        args.down_ratio,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
