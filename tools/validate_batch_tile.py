#!/usr/bin/env python3
"""Validate one batch tile before and after transient artifact cleanup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import rasterio
from rasterio.warp import transform_bounds

from validate_patch_grid import expected_patch_records, validate_patch_grid


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"{label} is missing or empty: {path}")


def count_matching(folder: Path, pattern: str) -> int:
    return sum(1 for path in folder.glob(pattern) if path.is_file()) if folder.is_dir() else 0


def validate_geojson(path: Path, *, require_height: bool) -> tuple[int, int]:
    require_file(path, "GeoJSON")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"GeoJSON has no CRS: {path}")
    if gdf.empty:
        return 0, 0
    missing = {"Offset_X", "Offset_Y"} - set(gdf.columns)
    if missing:
        raise ValueError(f"GeoJSON is missing columns {sorted(missing)}: {path}")

    valid_heights = 0
    if require_height:
        if "Extracted_Height" not in gdf.columns:
            raise ValueError(f"GeoJSON is missing Extracted_Height: {path}")
        height_values = np.asarray(gdf["Extracted_Height"], dtype=float)
        valid_heights = int(np.isfinite(height_values).sum())
        if valid_heights == 0:
            raise ValueError(f"GeoJSON has no finite Extracted_Height values: {path}")
    return len(gdf), valid_heights


def validate_height_tif(path: Path, input_tif: Path) -> dict:
    require_file(path, "absolute-height GeoTIFF")
    with rasterio.open(input_tif) as source, rasterio.open(path) as height:
        if height.crs != source.crs:
            raise ValueError(f"height CRS {height.crs} does not match input CRS {source.crs}")
        if height.transform != source.transform:
            raise ValueError("height transform does not match the input GeoTIFF")
        if height.width != source.width or height.height != source.height:
            raise ValueError(
                f"height dimensions {height.width}x{height.height} do not match "
                f"input {source.width}x{source.height}"
            )
        band = height.read(1, masked=True)
        valid = np.ma.compressed(band)
        finite_count = int(np.isfinite(valid).sum())
        if finite_count == 0:
            raise ValueError(f"height GeoTIFF has no finite pixels: {path}")
        return {
            "height_width": height.width,
            "height_height": height.height,
            "height_bands": height.count,
            "height_dtype": height.dtypes[0],
            "height_crs": str(height.crs),
            "height_finite_pixels": finite_count,
        }


def validate_control_tif(path: Path, input_tif: Path) -> int:
    require_file(path, "sparse-control GeoTIFF")
    with rasterio.open(input_tif) as source, rasterio.open(path) as control:
        if control.crs != source.crs or control.transform != source.transform:
            raise ValueError("sparse-control GeoTIFF georeferencing does not match the input")
        values = control.read(1, masked=True)
        valid_controls = int(np.count_nonzero(np.ma.filled(values, np.nan) >= 5))
        return valid_controls


def h5_overlaps_input(atl03_dir: Path, input_tif: Path) -> tuple[int, bool]:
    h5_files = sorted(atl03_dir.rglob("*.h5")) if atl03_dir.is_dir() else []
    if not h5_files:
        raise ValueError(f"no ATL03 HDF5 files found under {atl03_dir}")

    with rasterio.open(input_tif) as source:
        west, south, east, north = transform_bounds(
            source.crs, "EPSG:4326", *source.bounds, densify_pts=21
        )

    overlaps = False
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as h5_file:
            for beam in ("gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"):
                lat_dset = h5_file.get(f"{beam}/heights/lat_ph")
                lon_dset = h5_file.get(f"{beam}/heights/lon_ph")
                if lat_dset is None or lon_dset is None:
                    continue
                latitudes = np.asarray(lat_dset)
                longitudes = np.asarray(lon_dset)
                if np.any(
                    (latitudes >= south)
                    & (latitudes <= north)
                    & (longitudes >= west)
                    & (longitudes <= east)
                ):
                    overlaps = True
                    break
        if overlaps:
            break
    if not overlaps:
        raise ValueError("downloaded ATL03 files contain no photons inside the input bounds")
    return len(h5_files), overlaps


def expected_patch_count(input_tif: Path, window_size: int, overlap: float) -> int:
    return len(expected_patch_records(input_tif, window_size, overlap, 1.0))


def validate(args: argparse.Namespace) -> dict:
    job_dir = Path(args.job_dir)
    input_tif = Path(args.input_tif)
    stem = args.tile_stem
    atl03_dir = Path(args.atl03_dir) if args.atl03_dir else job_dir / "ICESat-2"
    require_file(input_tif, "input GeoTIFF")

    footprint = job_dir / "footprint_direct_nms.geojson"
    control = job_dir / "sparse_control_height.tif"
    photon_table = Path(args.photon_table) if args.photon_table else job_dir / "extracted_height.xlsx"
    stable_height = job_dir / f"{stem}_absolute_height.tif"
    stable_geojson = job_dir / f"{stem}_building_heights.geojson"

    footprint_count, _ = validate_geojson(footprint, require_height=False)
    valid_controls = None
    if args.control_source == "photon":
        require_file(photon_table, "photon table")
        valid_controls = validate_control_tif(control, input_tif)
        if args.photon_table and control.stat().st_mtime < photon_table.stat().st_mtime:
            raise ValueError(f"sparse-control GeoTIFF is older than shared photon table: {control}")
        if valid_controls < args.min_controls and not args.allow_model_fallback:
            raise ValueError(
                f"sparse-control GeoTIFF has {valid_controls} controls >= 5 m; "
                f"at least {args.min_controls} are required: {control}"
            )

    if args.retained_only:
        height_summary = validate_height_tif(stable_height, input_tif)
        building_count, valid_heights = validate_geojson(stable_geojson, require_height=True)
        result = {
            "mode": "retained",
            "model_name": args.model_name,
            "control_source": args.control_source,
            "footprint_count": footprint_count,
            "building_count": building_count,
            "valid_building_heights": valid_heights,
            **height_summary,
        }
        if valid_controls is not None:
            result["valid_controls"] = valid_controls
            result["effective_control_source"] = (
                "photon" if valid_controls >= args.min_controls else "model"
            )
        return result

    patch_dir = job_dir / f"windowsize-{args.window_size}-overlap-{args.overlap}-downratio-{args.down_ratio}-patches"
    depth_root = job_dir / f"{args.model_name}-windowsize-{args.window_size}-overlap-{args.overlap}-downratio-{args.down_ratio}-depth"
    patch_summary = validate_patch_grid(
        input_tif,
        patch_dir,
        args.window_size,
        args.overlap,
        float(args.down_ratio),
    )
    expected = patch_summary["patch_count"]
    patch_count = count_matching(patch_dir, "*.tif")
    depth_count = count_matching(depth_root / "depth", "*_depth.tif")
    uncertainty_count = count_matching(depth_root / "uncertainty", "*_uncertainty.tif")
    counts_valid = patch_count == expected and depth_count == expected
    if args.compute_uncertainty:
        counts_valid = counts_valid and uncertainty_count == expected
    if not counts_valid:
        raise ValueError(
            "patch count mismatch: "
            f"expected={expected}, patches={patch_count}, depth={depth_count}, "
            f"uncertainty={uncertainty_count}, require_uncertainty={args.compute_uncertainty}"
        )

    h5_count = None
    if args.control_source == "photon":
        if args.photon_table:
            h5_count = count_matching(atl03_dir, "*.h5")
        else:
            h5_count, _ = h5_overlaps_input(atl03_dir, input_tif)
    absolute_height = depth_root / "AbsoluteHeight" / "output.tif"
    final_geojson = job_dir / "footprint_direct_nms_absolute_height.geojson"
    height_summary = validate_height_tif(absolute_height, input_tif)
    building_count, valid_heights = validate_geojson(final_geojson, require_height=True)
    result = {
        "mode": "full",
        "model_name": args.model_name,
        "control_source": args.control_source,
        "expected_patches": expected,
        "patch_count": patch_count,
        "depth_count": depth_count,
        "uncertainty_count": uncertainty_count,
        "footprint_count": footprint_count,
        "building_count": building_count,
        "valid_building_heights": valid_heights,
        **height_summary,
    }
    if h5_count is not None:
        result["atl03_h5_count"] = h5_count
    if valid_controls is not None:
        result["valid_controls"] = valid_controls
        result["effective_control_source"] = (
            "photon" if valid_controls >= args.min_controls else "model"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--input-tif", required=True)
    parser.add_argument("--tile-stem", required=True)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.5)
    # This value is part of directory names produced by run_full_pipeline.sh.
    # Keep the caller's spelling (for example, "1" rather than converting it
    # to float "1.0") so validation resolves the exact same directory.
    parser.add_argument("--down-ratio", default="1")
    parser.add_argument("--model-name", default="Distill-Any-Depth")
    parser.add_argument("--control-source", choices=("photon", "model"), default="photon")
    parser.add_argument("--atl03-dir", help="ATL03 directory; defaults to JOB_DIR/ICESat-2.")
    parser.add_argument("--photon-table", help="Shared city-wide photon table.")
    parser.add_argument("--min-controls", type=int, default=4)
    parser.add_argument("--allow-model-fallback", action="store_true")
    parser.set_defaults(compute_uncertainty=True)
    parser.add_argument("--compute-uncertainty", dest="compute_uncertainty", action="store_true")
    parser.add_argument("--no-uncertainty", dest="compute_uncertainty", action="store_false")
    parser.add_argument("--retained-only", action="store_true")
    return parser


if __name__ == "__main__":
    result = validate(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
