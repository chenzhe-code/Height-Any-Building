#!/usr/bin/env python3
"""Calibrate synthetic relative-height patches and assign building heights on CPU."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
PATCH_SIZE = 64
CRS = "EPSG:32650"


def write_raster(path, values, transform, nodata=None):
    height, width = values.shape
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=1,
        dtype=values.dtype, crs=CRS, transform=transform, nodata=nodata,
    ) as raster:
        raster.write(values, 1)


def make_inputs(work_dir):
    transform = from_origin(400000, 3400000, 1, 1)
    truth = np.zeros((PATCH_SIZE, 96), dtype=np.float32)
    controls = np.full_like(truth, -9999)
    locations = [(8, 10), (20, 24), (34, 40), (10, 53), (42, 65), (47, 79)]
    heights = [12, 18, 24, 30, 36, 42]
    polygons = []
    for (row, col), height in zip(locations, heights):
        truth[row:row + 8, col:col + 8] = height
        controls[row + 4, col + 4] = height
        west, north = transform * (col, row)
        east, south = transform * (col + 8, row + 8)
        polygons.append(box(west, south, east, north))

    for index, (col, scale) in enumerate(((0, 0.5), (32, 0.75))):
        path = work_dir / "depth" / f"window_{index}_x{col}_y0_depth.tif"
        window = rasterio.windows.Window(col, 0, PATCH_SIZE, PATCH_SIZE)
        write_raster(path, scale * truth[:, col:col + PATCH_SIZE],
                     rasterio.windows.transform(window, transform))

    write_raster(work_dir / "controls.tif", controls, transform, nodata=-9999)
    gpd.GeoDataFrame(
        {"Reference_Height": heights}, geometry=polygons, crs=CRS,
    ).to_file(work_dir / "footprints.geojson", driver="GeoJSON", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()
    try:
        work_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Choose a new output directory: {work_dir}")
    (work_dir / "depth").mkdir()
    make_inputs(work_dir)

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    subprocess.run(
        [sys.executable, str(ROOT / "tools/calibrate_heights.py"), "--work_dir", str(work_dir),
         "--ch_img_path", str(work_dir / "controls.tif"), "--window_size", str(PATCH_SIZE),
         "--overlap", "0.5", "--block_size=-1,-1", "--skip-ground-filter"],
        check=True, env=env,
    )
    calibrated_raster = work_dir / "AbsoluteHeight/output.tif"
    height_raster = work_dir / "building_height_raster.tif"
    building_heights = work_dir / "per_building_heights.geojson"
    subprocess.run(
        [sys.executable, str(ROOT / "tools/assign_building_heights.py"),
         "--geojson-path", str(work_dir / "footprints.geojson"),
         "--geotiff-path", str(calibrated_raster), "--output-path", str(building_heights),
         "--apply-offset", "--quiet"],
        check=True, env=env,
    )

    result = gpd.read_file(building_heights)
    predicted = result["Extracted_Height"].to_numpy(dtype=float)
    if len(result) != 6 or not np.isfinite(predicted).all():
        raise SystemExit(f"Expected six buildings with finite heights in {building_heights}")
    error = np.abs(predicted - result["Reference_Height"])
    if error.max() > 1:
        raise SystemExit(f"Unexpected synthetic calibration error: {error.max():.2f} m")
    calibrated_raster.replace(height_raster)
    print(f"CPU demo passed. {len(result)} buildings, maximum error {error.max():.2f} m.")
    print(height_raster)
    print(building_heights)


if __name__ == "__main__":
    main()
