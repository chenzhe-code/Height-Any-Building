#!/usr/bin/env python3
"""Merge retained per-tile heights and buildings into physical city outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
from osgeo import gdal
from shapely.geometry import box
from shapely.ops import unary_union


UINT16_NODATA = 65535


def discover_outputs(input_dir: Path, results_root: Path) -> list[tuple[str, Path, Path]]:
    inputs = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    outputs = []
    missing = []
    for input_path in inputs:
        stem = input_path.stem
        height = results_root / stem / f"{stem}_absolute_height.tif"
        buildings = results_root / stem / f"{stem}_building_heights.geojson"
        if not height.is_file() or height.stat().st_size == 0:
            missing.append(str(height))
        if not buildings.is_file() or buildings.stat().st_size == 0:
            missing.append(str(buildings))
        outputs.append((stem, height, buildings))
    if missing:
        raise ValueError(
            f"missing {len(missing)} retained tile outputs; examples: {missing[:5]}"
        )
    return outputs


def validate_height_tiles(outputs: list[tuple[str, Path, Path]]) -> dict:
    crs = None
    resolution = None
    bounds = []
    total_pixels = 0
    for stem, height_path, _ in outputs:
        with rasterio.open(height_path) as source:
            if source.count < 1:
                raise ValueError(f"height raster has no bands: {height_path}")
            if crs is None:
                crs = source.crs
                resolution = (abs(source.transform.a), abs(source.transform.e))
            if source.crs != crs:
                raise ValueError(f"CRS mismatch for {stem}: {source.crs} != {crs}")
            current_resolution = (abs(source.transform.a), abs(source.transform.e))
            if current_resolution != resolution:
                raise ValueError(
                    f"resolution mismatch for {stem}: {current_resolution} != {resolution}"
                )
            if source.transform.b != 0 or source.transform.d != 0:
                raise ValueError(f"rotated rasters are not supported: {height_path}")
            bounds.append(box(*source.bounds))
            total_pixels += source.width * source.height

    if crs is None or resolution is None:
        raise ValueError("no height tiles found")
    if resolution != (1.0, 1.0):
        raise ValueError(f"city output requires 1 m inputs, got {resolution}")

    union = unary_union(bounds)
    summed_area = sum(item.area for item in bounds)
    if not np.isclose(union.area, summed_area):
        raise ValueError(
            f"height tiles overlap: union area {union.area} != summed area {summed_area}"
        )
    left = min(item.bounds[0] for item in bounds)
    bottom = min(item.bounds[1] for item in bounds)
    right = max(item.bounds[2] for item in bounds)
    top = max(item.bounds[3] for item in bounds)
    width = int(round((right - left) / resolution[0]))
    height = int(round((top - bottom) / resolution[1]))
    return {
        "crs": crs,
        "bounds": (left, bottom, right, top),
        "width": width,
        "height": height,
        "total_source_pixels": total_pixels,
    }


def merge_height_raster(
    outputs: list[tuple[str, Path, Path]],
    output_path: Path,
    grid: dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + ".tmp.tif")
    if temporary.exists():
        temporary.unlink()

    gdal.UseExceptions()
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        str(temporary),
        grid["width"],
        grid["height"],
        1,
        gdal.GDT_UInt16,
        options=[
            "BIGTIFF=YES",
            "TILED=YES",
            "BLOCKXSIZE=512",
            "BLOCKYSIZE=512",
            "COMPRESS=ZSTD",
            "ZSTD_LEVEL=9",
            "PREDICTOR=2",
            "NUM_THREADS=ALL_CPUS",
        ],
    )
    if dataset is None:
        raise RuntimeError("GDAL failed to create the city height BigTIFF")
    left, _, _, top = grid["bounds"]
    transform = from_origin(left, top, 1.0, 1.0)
    dataset.SetGeoTransform(transform.to_gdal())
    dataset.SetProjection(grid["crs"].to_wkt())
    dataset.SetMetadataItem("HEIGHT_UNITS", "metre")
    dataset.SetMetadataItem("VALUE_ENCODING", "rounded whole metres; 0-65534")
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(UINT16_NODATA)
    band.Fill(UINT16_NODATA)
    dataset.FlushCache()
    band = None
    dataset = None

    with rasterio.open(temporary, "r+") as target:
        for stem, height_path, _ in outputs:
            with rasterio.open(height_path) as source:
                col_offset_float = (source.bounds.left - left) / target.res[0]
                row_offset_float = (top - source.bounds.top) / target.res[1]
                col_offset = int(round(col_offset_float))
                row_offset = int(round(row_offset_float))
                if not np.isclose(col_offset_float, col_offset) or not np.isclose(
                    row_offset_float, row_offset
                ):
                    raise ValueError(f"tile is not aligned to the 1 m city grid: {stem}")
                if (
                    col_offset < 0
                    or row_offset < 0
                    or col_offset + source.width > target.width
                    or row_offset + source.height > target.height
                ):
                    raise ValueError(f"tile lies outside the city grid: {stem}")

                for _, source_window in source.block_windows(1):
                    source_values = source.read(1, window=source_window, masked=True)
                    raw = np.asarray(source_values.data, dtype=np.float64)
                    valid = ~np.ma.getmaskarray(source_values) & np.isfinite(raw)
                    encoded = np.full(raw.shape, UINT16_NODATA, dtype=np.uint16)
                    encoded[valid] = np.clip(
                        np.rint(raw[valid]), 0, UINT16_NODATA - 1
                    ).astype(np.uint16)
                    target_window = Window(
                        col_offset + source_window.col_off,
                        row_offset + source_window.row_off,
                        source_window.width,
                        source_window.height,
                    )
                    target.write(encoded, 1, window=target_window)

        target.update_tags(
            HEIGHT_UNITS="metre",
            VALUE_ENCODING="rounded whole metres; 0-65534",
        )

    previous_config = gdal.GetConfigOption("COMPRESS_OVERVIEW")
    gdal.SetConfigOption("COMPRESS_OVERVIEW", "ZSTD")
    try:
        dataset = gdal.Open(str(temporary), gdal.GA_Update)
        if dataset is None or dataset.BuildOverviews("AVERAGE", [2, 4, 8, 16, 32]) != 0:
            raise RuntimeError("failed to build internal height overviews")
        dataset.FlushCache()
        dataset = None
    finally:
        gdal.SetConfigOption("COMPRESS_OVERVIEW", previous_config)

    os.replace(temporary, output_path)


def merge_buildings(
    outputs: list[tuple[str, Path, Path]],
    output_path: Path,
    expected_crs,
) -> int:
    frames = []
    for stem, _, buildings_path in outputs:
        frame = gpd.read_file(buildings_path)
        if frame.crs is None:
            raise ValueError(f"building GeoJSON has no CRS: {buildings_path}")
        if frame.crs != expected_crs:
            frame = frame.to_crs(expected_crs)
        if frame.empty:
            continue
        if "Extracted_Height" not in frame.columns:
            raise ValueError(f"missing Extracted_Height: {buildings_path}")
        frame["source_tile"] = stem
        frame["GlobalBuildingId"] = [
            f"{stem}:{local_index}" for local_index in range(len(frame))
        ]
        heights = pd.to_numeric(frame["Extracted_Height"], errors="coerce")
        valid = heights.notna()
        frame.loc[valid, "Extracted_Height"] = np.clip(
            np.rint(heights[valid]), 0, UINT16_NODATA - 1
        ).astype(np.uint16)
        frames.append(frame)

    if frames:
        merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=expected_crs)
    else:
        merged = gpd.GeoDataFrame(geometry=[], crs=expected_crs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + ".tmp.geojson")
    if temporary.exists():
        temporary.unlink()
    merged.to_file(temporary, driver="GeoJSON")
    os.replace(temporary, output_path)
    return len(merged)


def validate_city_outputs(
    raster_path: Path,
    geojson_path: Path,
    grid: dict,
    expected_features: int,
) -> None:
    with rasterio.open(raster_path) as source:
        if source.count != 1 or source.dtypes[0] != "uint16":
            raise ValueError("city raster must be one-band UInt16")
        if source.res != (1.0, 1.0):
            raise ValueError(f"city raster resolution is not 1 m: {source.res}")
        if source.nodata != UINT16_NODATA:
            raise ValueError(f"city raster NoData is not {UINT16_NODATA}")
        if source.width != grid["width"] or source.height != grid["height"]:
            raise ValueError("city raster dimensions do not match the planned grid")
        if source.crs != grid["crs"]:
            raise ValueError("city raster CRS does not match source tiles")
    merged = gpd.read_file(geojson_path)
    if len(merged) != expected_features:
        raise ValueError(
            f"merged feature count {len(merged)} != expected {expected_features}"
        )
    if expected_features == 0:
        return
    required = {"Extracted_Height", "source_tile", "GlobalBuildingId"}
    missing = required - set(merged.columns)
    if missing:
        raise ValueError(f"merged GeoJSON is missing fields: {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-tif", required=True)
    parser.add_argument("--output-geojson", required=True)
    parser.add_argument("--expected-tiles", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    results_root = Path(args.results_root).resolve()
    output_tif = Path(args.output_tif).resolve()
    output_geojson = Path(args.output_geojson).resolve()
    if not args.overwrite:
        existing = [path for path in (output_tif, output_geojson) if path.exists()]
        if existing:
            parser.error(f"outputs already exist; use --overwrite: {existing}")

    outputs = discover_outputs(input_dir, results_root)
    if args.expected_tiles and len(outputs) != args.expected_tiles:
        raise ValueError(f"found {len(outputs)} tiles, expected {args.expected_tiles}")
    grid = validate_height_tiles(outputs)
    merge_height_raster(outputs, output_tif, grid)
    feature_count = merge_buildings(outputs, output_geojson, grid["crs"])
    validate_city_outputs(output_tif, output_geojson, grid, feature_count)
    print(
        f"Merged {len(outputs)} tiles into {output_tif} "
        f"({grid['width']}x{grid['height']}, UInt16, 1 m); "
        f"buildings={feature_count}: {output_geojson}"
    )


if __name__ == "__main__":
    main()
