#!/usr/bin/env python3
"""Recalculate city building heights from a UInt16 height raster."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.windows import Window
from shapely.affinity import translate
from shapely.geometry import box, mapping

UINT16_NODATA = 65535


def aligned_tile_window(city, tile):
    if city.crs != tile.crs or city.res != (1.0, 1.0) or tile.res != (1.0, 1.0):
        raise ValueError(f"city/tile grid or CRS mismatch: {tile.name}")
    col_float = (tile.bounds.left - city.bounds.left) / city.res[0]
    row_float = (city.bounds.top - tile.bounds.top) / city.res[1]
    col, row = round(col_float), round(row_float)
    if abs(col_float - col) > 1e-5 or abs(row_float - row) > 1e-5:
        raise ValueError(f"tile does not align to city grid: {tile.name}")
    if col < 0 or row < 0 or col + tile.width > city.width or row + tile.height > city.height:
        raise ValueError(f"tile is outside city raster: {tile.name}")
    return Window(col, row, tile.width, tile.height)


def pixel_offset(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def polygon_height(values, transform, polygon, nodata=UINT16_NODATA):
    """Return the maximum valid pixel, with rasterio.mask's center-pixel rule."""
    if polygon is None or polygon.is_empty:
        return None
    left = transform.c
    top = transform.f
    width = values.shape[1]
    height = values.shape[0]
    if not polygon.intersects(box(left, top - height, left + width, top)):
        return None
    minx, miny, maxx, maxy = polygon.bounds
    col0 = max(0, int(np.floor(minx - left)))
    col1 = min(width, int(np.ceil(maxx - left)))
    row0 = max(0, int(np.floor(top - maxy)))
    row1 = min(height, int(np.ceil(top - miny)))
    if col0 >= col1 or row0 >= row1:
        return None
    subset = values[row0:row1, col0:col1]
    subset_transform = transform * Affine.translation(col0, row0)
    inside = geometry_mask(
        [mapping(polygon)], out_shape=subset.shape,
        transform=subset_transform, invert=True, all_touched=False,
    )
    valid = inside & (subset != nodata)
    return int(np.max(subset[valid])) if np.any(valid) else None


def write_city_buildings(input_dir: Path, results_root: Path, height_tif: Path,
                         output_geojson: Path, expected_tiles: int = 0) -> dict:
    input_dir = Path(input_dir).resolve()
    results_root = Path(results_root).resolve()
    height_tif = Path(height_tif).resolve()
    output_geojson = Path(output_geojson).resolve()
    images = sorted(path for path in input_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in {".tif", ".tiff"})
    if not images or (expected_tiles and len(images) != expected_tiles):
        raise ValueError(f"input tile count {len(images)} != expected {expected_tiles}")
    frames = []
    missing_height = 0
    total = 0
    with rasterio.open(height_tif) as city:
        if city.count != 1 or city.dtypes != ("uint16",) or city.nodata != UINT16_NODATA:
            raise ValueError("city raster must be one-band UInt16 with NoData 65535")
        for number, image_path in enumerate(images, 1):
            stem = image_path.stem
            buildings_path = results_root / stem / f"{stem}_building_heights.geojson"
            if not buildings_path.is_file():
                raise ValueError(f"missing retained building polygons: {buildings_path}")
            frame = gpd.read_file(buildings_path)
            if frame.crs is None:
                raise ValueError(f"building GeoJSON has no CRS: {buildings_path}")
            with rasterio.open(image_path) as image:
                window = aligned_tile_window(city, image)
                tile_transform = image.transform
                tile_crs = image.crs
                data = city.read(1, window=window)
            if frame.crs != tile_crs:
                frame = frame.to_crs(tile_crs)
            heights = []
            for geometry, offset_x, offset_y in zip(
                    frame.geometry,
                    frame.get("Offset_X", pd.Series(0, index=frame.index)),
                    frame.get("Offset_Y", pd.Series(0, index=frame.index))):
                shifted = translate(
                    geometry,
                    xoff=pixel_offset(offset_x) * tile_transform.a,
                    yoff=pixel_offset(offset_y) * tile_transform.e,
                ) if geometry is not None else None
                value = polygon_height(data, tile_transform, shifted)
                heights.append(value)
                missing_height += value is None
            frame["Extracted_Height"] = pd.Series(heights, index=frame.index, dtype="Int64")
            frame["source_tile"] = stem
            frame["GlobalBuildingId"] = [f"{stem}:{i}" for i in range(len(frame))]
            total += len(frame)
            frames.append(frame)
            print(f"Reheighted {number}/{len(images)} tiles: {stem}, buildings={len(frame)}", flush=True)
    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=tile_crs)
    if len(merged) != total or merged["GlobalBuildingId"].duplicated().any():
        raise ValueError("building feature count or IDs are invalid")
    valid = merged["Extracted_Height"].dropna().astype(int)
    if not valid.between(0, UINT16_NODATA - 1).all():
        raise ValueError("building heights exceed UInt16 valid range")
    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_geojson.with_name(output_geojson.stem + ".tmp.geojson")
    if temporary.exists():
        temporary.unlink()
    merged.to_file(temporary, driver="GeoJSON")
    with rasterio.Env():
        import pyogrio
        info = pyogrio.read_info(temporary)
    if info["features"] != total or info["crs"] != tile_crs:
        temporary.unlink(missing_ok=True)
        raise ValueError("building GeoJSON failed feature count/CRS validation")
    os.replace(temporary, output_geojson)
    return {"features": total, "missing_height": missing_height, "output": str(output_geojson)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--height-tif", type=Path, required=True)
    parser.add_argument("--output-geojson", type=Path, required=True)
    parser.add_argument("--expected-tiles", type=int, default=0)
    args = parser.parse_args()
    result = write_city_buildings(
        args.input_dir, args.results_root, args.height_tif, args.output_geojson,
        expected_tiles=args.expected_tiles,
    )
    print(f"Published building GeoJSON: {result}", flush=True)


if __name__ == "__main__":
    main()
