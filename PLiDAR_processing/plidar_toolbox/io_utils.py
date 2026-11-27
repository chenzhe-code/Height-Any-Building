import glob
import os

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio import warp


def get_geotiff_bounds(file_path):
    """Read GeoTIFF bounds in WGS84 coordinates."""
    with rasterio.open(file_path) as src:
        bounds = src.bounds
        if not src.crs.is_geographic:
            left, bottom, right, top = warp.transform_bounds(
                src_crs=src.crs,
                dst_crs="EPSG:4326",
                left=bounds.left,
                bottom=bounds.bottom,
                right=bounds.right,
                top=bounds.top,
            )
        else:
            left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top

        return {
            "min_lon": left,
            "min_lat": bottom,
            "max_lon": right,
            "max_lat": top,
        }


def get_shapefile_bounds(shp_path):
    gdf = gpd.read_file(shp_path)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    minx, miny, maxx, maxy = gdf.total_bounds
    return {
        "min_lon": minx,
        "min_lat": miny,
        "max_lon": maxx,
        "max_lat": maxy,
    }


DEFAULT_BEAMS = ("gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r")


def check_h5_files(folder_path, bounds, beams=DEFAULT_BEAMS):
    """Find HDF5 files that contain photons in any requested beam inside the bounds."""
    if not os.path.isdir(folder_path):
        print("Error: the specified folder path does not exist.")
        return []

    h5_files = glob.glob(os.path.join(folder_path, "**", "*.h5"), recursive=True)
    if not h5_files:
        raise FileNotFoundError("No .h5 files found.")

    print(f"Checking {len(h5_files)} files...")
    lat_min = bounds["min_lat"]
    lat_max = bounds["max_lat"]
    lon_min = bounds["min_lon"]
    lon_max = bounds["max_lon"]
    matching_files = []

    for file_path in h5_files:
        try:
            with h5py.File(file_path, "r") as h5_file:
                matched = False
                for beam in beams:
                    lat_dset = h5_file.get(f"{beam}/heights/lat_ph")
                    lon_dset = h5_file.get(f"{beam}/heights/lon_ph")
                    if lat_dset is None or lon_dset is None:
                        continue

                    latitudes = np.array(lat_dset)
                    longitudes = np.array(lon_dset)
                    mask = (
                        (latitudes >= lat_min)
                        & (latitudes <= lat_max)
                        & (longitudes >= lon_min)
                        & (longitudes <= lon_max)
                    )

                    if np.any(mask):
                        matched = True
                        break

                if matched:
                    matching_files.append(file_path)
                    print(f"Matched file: {file_path}")
                else:
                    print(f"File {file_path} has no requested beam photons in bounds; skipped.")

        except Exception as exc:
            print(f"Error processing {file_path}: {exc}")

    if not matching_files:
        print("No HDF5 files matched the requested bounds.")
    else:
        print(f"Found {len(matching_files)} matching files.")

    return matching_files


def save_csv(ground_data, building_data, save_path=None):
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "h": ground_data["h"],
                    "lat": ground_data["lat"],
                    "lon": ground_data["lon"],
                    "label": 0,
                }
            ),
            pd.DataFrame(
                {
                    "h": building_data["h"],
                    "lat": building_data["lat"],
                    "lon": building_data["lon"],
                    "label": 1,
                }
            ),
        ],
        ignore_index=True,
    )

    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    lower_path = save_path.lower()
    if lower_path.endswith((".csv", ".csv.gz")):
        df.to_csv(save_path, index=False)
    elif lower_path.endswith((".xlsx", ".xls")):
        if len(df) > 1_048_575:
            raise ValueError(
                f"{len(df)} data rows exceed the Excel worksheet limit; "
                "use a .csv.gz save path"
            )
        df.to_excel(save_path, index=False, engine="openpyxl")
    else:
        raise ValueError("save_path must end in .xlsx, .xls, .csv, or .csv.gz")
    print(f"File saved to: {os.path.abspath(save_path)}")
