import argparse
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from scipy.spatial import cKDTree
from shapely.geometry import Point


def _require_columns(frame, required_cols, source_name):
    missing = [col for col in required_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"{source_name} missing required columns: {missing}")


def _point_xy_frame(points_gdf):
    return np.column_stack((points_gdf.geometry.x.to_numpy(), points_gdf.geometry.y.to_numpy()))


def _read_photon_table(path, wgs84_bounds):
    """Read photons, filtering large CSV inputs to the current tile while streaming."""
    path = str(path)
    left, bottom, right, top = wgs84_bounds

    def within_bounds(frame):
        _require_columns(frame, ["lon", "lat", "h", "label"], path)
        return frame[
            frame["lon"].between(left, right)
            & frame["lat"].between(bottom, top)
        ].copy()

    if path.lower().endswith((".csv", ".csv.gz")):
        selected = [within_bounds(chunk) for chunk in pd.read_csv(path, chunksize=200_000)]
        selected = [frame for frame in selected if not frame.empty]
        if not selected:
            return pd.DataFrame(columns=["h", "lat", "lon", "label"])
        return pd.concat(selected, ignore_index=True)
    if path.lower().endswith((".xlsx", ".xls")):
        return within_bounds(pd.read_excel(path))
    raise ValueError("Photon table must end in .xlsx, .xls, .csv, or .csv.gz")


def photon_lidar_to_control_height(
    geojson_path,
    excel_path,
    reference_tif_path,
    output_tif_path,
    nodata=-9999,
    allow_empty=False,
):
    """Match PLiDAR photons to MLS-BRN footprints and save sparse control heights."""
    with rasterio.open(reference_tif_path) as ref_src:
        ref_transform = ref_src.transform
        ref_crs = ref_src.crs
        ref_width = ref_src.width
        ref_height = ref_src.height
        ref_bounds = ref_src.bounds
        resolution_x = ref_transform.a
        resolution_y = -ref_transform.e
        wgs84_bounds = transform_bounds(ref_crs, "EPSG:4326", *ref_bounds, densify_pts=21)

    if ref_crs is None:
        raise ValueError(f"Reference GeoTIFF has no CRS: {reference_tif_path}")

    gdf = gpd.read_file(geojson_path)
    if gdf.crs is None:
        raise ValueError(f"GeoJSON has no CRS: {geojson_path}")
    if gdf.empty:
        if not allow_empty:
            raise ValueError("Building GeoJSON contains no features.")
        raster = np.full((ref_height, ref_width), nodata, dtype=np.float32)
        output_dir = os.path.dirname(output_tif_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        temporary_path = output_tif_path + ".part"
        with rasterio.open(
            temporary_path,
            "w",
            driver="GTiff",
            height=ref_height,
            width=ref_width,
            count=1,
            dtype=raster.dtype,
            crs=ref_crs,
            transform=ref_transform,
            nodata=nodata,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="ZSTD",
            predictor=3,
            zstd_level=9,
            sparse_ok=True,
            BIGTIFF="IF_SAFER",
        ) as dst:
            dst.write(raster, 1)
        os.replace(temporary_path, output_tif_path)
        print(f"Empty building GeoJSON; wrote empty control raster: {output_tif_path}")
        return
    df = _read_photon_table(excel_path, wgs84_bounds)

    print("Photon table columns:", list(df.columns))
    print(f"Photons within tile bounds: {len(df)}")
    _require_columns(df, ["lon", "lat", "h", "label"], "Photon table")

    print("GeoJSON columns:", list(gdf.columns))
    _require_columns(gdf, ["Offset_X", "Offset_Y"], "GeoJSON")

    if gdf.crs is None:
        raise ValueError(f"GeoJSON has no CRS: {geojson_path}")

    points_gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])],
        crs="EPSG:4326",
    )

    if points_gdf.crs != ref_crs:
        points_gdf = points_gdf.to_crs(ref_crs)
    if gdf.crs != ref_crs:
        gdf = gdf.to_crs(ref_crs)

    polygons = gdf[["geometry"]].reset_index(names="polygon_id")
    joined = gpd.sjoin(
        points_gdf[["geometry"]],
        polygons,
        how="left",
        predicate="within",
    )
    # Footprints are expected not to overlap. If they do, match the original
    # loop's deterministic last-polygon-wins behavior.
    matched = joined.dropna(subset=["polygon_id"]).groupby(level=0)["polygon_id"].max()
    points_gdf["in_any_polygon"] = points_gdf.index.isin(matched.index)
    points_gdf["polygon_id"] = -1
    points_gdf.loc[matched.index, "polygon_id"] = matched.astype(int)

    in_count = points_gdf["in_any_polygon"].sum()
    out_count = len(points_gdf) - in_count
    print(f"Photons in polygons: {in_count}, photons outside polygons: {out_count}")

    points_gdf["adjusted_height"] = np.nan

    outside_points = points_gdf[~points_gdf["in_any_polygon"]].copy()
    outside_label_0 = outside_points[outside_points["label"] == 0].copy()
    if outside_label_0.empty and not allow_empty:
        raise ValueError("No outside-polygon ground photons (label=0) were found.")

    print(f"Outside-polygon ground photons: {len(outside_label_0)}")
    tree = None if outside_label_0.empty else cKDTree(_point_xy_frame(outside_label_0))

    for poly_idx in points_gdf[points_gdf["in_any_polygon"]]["polygon_id"].unique():
        if poly_idx == -1:
            continue

        poly_points = points_gdf[points_gdf["polygon_id"] == poly_idx].copy()
        poly_label_1 = poly_points[poly_points["label"] == 1].copy()
        if poly_label_1.empty:
            print(f"Polygon {poly_idx}: no building photons (label=1).")
            continue

        max_height_idx = poly_label_1["h"].idxmax()
        max_point = poly_label_1.loc[max_height_idx]

        if tree is None:
            continue
        _, nearest_idx = tree.query([max_point.geometry.x, max_point.geometry.y])
        nearest_point = outside_label_0.iloc[nearest_idx]
        height_diff = max_point["h"] - nearest_point["h"]
        points_gdf.loc[max_height_idx, "adjusted_height"] = height_diff

        offset_x = gdf.loc[poly_idx, "Offset_X"] * resolution_x
        offset_y = gdf.loc[poly_idx, "Offset_Y"] * resolution_y
        new_x = max_point.geometry.x + offset_x
        new_y = max_point.geometry.y - offset_y
        points_gdf.loc[max_height_idx, "geometry"] = Point(new_x, new_y)

        nearest_point_idx = nearest_point.name
        points_gdf.loc[nearest_point_idx, "adjusted_height"] = 0

        print(
            f"Polygon {poly_idx} photon {max_height_idx}: "
            f"source=({max_point.geometry.x}, {max_point.geometry.y}), "
            f"adjusted=({new_x}, {new_y}), "
            f"h_building={max_point['h']}, h_ground={nearest_point['h']}, "
            f"height_diff={height_diff}, "
            f"Offset_X={gdf.loc[poly_idx, 'Offset_X']}, "
            f"Offset_Y={gdf.loc[poly_idx, 'Offset_Y']}"
        )

    raster = np.full((ref_height, ref_width), nodata, dtype=np.float32)

    for _, point in points_gdf[points_gdf["adjusted_height"].notna()].iterrows():
        col = int((point.geometry.x - ref_bounds.left) / resolution_x)
        row = int((ref_bounds.top - point.geometry.y) / resolution_y)
        if 0 <= row < ref_height and 0 <= col < ref_width:
            raster[row, col] = point["adjusted_height"]
        else:
            print(f"Photon {point.name} is outside the reference raster: {point.geometry}")

    output_dir = os.path.dirname(output_tif_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    temporary_path = output_tif_path + ".part"
    with rasterio.open(
        temporary_path,
        "w",
        driver="GTiff",
        height=ref_height,
        width=ref_width,
        count=1,
        dtype=raster.dtype,
        crs=ref_crs,
        transform=ref_transform,
        nodata=nodata,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="ZSTD",
        predictor=3,
        zstd_level=9,
        sparse_ok=True,
        BIGTIFF="IF_SAFER",
    ) as dst:
        dst.write(raster, 1)
    os.replace(temporary_path, output_tif_path)

    print(f"Control height GeoTIFF saved to: {output_tif_path}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Convert PLiDAR building/ground photons and MLS-BRN offsets into a sparse control height GeoTIFF."
    )
    parser.add_argument("--geojson-path", "--geojson_path", dest="geojson_path", required=True)
    parser.add_argument(
        "--photon-path", "--photon_path", "--excel-path", "--excel_path",
        dest="photon_path", required=True,
        help="Photon table (.xlsx, .csv, or .csv.gz)",
    )
    parser.add_argument("--reference-tif-path", "--reference_tif_path", dest="reference_tif_path", required=True)
    parser.add_argument("--output-tif-path", "--output_tif_path", dest="output_tif_path", required=True)
    parser.add_argument("--nodata", type=float, default=-9999)
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write an empty sparse-control raster when no usable photons exist.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    photon_lidar_to_control_height(
        geojson_path=args.geojson_path,
        excel_path=args.photon_path,
        reference_tif_path=args.reference_tif_path,
        output_tif_path=args.output_tif_path,
        nodata=args.nodata,
        allow_empty=args.allow_empty,
    )


if __name__ == "__main__":
    main()
