import argparse
import ast
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.errors import ShapelyError
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from shapely.wkt import loads


REQUIRED_COLUMNS = ["ImageId", "BuildingId", "PolygonWKT_Pix", "Confidence", "Offset", "Height"]


def parse_offset(value):
    if isinstance(value, str):
        if value.strip() == "[0 0]":
            return 0, 0
        parsed = ast.literal_eval(value)
    else:
        parsed = value

    if len(parsed) != 2:
        raise ValueError(f"Offset must have two values, got: {value}")
    return float(parsed[0]), float(parsed[1])


def require_columns(df, required_columns):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")


def pixel_to_geo(transform, pixel_coords, offset_x=0, offset_y=0):
    """将像素坐标转换为地理坐标，可选应用Offset。"""
    geo_coords = []
    for x, y in pixel_coords:
        geo_x, geo_y = rasterio.transform.xy(transform, y + offset_y, x + offset_x)
        geo_coords.append((geo_x, geo_y))
    return geo_coords


def calculate_iou(poly1, poly2):
    intersection = poly1.intersection(poly2).area
    if intersection <= 0:
        return 0

    union = poly1.area + poly2.area - intersection
    return intersection / union if union > 0 else 0


def nms_polygons(gdf, confidence_col="Confidence", iou_threshold=0.5):
    """对GeoDataFrame中的多边形执行非极大值抑制。"""
    if gdf.empty:
        return gdf

    gdf_sorted = gdf.sort_values(by=confidence_col, ascending=False).copy()
    geometries = list(gdf_sorted.geometry.values)
    tree = STRtree(geometries)
    geometry_id_to_pos = {id(geom): pos for pos, geom in enumerate(geometries)}
    suppressed = np.zeros(len(geometries), dtype=bool)
    keep_positions = []

    for current_pos, current_poly in enumerate(geometries):
        if suppressed[current_pos]:
            continue

        keep_positions.append(current_pos)
        for candidate in tree.query(current_poly):
            if isinstance(candidate, (int, np.integer)):
                candidate_pos = int(candidate)
            else:
                candidate_pos = geometry_id_to_pos.get(id(candidate))
                if candidate_pos is None:
                    continue

            if candidate_pos <= current_pos or suppressed[candidate_pos]:
                continue
            if calculate_iou(current_poly, geometries[candidate_pos]) > iou_threshold:
                suppressed[candidate_pos] = True

    return gdf_sorted.iloc[keep_positions]


def csv_and_geotiff_to_geojson_or_shp(
    csv_path,
    geotiff_folder,
    output_path,
    output_format="geojson",
    apply_offset=False,
    use_nms=True,
    iou_threshold=0.2,
    confidence_col="Confidence",
):
    """将MLS-BRN CSV和GeoTIFF patch转换为GeoJSON或Shapefile。"""
    df = pd.read_csv(csv_path)
    require_columns(df, REQUIRED_COLUMNS)
    unmapped = df.loc[df["ImageId"].isna(), "BuildingId"]
    if not unmapped.empty:
        raise ValueError(f"Missing ImageId for BuildingId={unmapped.iat[0]!r} in {csv_path}")

    geometries = []
    attributes = []
    crs = None

    for image_id, group in df.groupby("ImageId"):
        print("Processing img:", image_id)
        geotiff_path = os.path.join(geotiff_folder, f"{image_id}.tif")
        try:
            with rasterio.open(geotiff_path) as src:
                transform = src.transform
                patch_crs = src.crs
        except rasterio.errors.RasterioError as exc:
            raise ValueError(f"Cannot open GeoTIFF patch for ImageId={image_id!r}: {geotiff_path}: {exc}") from exc
        if patch_crs is None:
            raise ValueError(f"Missing CRS for ImageId={image_id!r}: {geotiff_path}")
        if crs is None:
            crs = patch_crs
        elif patch_crs != crs:
            raise ValueError(f"Patch CRS mismatch for ImageId={image_id!r}: {geotiff_path} has {patch_crs}, expected {crs}")

        for _, row in group.iterrows():
            try:
                pixel_poly = loads(row["PolygonWKT_Pix"])
                if not isinstance(pixel_poly, Polygon) or pixel_poly.is_empty:
                    raise ValueError("PolygonWKT_Pix must describe a non-empty polygon")
                pixel_coords = list(pixel_poly.exterior.coords)
                raw_offset_x, raw_offset_y = parse_offset(row["Offset"])
                if not np.isfinite([raw_offset_x, raw_offset_y]).all():
                    raise ValueError("Offset must contain finite values")
                offset_x, offset_y = (raw_offset_x, raw_offset_y) if apply_offset else (0, 0)
                geo_coords = pixel_to_geo(transform, pixel_coords, offset_x, offset_y)
                geometry = Polygon(geo_coords)
                if not geometry.is_valid:
                    raise ValueError("Converted footprint is an invalid polygon")
            except (ShapelyError, rasterio.errors.RasterioError, SyntaxError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid footprint for ImageId={image_id!r}, BuildingId={row['BuildingId']!r}: {exc}"
                ) from exc
            geometries.append(geometry)
            attributes.append(
                {
                    "ImageId": row["ImageId"],
                    "BuildingId": row["BuildingId"],
                    "Confidence": row["Confidence"],
                    "Offset_X": raw_offset_x,
                    "Offset_Y": raw_offset_y,
                    "Height": row["Height"],
                }
            )

    if not geometries and crs is None:
        for candidate in sorted(os.listdir(geotiff_folder)):
            if not candidate.lower().endswith((".tif", ".tiff")):
                continue
            with rasterio.open(os.path.join(geotiff_folder, candidate)) as src:
                crs = src.crs
            break
    if crs is None:
        raise ValueError("无法从GeoTIFF patch读取CRS")

    if geometries:
        gdf = gpd.GeoDataFrame(attributes, geometry=geometries, crs=crs)
    else:
        gdf = gpd.GeoDataFrame(
            {
                "ImageId": pd.Series(dtype="str"),
                "BuildingId": pd.Series(dtype="int64"),
                "Confidence": pd.Series(dtype="float64"),
                "Offset_X": pd.Series(dtype="float64"),
                "Offset_Y": pd.Series(dtype="float64"),
                "Height": pd.Series(dtype="float64"),
            },
            geometry=gpd.GeoSeries([], crs=crs),
            crs=crs,
        )

    if use_nms and not gdf.empty:
        print("Start nms, please wait...")
        gdf = nms_polygons(gdf, confidence_col=confidence_col, iou_threshold=iou_threshold)
        print(f"NMS后: {len(gdf)} 个多边形")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if output_format.lower() == "geojson":
        gdf.to_file(output_path, driver="GeoJSON")
        print(f"已保存GeoJSON到: {output_path}")
    elif output_format.lower() == "shp":
        gdf.to_file(output_path, driver="ESRI Shapefile")
        print(f"已保存Shapefile到: {output_path}")
    else:
        raise ValueError("不支持的输出格式，请选择 'geojson' 或 'shp'")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Convert MLS-BRN footprint CSV from pixel coordinates to georeferenced GeoJSON/Shapefile."
    )
    parser.add_argument("--csv-path", "--csv_path", dest="csv_path", required=True)
    parser.add_argument("--geotiff-folder", "--geotiff_folder", dest="geotiff_folder", required=True)
    parser.add_argument("--output-path", "--output_path", dest="output_path", required=True)
    parser.add_argument("--output-format", choices=["geojson", "shp"], default="geojson")
    parser.add_argument("--apply-offset", action="store_true")
    parser.add_argument("--no-nms", action="store_true", help="Disable polygon NMS.")
    parser.add_argument("--iou-threshold", type=float, default=0.2)
    parser.add_argument("--confidence-col", default="Confidence")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    csv_and_geotiff_to_geojson_or_shp(
        csv_path=args.csv_path,
        geotiff_folder=args.geotiff_folder,
        output_path=args.output_path,
        output_format=args.output_format,
        apply_offset=args.apply_offset,
        use_nms=not args.no_nms,
        iou_threshold=args.iou_threshold,
        confidence_col=args.confidence_col,
    )
