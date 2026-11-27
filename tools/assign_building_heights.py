import argparse
import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
from shapely.geometry import mapping, box
from shapely.affinity import translate
from rasterio.errors import RasterioError

def pixel_offset_to_geo_offset(transform, offset_x, offset_y):
    """将像素坐标系的Offset转换为地理坐标系的Offset"""
    geo_dx = offset_x * transform.a  # x方向地理单位/像素
    geo_dy = offset_y * transform.e  # y方向地理单位/像素（通常为负）
    return geo_dx, geo_dy

def apply_offset(geometry, geo_dx, geo_dy):
    """对多边形应用地理坐标系的偏移"""
    return translate(geometry, xoff=geo_dx, yoff=geo_dy)

def _polygon_overlaps_raster(src, polygon):
    """快速判定多边形与栅格是否有空间重叠（基于外接框）"""
    if polygon is None or polygon.is_empty:
        return False
    raster_bbox = box(*src.bounds)  # left, bottom, right, top
    return polygon.intersects(raster_bbox)

def extract_height_from_geotiff(geotiff_path, polygon):
    """Read the maximum valid height inside a polygon."""
    with rasterio.open(geotiff_path) as src:
        return _extract_height(src, polygon)

def _extract_height(src, polygon):
    if not _polygon_overlaps_raster(src, polygon):
        return None
    try:
        heights, _ = mask(src, [mapping(polygon)], crop=True, filled=False, indexes=1)
    except (ValueError, RasterioError):
        return None
    valid_heights = heights.compressed()
    return float(np.max(valid_heights)) if valid_heights.size else None

def match_polygons_with_height(geojson_path, geotiff_path, output_geojson_path,
                               apply_offset_flag=True, verbose=True, *, height_column=None):
    """
    将GeoJSON中的多边形与单一GeoTIFF高度匹配,处理图像坐标系Offset,确保高度为浮点型。
    若多边形与TIFF无重叠或仅含NoData，则写入None并跳过。
    """
    def log(message):
        if verbose:
            print(message)

    gdf = gpd.read_file(geojson_path)
    height_col_name = height_column
    if height_col_name is None:
        height_col_name = "Extracted_Height" if apply_offset_flag else "LiDAR_Height"
    if not height_col_name.strip() or height_col_name == gdf.geometry.name:
        raise ValueError("The height column must be a nonempty name distinct from the geometry column.")
    log(f"高度字段: {height_col_name}，应用偏移: {apply_offset_flag}")

    if not os.path.exists(geotiff_path):
        raise FileNotFoundError(f"未找到高度GeoTIFF文件: {geotiff_path}")

    try:
        with rasterio.open(geotiff_path) as src:
            transform = src.transform
            geotiff_crs = src.crs
            if gdf.crs != geotiff_crs:
                log(f"CRS不一致,正在转换: {gdf.crs} -> {geotiff_crs}")
                gdf = gdf.to_crs(geotiff_crs)

            heights = []
            for idx, row in gdf.iterrows():
                polygon = row.geometry
                bid = row.get('BuildingId', idx)
                if apply_offset_flag and polygon is not None:
                    pixel_x, pixel_y = row.get('Offset_X', 0), row.get('Offset_Y', 0)
                    geo_dx, geo_dy = pixel_offset_to_geo_offset(transform, pixel_x, pixel_y)
                    polygon = apply_offset(polygon, geo_dx, geo_dy)
                    log(f"BuildingId: {bid} - Pixel Offset: ({pixel_x}, {pixel_y}) "
                        f"-> Geo Offset: ({geo_dx}, {geo_dy})")
                height = _extract_height(src, polygon)
                heights.append(height)
                log(f"BuildingId: {bid} - 提取高度: {height}")
        gdf[height_col_name] = np.asarray(heights, dtype=float)

    except Exception as e:
        raise RuntimeError(f"处理 {geotiff_path} 失败: {e}") from e

    output_dir = os.path.dirname(output_geojson_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    gdf.to_file(output_geojson_path, driver='GeoJSON')
    print(f"已保存更新后的GeoJSON到: {output_geojson_path}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Attach heights from an absolute-height GeoTIFF to building polygons."
    )
    parser.add_argument("--geojson-path", "--geojson_path", dest="geojson_path", required=True)
    parser.add_argument("--geotiff-path", "--geotiff_path", dest="geotiff_path", required=True)
    parser.add_argument("--output-path", "--output_path", dest="output_path", required=True)

    offset_group = parser.add_mutually_exclusive_group()
    offset_group.add_argument(
        "--apply-offset",
        dest="apply_offset",
        action="store_true",
        default=True,
        help="Apply Offset_X/Offset_Y before extracting heights. This is the default.",
    )
    offset_group.add_argument(
        "--no-apply-offset",
        dest="apply_offset",
        action="store_false",
        help="Extract heights directly from the input polygon geometry.",
    )
    parser.add_argument(
        "--height-column",
        help="Output property name. Defaults to Extracted_Height with offsets and LiDAR_Height without offsets.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-building extraction logs.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    match_polygons_with_height(
        args.geojson_path,
        args.geotiff_path,
        args.output_path,
        args.apply_offset,
        verbose=not args.quiet,
        height_column=args.height_column,
    )
