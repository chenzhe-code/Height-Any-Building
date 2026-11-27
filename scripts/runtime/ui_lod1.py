"""Build a metre-based LOD1 preview mesh from the building-height GeoJSON."""

from array import array
from base64 import b64encode
from math import hypot
import sys

HEIGHT_FIELD = "Extracted_Height"


def polygons(geometry):
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        for part in geometry.geoms:
            yield from polygons(part)


def render(path):
    try:
        import geopandas as gpd
        import numpy as np
        import pandas as pd
        from shapely import constrained_delaunay_triangles, make_valid, orient_polygons
    except ImportError as error:
        raise ValueError("LOD1 预览需要安装项目的核心 Python 环境。") from error

    buildings = gpd.read_file(path, columns=[HEIGHT_FIELD])
    if HEIGHT_FIELD not in buildings:
        raise ValueError("成果中缺少逐栋提取高度，无法生成 LOD1。")
    count = len(buildings)
    heights = pd.to_numeric(buildings[HEIGHT_FIELD], errors="coerce")
    valid = np.isfinite(heights) & (heights > 0)
    buildings = buildings.loc[valid & buildings.geometry.notna() & ~buildings.geometry.is_empty].copy()
    if buildings.empty:
        return {"buildings": 0, "skipped": count}
    if buildings.crs is None:
        raise ValueError("LOD1 预览需要建筑足迹的坐标系。")
    crs = buildings.crs
    if not crs.is_projected or any(axis.unit_conversion_factor != 1 for axis in crs.axis_info[:2]):
        buildings = buildings.to_crs(buildings.estimate_utm_crs())

    left, bottom, right, top = buildings.total_bounds
    center_x, center_y = (left + right) / 2, (bottom + top) / 2
    vertices = array("f")
    maximum = 0.0
    rendered = 0
    for footprint, height in zip(buildings.geometry, heights.loc[buildings.index]):
        start = len(vertices)
        geometry = footprint if footprint.is_valid else make_valid(footprint)
        for poly in polygons(orient_polygons(geometry)):
            if poly.is_empty or poly.area <= 0:
                continue
            for triangle in constrained_delaunay_triangles(poly).geoms:
                face = list(triangle.exterior.coords)[:3]
                for level, shade in ((height, 1.0), (0, 0.58)):
                    for x, y, *_ in face:
                        vertices.extend((x - center_x, y - center_y, level, height, shade))
            for ring in (poly.exterior, *poly.interiors):
                points = [(x - center_x, y - center_y) for x, y, *_ in ring.coords]
                for a, b in zip(points, points[1:]):
                    dx, dy = b[0] - a[0], b[1] - a[1]
                    length = hypot(dx, dy)
                    if length == 0:
                        continue
                    shade = 0.70 + 0.20 * (0.4 * dy - 0.9 * dx) / length
                    for point, z in ((a, 0), (b, 0), (b, height), (a, 0), (b, height), (a, height)):
                        vertices.extend((*point, z, height, shade))
        if len(vertices) > start:
            rendered += 1
            maximum = max(maximum, float(height))

    if sys.byteorder != "little":
        vertices.byteswap()
    return {"mesh": b64encode(vertices).decode("ascii"), "extent": [right - left, top - bottom],
            "height_max": maximum, "buildings": rendered, "skipped": count - rendered}
