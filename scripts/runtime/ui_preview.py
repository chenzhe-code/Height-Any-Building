"""Render bounded previews from the GeoTIFF and GeoJSON of a finished UI job."""

from base64 import b64encode


MAX_EDGE = 1200
HEIGHT_PALETTE = ("#17274d", "#245d8b", "#219783", "#94c769", "#ffe19a")
FOOTPRINT_COLOR = (255, 255, 255)


def render(height_path, building_path):
    """Sample band one and align the footprint overlay to the same preview grid."""
    # The job runner can start before the geospatial dependencies are installed.
    try:
        import cv2
        import geopandas as gpd
        import numpy as np
        import rasterio
        from rasterio.features import rasterize
        from rasterio.transform import Affine
    except ImportError as error:
        raise ValueError("结果预览需要项目的核心 Python 环境。") from error

    def png(rgba):
        ok, payload = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        if not ok:
            raise ValueError("无法生成结果预览图。")
        return "data:image/png;base64," + b64encode(payload).decode("ascii")

    with rasterio.open(height_path) as source:
        factor = min(1.0, MAX_EDGE / max(source.width, source.height))
        cols = max(1, round(source.width * factor))
        rows = max(1, round(source.height * factor))
        height = source.read(1, out_shape=(rows, cols), masked=True).astype("float32")
        height = height * source.scales[0] + source.offsets[0]
        grid = source.transform * Affine.scale(source.width / cols, source.height / rows)
        crs = source.crs
        width, length = source.width, source.height

    valid = ~np.ma.getmaskarray(height) & np.isfinite(height.data)
    maximum = max(0.0, float(height.data[valid].max())) if valid.any() else 0.0
    normalized = np.clip(np.where(valid, height.data, 0) / (maximum or 1), 0, 1)
    colors = np.array([tuple(bytes.fromhex(color[1:])) for color in HEIGHT_PALETTE])
    stops = np.linspace(0, 1, len(colors))
    raster = np.empty((rows, cols, 4), dtype="uint8")
    for channel in range(3):
        raster[:, :, channel] = np.interp(normalized, stops, colors[:, channel]).round()
    raster[:, :, 3] = valid.astype("uint8") * 255

    footprint_png = None
    count = None
    if building_path is not None:
        buildings = gpd.read_file(building_path, columns=[]).geometry
        count = len(buildings)
        if count:
            if crs is None or buildings.crs is None:
                raise ValueError("叠加建筑物足迹需要栅格和 GeoJSON 的坐标系。")
            if buildings.crs != crs:
                buildings = buildings.to_crs(crs)
            outlines = buildings[buildings.geom_type.isin(("Polygon", "MultiPolygon"))].boundary
            if not outlines.empty:
                edges = rasterize(outlines, out_shape=(rows, cols), transform=grid,
                                  all_touched=True, default_value=220, dtype="uint8")
                overlay = np.empty_like(raster)
                overlay[:, :, :3] = FOOTPRINT_COLOR
                overlay[:, :, 3] = edges
                footprint_png = png(overlay)

    return {"height_png": png(raster), "footprint_png": footprint_png, "buildings": count,
            "height_max": maximum, "has_heights": bool(valid.any()), "palette": HEIGHT_PALETTE,
            "width": width, "height": length}
