"""Shared GeoTIFF reading and writing for imagery and height rasters."""

import rasterio


def read_geotiff(file_path, *, band=None):
    """Read all bands, or one 1-based band as a two-dimensional array."""
    with rasterio.open(file_path) as source:
        return source.read(band), source.meta


def save_geotiff(file_path, data, meta):
    with rasterio.open(file_path, "w", **meta) as destination:
        destination.write(data)
