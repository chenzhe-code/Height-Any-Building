import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
import rasterio
from rasterio.transform import from_origin

from tools.convert_footprints import REQUIRED_COLUMNS, csv_and_geotiff_to_geojson_or_shp


SQUARE = "POLYGON ((1 1, 4 1, 4 4, 1 4, 1 1))"


class FootprintConversionTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.csv = self.root / "footprints.csv"
        self.output = self.root / "footprints.geojson"
        self.create_patch("a")

    def create_patch(self, name, crs="EPSG:32650"):
        with rasterio.open(self.root / f"{name}.tif", "w", driver="GTiff", width=8,
                           height=8, count=1, dtype="uint8", crs=crs,
                           transform=from_origin(400000, 3300000, 1, 1)):
            pass

    def convert(self, *rows):
        pd.DataFrame(rows, columns=REQUIRED_COLUMNS).to_csv(self.csv, index=False)
        csv_and_geotiff_to_geojson_or_shp(self.csv, self.root, self.output)

    def test_missing_patch_aborts_even_after_processing_a_valid_patch(self):
        with self.assertRaisesRegex(ValueError, r"Cannot open GeoTIFF patch.*ImageId='missing'.*missing\.tif"):
            self.convert(("a", 1, SQUARE, 0.9, "[0 0]", 10), ("missing", 2, SQUARE, 0.9, "[0 0]", 10))
        self.assertFalse(self.output.exists())

    def test_missing_or_mismatched_crs_aborts_conversion(self):
        for crs, message in ((None, "Missing CRS"), ("EPSG:32651", "Patch CRS mismatch")):
            with self.subTest(crs=crs):
                self.create_patch("b", crs)
                with self.assertRaisesRegex(ValueError, rf"{message}.*ImageId='b'"):
                    self.convert(("a", 1, SQUARE, 0.9, "[0 0]", 10), ("b", 2, SQUARE, 0.9, "[0 0]", 10))
                self.assertFalse(self.output.exists())

    def test_bad_row_identifies_the_building_and_does_not_publish(self):
        for wkt, offset in (("NOT_WKT", "[0 0]"), ("POLYGON EMPTY", "[0 0]"), (SQUARE, "[1]")):
            with self.subTest(wkt=wkt, offset=offset):
                with self.assertRaisesRegex(ValueError, r"Invalid footprint for ImageId='a', BuildingId=2"):
                    self.convert(("a", 1, SQUARE, 0.9, "[0 0]", 10), ("a", 2, wkt, 0.9, offset, 10),
                                 ("a", 3, SQUARE, 0.9, "[0 0]", 10))
                self.assertFalse(self.output.exists())

    def test_missing_image_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"Missing ImageId for BuildingId=.*2"):
            self.convert((None, 2, SQUARE, 0.9, "[0 0]", 10))
        self.assertFalse(self.output.exists())

    def test_empty_predictions_write_a_valid_empty_geojson(self):
        self.convert()
        result = json.loads(self.output.read_text())
        self.assertEqual(result["type"], "FeatureCollection")
        self.assertEqual(result["features"], [])
