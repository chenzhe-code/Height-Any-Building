"""Test Harmony job outcomes without starting remote jobs."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

from tools import download_atl03_by_tif as atl03


class HarmonyDownloadTests(TestCase):
    def test_partial_job_requires_opt_in_and_cannot_be_reused_as_complete(self):
        for allow_partial in (False, True):
            with self.subTest(allow_partial=allow_partial), TemporaryDirectory() as folder:
                output = Path(folder)
                tif = output / "image.tif"
                track = output / "track.h5"
                track.write_bytes(b"sample")
                response = {
                    "status": "successful with errors",
                    "links": [{"href": "https://example.invalid/track.h5", "rel": "data"}],
                    "errors": [{"message": "second granule failed"}],
                }
                argv = ["download_atl03_by_tif.py", "--tif-path", str(tif),
                        "--output-dir", folder, "--start-date", "2022-01-10",
                        "--end-date", "2022-01-11", "--download", "--harmony-subset"]
                if allow_partial:
                    argv.append("--harmony-ignore-errors")
                with (mock.patch.object(sys, "argv", argv),
                      mock.patch.object(atl03, "resolve_tif_paths", return_value=[tif]),
                      mock.patch.object(atl03, "geotiffs_wgs84_bbox", return_value=(114, 30, 115, 31)),
                      mock.patch.object(atl03, "find_collection", return_value={"id": "collection"}),
                      mock.patch.object(atl03, "search_granules", return_value=[{"id": "G1"}, {"id": "G2"}]),
                      mock.patch.object(atl03, "make_session"),
                      mock.patch.object(atl03, "submit_harmony_subset_job", return_value={"jobID": "mock-job"}),
                      mock.patch.object(atl03, "get_harmony_job", return_value=response),
                      mock.patch.object(atl03, "download_urls", return_value=[track]) as download):
                    if allow_partial:
                        self.assertEqual(atl03.main(), 0)
                        download.assert_called_once()
                    else:
                        with self.assertRaisesRegex(RuntimeError, "did not complete all inputs"):
                            atl03.main()
                        download.assert_not_called()
                marker = json.loads((output / "atl03_download_partial.json").read_text())
                self.assertEqual(marker["status"], "partial")
                self.assertEqual(marker["errors"], response["errors"])
                self.assertEqual(len(marker["files"]), int(allow_partial))
                self.assertIsNone(atl03.reusable_harmony_downloads(output, marker["request"]))
