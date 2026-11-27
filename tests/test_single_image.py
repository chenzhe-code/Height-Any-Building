"""Input-only integration checks: real GeoTIFF tiling and large-image routing."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import rasterio
from rasterio.transform import from_origin

from types import SimpleNamespace

from scripts.runtime import model_config
from scripts.runtime.run_depth_backend import command_for
from tools import run_image
from tools.run_image import axis_edges, grid_plan, pipeline_plan, prepare_tiles

ROOT = Path(__file__).resolve().parents[1]

def image(path, width=2200, height=1024, crs='EPSG:32650'):
    transform = from_origin(400000, 3400000, 1, 1)
    data = np.repeat((np.arange(width, dtype=np.uint16) % 251).astype('uint8')[None, None, :], height, axis=1)
    data = np.repeat(data, 3, axis=0)
    with rasterio.open(path, 'w', driver='GTiff', width=width, height=height,
                       count=3, dtype='uint8', crs=crs, transform=transform) as output:
        output.write(data)
    return data, transform


class ImageEntryTests(unittest.TestCase):
    def test_balanced_tiles_keep_edges_without_small_remainder(self):
        for length in (1024, 5000, 5001, 10001, 25317):
            edges = axis_edges(length, 5000)
            self.assertEqual((edges[0], edges[-1]), (0, length))
            self.assertTrue(all(1024 <= b-a <= 5000 for a, b in zip(edges, edges[1:])))

    def test_too_small_input_and_tile_size_rejected(self):
        for length, size in ((1023, 5000), (5000, 1024)):
            with self.assertRaises(ValueError):
                axis_edges(length, size)

    def test_prepare_reconstructs_exact_pixels_and_transforms(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.tif'
            data, transform = image(path)
            plan = grid_plan(path, 2048)
            work = Path(directory) / 'job'
            tiles = prepare_tiles(plan, work)
            combined = np.zeros_like(data)
            for record in plan['tiles']:
                with rasterio.open(tiles / (record['name']+'.tif')) as source:
                    self.assertEqual(source.transform, transform * rasterio.Affine.translation(record['x'], record['y']))
                    combined[:, record['y']:record['y']+source.height, record['x']:record['x']+source.width] = source.read()
            np.testing.assert_array_equal(combined, data)
            self.assertEqual(prepare_tiles(plan, work), tiles)
            other = dict(plan, tile_size=5000)
            with self.assertRaisesRegex(ValueError, 'different image'):
                prepare_tiles(other, work)

    def test_large_default_and_relative_model_opt_in(self):
        plan = dict(tiles=[dict(name='left'), dict(name='right')])
        run = pipeline_plan(plan, Path('/tmp/example'))
        self.assertEqual((run['model'], run['global_mode']), ('SynRS3D', 'patch'))
        self.assertIn('--global-patch-optimize', run['command'])
        self.assertIn('--merge-city', run['command'])
        relative = pipeline_plan(plan, Path('/tmp/example'), model='Distill-Any-Depth')
        self.assertEqual(relative['global_mode'], 'none')
        self.assertIsNone(relative['seam_model'])
        enabled = pipeline_plan(plan, Path('/tmp/example'), model='Distill-Any-Depth', global_mode='patch')
        self.assertEqual(enabled['seam_model'], 'SynRS3D')
        self.assertIn('--global-patch-optimize', enabled['command'])
        self.assertNotIn('--export-synrs3d-raw-city', enabled['command'])
        with self.assertRaisesRegex(ValueError, '多块'):
            pipeline_plan(dict(tiles=plan['tiles'][:1]), Path('/tmp/example'), global_mode='patch')

    def test_each_primary_model_plans_its_own_environment_and_seam_stays_separate(self):
        grid = dict(tiles=[dict(name='left'), dict(name='right')])
        names = {
            'Distill-Any-Depth': 'distill-any-depth',
            'DEPTH-ANYTHING-V1': 'depth-anything-v1',
            'DEPTH-ANYTHING-V2': 'depth-anything-v2',
            'SynRS3D': 'synrs3d',
        }
        with mock.patch.dict(os.environ, {}, clear=True):
            for model, environment in names.items():
                with self.subTest(model=model):
                    run = pipeline_plan(grid, Path('/tmp/example'), model=model, global_mode='patch')
                    self.assertEqual(run['conda_environments'], {
                        'geospatial': 'hab-core', 'depth': environment,
                        'footprints': 'mlsbrn', 'seam': 'synrs3d',
                    })
        with mock.patch.dict(os.environ, {'GEO_ENV': 'DAB', 'HAB_DEPTH_ENV': 'DAB'}, clear=True):
            run = pipeline_plan(grid, Path('/tmp/example'), model='Distill-Any-Depth', global_mode='patch')
            self.assertEqual(run['conda_environments']['geospatial'], 'DAB')
            self.assertEqual(run['conda_environments']['depth'], 'DAB')
            self.assertEqual(run['conda_environments']['seam'], 'synrs3d')

    def test_custom_window_and_control_parameters_reach_both_routes(self):
        options = dict(window_size=512, overlap=0.25, min_photon_controls=6, ground_filter=True)
        for tiles in ([dict(name="left")], [dict(name="left"), dict(name="right")]):
            run = pipeline_plan(dict(tiles=tiles), Path("/tmp/example"), global_mode="none", **options)
            self.assertEqual([run["environment"][k] for k in ("WS", "OL", "MIN_PHOTON_CONTROLS")],
                             ["512", "0.25", "6"])
            if len(tiles) == 1:
                self.assertEqual(run["environment"]["GROUND_FILTER"], "1")
            else:
                self.assertIn("--ground-filter", run["command"])
                optimized = pipeline_plan(dict(tiles=tiles), Path("/tmp/example"), **options)
                self.assertIn("--global-patch-optimize", optimized["command"])
                self.assertNotIn("--export-synrs3d-raw-city", optimized["command"])
        with self.assertRaisesRegex(ValueError, "smaller than"):
            axis_edges(2100, 2048, 1200)

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.tif'
            image(path)
            work = Path(directory) / 'job'
            output = subprocess.check_output([sys.executable, str(ROOT/'tools/run_image.py'),
                '--image', str(path), '--work-dir', str(work), '--tile-size', '2048', '--dry-run'], text=True)
            self.assertEqual(json.loads(output)['pipeline']['model'], 'SynRS3D')
            self.assertFalse(work.exists())

            local = Path(directory) / 'atl03'
            local.mkdir()
            (local / 'track.h5').touch()
            command = [sys.executable, str(ROOT/'tools/run_image.py'), '--image', str(path),
                       '--work-dir', str(work), '--local-atl03-dir', str(local), '--dry-run']
            one = json.loads(subprocess.check_output(command, text=True))['pipeline']
            self.assertEqual(one['environment']['ICESAT_PATH'], str(local))
            many = json.loads(subprocess.check_output(command + ['--tile-size', '2048'], text=True))['pipeline']
            self.assertEqual(many['command'][-2:], ['--local-atl03-dir', str(local)])
            self.assertFalse(work.exists())

    def test_geographic_image_is_prepared_on_a_metric_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'geographic.tif'
            with rasterio.open(path, 'w', driver='GTiff', width=1200, height=1200,
                               count=3, dtype='uint8', crs='EPSG:4326',
                               transform=from_origin(114, 30, 0.00001, 0.00001)) as output:
                output.write(np.full((3, 1200, 1200), 100, dtype='uint8'))
            plan = grid_plan(path)
            self.assertTrue(plan['reproject'])
            self.assertEqual(plan['crs'], 'EPSG:32650')
            tiles = prepare_tiles(plan, Path(directory) / 'job')
            with rasterio.open(next(tiles.glob('*.tif'))) as prepared:
                self.assertEqual(prepared.res, (1, 1))
                self.assertTrue(prepared.crs.is_projected)
                self.assertEqual(prepared.shape, (plan['height'], plan['width']))
            with mock.patch.object(run_image, 'reproject') as reproject:
                self.assertEqual(prepare_tiles(plan, Path(directory) / 'job'), tiles)
                reproject.assert_not_called()

    def test_runtime_check_precedes_preparation_and_download_for_one_image(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.tif'
            image(path)
            work = Path(directory) / 'job'
            argv = ['run_image.py', '--image', str(path), '--work-dir', str(work)]
            with (mock.patch.object(sys, 'argv', argv),
                  mock.patch.object(run_image, 'check_runtime', side_effect=ValueError('missing checkpoint')),
                  mock.patch.object(run_image.subprocess, 'check_output') as download):
                with self.assertRaises(SystemExit) as error:
                    run_image.main()
                self.assertEqual(error.exception.code, 2)
                self.assertFalse(work.exists())
                download.assert_not_called()

    def test_nonmetric_explicit_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.tif'
            image(path)
            with self.assertRaisesRegex(ValueError, 'metre units'):
                grid_plan(path, target_crs='EPSG:4326')

    def test_depth_checkpoint_prefers_documented_path_and_accepts_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / 'checkpoints/SynRS3D/RS3DAda_vitl_DPT_height.pth'
            legacy = root / 'MDE/SynRS3D/pretrain/RS3DAda_vitl_DPT_height.pth'
            with mock.patch.object(model_config, 'ROOT', root):
                self.assertEqual(model_config.checkpoint_path('SynRS3D'), canonical)
                legacy.parent.mkdir(parents=True)
                legacy.touch()
                self.assertEqual(model_config.checkpoint_path('SynRS3D'), legacy)
                canonical.parent.mkdir(parents=True)
                canonical.touch()
                self.assertEqual(model_config.checkpoint_path('SynRS3D'), canonical)

    def test_mls_checkpoint_prefers_documented_path_and_accepts_previous_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mission = root / 'old_mls'
            mission.mkdir()
            canonical = root / 'MLS-BRN/checkpoints/mlsbrn-inference.pth'
            previous = root / 'checkpoints/MLS-BRN/latest.pth'
            with mock.patch.multiple(model_config, ROOT=root, MLS_MISSION=mission):
                self.assertEqual(model_config.checkpoint_path('MLS-BRN'), canonical)
                for checkpoint in (mission / 'epoch_24.pth', mission / 'latest.pth', previous, canonical):
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint.touch()
                    self.assertEqual(model_config.checkpoint_path('MLS-BRN'), checkpoint)

    def test_backend_cli_matches_each_model(self):
        for model in ['SynRS3D','Distill-Any-Depth','DEPTH-ANYTHING-V1','DEPTH-ANYTHING-V2']:
            args = SimpleNamespace(model=model, img_path='patches', output_dir='prediction',
                                   checkpoint='model.pth', processing_res=700, seed=1234, get_conf=False)
            cwd, argv = command_for(args)
            self.assertTrue(Path(argv[1]).is_file())
            self.assertIn('--checkpoint', argv)
            self.assertTrue(cwd.is_dir())
            if model.endswith('V1'):
                self.assertTrue(argv[argv.index('--outdir')+1].endswith('/prediction/depth'))
                args.get_conf=True
                with self.assertRaises(ValueError):
                    command_for(args)


if __name__ == '__main__':
    unittest.main()
