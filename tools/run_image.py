#!/usr/bin/env python3
"""Run HAB from one RGB GeoTIFF, optionally refining seams across calibrated tiles."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from rasterio.windows import Window

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.runtime.check_runtime import check_runtime
from scripts.runtime.model_config import (
    DEFAULT_GEO_ENV, DEFAULT_MLSBRN_ENV, DEPTH_MODELS, ROOT, depth_environment,
)
from utils.window_grid import window_step

PATCH_SIZE = 1024


def axis_edges(length, tile_size, window_size=PATCH_SIZE):
    if length < window_size:
        raise ValueError(f'Image dimension {length} is smaller than a {window_size}-pixel window')
    if tile_size < 2 * PATCH_SIZE:
        raise ValueError(f'--tile-size must be at least {2 * PATCH_SIZE}')
    count = math.ceil(length / tile_size)
    # Balanced, disjoint intervals keep the complete extent and avoid narrow edge tiles.
    edges = [index * length // count for index in range(count + 1)]
    if length // count < window_size:
        raise ValueError(f'An output tile is smaller than the {window_size}-pixel window; adjust --tile-size or --window-size')
    return edges


def metric_crs(crs):
    return crs.is_projected and crs.linear_units.lower() in ('metre', 'meter', 'metres', 'meters')


def grid_plan(image, tile_size=5000, target_crs=None, window_size=PATCH_SIZE):
    image = Path(image).resolve()
    with rasterio.open(image) as source:
        if source.crs is None:
            raise ValueError('Input GeoTIFF has no CRS')
        if source.count != 3 or source.dtypes != ('uint8',) * 3:
            raise ValueError('Input must contain exactly three UInt8 RGB bands; convert radiometry and band order before running HAB')
        target = CRS.from_user_input(target_crs) if target_crs else source.crs
        if not target_crs and not metric_crs(target):
            west, south, east, north = transform_bounds(source.crs, 'EPSG:4326', *source.bounds)
            if east - west > 6 or south < -80 or north > 84:
                raise ValueError('Automatic UTM selection requires a regional extent within 6 degrees longitude and UTM latitudes; supply --target-crs')
            lon, lat = (west + east) / 2, (south + north) / 2
            zone = max(1, min(60, int((lon + 180) // 6) + 1))
            target = CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)
        if not metric_crs(target):
            raise ValueError('--target-crs must be a projected coordinate system with metre units')
        native = (source.crs == target and source.transform.b == source.transform.d == 0
                  and source.transform.a == 1 and source.transform.e == -1)
        if native:
            transform, width, height = source.transform, source.width, source.height
        else:
            transform, width, height = calculate_default_transform(
                source.crs, target, source.width, source.height, *source.bounds, resolution=1.0)
        xs = axis_edges(width, tile_size, window_size)
        ys = axis_edges(height, tile_size, window_size)
        tiles = []
        for row, (top, bottom) in enumerate(zip(ys, ys[1:])):
            for col, (left, right) in enumerate(zip(xs, xs[1:])):
                tiles.append(dict(name=f'tile_r{row:04d}_c{col:04d}', x=left, y=top,
                                  width=right-left, height=bottom-top))
        stat = image.stat()
        return dict(preparation_version=2, input=str(image), input_size=stat.st_size, input_mtime_ns=stat.st_mtime_ns,
                    crs=target.to_string(), transform=list(transform)[:6], width=width,
                    height=height, reproject=not native, tile_size=tile_size, tiles=tiles)


@contextmanager
def aligned_source(source, plan, temporary):
    if not plan['reproject']:
        yield source
    else:
        from affine import Affine
        # Use GDAL's streaming reproject path, matching the production image
        # preparation. The VRT read path produced different downsampled pixels
        # on the Wuhan input despite using the same target grid and resampler.
        profile = dict(driver='GTiff', crs=plan['crs'], transform=Affine(*plan['transform']),
                       width=plan['width'], height=plan['height'], count=3, dtype='uint8',
                       tiled=True, blockxsize=256, blockysize=256,
                       compress='deflate', BIGTIFF='IF_SAFER', nodata=None)
        try:
            with rasterio.open(temporary, 'w', **profile) as destination:
                for band in range(1, 4):
                    reproject(rasterio.band(source, band), rasterio.band(destination, band),
                              resampling=Resampling.bilinear, num_threads=16)
            with rasterio.open(temporary) as warped:
                yield warped
        finally:
            Path(temporary).unlink(missing_ok=True)


def prepare_tiles(plan, work_dir):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    with (work_dir / '.prepare.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _prepare_locked(plan, work_dir)


def _prepare_locked(plan, work_dir):
    directory = work_dir / 'input_tiles'
    manifest = work_dir / 'input_grid.json'
    if manifest.exists() and json.loads(manifest.read_text()) != plan:
        raise ValueError('Work directory belongs to a different image or grid; choose a new --work-dir')
    existing = list(directory.glob('*.tif')) if directory.exists() else []
    if existing and not manifest.exists():
        raise ValueError('Untracked input tiles already exist; choose a new --work-dir')
    directory.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        temporary_manifest = manifest.with_suffix('.tmp')
        temporary_manifest.write_text(json.dumps(plan, indent=2) + '\n')
        temporary_manifest.replace(manifest)
    from affine import Affine
    grid_transform = Affine(*plan['transform'])
    crs = CRS.from_user_input(plan['crs'])
    missing = []
    for record in plan['tiles']:
        destination = directory / (record['name'] + '.tif')
        window = Window(record['x'], record['y'], record['width'], record['height'])
        transform = rasterio.windows.transform(window, grid_transform)
        if destination.is_file():
            with rasterio.open(destination) as current:
                if (current.shape != (record['height'], record['width']) or current.count != 3
                        or current.dtypes != ('uint8',) * 3 or current.transform != transform
                        or current.crs != crs):
                    raise ValueError(f'Existing tile grid mismatch: {destination}')
        else:
            missing.append((destination, window, transform))
    if not missing:
        return directory

    with rasterio.open(plan['input']) as source, aligned_source(
            source, plan, work_dir / 'aligned_input.tmp.tif') as aligned:
        for destination, window, transform in missing:
            temporary = destination.with_suffix('.tif.tmp')
            profile = dict(driver='GTiff', width=window.width, height=window.height,
                           count=3, dtype='uint8', crs=plan['crs'], transform=transform,
                           tiled=True, blockxsize=256, blockysize=256, compress='deflate', BIGTIFF='IF_SAFER')
            if aligned.nodata is not None:
                profile['nodata'] = aligned.nodata
            with rasterio.open(temporary, 'w', **profile) as output:
                for _, block in output.block_windows(1):
                    input_window = Window(window.col_off + block.col_off, window.row_off + block.row_off,
                                          block.width, block.height)
                    output.write(aligned.read(window=input_window), window=block)
            temporary.replace(destination)
    return directory


def pipeline_plan(plan, work_dir, model=None, global_mode=None, gpus='0', *,
                  window_size=PATCH_SIZE, overlap=0.5, min_photon_controls=None, ground_filter=None):
    window_step(window_size, overlap)
    if min_photon_controls is not None and min_photon_controls <= 0:
        raise ValueError('--min-photon-controls must be a positive integer')
    work_dir = Path(work_dir).resolve()
    large = len(plan['tiles']) > 1
    model = model or ('SynRS3D' if large else 'Distill-Any-Depth')
    global_mode = global_mode or ('patch' if large and model == 'SynRS3D' else 'none')
    if global_mode != 'none' and not large:
        raise ValueError('跨瓦片联合优化需要实际切出多块相邻瓦片。')
    if global_mode == 'tile' and model != 'SynRS3D':
        raise ValueError('Legacy --global-mode tile requires SynRS3D. Use --global-mode patch with other models.')
    env = dict(WS=str(window_size), OL=str(overlap), DR='1', COMPUTE_UNCERTAINTY='0',
               FUSION_MODE='mean', HEIGHT_SOURCE='absolute', CONTROL_SOURCE='photon')
    if min_photon_controls is not None:
        env['MIN_PHOTON_CONTROLS'] = str(min_photon_controls)
    if large:
        command = ['bash', str(ROOT / 'scripts/run_batch_work_dir.sh'),
                   '--input-dir', str(work_dir / 'input_tiles'),
                   '--output-root', str(work_dir / 'results'), '--model', model,
                   '--control-source', 'photon', '--no-uncertainty', '--fusion', 'mean',
                   '--gpus', gpus, '--atl03-scope', 'shared', '--merge-city']
        if model == 'SynRS3D':
            command += ['--photon-fallback']
        if ground_filter is not None or model == 'SynRS3D':
            command += ['--ground-filter' if ground_filter else '--no-ground-filter']
        if global_mode == 'patch':
            command += ['--global-patch-optimize']
            if model == 'SynRS3D' and window_size == PATCH_SIZE and overlap == 0.5 and ground_filter is not True:
                command += ['--export-synrs3d-raw-city']
        elif global_mode == 'tile':
            command += ['--global-optimize']
    else:
        tile = work_dir / 'input_tiles' / (plan['tiles'][0]['name'] + '.tif')
        env.update(JOB_DIR=str(work_dir), INPUT_TIF=str(tile), TIF_IMG=tile.name,
                   MODEL_NAME=model, GPU_ID=gpus.split(',')[0],
                   MLS_OUTPUT_DIR=str(work_dir / 'mls_results'),
                   GROUND_FILTER='auto' if ground_filter is None else str(int(ground_filter)))
        # The full pipeline consumes local ATL03, so the single-tile entry downloads it first.
        command = ['bash', str(ROOT / 'scripts/run_full_pipeline.sh')]
    seam_model = 'SynRS3D' if global_mode != 'none' else None
    conda_environments = dict(
        geospatial=os.environ.get('GEO_ENV') or DEFAULT_GEO_ENV,
        depth=depth_environment(model),
        footprints=os.environ.get('MLSBRN_ENV') or DEFAULT_MLSBRN_ENV,
        seam=DEPTH_MODELS[seam_model][1] if seam_model else None,
    )
    return dict(model=model, global_mode=global_mode, seam_model=seam_model,
                conda_environments=conda_environments,
                command=command, environment=env)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--tile-size', type=int, default=5000)
    p.add_argument('--window-size', type=int, default=PATCH_SIZE, help='Inference window width and height in pixels')
    p.add_argument('--overlap', type=float, default=0.5, help='Inference window overlap fraction')
    p.add_argument('--min-photon-controls', type=int, help='Minimum valid controls for photon calibration')
    ground = p.add_mutually_exclusive_group()
    ground.add_argument('--ground-filter', dest='ground_filter', action='store_true', default=None)
    ground.add_argument('--no-ground-filter', dest='ground_filter', action='store_false', default=None)
    p.add_argument('--target-crs', help='Projected metre CRS; regional geographic images default to their local UTM zone')
    p.add_argument('--model', choices=DEPTH_MODELS)
    p.add_argument('--global-mode', choices=['patch', 'tile', 'none'])
    p.add_argument('--gpus', default='0')
    p.add_argument('--atl03-start-date', default='2022-01-01')
    p.add_argument('--atl03-end-date', default=date.today().isoformat())
    p.add_argument('--local-atl03-dir', type=Path,
                   help='Use these already selected ATL03 HDF5 files instead of downloading')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Print grid and pipeline plan; write nothing')
    mode.add_argument('--check', action='store_true', help='Check the selected GPU environments and weights without preparing inputs or downloading ATL03')
    mode.add_argument('--prepare-only', action='store_true', help='Only reproject and split input; do not run models or download ATL03')
    args = p.parse_args()
    if not args.gpus or any(not value.isdigit() for value in args.gpus.split(',')):
        p.error('--gpus must be comma-separated nonnegative GPU IDs')
    try:
        if date.fromisoformat(args.atl03_start_date) > date.fromisoformat(args.atl03_end_date):
            raise ValueError('ATL03 start date must be no later than its end date')
        window_step(args.window_size, args.overlap)
        plan = grid_plan(args.image, args.tile_size, args.target_crs, args.window_size)
        run = pipeline_plan(plan, args.work_dir, args.model, args.global_mode, args.gpus,
                            window_size=args.window_size, overlap=args.overlap,
                            min_photon_controls=args.min_photon_controls, ground_filter=args.ground_filter)
        run['environment'].update(ATL03_START_DATE=args.atl03_start_date,
                                  ATL03_END_DATE=args.atl03_end_date,
                                  START_DATE=args.atl03_start_date, END_DATE=args.atl03_end_date)
        if args.local_atl03_dir is not None:
            local = args.local_atl03_dir.resolve()
            if not any(path.is_file() for path in local.glob('*.h5')):
                raise ValueError(f'Local ATL03 directory must contain HDF5 granules: {local}')
            if len(plan['tiles']) == 1:
                run['environment']['ICESAT_PATH'] = str(local)
            else:
                run['command'] += ['--local-atl03-dir', str(local)]
        if args.dry_run:
            print(json.dumps(dict(grid=plan, pipeline=run), indent=2))
            return
        single_tile = len(plan['tiles']) == 1
        if args.check or (single_tile and not args.prepare_only):
            check_runtime(run['model'], run['environment'].get('GPU_ID', args.gpus),
                          seam=run['seam_model'] is not None)
        if args.check:
            print(f"Runtime check passed for {run['model']} and MLS-BRN"
                  + ("; SynRS3D seam model ready" if run['seam_model'] is not None else ""))
            return
        prepare_tiles(plan, args.work_dir)
    except (OSError, ValueError, rasterio.errors.RasterioError) as error:
        p.error(str(error))
    if args.prepare_only:
        print(f'Prepared {len(plan["tiles"])} input tiles: {args.work_dir / "input_tiles"}')
        return
    env = os.environ.copy()
    env.update(run['environment'])
    env.setdefault('MPLBACKEND', 'Agg')
    if len(plan['tiles']) == 1 and args.local_atl03_dir is None:
        env['ICESAT_PATH'] = subprocess.check_output(
            [sys.executable, str(ROOT / 'scripts/runtime/pipeline_cache.py'), 'download-atl03',
             str(args.work_dir.resolve() / 'ICESat-2'), '--tif', env['INPUT_TIF']],
            cwd=ROOT, env=env, text=True,
        ).strip()
    subprocess.run(run['command'], cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
