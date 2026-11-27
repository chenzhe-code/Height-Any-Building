#!/usr/bin/env python3
"""Translate HAB's common options to each inference backend's actual CLI."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

if __package__:
    from .model_config import DEPTH_MODELS, checkpoint_path, depth_directory, depth_runner
else:
    from model_config import DEPTH_MODELS, checkpoint_path, depth_directory, depth_runner


def command_for(args):
    runner = depth_runner(args.model)
    backend = depth_directory(args.model)
    image_dir = str(Path(args.img_path).resolve())
    output_dir = str(Path(args.output_dir).resolve())
    checkpoint = str(Path(args.checkpoint).resolve())
    if args.model in ('Distill-Any-Depth', 'SynRS3D'):
        cmd = [sys.executable, str(runner), '--img_path', image_dir,
               '--output_dir', output_dir, '--checkpoint', checkpoint,
               '--processing_res', str(args.processing_res), '--seed', str(args.seed),
               '--grayscale', '--save_tif']
        if args.model == 'Distill-Any-Depth':
            cmd += ['--arch_name', 'depthanything-large']
        if args.get_conf:
            cmd += ['--get_conf']
    else:
        if args.model == 'DEPTH-ANYTHING-V1' and args.get_conf:
            raise ValueError('The V1 adapter does not implement uncertainty; use COMPUTE_UNCERTAINTY=0 and FUSION_MODE=mean.')
        cmd = [sys.executable, str(runner), '--img-path', image_dir,
               '--outdir', str(Path(output_dir) / 'depth') if args.model.endswith('V1') else output_dir,
               '--checkpoint', checkpoint, '--encoder', 'vitl', '--pred-only', '--grayscale', '--save_tif']
        if args.model.endswith('V2'):
            cmd += ['--input-size', str(args.processing_res)]
            if args.get_conf:
                cmd += ['--get-conf']
    return backend, cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, choices=DEPTH_MODELS)
    p.add_argument('--img_path', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--checkpoint', type=Path, help='Override the checkpoint under the release checkpoints directory')
    p.add_argument('--processing_res', type=int, default=700)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--get_conf', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.checkpoint is None:
        args.checkpoint = checkpoint_path(args.model)
    try:
        backend, cmd = command_for(args)
    except ValueError as error:
        p.error(str(error))
    if args.dry_run:
        import json
        print(json.dumps({'cwd': str(backend), 'argv': cmd}, indent=2))
        return
    if not Path(args.img_path).is_dir():
        p.error('--img_path must be an existing directory of GeoTIFF patches')
    if not Path(args.checkpoint).is_file():
        p.error(f'Checkpoint is missing: {args.checkpoint}. See the README model download table.')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(backend) + os.pathsep + env.get('PYTHONPATH', '')
    env.setdefault('MPLBACKEND', 'Agg')
    subprocess.run(cmd, cwd=backend, env=env, check=True)


if __name__ == '__main__':
    main()
