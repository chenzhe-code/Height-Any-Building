#!/usr/bin/env python3
"""Install external MLS-BRN runtime sources into this checkout, without weights.

Sources must be separate, user-obtained upstream checkouts. This command copies
runtime packages while preserving the release's maintained files in place.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MAINTAINED_FILES = (
    "bstool/bstool/datasets/dump.py",
    "bstool/bstool/datasets/parse.py",
    "bstool/setup.py",
    "mmdet/apis/test.py",
    "mmdet/models/roi_heads/loft_hfm_roi_head.py",
    "tools/bonai/bonai_evaluation.py",
    "tools/bonai/bonai_test.py",
    "wwtool/setup.py",
)


def revision(path):
    result = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                            text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else 'unknown (not a git checkout)'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mlsbrn-source', type=Path, required=True)
    p.add_argument('--bstool-source', type=Path, required=True)
    p.add_argument('--wwtool-source', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    sources = {'MLS-BRN': args.mlsbrn_source.resolve(),
               'bstool': args.bstool_source.resolve(), 'wwtool': args.wwtool_source.resolve()}
    target = ROOT / 'MLS-BRN'
    protected = {target / path for path in MAINTAINED_FILES}
    missing = sorted(path.relative_to(ROOT).as_posix() for path in protected if not path.is_file())
    if missing:
        p.error(f'Missing maintained release files: {", ".join(missing)}')
    jobs = []
    for name, source in sources.items():
        package = 'mmdet' if name == 'MLS-BRN' else name
        if not (source / package / '__init__.py').is_file():
            p.error(f'{name} source must contain {package}/__init__.py: {source}')
        destination = target if name == 'MLS-BRN' else target / name
        if source == destination.resolve() or ROOT == source or ROOT in source.parents:
            p.error('Use independent upstream checkouts outside the release directory')
        for relative in [package, 'setup.py', 'setup.cfg', 'requirements.txt', 'requirements',
                         'MANIFEST.in', 'README.md', 'LICENSE', 'LICENSE.txt', 'NOTICE']:
            file = source / relative
            if file.exists():
                jobs.append((file, destination / relative))
    plan = {key: {'path': str(value), 'revision': revision(value)} for key, value in sources.items()}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '*.so', '*.egg-info', '.git', 'build')

    def copy_upstream(source, destination):
        if Path(destination) not in protected:
            shutil.copy2(source, destination)

    for source, destination in jobs:
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True, ignore=ignore, copy_function=copy_upstream)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy_upstream(source, destination)
    (target / 'backend_sources.local.json').write_text(json.dumps(plan, indent=2) + '\n')
    print('External runtime prepared. Install MLS-BRN, bstool and wwtool in the mlsbrn environment; see environments/README.md.')


if __name__ == '__main__':
    main()
