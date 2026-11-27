#!/usr/bin/env python3
"""Extract PLiDAR photon heights from ATL03 files in parallel and merge them."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds


THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}



def build_union_bounds(bound_dir: Path, output: Path) -> Path:
    """Create a tiny WGS84 GeoTIFF whose bounds cover every input tile."""
    sources = sorted(
        path
        for path in bound_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    if not sources:
        raise ValueError(f"no GeoTIFF files found in boundary directory: {bound_dir}")

    bounds = []
    for source in sources:
        with rasterio.open(source) as dataset:
            if dataset.crs is None:
                raise ValueError(f"boundary GeoTIFF has no CRS: {source}")
            bounds.append(
                transform_bounds(dataset.crs, "EPSG:4326", *dataset.bounds, densify_pts=21)
            )
    west = min(item[0] for item in bounds)
    south = min(item[1] for item in bounds)
    east = max(item[2] for item in bounds)
    north = max(item[3] for item in bounds)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with rasterio.open(
        temporary,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_bounds(west, south, east, north, 2, 2),
        compress="DEFLATE",
    ) as target:
        target.write(np.zeros((1, 2, 2), dtype=np.uint8))
    os.replace(temporary, output)
    print(
        f"Combined {len(sources)} tile bounds: "
        f"{west:.8f},{south:.8f},{east:.8f},{north:.8f}",
        flush=True,
    )
    return output


def run_one(
    index: int,
    source: Path,
    bound_file: Path,
    parts_dir: Path,
    extractor: Path,
    seg_len: int,
) -> tuple[int, Path, str]:
    part = parts_dir / f"{index:04d}_{source.stem}.csv.gz"
    legacy_part = parts_dir / f"{index:04d}_{source.stem}.xlsx"
    log = parts_dir / f"{index:04d}_{source.stem}.log"
    if part.is_file() and part.stat().st_size > 0:
        return index, part, "cached"
    if legacy_part.is_file() and legacy_part.stat().st_size > 0:
        return index, legacy_part, "cached-legacy"

    input_dir = parts_dir / "inputs" / f"{index:04d}"
    input_dir.mkdir(parents=True, exist_ok=True)
    link = input_dir / source.name
    if not link.exists():
        link.symlink_to(source.resolve())

    temporary = part.with_name(part.name.removesuffix(".csv.gz") + ".tmp.csv.gz")
    command = [
        sys.executable,
        str(extractor),
        "--bound-file",
        str(bound_file),
        "--plidar-folder",
        str(input_dir),
        "--save-path",
        str(temporary),
        "--seg-len",
        str(seg_len),
    ]
    environment = os.environ.copy()
    environment.update(THREAD_ENV)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=extractor.parents[1],
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0 or not temporary.is_file():
        raise RuntimeError(f"{source.name} failed (exit {result.returncode}); see {log}")
    os.replace(temporary, part)
    return index, part, "done"


def merge_parts(parts: list[Path], output: Path) -> int:
    def read_part(path: Path) -> pd.DataFrame:
        if path.name.endswith((".csv", ".csv.gz")):
            return pd.read_csv(path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        raise ValueError(f"unsupported photon part format: {path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.name.endswith(".csv.gz"):
        temporary = output.with_name(output.name.removesuffix(".csv.gz") + ".tmp.csv.gz")
        row_count = 0
        wrote_header = False
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as stream:
            for part in parts:
                frame = read_part(part)
                frame.to_csv(stream, index=False, header=not wrote_header)
                wrote_header = True
                row_count += len(frame)
    elif output.suffix.lower() == ".csv":
        temporary = output.with_name(output.stem + ".tmp.csv")
        row_count = 0
        wrote_header = False
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            for part in parts:
                frame = read_part(part)
                frame.to_csv(stream, index=False, header=not wrote_header)
                wrote_header = True
                row_count += len(frame)
    elif output.suffix.lower() == ".xlsx":
        frames = [read_part(part) for part in parts]
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if len(frame) > 1_048_575:
            raise ValueError(
                f"{len(frame)} data rows exceed the Excel worksheet limit; "
                "use an output ending in .csv.gz"
            )
        temporary = output.with_name(output.stem + ".tmp.xlsx")
        frame.to_excel(temporary, index=False, engine="openpyxl")
        row_count = len(frame)
    else:
        raise ValueError("output must end in .xlsx, .csv, or .csv.gz")
    os.replace(temporary, output)
    return row_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    boundary = parser.add_mutually_exclusive_group(required=True)
    boundary.add_argument("--bound-file")
    boundary.add_argument(
        "--bound-dir",
        help="Directory of GeoTIFF tiles; their combined WGS84 extent is used once.",
    )
    parser.add_argument("--atl03-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seg-len", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be positive")
    project_dir = Path(__file__).resolve().parents[1]
    extractor = project_dir / "PLiDAR_processing" / "PLiDAR_HeightExtract.py"
    atl03_dir = Path(args.atl03_dir).resolve()
    output = Path(args.output).resolve()
    if not atl03_dir.is_dir():
        parser.error("ATL03 directory does not exist")
    sources = sorted(path for path in atl03_dir.glob("*.h5") if path.stat().st_size > 0)
    if not sources:
        parser.error("no non-empty .h5 files found")
    if output.exists() and not args.overwrite:
        print(f"Output already exists: {output}; use --overwrite to rebuild")
        return

    if output.name.endswith(".csv.gz"):
        parts_dir = output.with_name(output.name.removesuffix(".csv.gz") + ".parts")
    else:
        parts_dir = output.with_suffix(".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    if args.bound_dir:
        bound_dir = Path(args.bound_dir).resolve()
        if not bound_dir.is_dir():
            parser.error("boundary directory does not exist")
        bound_file = build_union_bounds(bound_dir, parts_dir / "city_bounds.tif")
    else:
        bound_file = Path(args.bound_file).resolve()
        if not bound_file.is_file():
            parser.error("boundary file does not exist")
    print(f"Extracting {len(sources)} ATL03 files with {args.workers} workers")
    results: dict[int, Path] = {}
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_one,
                index,
                source,
                bound_file,
                parts_dir,
                extractor,
                args.seg_len,
            ): (index, source)
            for index, source in enumerate(sources)
        }
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            index, source = futures[future]
            completed += 1
            try:
                _, part, state = future.result()
                results[index] = part
                print(f"[{completed}/{len(sources)}] {state}: {source.name}", flush=True)
            except Exception as error:
                failures.append(str(error))
                print(f"[{completed}/{len(sources)}] FAIL: {error}", file=sys.stderr, flush=True)

    status_path = parts_dir / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "input_count": len(sources),
                "completed_count": len(results),
                "failures": failures,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if failures:
        raise SystemExit(f"{len(failures)} ATL03 files failed; rerun to resume")

    ordered_parts = [results[index] for index in range(len(sources))]
    row_count = merge_parts(ordered_parts, output)
    print(f"Merged {row_count} photon rows into {output}")


if __name__ == "__main__":
    main()
