#!/usr/bin/env python3
"""City-wide SynRS3D patch calibration without tile-level scale parameters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

os.environ.pop("PROJ_DATA", None)

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import lsqr
from shapely.geometry import box
from shapely.strtree import STRtree

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.merge_city_height_results import UINT16_NODATA, merge_height_raster, validate_height_tiles
from utils.window_grid import window_positions

PATCH_SIZE = 1024
HALF = PATCH_SIZE // 2
VERSION = "london-patch-global-v1"
NAME_PATTERN = re.compile(r"window_\d+_x(\d+)_y(\d+)_depth\.tif$")
SCALE_MIN = 0.4
SCALE_MAX = 2.5
SCALE_PRIOR_WEIGHT = 5.0
HAB_SCALE_PRIOR_WEIGHT = 20.0
HAB_SCALE_PRIOR_WEIGHTS = tuple(HAB_SCALE_PRIOR_WEIGHT * 2 ** index for index in range(7))
HAB_PRIOR_SAMPLE_STEP = 16
HAB_PRIOR_MIN_SAMPLES = 32
CONTROL_ROW_WEIGHT = 1.5
CONTROL_ROBUST_THRESHOLD_M = 4.0
CONTROL_ROBUST_POWER = 2.0
SOLVER_INPUT_VERSION = b"london-graph-inputs-v2-finite-control-predictions"


@dataclass(frozen=True)
class Tile:
    name: str
    image: str
    left: float
    top: float
    width: int
    height_px: int
    crs: str
    control: str
    hab: str
    status: str

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top - self.height_px


@dataclass(frozen=True)
class Patch:
    name: str
    cache: str
    left: float
    top: float
    size: int
    source: str
    tile: str

    @property
    def right(self):
        return self.left + self.size

    @property
    def bottom(self):
        return self.top - self.size


def same(a, b):
    return abs(a - b) < 1e-5


def expected_positions(width, height):
    return set(window_positions(width, height, PATCH_SIZE, 0.5))


def archive_tile(args):
    image = Path(args.input_tif)
    depth_dir = Path(args.depth_dir)
    target = Path(args.cache_root) / "interior" / args.tile_name
    with rasterio.open(image) as src:
        if src.res != (1.0, 1.0) or src.crs is None:
            raise ValueError(f"invalid tile grid: {image}")
        positions = expected_positions(src.width, src.height)
        transform = src.transform
        crs = src.crs
    files = sorted((depth_dir / "depth").glob("*_depth.tif"))
    if len(files) != len(positions):
        raise ValueError(f"{image.name}: expected {len(positions)} raw patches, found {len(files)}")
    target.mkdir(parents=True, exist_ok=True)
    found = set()
    records = []
    for path in files:
        match = NAME_PATTERN.fullmatch(path.name)
        if not match:
            raise ValueError(f"unexpected raw patch name: {path}")
        x, y = map(int, match.groups())
        if (x, y) not in positions or (x, y) in found:
            raise ValueError(f"unexpected/duplicate raw patch position: {path}")
        found.add((x, y))
        with rasterio.open(path) as src:
            if (src.count, src.width, src.height) != (1, PATCH_SIZE, PATCH_SIZE):
                raise ValueError(f"invalid patch shape: {path}")
            if src.crs != crs or src.transform != transform * rasterio.Affine.translation(x, y):
                raise ValueError(f"raw patch georeferencing mismatch: {path}")
            prediction = src.read(1).astype(np.float32)
            if not np.isfinite(prediction).all():
                raise ValueError(f"nonfinite raw prediction: {path}")
        cache = target / f"x{x}_y{y}.npz"
        cached_ok = False
        if cache.is_file() and path.stat().st_mtime_ns <= cache.stat().st_mtime_ns:
            try:
                with np.load(cache) as saved:
                    old = saved["prediction"]
                cached_ok = old.shape == prediction.shape and np.isfinite(old).all()
            except (OSError, ValueError, KeyError):
                cached_ok = False
        if not cached_ok:
            temporary = cache.with_suffix(".tmp")
            with temporary.open("wb") as output:
                np.savez_compressed(output, prediction=prediction)
            os.replace(temporary, cache)
        records.append({"x": x, "y": y, "cache": str(cache)})
    if found != positions:
        raise ValueError(f"incomplete raw patch coverage: {image}")
    manifest = {
        "version": VERSION,
        "input": str(image.resolve()),
        "tile": args.tile_name,
        "width": int((max(x for x, _ in positions) + PATCH_SIZE)),
        "height": int((max(y for _, y in positions) + PATCH_SIZE)),
        "patches": records,
    }
    temporary = target / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target / "manifest.json")
    print(f"Archived {len(records)} raw SynRS3D patches: {args.tile_name}")


def discover(input_dir, results_root, retained, allow_disconnected=False):
    paths = sorted([*input_dir.glob("*.tif"), *input_dir.glob("*.tiff")])
    if not paths:
        raise ValueError(f"no RGB tiles: {input_dir}")
    tiles = {}
    crs = None
    origin = None
    for path in paths:
        with rasterio.open(path) as src:
            if src.count != 3 or src.dtypes != ("uint8",) * 3:
                raise ValueError(f"expected 3-band UInt8 RGB: {path}")
            if (src.crs is None or not src.crs.is_projected
                    or src.crs.linear_units_factor[1] != 1.0
                    or (src.transform.a, src.transform.e) != (1.0, -1.0)
                    or src.transform.b or src.transform.d):
                raise ValueError(f"expected north-up 1m projected tile: {path}")
            if crs is None:
                crs = src.crs
                origin = src.bounds.left, src.bounds.top
            if src.crs != crs:
                raise ValueError(f"CRS mismatch: {path}")
            x, y = src.bounds.left - origin[0], src.bounds.top - origin[1]
            if not same(x, round(x)) or not same(y, round(y)):
                raise ValueError(f"tile is not aligned to the common 1m grid: {path}")
            tile = Tile(
                path.stem, str(path.resolve()), src.bounds.left, src.bounds.top,
                src.width, src.height, src.crs.to_string(),
                str(results_root / path.stem / "sparse_control_height.tif"),
                str(results_root / path.stem / f"{path.stem}_absolute_height.tif"),
                str(results_root / path.stem / "status.json"),
            )
            tiles[tile.name] = tile
    edges = []
    values = list(tiles.values())
    for i, a in enumerate(values):
        for b in values[i+1:]:
            overlap_x = min(a.right, b.right) - max(a.left, b.left)
            overlap_y = min(a.top, b.top) - max(a.bottom, b.bottom)
            if overlap_x > 1e-5 and overlap_y > 1e-5:
                raise ValueError(f"overlapping input tiles: {a.name}, {b.name}")
            if same(a.right, b.left) and same(a.top, b.top) and same(a.bottom, b.bottom):
                edges.append((a.name, b.name, "vertical"))
            elif same(b.right, a.left) and same(a.top, b.top) and same(a.bottom, b.bottom):
                edges.append((b.name, a.name, "vertical"))
            elif same(a.bottom, b.top) and same(a.left, b.left) and same(a.right, b.right):
                edges.append((a.name, b.name, "horizontal"))
            elif same(b.bottom, a.top) and same(a.left, b.left) and same(a.right, b.right):
                edges.append((b.name, a.name, "horizontal"))
    if len(tile_components(tiles, edges)) != 1 and not allow_disconnected:
        raise ValueError("input tile adjacency graph is disconnected")
    if retained:
        for tile in values:
            status = Path(tile.status)
            if not status.is_file() or json.loads(status.read_text()).get("status") != "success":
                raise ValueError(f"tile is not complete: {tile.name}")
            for item in (tile.hab, tile.control):
                if not Path(item).is_file():
                    raise ValueError(f"missing retained tile input: {item}")
    return tiles, sorted(edges)


def tile_components(tiles, edges):
    """Partition full-edge neighbours, keeping corner contacts and islands separate."""
    neighbours = {name: set() for name in tiles}
    for a, b, _ in edges:
        neighbours[a].add(b)
        neighbours[b].add(a)
    seen = set()
    groups = []
    for name in sorted(tiles):
        if name in seen:
            continue
        pending = [name]
        seen.add(name)
        group = []
        while pending:
            current = pending.pop()
            group.append(current)
            for other in sorted(neighbours[current] - seen):
                seen.add(other)
                pending.append(other)
        groups.append({key: tiles[key] for key in sorted(group)})
    return groups


def seam_starts(length):
    starts = list(range(0, length - PATCH_SIZE + 1, HALF))
    if not starts or starts[-1] != length - PATCH_SIZE:
        starts.append(length - PATCH_SIZE)
    return starts


def seam_records(tiles, edges, *, skip_short=False):
    result = []
    for a, b, axis in edges:
        length = tiles[a].height_px if axis == "vertical" else tiles[a].width
        depth = (min(tiles[a].width, tiles[b].width) if axis == "vertical"
                 else min(tiles[a].height_px, tiles[b].height_px))
        if skip_short and (length < PATCH_SIZE or depth < HALF):
            continue
        if length < PATCH_SIZE:
            raise ValueError("shared edge is too short for a crossing patch")
        for start in seam_starts(length):
            name = f"{a}__{b}__{axis}__{start:04d}"
            tile = tiles[a]
            left = tile.right - HALF if axis == "vertical" else tile.left + start
            top = tile.top - start if axis == "vertical" else tile.bottom + HALF
            result.append(({"name": name, "a": a, "b": b, "axis": axis, "start": start}, left, top))
    return result


def load_interior(tiles, cache_root):
    patches = []
    for tile in tiles.values():
        manifest = cache_root / "interior" / tile.name / "manifest.json"
        if not manifest.is_file():
            raise ValueError(f"missing raw SynRS3D patch archive: {manifest}")
        data = json.loads(manifest.read_text())
        if data.get("version") != VERSION or Path(data.get("input", "")) != Path(tile.image):
            raise ValueError(f"stale raw patch archive: {manifest}")
        positions = expected_positions(tile.width, tile.height_px)
        recorded = {(p["x"], p["y"]) for p in data["patches"]}
        if recorded != positions or len(data["patches"]) != len(positions):
            raise ValueError(f"incomplete patch archive: {manifest}")
        for p in data["patches"]:
            cache = Path(p["cache"])
            if not cache.is_file():
                raise ValueError(f"missing cached raw patch: {cache}")
            patches.append(Patch(
                f"{tile.name}:x{p['x']}_y{p['y']}", str(cache),
                tile.left + p["x"], tile.top - p["y"], PATCH_SIZE, "interior", tile.name,
            ))
    return patches


def load_calibrated(tiles, results_root, work_directory, window_size, overlap):
    patches = []
    for tile in tiles.values():
        directory = results_root / tile.name / work_directory / "CalibratedWindows"
        inventory = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (inventory.get("version"), inventory.get("units")) != (1, "m"):
            raise ValueError(f"Invalid calibrated height archive: {directory}")
        expected = set(window_positions(tile.width, tile.height_px, window_size, overlap))
        records = inventory["patches"]
        if len(records) != len(expected) or {(r["x"], r["y"]) for r in records} != expected:
            raise ValueError(f"Calibrated window coverage does not match the tile: {tile.name}")
        for record in records:
            x, y = record["x"], record["y"]
            name = f"x{x}_y{y}.npz"
            left, top = tile.left + x, tile.top - y
            transform = list(from_origin(left, top, 1, 1))[:6]
            if (record["file"] != name or record["shape"] != [window_size, window_size]
                    or record["crs"] != tile.crs
                    or not np.allclose(record["transform"], transform, rtol=0, atol=1e-6)):
                raise ValueError(f"Calibrated window grid mismatch: {directory / name}")
            cache = directory / name
            if not cache.is_file():
                raise ValueError(f"Missing calibrated window: {cache}")
            patches.append(Patch(f"{tile.name}:x{x}_y{y}", str(cache), left, top,
                                 window_size, "interior", tile.name))
    return patches


def infer_seams(tiles, seams, cache_root, checkpoint, gpus, conda_bin):
    import hashlib
    cache_dir = cache_root / "seams"
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_stat = checkpoint.stat()
    checkpoint_signature = (str(checkpoint.resolve()), checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns)
    records = []
    pending = []
    for seam, left, top in seams:
        signature = [VERSION, seam, checkpoint_signature]
        for name in (seam["a"], seam["b"]):
            path = Path(tiles[name].image)
            stat = path.stat()
            signature.append((str(path), stat.st_size, stat.st_mtime_ns))
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
        cache = cache_dir / f"{seam['name']}__{key}.npz"
        records.append(Patch(seam["name"], str(cache), left, top, PATCH_SIZE, "seam", ""))
        if cache.is_file():
            try:
                with np.load(cache) as source:
                    values = source["prediction"]
                if values.shape != (PATCH_SIZE, PATCH_SIZE) or not np.isfinite(values).all():
                    cache.unlink()
            except (OSError, ValueError, KeyError):
                cache.unlink()
        if not cache.is_file():
            pending.append((seam, cache))
    print(f"Crossing patches: {len(records)}, cached={len(records)-len(pending)}, pending={len(pending)}", flush=True)
    if pending:
        batches = [[] for _ in gpus]
        for i, item in enumerate(pending):
            batches[i % len(gpus)].append(item)
        processes = []
        for gpu, batch in zip(gpus, batches):
            if not batch:
                continue
            manifest = cache_dir / f"worker_gpu_{gpu}.json"
            manifest.write_text(json.dumps({
                "patch_size": PATCH_SIZE,
                "tiles": {name: asdict(tile) for name, tile in tiles.items()},
                "seams": [{"seam": seam, "cache": str(path)} for seam, path in batch],
            }), encoding="utf-8")
            # The existing worker expects width and height_px and reads original RGB directly.
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["NO_ALBUMENTATIONS_UPDATE"] = "1"
            env.pop("PROJ_DATA", None)
            command = [str(conda_bin), "run", "--no-capture-output", "-n", "synrs3d",
                       "python", str(Path(__file__).with_name("infer_global_seam_windows.py")),
                       "--task-file", str(manifest), "--checkpoint", str(checkpoint)]
            processes.append(subprocess.Popen(command, env=env))
        codes = [p.wait() for p in processes]
        if any(code != 0 for code in codes):
            raise RuntimeError(f"crossing SynRS3D worker failed: {codes}")
    for patch in records:
        if not Path(patch.cache).is_file():
            raise ValueError(f"missing crossing prediction: {patch.cache}")
    return records


@lru_cache(maxsize=24)
def prediction(path, size=PATCH_SIZE):
    with np.load(path) as source:
        data = source["prediction"].astype(np.float32)
    if data.shape != (size, size) or not np.isfinite(data).all():
        raise ValueError(f"invalid cached prediction: {path}")
    return data


def intersect(a, b):
    left, right = max(a.left, b.left), min(a.right, b.right)
    top, bottom = min(a.top, b.top), max(a.bottom, b.bottom)
    return (left, right, top, bottom) if right > left and top > bottom else None


def sample_overlap(a, b, bounds, count=12):
    left, right, top, bottom = bounds
    xx = np.linspace(left + 0.5, right - 0.5, min(count, round(right - left)))
    yy = np.linspace(top - 0.5, bottom + 0.5, min(count, round(top - bottom)))
    x, y = np.meshgrid(xx, yy)
    ai = np.clip(np.floor(x - a.left).astype(int), 0, a.size - 1)
    aj = np.clip(np.floor(a.top - y).astype(int), 0, a.size - 1)
    bi = np.clip(np.floor(x - b.left).astype(int), 0, b.size - 1)
    bj = np.clip(np.floor(b.top - y).astype(int), 0, b.size - 1)
    u = prediction(a.cache, a.size)[aj, ai].ravel()
    v = prediction(b.cache, b.size)[bj, bi].ravel()
    valid = (u >= 3) & (u <= 300) & (v >= 3) & (v <= 300)
    return u[valid], v[valid]


def overlap_equations(patches, *, calibrated=False):
    geometries = [box(p.left, p.bottom, p.right, p.top) for p in patches]
    tree = STRtree(geometries)
    edges = []
    coeffs = []
    baseline = []
    for i, patch in enumerate(patches):
        for j in tree.query(geometries[i], predicate="intersects"):
            j = int(j)
            if j <= i:
                continue
            other = patches[j]
            bounds = intersect(patch, other)
            if bounds is None:
                continue
            left, right, top, bottom = bounds
            minimum_side = min(128, max(1, min(patch.size, other.size) // 8)) if calibrated else 128
            if (right - left) * (top - bottom) < minimum_side ** 2:
                continue
            u, v = sample_overlap(patch, other, bounds)
            if len(u) < 12:
                # A weak scale-continuity edge keeps empty/water overlaps
                # connected while the model prior anchors their absolute scale.
                ai, bj, disagreement = 2.0, 2.0, 0.0
            else:
                # The median paired residual resists roof/non-roof outliers.
                ai = float(np.median(u))
                bj = float(np.median(v))
                disagreement = float(np.median(np.abs(u - v)))
            edges.append((i, j))
            coeffs.append((ai, bj))
            baseline.append(disagreement)
        if i and i % 1000 == 0:
            print(f"Overlap graph: {i}/{len(patches)} patches", flush=True)
    if not edges and not calibrated:
        raise ValueError("no valid patch overlap constraints")
    return np.asarray(edges, int).reshape(-1, 2), np.asarray(coeffs, float).reshape(-1, 2), np.asarray(baseline, float)


def component_count(n, edges):
    parent = list(range(n))
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        ra, rb = root(int(a)), root(int(b))
        parent[ra] = rb
    return len({root(i) for i in range(n)})


def control_equations(tiles, patches, require_controls=True):
    per_tile = defaultdict(list)
    for i, patch in enumerate(patches):
        for tile in tiles.values():
            if intersect(patch, tile):
                per_tile[tile.name].append(i)
    rows = []
    holdout = []
    hab_errors = []
    for tile in tiles.values():
        with rasterio.open(tile.control) as source:
            with rasterio.open(tile.image) as image:
                if source.crs != image.crs or source.transform != image.transform or source.count != 1:
                    raise ValueError(f"control grid mismatch: {tile.name}")
            values = source.read(1, masked=True)
            arr = np.asarray(values.data)
            valid = ~np.ma.getmaskarray(values) & np.isfinite(arr) & (arr >= 5) & (arr <= 500)
            rr, cc = np.nonzero(valid)
            heights = arr[rr, cc].astype(np.float32)
        if len(rr) > 20000:
            selected = np.linspace(0, len(rr) - 1, 20000).astype(int)
            rr, cc, heights = rr[selected], cc[selected], heights[selected]
        hab = None
        if Path(tile.hab).is_file() and len(rr):
            with rasterio.open(tile.hab) as source:
                hab = source.read(1)[rr, cc].astype(np.float32)
        for patch_id in per_tile[tile.name]:
            patch = patches[patch_id]
            x0 = max(0, int(round(patch.left - tile.left)))
            x1 = min(tile.width, int(round(patch.right - tile.left)))
            y0 = max(0, int(round(tile.top - patch.top)))
            y1 = min(tile.height_px, int(round(tile.top - patch.bottom)))
            inside = np.flatnonzero((cc >= x0) & (cc < x1) & (rr >= y0) & (rr < y1))
            if len(inside) > 96:
                inside = inside[np.linspace(0, len(inside) - 1, 96).astype(int)]
            if not len(inside):
                continue
            pc = cc[inside] - int(round(patch.left - tile.left))
            pr = rr[inside] - int(round(tile.top - patch.top))
            pred = prediction(patch.cache, patch.size)[pr, pc]
            # A finite model prediction may support a valid control at any height.
            mask = np.isfinite(pred)
            for local, index in enumerate(inside):
                if not mask[local]:
                    continue
                item = (patch_id, float(pred[local]), float(heights[index]))
                if (((int(rr[index]) * 73856093) ^ (int(cc[index]) * 19349663)) % 5) == 0:
                    holdout.append(item)
                    if hab is not None and 0 <= hab[index] < 65535:
                        hab_errors.append(abs(float(hab[index]) - float(heights[index])))
                else:
                    rows.append(item)
    if require_controls and (not rows or not holdout):
        raise ValueError("insufficient training/held-out ATL03 control points")
    return rows, holdout, hab_errors



def _weighted_log_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (not len(values) or values.shape != weights.shape
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or np.any(values <= 0) or np.any(weights <= 0)):
        raise ValueError("invalid values for weighted scale aggregation")
    return float(np.exp(np.average(np.log(values), weights=weights)))


def hab_patch_priors(tiles, patches, sample_step=HAB_PRIOR_SAMPLE_STEP,
                     min_samples=HAB_PRIOR_MIN_SAMPLES):
    """Estimate one multiplicative raw-patch scale from retained tile HAB rasters."""
    if sample_step < 1 or min_samples < 1:
        raise ValueError("HAB prior sampling parameters must be positive")
    estimates = [[] for _ in patches]
    sample_counts = np.zeros(len(patches), dtype=np.int64)
    offset = sample_step // 2
    for tile_number, tile in enumerate(tiles.values(), 1):
        relevant = [(i, patch) for i, patch in enumerate(patches) if intersect(tile, patch)]
        with rasterio.open(tile.hab) as source:
            expected_transform = from_origin(tile.left, tile.top, 1, 1)
            if (source.shape != (tile.height_px, tile.width)
                    or source.count not in (1, 3)
                    or source.crs is None or source.crs.to_string() != tile.crs
                    or source.transform != expected_transform):
                raise ValueError(f"HAB grid mismatch while deriving patch priors: {tile.name}")
            hab = source.read(1).astype(np.float32)
            nodata = source.nodata
        for patch_id, patch in relevant:
            bounds = intersect(tile, patch)
            left, right, top, bottom = bounds
            tc = int(round(left - tile.left))
            tr = int(round(tile.top - top))
            pc = int(round(left - patch.left))
            pr = int(round(patch.top - top))
            width = int(round(right - left))
            height = int(round(top - bottom))
            if width <= offset or height <= offset:
                continue
            tile_rows = slice(tr + offset, tr + height, sample_step)
            tile_cols = slice(tc + offset, tc + width, sample_step)
            patch_rows = slice(pr + offset, pr + height, sample_step)
            patch_cols = slice(pc + offset, pc + width, sample_step)
            hab_values = hab[tile_rows, tile_cols]
            raw_values = prediction(patch.cache)[patch_rows, patch_cols]
            if hab_values.shape != raw_values.shape:
                raise ValueError(f"HAB/raw sample shape mismatch: {patch.name}, {tile.name}")
            valid = (np.isfinite(hab_values) & np.isfinite(raw_values)
                     & (hab_values >= 3) & (hab_values <= 300)
                     & (raw_values >= 3) & (raw_values <= 300))
            if nodata is not None:
                valid &= hab_values != nodata
            ratios = hab_values[valid] / raw_values[valid]
            ratios = ratios[(ratios >= SCALE_MIN / 2) & (ratios <= SCALE_MAX * 2)]
            if len(ratios) < min_samples:
                continue
            estimates[patch_id].append((float(np.median(ratios)), int(len(ratios))))
            sample_counts[patch_id] += len(ratios)
        del hab
        print(f"HAB patch priors: sampled {tile_number}/{len(tiles)} tiles", flush=True)

    priors = np.full(len(patches), np.nan, dtype=np.float64)
    direct = np.zeros(len(patches), dtype=bool)
    for patch_id, records in enumerate(estimates):
        if records:
            priors[patch_id] = _weighted_log_mean(
                [value for value, _ in records], [count for _, count in records]
            )
            direct[patch_id] = True
    if not np.any(direct):
        raise ValueError("no valid raw-patch/HAB samples for HAB-derived priors")

    global_default = float(np.median(priors[direct]))
    tile_defaults = {}
    for name in tiles:
        values = [priors[i] for i, patch in enumerate(patches)
                  if direct[i] and patch.tile == name]
        tile_defaults[name] = float(np.median(values)) if values else global_default
    for patch_id, patch in enumerate(patches):
        if direct[patch_id]:
            continue
        names = ([patch.tile] if patch.tile else
                 [name for name, tile in tiles.items() if intersect(tile, patch)])
        defaults = [tile_defaults[name] for name in names if name in tile_defaults]
        priors[patch_id] = float(np.median(defaults)) if defaults else global_default

    priors = np.clip(priors, SCALE_MIN, SCALE_MAX)
    prediction.cache_clear()
    summary = {
        "hab_prior_direct_patches": int(np.count_nonzero(direct)),
        "hab_prior_fallback_patches": int(np.count_nonzero(~direct)),
        "hab_prior_sample_count": int(np.sum(sample_counts)),
        "hab_prior_scale_min": float(np.min(priors)),
        "hab_prior_scale_median": float(np.median(priors)),
        "hab_prior_scale_max": float(np.max(priors)),
    }
    return priors, summary

def solve_scales(n, edges, coeffs, controls, overlap_weight=5.0, initial_scales=None,
                 prior_scales=None, prior_weight=SCALE_PRIOR_WEIGHT):
    # One unknown per patch. No tile-scale or tile-offset variable is introduced.
    if prior_scales is None:
        prior_scales = np.ones(n, dtype=np.float64)
    else:
        prior_scales = np.asarray(prior_scales, dtype=np.float64)
        if prior_scales.shape != (n,) or not np.isfinite(prior_scales).all():
            raise ValueError("prior scales must be one finite value per patch")
        prior_scales = np.clip(prior_scales, SCALE_MIN, SCALE_MAX)
    prior_weight = np.broadcast_to(prior_weight, (n,))
    if not np.isfinite(prior_weight).all() or np.any(prior_weight <= 0):
        raise ValueError("prior weight must be positive and finite")
    m = len(edges) + len(controls) + n
    row = np.empty(2 * len(edges) + len(controls) + n, dtype=np.int32)
    col = np.empty_like(row)
    val = np.empty(len(row), dtype=np.float64)
    target = np.zeros(m, dtype=np.float64)
    k = 0
    for r, ((a, b), (pa, pb)) in enumerate(zip(edges, coeffs)):
        row[k:k+2] = r
        col[k:k+2] = (a, b)
        val[k:k+2] = (pa * overlap_weight, -pb * overlap_weight)
        k += 2
    start = len(edges)
    for t, (i, pred, truth) in enumerate(controls):
        row[k] = start + t
        col[k] = i
        val[k] = pred * CONTROL_ROW_WEIGHT
        target[start+t] = truth * CONTROL_ROW_WEIGHT
        k += 1
    start += len(controls)
    row[k:] = np.arange(start, m)
    col[k:] = np.arange(n)
    val[k:] = prior_weight
    target[start:] = prior_weight * prior_scales
    matrix = coo_matrix((val, (row, col)), shape=(m, n)).tocsr()
    weights = np.ones(m, dtype=np.float64)
    scale = prior_scales.copy()
    if initial_scales is None:
        initial_scales = prior_scales
    else:
        initial_scales = np.asarray(initial_scales, dtype=np.float64)
        if initial_scales.shape != (n,) or not np.isfinite(initial_scales).all():
            raise ValueError("initial scales must be one finite value per patch")
        initial_scales = np.clip(initial_scales, SCALE_MIN, SCALE_MAX)
    for iteration in range(5):
        fitted = lsqr(diags(np.sqrt(weights)) @ matrix, np.sqrt(weights) * target,
                      atol=1e-5, btol=1e-5, iter_lim=600,
                      x0=initial_scales if iteration == 0 else None)[0]
        scale = np.clip(fitted, SCALE_MIN, SCALE_MAX)
        residual = np.abs(matrix @ scale - target)
        robust = residual[:start]
        weights[:start] = np.minimum(1.0, 6.0 / np.maximum(robust, 1e-6))
        # Faster decay only for training controls; overlap and prior rows retain
        # their existing rules. Convert the equation residual back to metres.
        control_error_m = residual[len(edges):start] / CONTROL_ROW_WEIGHT
        weights[len(edges):start] = np.minimum(
            1.0, CONTROL_ROBUST_THRESHOLD_M / np.maximum(control_error_m, 1e-6)
        ) ** CONTROL_ROBUST_POWER
    return scale

def evaluate(edges, coeffs, baseline, holdout, scales, hab_errors, seam_mask=None,
             reference_scales=None, quality_reference="synrs3d_metric",
             prior_weight=SCALE_PRIOR_WEIGHT, validate_quality=True):
    if reference_scales is None:
        reference_scales = np.ones(len(scales), dtype=np.float64)
    else:
        reference_scales = np.asarray(reference_scales, dtype=np.float64)
        if reference_scales.shape != np.asarray(scales).shape or not np.isfinite(reference_scales).all():
            raise ValueError("reference scales must match the fitted patch scales")
    before = np.abs(coeffs[:, 0] * reference_scales[edges[:, 0]]
                    - coeffs[:, 1] * reference_scales[edges[:, 1]])
    after = np.abs(coeffs[:, 0] * scales[edges[:, 0]]
                   - coeffs[:, 1] * scales[edges[:, 1]])
    # overlap_equations() emits (2, 2) with a zero baseline when an overlap has
    # too few valid height samples. Those weak edges keep the patch graph
    # connected, but their artificial zero must not be treated as a measured
    # pre-optimization error: any unequal fitted scales would otherwise make a
    # city with >=50% weak edges fail the median quality gate by construction.
    weak = ((baseline == 0)
            & np.isclose(coeffs[:, 0], 2.0)
            & np.isclose(coeffs[:, 1], 2.0))
    observed = np.isfinite(baseline) & ~weak
    if not np.any(observed):
        raise ValueError("no measured patch overlaps are available for quality validation")
    raw_photon = np.array([abs(pred - truth) for _, pred, truth in holdout])
    prior_photon = np.array([abs(pred * reference_scales[i] - truth)
                             for i, pred, truth in holdout])
    fit_photon = np.array([abs(pred * scales[i] - truth) for i, pred, truth in holdout])
    report = {
        "version": VERSION, "patches": len(scales), "overlap_edges": len(edges),
        "quality_reference": quality_reference,
        "observed_overlap_edges": int(np.count_nonzero(observed)),
        "weak_overlap_edges": int(np.count_nonzero(weak)),
        "holdout_control_samples": len(holdout),
        "overlap_median_abs_before_m": float(np.median(before[observed])),
        "overlap_median_abs_after_m": float(np.median(after[observed])),
        "sample_overlap_median_abs_before_m": float(np.median(baseline[observed])),
        "holdout_control_mae_raw_m": float(np.mean(raw_photon)),
        "holdout_control_mae_prior_m": float(np.mean(prior_photon)),
        "holdout_control_mae_global_m": float(np.mean(fit_photon)),
        "holdout_control_mae_tile_hab_m": float(np.mean(hab_errors)) if len(hab_errors) else None,
        "scale_min": float(np.min(scales)), "scale_median": float(np.median(scales)),
        "scale_max": float(np.max(scales)),
        "overlap_constraint_weight": 5.0,
        "control_robust_threshold_m": CONTROL_ROBUST_THRESHOLD_M,
        "control_robust_power": CONTROL_ROBUST_POWER,
        "scale_prior_weight": float(prior_weight),
    }
    if seam_mask is not None and np.any(seam_mask):
        report["seam_overlap_edges"] = int(np.count_nonzero(seam_mask))
        observed_seams = seam_mask & observed
        report["observed_seam_overlap_edges"] = int(np.count_nonzero(observed_seams))
        report["weak_seam_overlap_edges"] = int(np.count_nonzero(seam_mask & weak))
        if not np.any(observed_seams):
            raise ValueError("no measured cross-tile overlaps are available for quality validation")
        report["seam_overlap_median_abs_before_m"] = float(np.median(before[observed_seams]))
        report["seam_overlap_median_abs_after_m"] = float(np.median(after[observed_seams]))
    if validate_quality:
        raise_for_quality_failure(report)
    return report


def quality_failures(report):
    failures = []
    if ("seam_overlap_median_abs_after_m" in report
            and report["seam_overlap_median_abs_after_m"]
            > report["seam_overlap_median_abs_before_m"] + 0.02):
        failures.append("cross_tile_overlap")
    if (report["overlap_median_abs_after_m"]
            > report["overlap_median_abs_before_m"] + 0.02):
        failures.append("global_overlap")
    if (report["holdout_control_mae_global_m"]
            > report["holdout_control_mae_prior_m"] + 0.5):
        failures.append("held_out_photon")
    return failures


def raise_for_quality_failure(report):
    failures = quality_failures(report)
    labels = {
        "cross_tile_overlap": "cross-tile overlap error worsened",
        "global_overlap": "global overlap error worsened",
        "held_out_photon": "held-out photon error worsened",
    }
    if failures:
        raise ValueError(f"{labels[failures[0]]}: {report}")


def solve_hab_prior_scales(n, edges, coeffs, controls, baseline, holdout,
                           hab_errors, seam_mask, prior_scales,
                           prior_weights=HAB_SCALE_PRIOR_WEIGHTS):
    """Use the weakest HAB prior that passes every existing quality guardrail."""
    weights = tuple(float(weight) for weight in prior_weights)
    if (not weights or any(not np.isfinite(weight) or weight <= 0 for weight in weights)
            or any(right <= left for left, right in zip(weights, weights[1:]))):
        raise ValueError("HAB prior weights must be finite, positive, and strictly increasing")
    attempts = []
    for prior_weight in weights:
        scales = solve_scales(
            n, edges, coeffs, controls,
            initial_scales=prior_scales, prior_scales=prior_scales,
            prior_weight=prior_weight,
        )
        report = evaluate(
            edges, coeffs, baseline, holdout, scales, hab_errors,
            seam_mask=seam_mask, reference_scales=prior_scales,
            quality_reference="tile_hab", prior_weight=prior_weight,
            validate_quality=False,
        )
        failures = quality_failures(report)
        attempt = {
            "scale_prior_weight": prior_weight,
            "overlap_median_abs_after_m": report["overlap_median_abs_after_m"],
            "holdout_control_mae_global_m": report["holdout_control_mae_global_m"],
            "quality_failures": failures,
            "passed": not failures,
        }
        if "seam_overlap_median_abs_after_m" in report:
            attempt["seam_overlap_median_abs_after_m"] = report[
                "seam_overlap_median_abs_after_m"
            ]
        attempts.append(attempt)
        print(
            f"HAB prior weight {prior_weight:g}: "
            f"overlap={attempt['overlap_median_abs_after_m']:.3f}m, "
            f"photon={attempt['holdout_control_mae_global_m']:.3f}m, "
            f"quality={'pass' if not failures else ','.join(failures)}",
            flush=True,
        )
        if not failures:
            report["scale_prior_weight_selection"] = "minimum_passing_quality_gate"
            report["scale_prior_weight_candidates"] = list(weights)
            report["scale_prior_weight_attempts"] = attempts
            return scales, report
    raise_for_quality_failure(report)
    raise AssertionError("unreachable HAB prior quality selection state")


def solver_input_cache_path(tiles, patches, cache_root, require_controls=True, *, calibrated=False):
    digest = hashlib.sha256(SOLVER_INPUT_VERSION)
    if not require_controls:
        digest.update(b"allow-sparse-controls")
    if calibrated:
        digest.update(b"calibrated-windows-v1")
    for path in [*(p.cache for p in patches), *(t.control for t in tiles.values()),
                 *(t.hab for t in tiles.values())]:
        item = Path(path)
        stat = item.stat()
        digest.update(str(item).encode())
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return cache_root / f"solver_inputs_{digest.hexdigest()[:20]}.npz"


def solver_inputs(tiles, patches, cache_root, require_controls=True, *, calibrated=False):
    cache = solver_input_cache_path(tiles, patches, cache_root, require_controls, calibrated=calibrated)
    if cache.is_file():
        try:
            with np.load(cache) as saved:
                edges = saved["edges"]
                coefficients = saved["coefficients"]
                baseline = saved["baseline"]
                controls = [(int(i), float(pred), float(truth))
                            for i, pred, truth in saved["controls"]]
                holdout = [(int(i), float(pred), float(truth))
                           for i, pred, truth in saved["holdout"]]
                hab_errors = saved["hab_errors"].tolist()
            if (edges.ndim != 2 or edges.shape[1] != 2
                    or coefficients.shape != edges.shape
                    or baseline.shape != (len(edges),)
                    or (edges.size and np.max(edges) >= len(patches))
                    or not np.isfinite(coefficients).all()):
                raise ValueError("invalid cached solver inputs")
            print(f"Reusing {len(edges)} overlap edges and {len(controls)} training controls: {cache}", flush=True)
            return edges, coefficients, baseline, controls, holdout, hab_errors
        except (OSError, ValueError, KeyError):
            cache.unlink()
    edges, coefficients, baseline = overlap_equations(patches, calibrated=calibrated)
    controls, holdout, hab_errors = control_equations(tiles, patches, require_controls)
    temporary = cache.with_suffix(".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(
            output, edges=edges, coefficients=coefficients, baseline=baseline,
            controls=np.asarray(controls, dtype=np.float64),
            holdout=np.asarray(holdout, dtype=np.float64),
            hab_errors=np.asarray(hab_errors, dtype=np.float64),
        )
    os.replace(temporary, cache)
    return edges, coefficients, baseline, controls, holdout, hab_errors


def patch_weights(size=PATCH_SIZE):
    distance = np.minimum(np.arange(size) + 1, np.arange(size, 0, -1))
    ramp = np.clip(distance / 128.0, 0.02, 1.0).astype(np.float32)
    return ramp[:, None] * ramp[None, :]


def render_tile(tile, patches, scales, output, feathered=True, *, calibrated=False):
    total = np.zeros((tile.height_px, tile.width), np.float32)
    weight = np.zeros_like(total)
    feather = patch_weights() if feathered else None
    for i, patch in enumerate(patches):
        overlap = intersect(tile, patch)
        if overlap is None:
            continue
        left, right, top, bottom = overlap
        tc = int(round(left - tile.left))
        tr = int(round(tile.top - top))
        pc = int(round(left - patch.left))
        pr = int(round(patch.top - top))
        w = int(round(right - left))
        h = int(round(top - bottom))
        prediction_part = prediction(patch.cache, patch.size)[pr:pr+h, pc:pc+w]
        weights = 1.0 if feather is None or (calibrated and patch.source == "interior") else feather[pr:pr+h, pc:pc+w]
        total[tr:tr+h, tc:tc+w] += prediction_part * scales[i] * weights
        weight[tr:tr+h, tc:tc+w] += weights
    if np.any(weight <= 0):
        raise ValueError(f"uncovered tile pixels: {tile.name}")
    np.divide(total, weight, out=total)
    encoded = np.clip(np.rint(total), 0, UINT16_NODATA - 1).astype(np.uint16)
    profile = {
        "driver": "GTiff", "height": tile.height_px, "width": tile.width,
        "count": 1, "dtype": "uint16", "crs": tile.crs,
        "transform": from_origin(tile.left, tile.top, 1, 1),
        "nodata": UINT16_NODATA, "tiled": True, "blockxsize": 512,
        "blockysize": 512, "compress": "zstd", "predictor": 2,
        "BIGTIFF": "YES",
    }
    with rasterio.open(output, "w", **profile) as target:
        target.write(encoded, 1)
    del total, weight, encoded
    prediction.cache_clear()


def merge_rendered(tiles, patches, scales, output, temp_root, feathered=True,
                   retained_tiles=(), *, calibrated=False):
    paths = []
    for number, tile in enumerate(tiles.values(), 1):
        if tile.name in retained_tiles:
            paths.append((tile.name, Path(tile.hab), Path(tile.hab)))
            print(f"Retained HAB {number}/{len(tiles)}: {tile.name}", flush=True)
            continue
        temp_tile = temp_root / f"{tile.name}.tif"
        relevant = [(i, patch) for i, patch in enumerate(patches) if intersect(tile, patch)]
        ids = [i for i, _ in relevant]
        render_tile(tile, [p for _, p in relevant], scales[ids], temp_tile, feathered,
                    calibrated=calibrated)
        paths.append((tile.name, temp_tile, Path(tile.hab)))
        print(f"Rendered {number}/{len(tiles)} tiles for {output.name}", flush=True)
    grid = validate_height_tiles(paths)
    staged = temp_root / "city" / (output.stem + ".staging.tif")
    merge_height_raster(paths, staged, grid)
    with rasterio.open(staged) as source:
        if source.count != 1 or source.dtypes != ("uint16",) or source.nodata != UINT16_NODATA or source.crs != grid["crs"]:
            raise ValueError(f"invalid final city TIFF: {staged}")
        if source.width != grid["width"] or source.height != grid["height"]:
            raise ValueError("city dimensions mismatch")
    return staged


def optimize_components(tiles, edges, interiors, crossing, cache_root, hab_prior):
    """Solve each geographic component independently before a single city mosaic."""
    patches = interiors + crossing
    scales = np.ones(len(patches), dtype=np.float64)
    retained = set()
    reports = []
    groups = tile_components(tiles, edges)
    for number, group in enumerate(groups, 1):
        ids = [i for i, patch in enumerate(patches)
               if any(intersect(tile, patch) for tile in group.values())]
        local = [patches[i] for i in ids]
        info = {
            "component": number, "tile_names": list(group), "tiles": len(group),
            "interior_patches": sum(p.source == "interior" for p in local),
            "crossing_patches": sum(p.source == "seam" for p in local),
        }
        print(f"Optimizing component {number}/{len(groups)}: "
              f"tiles={len(group)}, patches={len(local)}", flush=True)
        # Islands have no cross-tile evidence. Preserve their completed HAB if
        # there are no independent ATL03 training and validation samples.
        if len(group) == 1:
            controls, holdout, _ = control_equations(group, local, require_controls=False)
            if not controls or not holdout:
                retained.update(group)
                info.update({"method": "retained_hab", "graph_components": 1,
                             "training_control_samples": len(controls),
                             "holdout_control_samples": len(holdout),
                             "photon_validation": "unavailable",
                             "fallback_reason": "insufficient_training_or_held_out_controls"})
                reports.append(info)
                print(f"Retaining HAB for {list(group)}: insufficient controls", flush=True)
                continue
        graph, coefficients, baseline, controls, holdout, hab_errors = solver_inputs(
            group, local, cache_root
        )
        components = component_count(len(local), graph)
        if components != 1:
            raise ValueError(f"component {number}: patch graph has {components} disconnected components")
        prior_summary = {}
        seam_nodes = np.asarray([patch.source == "seam" for patch in local])
        seam_mask = seam_nodes[graph[:, 0]] | seam_nodes[graph[:, 1]]
        prior_mode = "tile_hab" if hab_prior else "synrs3d_metric"
        if hab_prior:
            priors, prior_summary = hab_patch_priors(group, local)
            fitted, report = solve_hab_prior_scales(
                len(local), graph, coefficients, controls, baseline,
                holdout, hab_errors, seam_mask, priors,
            )
        else:
            priors = np.ones(len(local), dtype=np.float64)
            fitted = solve_scales(
                len(local), graph, coefficients, controls,
                initial_scales=priors, prior_scales=priors,
            )
            report = evaluate(
                graph, coefficients, baseline, holdout, fitted, hab_errors,
                seam_mask=seam_mask, reference_scales=priors,
                quality_reference=prior_mode,
            )
        scales[ids] = fitted
        report.update({**info, "method": "patch_optimization",
                       "training_control_samples": len(controls),
                       "graph_components": components, "scale_prior": prior_mode,
                       "photon_validation": "passed", **prior_summary})
        reports.append(report)
    if len(groups) == 1 and reports[0]["method"] == "patch_optimization":
        summary = dict(reports[0])
    else:
        # Detailed metrics belong to their individual components. Do not average
        # medians or let a large region hide a small region's quality failure.
        summary = {"version": VERSION, "patches": len(patches),
                   "scale_prior": "tile_hab" if hab_prior else "synrs3d_metric",
                   "training_control_samples": sum(r["training_control_samples"] for r in reports),
                   "holdout_control_samples": sum(r["holdout_control_samples"] for r in reports)}
    summary.update({"tiles": len(tiles), "adjacent_pairs": len(edges),
                    "interior_patches": len(interiors), "crossing_patches": len(crossing),
                    "graph_components": len(groups), "components": reports,
                    "retained_hab_tiles": sorted(retained),
                    "optimized_components": sum(r["method"] == "patch_optimization" for r in reports)})
    return patches, scales, summary, retained


def optimize_calibrated(tiles, edges, interiors, crossing, cache_root):
    patches = interiors + crossing
    scales = np.ones(len(patches))
    retained = set()
    reports = []
    for group in tile_components(tiles, edges):
        ids = [i for i, patch in enumerate(patches)
               if patch.tile in group or (patch.source == "seam"
                                          and any(intersect(tile, patch) for tile in group.values()))]
        local = [patches[i] for i in ids]
        report = {"tile_names": list(group), "method": "retained_hab"}
        reports.append(report)
        group_edges = [(a, b) for a, b, _ in edges if a in group and b in group]
        if not group_edges:
            report["fallback_reason"] = "no_adjacent_tiles"
            retained.update(group)
            continue
        graph, coeffs, baseline, controls, holdout, hab_errors = solver_inputs(
            group, local, cache_root, require_controls=False, calibrated=True
        )
        report.update(training_control_samples=len(controls), holdout_control_samples=len(holdout))
        seam_nodes = np.asarray([patch.source == "seam" for patch in local], dtype=bool)
        seam_mask = seam_nodes[graph[:, 0]] ^ seam_nodes[graph[:, 1]]
        measured = ~((baseline == 0) & np.isclose(coeffs[:, 0], 2) & np.isclose(coeffs[:, 1], 2))
        seam_sides = defaultdict(set)
        for a, b in graph[measured & seam_mask]:
            seam, interior = (a, b) if seam_nodes[a] else (b, a)
            seam_sides[seam].add(local[interior].tile)
        unsupported = [(a, b) for a, b in group_edges
                       if not any({a, b} <= sides for sides in seam_sides.values())]
        if unsupported or not controls or not holdout:
            report["fallback_reason"] = ("insufficient_seam_observations" if unsupported
                                         else "insufficient_training_or_held_out_controls")
            report["unsupported_seams"] = unsupported
        else:
            attempts = []
            report["scale_prior_weight_attempts"] = attempts
            for weight in HAB_SCALE_PRIOR_WEIGHTS:
                fitted = solve_scales(len(local), graph, coeffs, controls,
                                      prior_weight=np.where(seam_nodes, SCALE_PRIOR_WEIGHT, weight))
                metrics = evaluate(graph, coeffs, baseline, holdout, fitted, hab_errors,
                                   seam_mask=seam_mask, quality_reference="calibrated_windows",
                                   prior_weight=weight, validate_quality=False)
                failures = quality_failures(metrics)
                attempts.append({"interior_prior_weight": weight, "quality_failures": failures,
                                 "seam_error_m": metrics["seam_overlap_median_abs_after_m"],
                                 "holdout_mae_m": metrics["holdout_control_mae_global_m"]})
                report.update(metrics)
                if not failures:
                    scales[ids] = fitted
                    report.update(method="patch_optimization", seam_prior_weight=SCALE_PRIOR_WEIGHT)
                    break
            if report["method"] == "retained_hab":
                report["fallback_reason"] = "quality_checks_failed"
        if report["method"] == "retained_hab":
            retained.update(group)
            print(f"[WARN] Keeping independent tile heights for {list(group)}: {report['fallback_reason']}",
                  flush=True)
    summary = {"scale_prior": "calibrated_windows", "seam_model": "SynRS3D",
               "interior_patches": len(interiors), "crossing_patches": len(crossing),
               "retained_hab_tiles": sorted(retained), "components": reports,
               "optimized_components": sum(r["method"] == "patch_optimization" for r in reports)}
    return patches, scales, summary, retained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-tile", action="store_true")
    parser.add_argument("--input-tif", type=Path)
    parser.add_argument("--depth-dir", type=Path)
    parser.add_argument("--tile-name")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--conda-bin", type=Path, default=Path(shutil.which("conda") or "conda"))
    parser.add_argument("--export-raw", action="store_true")
    parser.add_argument("--global-optimize", action="store_true")
    parser.add_argument("--calibrated-work-directory",
                        help="Per-tile depth work directory containing RH2AH's calibrated windows.")
    parser.add_argument("--window-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--by-component", action="store_true",
                        help="Optimize disconnected regions independently, then merge the city.")
    parser.add_argument(
        "--hab-prior", action="store_true",
        help="Center every patch scale prior on retained per-tile HAB heights.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--expected-tiles", type=int, default=0)
    args = parser.parse_args()
    if args.archive_tile:
        for name in ("input_tif", "depth_dir", "tile_name", "cache_root"):
            if getattr(args, name) is None:
                parser.error(f"--archive-tile requires --{name.replace('_','-')}")
        archive_tile(args)
        return
    if not args.export_raw and not args.global_optimize:
        parser.error("select --export-raw and/or --global-optimize")
    if args.hab_prior and not args.global_optimize:
        parser.error("--hab-prior requires --global-optimize")
    calibrated = bool(args.calibrated_work_directory)
    if calibrated and (not args.global_optimize or args.hab_prior):
        parser.error("Calibrated windows require --global-optimize and cannot be used with --hab-prior")
    if calibrated and Path(args.calibrated_work_directory).name != args.calibrated_work_directory:
        parser.error("--calibrated-work-directory must be a single directory name")
    for name in ("input_dir", "results_root"):
        if getattr(args, name) is None:
            parser.error(f"missing --{name.replace('_','-')}")
    root = args.results_root.resolve()
    cache_root = (args.cache_root or root / "_global_patch_cache").resolve()
    tiles, edges = discover(args.input_dir.resolve(), root, not args.dry_run,
                            allow_disconnected=calibrated or args.by_component or not args.global_optimize)
    if args.expected_tiles and len(tiles) != args.expected_tiles:
        raise ValueError(f"expected {args.expected_tiles} tiles, found {len(tiles)}")
    seams = seam_records(tiles, edges, skip_short=calibrated)
    interior_count = sum(len(window_positions(t.width, t.height_px, args.window_size, args.overlap)
                             if calibrated else expected_positions(t.width, t.height_px))
                         for t in tiles.values())
    prior_mode = "calibrated_windows" if calibrated else ("tile_hab" if args.hab_prior else "synrs3d_metric")
    print(f"Patch city plan: tiles={len(tiles)}, adjacent_pairs={len(edges)}, "
          f"interior_patches={interior_count}, crossing_patches={len(seams)}, "
          f"scale_prior={prior_mode}", flush=True)
    for number, group in enumerate(tile_components(tiles, edges), 1):
        print(f"Component {number}: tiles={len(group)}; {', '.join(group)}", flush=True)
    if args.dry_run:
        return
    raw = load_interior(tiles, cache_root) if args.export_raw or not calibrated else []
    interiors = (load_calibrated(tiles, root, args.calibrated_work_directory, args.window_size, args.overlap)
                 if calibrated else raw)
    if len(interiors) != interior_count:
        raise ValueError("interior patch count mismatch")
    cache_root.mkdir(parents=True, exist_ok=True)
    staged = []
    outputs = []
    with tempfile.TemporaryDirectory(prefix="global_patch_render_", dir=cache_root) as temp:
        temp_dir = Path(temp)
        if args.export_raw:
            raw_output = root / "city_synrs3d_raw_1m_uint16.tif"
            raw_stage = merge_rendered(tiles, raw, np.ones(len(raw)), raw_output, temp_dir,
                                        feathered=False)
            from tools.assign_city_building_heights import write_city_buildings
            raw_building_output = root / "city_synrs3d_raw_building_heights.geojson"
            raw_building_stage = temp_dir / "city" / "city_synrs3d_raw_building_heights.staging.geojson"
            write_city_buildings(
                args.input_dir.resolve(), root, raw_stage, raw_building_stage,
                expected_tiles=len(tiles),
            )
            staged.extend((raw_stage, raw_building_stage))
            outputs.extend((raw_output, raw_building_output))
            for path in temp_dir.glob("*.tif"):
                path.unlink()
        if args.global_optimize:
            if args.checkpoint is None or not args.checkpoint.is_file():
                raise ValueError("SynRS3D checkpoint is required for crossing inference")
            gpus = [int(value) for value in args.gpus.split(",")]
            crossing = infer_seams(tiles, seams, cache_root, args.checkpoint, gpus, args.conda_bin) if seams else []
            if calibrated:
                patches, scales, report, retained = optimize_calibrated(tiles, edges, interiors, crossing, cache_root)
            else:
                patches, scales, report, retained = optimize_components(
                    tiles, edges, interiors, crossing, cache_root, args.hab_prior
                )
            output_stem = "city_global_patch_hab" if args.hab_prior else "city_global_patch"
            output = root / f"{output_stem}_height_1m_uint16.tif"
            building_output = root / f"{output_stem}_building_heights.geojson"
            rejected = calibrated and not report["optimized_components"]
            if rejected:
                print("[WARN] No region passed cross-tile optimization. Independent tile heights remain available.",
                      flush=True)
            else:
                opt_stage = merge_rendered(tiles, patches, scales, output, temp_dir,
                                           retained_tiles=retained, calibrated=calibrated)
                from tools.assign_city_building_heights import write_city_buildings
                building_stage = temp_dir / "city" / f"{output_stem}_building_heights.staging.geojson"
                building_summary = write_city_buildings(
                    args.input_dir.resolve(), root, opt_stage, building_stage,
                    expected_tiles=len(tiles),
                )
                report["optimized_building_count"] = building_summary["features"]
                report["optimized_buildings_without_height"] = building_summary["missing_height"]
                staged.extend((opt_stage, building_stage))
                outputs.extend((output, building_output))
            report_name = ("global_patch_hab_diagnostics.json" if args.hab_prior
                           else "global_patch_diagnostics.json")
            report_path = root / report_name
            report_temp = temp_dir / "city" / f"{Path(report_name).stem}.staging.json"
            report_temp.parent.mkdir(exist_ok=True)
            report_temp.write_text(json.dumps(report, indent=2) + "\n")
        for before, after in zip(staged, outputs):
            os.replace(before, after)
            print(f"Published {after}", flush=True)
        if args.global_optimize:
            if rejected:
                output.unlink(missing_ok=True)
                building_output.unlink(missing_ok=True)
            os.replace(report_temp, report_path)
            print(f"Published {report_path}", flush=True)


if __name__ == "__main__":
    main()
