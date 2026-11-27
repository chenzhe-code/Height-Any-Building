#!/usr/bin/env python3
"""Infer cross-tile seam windows and jointly calibrate retained HAB height tiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

os.environ.pop("PROJ_DATA", None)

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.windows import Window
from scipy.optimize import lsq_linear
from scipy.sparse import coo_matrix, diags
from shapely.affinity import translate
from shapely.geometry import mapping

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.merge_city_height_results import (
    UINT16_NODATA,
    merge_buildings,
    merge_height_raster,
    validate_city_outputs,
    validate_height_tiles,
)

VERSION = "lagos-global-seams-v1"
PATCH_SIZE = 1022
HALF = PATCH_SIZE // 2
BLEND = 256
MAX_SAMPLES_PER_SIDE = 256


@dataclass(frozen=True)
class Tile:
    name: str
    image: str
    height: str
    buildings: str
    control: str
    old_control: str
    left: float
    right: float
    top: float
    bottom: float
    width: int
    height_px: int


@dataclass(frozen=True)
class Seam:
    name: str
    a: str  # left tile for vertical seams; top tile for horizontal seams
    b: str
    axis: str
    start: int
    last_start: int


def axis_starts(length: int) -> list[int]:
    if length < PATCH_SIZE:
        raise ValueError(f"shared edge is shorter than {PATCH_SIZE} pixels")
    starts = list(range(0, length - PATCH_SIZE + 1, HALF))
    if starts[-1] != length - PATCH_SIZE:
        starts.append(length - PATCH_SIZE)
    return starts


def same(x: float, y: float) -> bool:
    return abs(x - y) <= 1e-5


def discover(input_dir: Path, results_root: Path, control_root: Path,
             check_retained: bool) -> tuple[dict[str, Tile], list[Seam], int]:
    images = sorted(input_dir.glob("*.tif"))
    if not images:
        raise ValueError(f"no input GeoTIFFs: {input_dir}")
    tiles: dict[str, Tile] = {}
    common_crs = None
    common_shape = None
    for path in images:
        name = path.stem
        with rasterio.open(path) as src:
            if src.count != 3 or src.dtypes != ("uint8",) * 3 or src.crs is None:
                raise ValueError(f"expected georeferenced 3-band UInt8 RGB: {path}")
            if src.transform.b != 0 or src.transform.d != 0 or not same(src.res[0], 1) or not same(src.res[1], 1):
                raise ValueError(f"expected north-up 1 m image: {path}")
            if common_crs is None:
                common_crs = src.crs
                common_shape = (src.width, src.height)
            if src.crs != common_crs or (src.width, src.height) != common_shape:
                raise ValueError(f"tile grid mismatch: {path}")
            b = src.bounds
            tile = Tile(
                name=name, image=str(path.resolve()),
                height=str((results_root / name / f"{name}_absolute_height.tif").resolve()),
                buildings=str((results_root / name / f"{name}_building_heights.geojson").resolve()),
                control=str((control_root / name / "sparse_control_height.tif").resolve()),
                old_control=str((results_root / name / "sparse_control_height.tif").resolve()),
                left=b.left, right=b.right, top=b.top, bottom=b.bottom,
                width=src.width, height_px=src.height,
            )
            tiles[name] = tile
    values = list(tiles.values())
    edges: list[tuple[str, str, str]] = []
    for i, x in enumerate(values):
        for y in values[i + 1:]:
            if min(x.right, y.right) - max(x.left, y.left) > 1e-5 and min(x.top, y.top) - max(x.bottom, y.bottom) > 1e-5:
                raise ValueError(f"input tiles overlap: {x.name}, {y.name}")
            if same(x.right, y.left) and same(x.top, y.top) and same(x.bottom, y.bottom):
                edges.append((x.name, y.name, "vertical"))
            elif same(y.right, x.left) and same(x.top, y.top) and same(x.bottom, y.bottom):
                edges.append((y.name, x.name, "vertical"))
            elif same(x.bottom, y.top) and same(x.left, y.left) and same(x.right, y.right):
                edges.append((x.name, y.name, "horizontal"))
            elif same(y.bottom, x.top) and same(x.left, y.left) and same(x.right, y.right):
                edges.append((y.name, x.name, "horizontal"))
    if not edges:
        raise ValueError("no full-edge adjacent tile pairs")
    seams: list[Seam] = []
    for a, b, axis in sorted(edges):
        length = tiles[a].height_px if axis == "vertical" else tiles[a].width
        starts = axis_starts(length)
        for start in starts:
            seams.append(Seam(f"{a}__{b}__{axis}__{start:04d}", a, b, axis, start, starts[-1]))
    if check_retained:
        for tile in values:
            status = results_root / tile.name / "status.json"
            if not status.is_file():
                raise ValueError(f"missing completed tile status: {status}")
            payload = json.loads(status.read_text(encoding="utf-8"))
            if payload.get("status") != "success" or payload.get("model_name") != "SynRS3D":
                raise ValueError(f"tile is not a completed SynRS3D result: {tile.name}")
            for item in (tile.height, tile.buildings, tile.control):
                if not Path(item).is_file() or Path(item).stat().st_size == 0:
                    raise ValueError(f"missing retained input: {item}")
            with rasterio.open(tile.image) as image, rasterio.open(tile.height) as height, rasterio.open(tile.control) as control:
                for source, label in ((height, "HAB"), (control, "control")):
                    if source.crs != image.crs or source.transform != image.transform or source.width != image.width or source.height != image.height:
                        raise ValueError(f"{label} grid does not match RGB: {tile.name}")
                if height.count != 3 or height.dtypes != ("int16",) * 3:
                    raise ValueError(f"expected 3-band Int16 HAB raster: {tile.height}")
                if control.count != 1:
                    raise ValueError(f"expected 1-band sparse control: {tile.control}")
    return tiles, seams, len(edges)


def side_window(tile: Tile, seam: Seam) -> Window:
    if seam.axis == "vertical":
        return Window(tile.width - HALF if tile.name == seam.a else 0,
                      seam.start, HALF, PATCH_SIZE)
    return Window(seam.start, tile.height_px - HALF if tile.name == seam.a else 0,
                  PATCH_SIZE, HALF)


def read_rgb(seam: Seam, tiles: dict[str, Tile]) -> np.ndarray:
    with rasterio.open(tiles[seam.a].image) as a, rasterio.open(tiles[seam.b].image) as b:
        aa = a.read(window=side_window(tiles[seam.a], seam))
        bb = b.read(window=side_window(tiles[seam.b], seam))
    image = np.concatenate((aa, bb), axis=2 if seam.axis == "vertical" else 1)
    if image.shape != (3, PATCH_SIZE, PATCH_SIZE):
        raise ValueError(f"wrong crossing window shape: {seam.name}: {image.shape}")
    return np.moveaxis(image, 0, -1).copy()


def patch_side(pred: np.ndarray, seam: Seam, tile_name: str) -> np.ndarray:
    if seam.axis == "vertical":
        return pred[:, :HALF] if tile_name == seam.a else pred[:, HALF:]
    return pred[:HALF, :] if tile_name == seam.a else pred[HALF:, :]


def cache_key(seam: Seam, tiles: dict[str, Tile], checkpoint: Path) -> str:
    paths = (tiles[seam.a].image, tiles[seam.b].image, str(checkpoint))
    signature = [VERSION, asdict(seam)]
    for item in paths:
        stat = Path(item).stat()
        signature.append((str(item), stat.st_size, stat.st_mtime_ns))
    raw = json.dumps(signature, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def cache_path(seam: Seam, tiles: dict[str, Tile], checkpoint: Path, cache_dir: Path) -> Path:
    return cache_dir / f"{seam.name}__{cache_key(seam, tiles, checkpoint)}.npz"


def ensure_predictions(seams: list[Seam], tiles: dict[str, Tile], checkpoint: Path,
                       cache_dir: Path, gpu_ids: list[int], conda_bin: Path) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {seam.name: cache_path(seam, tiles, checkpoint, cache_dir) for seam in seams}
    pending = [seam for seam in seams if not paths[seam.name].is_file()]
    print(f"Crossing windows: {len(seams)}; cached: {len(seams) - len(pending)}; pending: {len(pending)}", flush=True)
    if pending:
        partitions = [[] for _ in gpu_ids]
        for index, seam in enumerate(pending):
            partitions[index % len(gpu_ids)].append(seam)
        processes = []
        for slot, partition in enumerate(partitions):
            if not partition:
                continue
            manifest = cache_dir / f"worker_gpu_{gpu_ids[slot]}.json"
            manifest.write_text(json.dumps({
                "tiles": {name: asdict(tile) for name, tile in tiles.items()},
                "seams": [{"seam": asdict(seam), "cache": str(paths[seam.name])} for seam in partition],
            }), encoding="utf-8")
            env = os.environ.copy()
            env.pop("PROJ_DATA", None)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[slot])
            env["NO_ALBUMENTATIONS_UPDATE"] = "1"
            command = [str(conda_bin), "run", "--no-capture-output", "-n", "synrs3d",
                       "python", str(Path(__file__).with_name("infer_global_seam_windows.py")),
                       "--task-file", str(manifest), "--checkpoint", str(checkpoint)]
            processes.append(subprocess.Popen(command, env=env))
        codes = [process.wait() for process in processes]
        if any(code != 0 for code in codes):
            raise RuntimeError(f"SynRS3D seam worker failed: {codes}")
    for seam in seams:
        path = paths[seam.name]
        if not path.is_file():
            raise ValueError(f"missing crossing prediction: {path}")
        with np.load(path) as cache:
            pred = cache["prediction"]
            if pred.shape != (PATCH_SIZE, PATCH_SIZE) or not np.isfinite(pred).all():
                raise ValueError(f"invalid cached prediction: {path}")
    return paths


def along_weights(seam: Seam) -> np.ndarray:
    positions = np.arange(PATCH_SIZE, dtype=np.float32)
    weights = np.ones(PATCH_SIZE, dtype=np.float32)
    if seam.start > 0:
        weights = np.minimum(weights, (positions + 0.5) / BLEND)
    if seam.start < seam.last_start:
        weights = np.minimum(weights, (PATCH_SIZE - positions - 0.5) / BLEND)
    return np.clip(weights, 0, 1)


def side_weight(seam: Seam, tile_name: str) -> np.ndarray:
    distance = np.arange(BLEND, dtype=np.float32) + 0.5
    cross = np.cos((math.pi / 2) * distance / BLEND)
    if tile_name == seam.a:
        cross = cross[::-1]
    along = along_weights(seam)
    return along[:, None] * cross[None, :] if seam.axis == "vertical" else cross[:, None] * along[None, :]


def inner_prediction(pred: np.ndarray, seam: Seam, tile_name: str) -> np.ndarray:
    side = patch_side(pred, seam, tile_name)
    if seam.axis == "vertical":
        return side[:, -BLEND:] if tile_name == seam.a else side[:, :BLEND]
    return side[-BLEND:, :] if tile_name == seam.a else side[:BLEND, :]


def inner_window(tile: Tile, seam: Seam) -> Window:
    if seam.axis == "vertical":
        return Window(tile.width - BLEND if tile.name == seam.a else 0,
                      seam.start, BLEND, PATCH_SIZE)
    return Window(seam.start, tile.height_px - BLEND if tile.name == seam.a else 0,
                  PATCH_SIZE, BLEND)


@lru_cache(maxsize=32)
def prediction_from_cache(path: str) -> np.ndarray:
    with np.load(path) as source:
        return source["prediction"].astype(np.float32)


class EquationRows:
    def __init__(self, parameter_count: int):
        self.parameter_count = parameter_count
        self.row_indices: list[int] = []
        self.col_indices: list[int] = []
        self.values: list[float] = []
        self.targets: list[float] = []
        self.robust_units: list[float] = []
        self.robust: list[bool] = []

    def add(self, terms: list[tuple[int, float]], target: float,
            group: float = 1.0, robust: bool = True) -> None:
        row = len(self.targets)
        factor = group / 5.0 if robust else 1.0
        for column, value in terms:
            if value:
                self.row_indices.append(row)
                self.col_indices.append(column)
                self.values.append(float(value) * factor)
        self.targets.append(float(target) * factor)
        self.robust_units.append(group if robust else 1.0)
        self.robust.append(robust)

    def arrays(self):
        shape = (len(self.targets), self.parameter_count)
        matrix = coo_matrix(
            (np.asarray(self.values, dtype=np.float64),
             (self.row_indices, self.col_indices)),
            shape=shape,
        ).tocsr()
        return (matrix, np.asarray(self.targets, dtype=np.float64),
                np.asarray(self.robust_units, dtype=np.float64),
                np.asarray(self.robust, dtype=bool))


def pixel_weight(seam: Seam, tile_name: str, local_row: int, local_col: int) -> float:
    if seam.axis == "vertical":
        distance = (BLEND - local_col - 0.5) if tile_name == seam.a else (local_col + 0.5)
        longitudinal = local_row
    else:
        distance = (BLEND - local_row - 0.5) if tile_name == seam.a else (local_row + 0.5)
        longitudinal = local_col
    return float(math.cos((math.pi / 2) * distance / BLEND) * along_weights(seam)[longitudinal])


def control_terms(tile: Tile, row: int, col: int, baseline: float, tile_index: int,
                  tile_seams: list[Seam], seam_index: dict[str, int],
                  cache_paths: dict[str, Path], tile_count: int) -> list[tuple[int, float]]:
    contributions = []
    for seam in tile_seams:
        window = inner_window(tile, seam)
        lr = row - int(window.row_off)
        lc = col - int(window.col_off)
        if not (0 <= lr < window.height and 0 <= lc < window.width):
            continue
        weight = pixel_weight(seam, tile.name, lr, lc)
        if weight <= 0:
            continue
        pred = inner_prediction(prediction_from_cache(str(cache_paths[seam.name])),
                                seam, tile.name)[lr, lc]
        if np.isfinite(pred):
            contributions.append((seam_index[seam.name], weight, float(pred)))
    weight_sum = sum(item[1] for item in contributions)
    alpha = min(1.0, weight_sum)
    terms = [(tile_index, baseline * (1 - alpha))]
    if weight_sum:
        for index, weight, pred in contributions:
            blend_weight = alpha * weight / weight_sum
            terms.extend(((tile_count + 2 * index, blend_weight * pred),
                          (tile_count + 2 * index + 1, blend_weight)))
    return terms


def build_equations(tiles: dict[str, Tile], seams: list[Seam],
                    cache_paths: dict[str, Path]) -> tuple[EquationRows, dict]:
    tile_names = sorted(tiles)
    tile_indices = {name: i for i, name in enumerate(tile_names)}
    seam_indices = {seam.name: i for i, seam in enumerate(seams)}
    n_tiles = len(tiles)
    rows = EquationRows(n_tiles + 2 * len(seams))
    heldout_seams = []
    control_records = []
    excluded_controls = []
    side_counts = {}
    tile_to_seams: dict[str, list[Seam]] = {name: [] for name in tiles}
    for seam in seams:
        tile_to_seams[seam.a].append(seam)
        tile_to_seams[seam.b].append(seam)

    from contextlib import ExitStack
    with ExitStack() as stack:
        heights = {name: stack.enter_context(rasterio.open(tiles[name].height))
                   for name in tile_names}
        for seam in seams:
            pred = prediction_from_cache(str(cache_paths[seam.name]))
            j = seam_indices[seam.name]
            for name in (seam.a, seam.b):
                tile = tiles[name]
                base = heights[name].read(1, window=side_window(tile, seam))
                model = patch_side(pred, seam, name)
                valid = np.flatnonzero((base >= 5) & np.isfinite(model) & (model >= 1) & (model <= 300))
                seed = int(hashlib.sha256(f"{seam.name}:{name}".encode()).hexdigest()[:16], 16)
                rng = np.random.default_rng(seed)
                chosen = rng.permutation(valid)[:MAX_SAMPLES_PER_SIDE + 64]
                training = chosen[:MAX_SAMPLES_PER_SIDE]
                validating = chosen[MAX_SAMPLES_PER_SIDE:]
                side_counts[f"{seam.name}:{name}"] = int(len(valid))
                if len(training) >= 16:
                    group = 1 / math.sqrt(len(training))
                    for position in training:
                        h = float(base.flat[position])
                        p = float(model.flat[position])
                        rows.add([
                            (tile_indices[name], h),
                            (n_tiles + 2 * j, -p),
                            (n_tiles + 2 * j + 1, -1),
                        ], 0, group)
                for position in validating:
                    heldout_seams.append((tile_indices[name], j,
                                          float(base.flat[position]),
                                          float(model.flat[position])))
        for name in tile_names:
            tile = tiles[name]
            with rasterio.open(tile.control) as source:
                controls = source.read(1, masked=True)
            old = None
            if Path(tile.old_control).is_file() and Path(tile.old_control) != Path(tile.control):
                with rasterio.open(tile.old_control) as source:
                    if source.transform == heights[name].transform and source.shape == heights[name].shape:
                        old = source.read(1, masked=True)
            valid = (~np.ma.getmaskarray(controls)) & np.isfinite(controls.data) & (controls.data >= 5)
            yy, xx = np.where(valid)
            for row, col in zip(yy, xx):
                observed = float(controls.data[row, col])
                label = f"{name}:{row}:{col}"
                if old is not None and not np.ma.getmaskarray(old)[row, col]:
                    prior = float(old.data[row, col])
                    if np.isfinite(prior) and prior >= 5 and abs(prior - observed) > 5:
                        excluded_controls.append((label, "old/new control conflict", observed, prior))
                        continue
                baseline = float(heights[name].read(
                    1, window=Window(int(col), int(row), 1, 1))[0, 0])
                terms = control_terms(tile, int(row), int(col), baseline,
                                      tile_indices[name], tile_to_seams[name],
                                      seam_indices, cache_paths, n_tiles)
                if baseline <= 0 and len(terms) == 1:
                    excluded_controls.append((label, "zero source height", observed, baseline))
                    continue
                heldout = int(hashlib.sha256(label.encode()).hexdigest()[:8], 16) % 5 == 0
                control_records.append((label, observed, baseline, terms, heldout))
                if not heldout:
                    rows.add(terms, observed, group=3.0)
    for index in range(n_tiles):
        rows.add([(index, 10.0)], 10.0, robust=False)
    for index in range(len(seams)):
        rows.add([(n_tiles + 2 * index, 2.0)], 2.0, robust=False)
        rows.add([(n_tiles + 2 * index + 1, 0.2)], 0.0, robust=False)
    return rows, {
        "tile_names": tile_names,
        "side_counts": side_counts,
        "seam_holdout": heldout_seams,
        "controls": control_records,
        "excluded_controls": excluded_controls,
    }


def fit_global(rows: EquationRows, tile_count: int, seam_count: int) -> np.ndarray:
    matrix, targets, robust_units, robust_mask = rows.arrays()
    lower = np.r_[np.full(tile_count, 0.7),
                  np.tile(np.array([0.3, -10.0]), seam_count)]
    upper = np.r_[np.full(tile_count, 1.3),
                  np.tile(np.array([3.0, 10.0]), seam_count)]
    result = np.r_[np.ones(tile_count),
                   np.tile(np.array([1.0, 0.0]), seam_count)]
    for iteration in range(5):
        residual = (matrix @ result - targets) / robust_units
        huber = np.ones(len(targets), dtype=np.float64)
        magnitude = np.abs(residual[robust_mask])
        huber[robust_mask] = np.sqrt(np.minimum(1.0, 1.0 / np.maximum(magnitude, 1e-9)))
        weighted = diags(huber) @ matrix
        fit = lsq_linear(weighted, targets * huber, bounds=(lower, upper),
                         method="trf", lsq_solver="lsmr", tol=1e-5,
                         lsmr_maxiter=300, max_iter=50)
        if fit.status <= 0 or not np.isfinite(fit.x).all():
            raise RuntimeError(f"global sparse optimization failed at iteration {iteration}: {fit.message}")
        difference = float(np.max(np.abs(fit.x - result)))
        result = fit.x
        print(f"Global IRLS {iteration + 1}/5: max parameter change {difference:.5f}", flush=True)
        if difference < 1e-4:
            break
    return result


def quality_report(solution: np.ndarray, tile_count: int, diagnostics: dict,
                   seams: list[Seam]) -> dict:
    seam_before = []
    seam_after = []
    for tile_index, window_index, h, p in diagnostics["seam_holdout"]:
        transformed = solution[tile_count + 2 * window_index] * p + solution[tile_count + 2 * window_index + 1]
        seam_before.append(abs(h - p))
        seam_after.append(abs(solution[tile_index] * h - transformed))
    control_before = []
    control_after = []
    holdout_before = []
    holdout_after = []
    for _, observed, baseline, terms, heldout in diagnostics["controls"]:
        refined = sum(solution[index] * value for index, value in terms)
        control_before.append(abs(baseline - observed))
        control_after.append(abs(refined - observed))
        if heldout:
            holdout_before.append(abs(baseline - observed))
            holdout_after.append(abs(refined - observed))
    report = {
        "version": VERSION,
        "tiles": tile_count,
        "windows": len(seams),
        "candidate_controls": len(diagnostics["controls"]) + len(diagnostics["excluded_controls"]),
        "used_controls": sum(not record[4] for record in diagnostics["controls"]),
        "heldout_controls": len(holdout_before),
        "excluded_controls": len(diagnostics["excluded_controls"]),
        "heldout_seam_samples": len(seam_before),
        "seam_mae_before": float(np.mean(seam_before)) if seam_before else None,
        "seam_mae_after": float(np.mean(seam_after)) if seam_after else None,
        "control_mae_before": float(np.mean(control_before)) if control_before else None,
        "control_mae_after": float(np.mean(control_after)) if control_after else None,
        "heldout_control_mae_before": float(np.mean(holdout_before)) if holdout_before else None,
        "heldout_control_mae_after": float(np.mean(holdout_after)) if holdout_after else None,
        "tile_scales": {name: float(solution[index]) for index, name in enumerate(diagnostics["tile_names"])},
        "excluded_control_details": diagnostics["excluded_controls"],
    }
    if not seam_before:
        raise ValueError("no held-out seam samples; cannot validate cross-tile calibration")
    if report["seam_mae_after"] > report["seam_mae_before"] + 0.5:
        raise ValueError(f"held-out seam error worsened: {report}")
    if len(holdout_before) >= 10 and report["heldout_control_mae_after"] > report["heldout_control_mae_before"] + 0.5:
        raise ValueError(f"held-out photon error worsened: {report}")
    return report


def write_uint16_tiles(stage: Path, tiles: dict[str, Tile], seams: list[Seam],
                       cache_paths: dict[str, Path], solution: np.ndarray,
                       tile_names: list[str]) -> list[tuple[str, Path, Path]]:
    tile_index = {name: index for index, name in enumerate(tile_names)}
    linked: dict[str, list[tuple[int, Seam]]] = {name: [] for name in tiles}
    for index, seam in enumerate(seams):
        linked[seam.a].append((index, seam))
        linked[seam.b].append((index, seam))
    out_dir = stage / "tiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for number, name in enumerate(tile_names, 1):
        tile = tiles[name]
        with rasterio.open(tile.height) as source:
            base = source.read(1)
            if not np.array_equal(base, source.read(2)) or not np.array_equal(base, source.read(3)):
                raise ValueError(f"HAB bands disagree; refusing to choose band 1: {name}")
            valid = np.isfinite(base)
            if source.nodata is not None:
                valid &= base != source.nodata
            transformed = base.astype(np.float32) * np.float32(solution[tile_index[name]])
            patch_sum = np.zeros_like(transformed)
            weight_sum = np.zeros_like(transformed)
            for seam_index, seam in linked[name]:
                pred = prediction_from_cache(str(cache_paths[seam.name]))
                patch = inner_prediction(pred, seam, name)
                calibrated = (solution[len(tiles) + 2 * seam_index] * patch
                              + solution[len(tiles) + 2 * seam_index + 1]).astype(np.float32)
                weight = side_weight(seam, name)
                window = inner_window(tile, seam)
                row = int(window.row_off)
                col = int(window.col_off)
                h, w = patch.shape
                view_sum = patch_sum[row:row + h, col:col + w]
                view_weight = weight_sum[row:row + h, col:col + w]
                good = np.isfinite(calibrated)
                view_sum += np.where(good, calibrated * weight, 0)
                view_weight += np.where(good, weight, 0)
            blended = weight_sum > 0
            alpha = np.minimum(weight_sum[blended], 1)
            transformed[blended] = ((1 - alpha) * transformed[blended]
                                    + alpha * patch_sum[blended] / weight_sum[blended])
            encoded = np.full(base.shape, UINT16_NODATA, dtype=np.uint16)
            encoded[valid] = np.clip(np.rint(transformed[valid]), 0, UINT16_NODATA - 1).astype(np.uint16)
            path = out_dir / f"{name}_global_height_uint16.tif"
            profile = source.profile.copy()
            profile.update(driver="GTiff", count=1, dtype="uint16", nodata=UINT16_NODATA,
                           tiled=True, blockxsize=512, blockysize=512, compress="ZSTD",
                           predictor=2, BIGTIFF="IF_SAFER")
            with rasterio.open(path, "w", **profile) as target:
                target.write(encoded, 1)
                target.update_tags(HEIGHT_UNITS="metre",
                                   VALUE_ENCODING="rounded whole metres; 0-65534")
        outputs.append((name, path, stage / "buildings" / f"{name}_global_buildings.geojson"))
        print(f"UInt16 refined tile {number}/{len(tile_names)}: {name}", flush=True)
    return outputs


def recompute_buildings(outputs: list[tuple[str, Path, Path]],
                        tiles: dict[str, Tile]) -> int:
    from rasterio.errors import RasterioError
    feature_count = 0
    for name, raster_path, target_path in outputs:
        frame = gpd.read_file(tiles[name].buildings)
        with rasterio.open(raster_path) as source:
            if frame.crs is None:
                raise ValueError(f"building GeoJSON lacks CRS: {tiles[name].buildings}")
            if frame.crs != source.crs:
                frame = frame.to_crs(source.crs)
            heights = []
            for _, row in frame.iterrows():
                geometry = row.geometry
                if geometry is None or geometry.is_empty:
                    heights.append(None)
                    continue
                ox = row.get("Offset_X", 0)
                oy = row.get("Offset_Y", 0)
                ox = float(ox) if ox is not None and np.isfinite(ox) else 0.0
                oy = float(oy) if oy is not None and np.isfinite(oy) else 0.0
                shifted = translate(geometry,
                                    xoff=ox * source.transform.a,
                                    yoff=oy * source.transform.e)
                try:
                    values, _ = mask(source, [mapping(shifted)], crop=True, filled=False)
                    valid = np.ma.compressed(values[0])
                    heights.append(float(np.max(valid)) if valid.size else None)
                except (ValueError, RasterioError):
                    heights.append(None)
        frame["Extracted_Height"] = heights
        target_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_file(target_path, driver="GeoJSON")
        feature_count += len(frame)
        print(f"Building heights: {name}, {len(frame)} features", flush=True)
    return feature_count


def run_fingerprint(tiles: dict[str, Tile], checkpoint: Path) -> str:
    signatures = [VERSION, str(checkpoint)]
    paths = [checkpoint]
    for name in sorted(tiles):
        tile = tiles[name]
        paths.extend((Path(tile.image), Path(tile.height),
                      Path(tile.buildings), Path(tile.control)))
    for path in paths:
        stat = path.stat()
        signatures.append((str(path), stat.st_size, stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(signatures).encode()).hexdigest()


def write_control_audit(stage: Path, report: dict, diagnostics: dict,
                        solution: np.ndarray) -> None:
    with (stage / "control_residuals.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("location", "control_m", "before_m", "after_m", "holdout"))
        for label, observed, baseline, terms, heldout in diagnostics["controls"]:
            refined = sum(solution[index] * value for index, value in terms)
            writer.writerow((label, observed, baseline, refined, int(heldout)))
    (stage / "diagnostics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--control-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--conda-bin", type=Path)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-retained", action="store_true")
    args = parser.parse_args()

    required = (args.input_dir, args.results_root, args.control_root,
                args.checkpoint, args.conda_bin)
    if any(value is None for value in required):
        parser.error("input-dir, results-root, control-root, checkpoint and conda-bin are required")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise ValueError(f"missing SynRS3D checkpoint: {checkpoint}")
    gpu_ids = [int(item) for item in args.gpus.split(",")]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU IDs must be a nonempty unique list")
    tiles, seams, edge_count = discover(
        args.input_dir.resolve(), args.results_root.resolve(),
        args.control_root.resolve(), args.check_retained or not args.dry_run)
    print(f"Global seam plan: {len(tiles)} tiles, {edge_count} adjacent pairs, "
          f"{len(seams)} crossing windows, GPUs {gpu_ids}", flush=True)
    if args.dry_run:
        print("Dry-run: no crossing inference or output writes", flush=True)
        return
    root = args.results_root.resolve() / "global_optimization"
    fingerprint = run_fingerprint(tiles, checkpoint)
    current = root / "current"
    if current.is_symlink():
        manifest = current / "manifest.json"
        if manifest.is_file():
            previous = json.loads(manifest.read_text(encoding="utf-8"))
            if previous.get("fingerprint") == fingerprint and previous.get("status") == "success":
                print(f"[REUSE] Global result: {current}", flush=True)
                return
    elif current.exists():
        raise ValueError(f"refusing to overwrite non-symlink output: {current}")

    cache_dir = root / "cache"
    cache_paths = ensure_predictions(seams, tiles, checkpoint, cache_dir,
                                     gpu_ids, args.conda_bin.resolve())
    rows, diagnostics = build_equations(tiles, seams, cache_paths)
    solution = fit_global(rows, len(tiles), len(seams))
    report = quality_report(solution, len(tiles), diagnostics, seams)
    report["adjacent_pairs"] = edge_count
    report["fingerprint"] = fingerprint
    print(json.dumps({key: report[key] for key in (
        "candidate_controls", "used_controls", "heldout_controls",
        "excluded_controls", "seam_mae_before", "seam_mae_after",
        "heldout_control_mae_before", "heldout_control_mae_after")},
        indent=2), flush=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + fingerprint[:8]
    stage = root / "runs" / f".staging_{run_id}"
    if stage.exists():
        raise ValueError(f"staging directory already exists: {stage}")
    stage.mkdir(parents=True)
    outputs = write_uint16_tiles(stage, tiles, seams, cache_paths,
                                 solution, diagnostics["tile_names"])
    grid = validate_height_tiles(outputs)
    city_tif = stage / "city_global_height_1m_uint16.tif"
    merge_height_raster(outputs, city_tif, grid)
    feature_count = recompute_buildings(outputs, tiles)
    city_geojson = stage / "city_global_building_heights.geojson"
    merge_buildings(outputs, city_geojson, grid["crs"])
    validate_city_outputs(city_tif, city_geojson, grid, feature_count)
    with rasterio.open(city_tif) as source:
        if source.count != 1 or source.dtypes != ("uint16",) or source.nodata != UINT16_NODATA:
            raise ValueError("invalid published city UInt16 GeoTIFF")
    report["features"] = feature_count
    write_control_audit(stage, report, diagnostics, solution)
    (stage / "manifest.json").write_text(
        json.dumps({"status": "success", "fingerprint": fingerprint,
                    "version": VERSION, "run_id": run_id}, indent=2) + "\n",
        encoding="utf-8")
    completed = root / "runs" / run_id
    os.replace(stage, completed)
    temporary_link = root / f".current_{run_id}"
    temporary_link.symlink_to(Path("runs") / run_id, target_is_directory=True)
    os.replace(temporary_link, current)
    print(f"[OK] Global UInt16 city height: {current / city_tif.name}", flush=True)
    print(f"[OK] Global building GeoJSON: {current / city_geojson.name}", flush=True)
    print(f"[OK] Diagnostics: {current / 'diagnostics.json'}", flush=True)


if __name__ == "__main__":
    main()
