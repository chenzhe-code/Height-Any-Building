"""Run the published HAB commands and keep local UI task records."""

from datetime import date, datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from uuid import uuid4

from utils.window_grid import window_step

from .model_config import DEPTH_MODELS, ROOT
from .pipeline_cache import write_json


ACTIVE = {"queued", "running"}
RASTERS = {".tif", ".tiff"}
EARTHDATA_FIELDS = {"EARTHDATA_USERNAME": "earthdata_username", "EARTHDATA_PASSWORD": "earthdata_password"}
BUILDING_MODELS = {"offset_model": ("MLS-BRN",), "footprint_model": ("MLS-BRN",)}
DEMO_OUTPUTS = ("building_height_raster.tif", "per_building_heights.geojson")
LEGACY_DEMO_OUTPUTS = ("AbsoluteHeight/output.tif", "building_heights.geojson")
CITY_OUTPUTS = (
    "city_absolute_height_1m_uint16.tif", "city_building_heights.geojson",
    "city_global_patch_height_1m_uint16.tif", "city_global_patch_building_heights.geojson",
    "city_synrs3d_raw_1m_uint16.tif", "city_synrs3d_raw_building_heights.geojson",
    "batch_status.csv", "global_patch_diagnostics.json",
)


class BusyError(ValueError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def log_path(folder, attempt):
    return folder / f"attempt-{attempt:03d}.log"


def cancel_path(folder, attempt):
    return folder / f"cancel-{attempt:03d}"


def allowed_path(value, roots):
    if not isinstance(value, str) or not value:
        raise ValueError("请选择本机文件或目录。")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("请输入本机的绝对路径。")
    path = path.resolve(strict=True)
    if not any(path.is_relative_to(root) for root in roots):
        raise ValueError("路径不在已开放的数据目录中。")
    return path


def list_directory(value, roots, kind):
    if kind not in {"image", "directory"}:
        raise ValueError("无效的选择类型。")
    if not value:
        return {"path": None, "parent": None,
                "entries": [{"name": str(root), "path": str(root), "directory": True} for root in roots]}
    folder = allowed_path(value, roots)
    if not folder.is_dir():
        raise ValueError("请选择目录。")
    entries = []
    for path in folder.iterdir():
        if path.name.startswith("."):
            continue
        try:
            allowed_path(str(path), roots)
        except (OSError, RuntimeError, ValueError):
            continue
        if path.is_dir() or (kind == "image" and path.is_file() and path.suffix.lower() in RASTERS):
            entries.append({"name": path.name, "path": str(path), "directory": path.is_dir()})
    entries.sort(key=lambda item: (not item["directory"], item["name"].casefold()))
    parent = folder.parent if any(folder != root and folder.is_relative_to(root) for root in roots) else None
    return {"path": str(folder), "parent": str(parent) if parent else None, "entries": entries}


def positive_integer(value, label, default):
    if value in ("", None):
        return default
    if not re.fullmatch(r"[0-9]+", str(value)) or int(value) < 1:
        raise ValueError(f"{label}必须是正整数。")
    return int(value)


def experiment_options(values):
    window = positive_integer(values.get("window_size"), "推理窗口", 1024)
    minimum = positive_integer(values.get("min_photon_controls"), "最少光子控制点", 4)
    raw_overlap = values.get("overlap", "")
    try:
        overlap = float(0.5 if raw_overlap in ("", None) else raw_overlap)
        window_step(window, overlap)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("窗口重叠率须在 0 到 1 之间（不含 1），且窗口步长须至少为 1 像素。") from None
    ground = values.get("ground_filter") or "auto"
    if ground not in ("auto", "on", "off"):
        raise ValueError("请选择有效的地面滤波选项。")
    return window, overlap, minimum, ground


def requires_earthdata(job):
    return job["mode"] in ("image", "batch") and job["action"] == "run" and "--local-atl03-dir" not in job["argv"]


def earthdata_environment(job, values):
    if not requires_earthdata(job):
        return {}
    env = {key: values.get(field) for key, field in EARTHDATA_FIELDS.items()}
    if any(not isinstance(value, str) or not value.strip() or
           any(ord(c) < 32 or ord(c) > 255 for c in value) for value in env.values()):
        raise ValueError("通过 Earthdata 下载时，请输入有效的登录名和密码。")
    env["EARTHDATA_USERNAME"] = env["EARTHDATA_USERNAME"].strip()
    return env


def build_command(values, roots, output, python=sys.executable):
    mode, action = values.get("mode"), values.get("action", "run")
    if mode not in ("image", "batch", "demo") or action not in ("run", "preview", "check"):
        raise ValueError("请选择有效的处理方式。")
    if mode == "demo":
        if action != "run":
            raise ValueError("CPU 示例可以直接运行。")
        return [python, str(ROOT / "scripts/run_cpu_demo.py"), "--work-dir", str(output)]

    # MLS-BRN emits the footprint and offset in one inference pass.
    for field, models in BUILDING_MODELS.items():
        if values.get(field, models[0]) not in models:
            raise ValueError("建筑物偏移和足迹目前都使用 MLS-BRN。")
    source = allowed_path(values.get("input"), roots)
    model = values.get("model") or ("SynRS3D" if mode == "batch" else "auto")
    if model != "auto" and model not in DEPTH_MODELS:
        raise ValueError("请选择有效的模型。")
    gpus = values.get("gpus") or "0"
    if not isinstance(gpus, str) or not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", gpus):
        raise ValueError("GPU 编号用英文逗号分隔，例如 0,1。")
    if len({int(value) for value in gpus.split(",")}) != len(gpus.split(",")):
        raise ValueError("GPU 编号不能重复。")

    window, overlap, minimum, ground = experiment_options(values)
    if mode == "image":
        if not source.is_file() or source.suffix.lower() not in RASTERS:
            raise ValueError("请选择 GeoTIFF 影像。")
        cmd = [python, str(ROOT / "tools/run_image.py"), "--image", str(source),
               "--work-dir", str(output)]
        global_mode = values.get("global_mode") or "auto"
        if global_mode not in ("auto", "patch", "none"):
            raise ValueError("请选择有效的跨瓦片优化设置。")
        if model != "auto":
            cmd += ["--model", model]
        if global_mode != "auto":
            cmd += ["--global-mode", global_mode]
        tile_size = values.get("tile_size")
        if tile_size not in ("", None):
            size = positive_integer(tile_size, "瓦片尺寸", 5000)
            if size < 2048:
                raise ValueError("瓦片尺寸至少为 2048 像素。")
            cmd += ["--tile-size", str(size)]
        target_crs = values.get("target_crs")
        if target_crs:
            if not isinstance(target_crs, str) or not re.fullmatch(r"EPSG:[0-9]+", target_crs, re.IGNORECASE):
                raise ValueError("目标坐标系请填写 EPSG 编号，例如 EPSG:32650。")
            cmd += ["--target-crs", target_crs.upper()]
    else:
        if not source.is_dir() or not any(p.is_file() and p.suffix.lower() in RASTERS for p in source.iterdir()):
            raise ValueError("请选择包含 GeoTIFF 瓦片的目录。")
        model = "SynRS3D" if model == "auto" else model
        cmd = ["bash", str(ROOT / "scripts/run_batch_work_dir.sh"), "--input-dir", str(source),
               "--output-root", str(output), "--model", model, "--atl03-scope", "shared",
               "--merge-city", "--no-uncertainty"]
        if values.get("global_optimize") is True:
            cmd += ["--global-patch-optimize"]
    cmd += ["--gpus", gpus, "--window-size", str(window), "--overlap", str(overlap),
            "--min-photon-controls", str(minimum)]
    if ground != "auto":
        cmd.append("--ground-filter" if ground == "on" else "--no-ground-filter")

    atl03_source = values.get("atl03_source", "online")
    if atl03_source == "local":
        local = allowed_path(values.get("atl03_dir"), roots)
        if not local.is_dir() or not any(p.is_file() for p in local.glob("*.h5")):
            raise ValueError("请选择直接包含 ATL03 HDF5 文件的目录。")
        cmd += ["--local-atl03-dir", str(local)]
    elif atl03_source == "online":
        dates = {}
        for key, flag in (("start_date", "--atl03-start-date"), ("end_date", "--atl03-end-date")):
            value = values.get(key)
            if value:
                if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                    raise ValueError("日期格式应为 YYYY-MM-DD。")
                dates[key] = date.fromisoformat(value)
                cmd += [flag, value]
        if dates.get("start_date", date.min) > dates.get("end_date", date.max):
            raise ValueError("ATL03 开始日期必须早于或等于结束日期。")
    else:
        raise ValueError("请选择 ATL03 数据来源。")

    if action != "run":
        cmd.append("--dry-run" if action == "preview" else "--check")
    return cmd


def collect_outputs(folder, job):
    if job["action"] != "run" or job["status"] in ACTIVE:
        return []
    output = folder / "output"
    result_root = output / "results" if job["mode"] == "image" else output
    paths = []
    if job["mode"] == "demo":
        demo_outputs = DEMO_OUTPUTS
        if not any((output / name).is_file() for name in demo_outputs):
            demo_outputs = LEGACY_DEMO_OUTPUTS
        paths.extend(output / name for name in demo_outputs)
    else:
        paths.extend(result_root / name for name in CITY_OUTPUTS)
        paths.append(output / "footprint_direct_nms_absolute_height.geojson")
        paths.extend(sorted(output.glob("*-depth/AbsoluteHeight/output.tif")))
        if not (result_root / "city_absolute_height_1m_uint16.tif").is_file():
            paths.extend(sorted(result_root.glob("*/*_absolute_height.tif")))
            paths.extend(sorted(result_root.glob("*/*_building_heights.geojson")))
    return [
        {"name": path.name,
         "id": str(path.relative_to(output)), "bytes": path.stat().st_size}
        for path in paths if path.is_file() and path.resolve().is_relative_to(output)
    ]


def footprint_artifact(raster):
    """Find the GeoJSON produced alongside a published height raster."""
    raster = Path(raster)
    if raster == Path(DEMO_OUTPUTS[0]):
        return DEMO_OUTPUTS[1]
    if raster == Path(LEGACY_DEMO_OUTPUTS[0]):
        return LEGACY_DEMO_OUTPUTS[1]
    if raster.name == "output.tif":
        return "footprint_direct_nms_absolute_height.geojson"
    city_names = {
        "city_absolute_height_1m_uint16.tif": "city_building_heights.geojson",
        "city_global_patch_height_1m_uint16.tif": "city_global_patch_building_heights.geojson",
        "city_synrs3d_raw_1m_uint16.tif": "city_synrs3d_raw_building_heights.geojson",
    }
    filename = city_names.get(raster.name)
    if not filename and raster.name.endswith("_absolute_height.tif"):
        filename = raster.name.removesuffix("_absolute_height.tif") + "_building_heights.geojson"
    return str(raster.with_name(filename)) if filename else None


def current_progress(folder, job):
    output = folder / "output"
    result_root = output / "results" if job["mode"] == "image" else output
    began = job.get("started_at") or job["created_at"]
    single = read_json(output / "stage_status.json")
    stages = list(single.get("stages", {}).values()) if single.get("run_id") == job["run_id"] else []
    tile_statuses = []
    for path in result_root.glob("*/status.json"):
        tile = read_json(path)
        if tile and tile.get("started_at", "") >= began[:19]:
            tile_statuses.append({k: tile.get(k) for k in ("tile", "status", "stage", "exit_code")})
    tile_statuses.sort(key=lambda item: item["tile"])
    return stages, tile_statuses


class Jobs:
    def __init__(self, root, data_roots):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.data_roots = tuple(data_roots)
        lock = self.lock(required=False)
        if lock:
            with lock:
                self.reconcile()

    def lock(self, required=True):
        stream = (self.root / ".runner.lock").open("a+")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            if required:
                raise BusyError("已有任务正在运行。")
            return None
        return stream

    def folders(self):
        return sorted((file.parent for file in self.root.glob("*/job.json")
                       if re.fullmatch(r"[a-f0-9]{32}", file.parent.name)
                       and not file.parent.is_symlink() and not file.is_symlink()),
                      key=lambda path: path.stat().st_mtime, reverse=True)

    def reconcile(self):
        for folder in self.folders():
            path = folder / "job.json"
            job = read_json(path)
            if job.get("status") in ACTIVE:
                job.update(status="interrupted", finished_at=now())
                write_json(path, job)

    def locate(self, job_id):
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise FileNotFoundError("找不到任务。")
        folder = self.root / job_id
        path = folder / "job.json"
        if folder.is_symlink() or path.is_symlink():
            raise FileNotFoundError("找不到任务。")
        job = read_json(path)
        if not job:
            raise FileNotFoundError("找不到任务。")
        return folder, job

    def spawn(self, folder, job, lock, credentials):
        job["run_id"] = uuid4().hex
        job.update(status="queued", exit_code=None, started_at=None, finished_at=None)
        path = folder / "job.json"
        write_json(path, job)
        env = {key: value for key, value in os.environ.items() if key not in EARTHDATA_FIELDS}
        try:
            child = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts/run_ui.py"), "--worker", str(folder), str(lock.fileno())],
                cwd=ROOT, env=env | credentials, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True, pass_fds=(lock.fileno(),),
            )
        except OSError:
            job.update(status="failed", finished_at=now())
            write_json(path, job)
            raise
        return child

    def start(self, values):
        job_id = uuid4().hex
        folder = self.root / job_id
        cmd = build_command(values, self.data_roots, folder / "output")
        job = {"id": job_id, "mode": values["mode"], "action": values.get("action", "run"),
               "created_at": now(), "attempt": 1, "argv": cmd}
        credentials = earthdata_environment(job, values)
        with self.lock() as lock:
            self.reconcile()
            folder.mkdir()
            return job, self.spawn(folder, job, lock, credentials)

    def resume(self, job_id, values=None):
        with self.lock() as lock:
            self.reconcile()
            folder, job = self.locate(job_id)
            if (job["mode"] == "demo" or job["action"] != "run"
                    or job["status"] not in {"failed", "cancelled", "interrupted"}):
                raise ValueError("该任务当前不需要续跑。")
            credentials = earthdata_environment(job, values or {})
            job["attempt"] += 1
            return job, self.spawn(folder, job, lock, credentials)

    def cancel(self, job_id):
        folder, job = self.locate(job_id)
        if job["status"] not in ACTIVE:
            raise ValueError("任务已经结束。")
        cancel_path(folder, job["attempt"]).touch()

    def list(self):
        lock = self.lock(required=False)
        if lock:
            with lock:
                self.reconcile()
        return [job for folder in self.folders() if (job := read_json(folder / "job.json"))]

    def detail(self, job_id):
        folder, job = self.locate(job_id)
        stages, tiles = current_progress(folder, job)
        log = log_path(folder, job["attempt"])
        tail = ""
        if log.is_file():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 60_000))
                tail = stream.read(60_000).decode("utf-8", errors="replace")
        return {**job, "command": shlex.join(job["argv"]), "output": str(folder / "output"),
                "requires_earthdata": requires_earthdata(job),
                "stages": stages, "tiles": tiles, "artifacts": collect_outputs(folder, job), "log": tail}


def stop_child(child):
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return child.wait()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        child.poll()
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            return child.wait()
        time.sleep(0.1)
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return child.wait()


def run_worker(folder, lock_fd):
    with os.fdopen(lock_fd):
        path = folder / "job.json"
        job = read_json(path)
        cancelled = cancel_path(folder, job["attempt"])
        job.update(status="running", started_at=now())
        write_json(path, job)
        code = None
        with log_path(folder, job["attempt"]).open("wb") as log:
            if not cancelled.exists():
                env = os.environ | {"HAB_PIPELINE_RUN_ID": job["run_id"], "PYTHONUNBUFFERED": "1"}
                child = None
                try:
                    child = subprocess.Popen(job["argv"], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                             pass_fds=(lock_fd,))
                    while child.poll() is None:
                        if cancelled.exists():
                            stop_child(child)
                            break
                        time.sleep(0.2)
                    code = child.returncode
                except OSError as error:
                    log.write(f"{error}\n".encode())
                    if child and child.poll() is None:
                        stop_child(child)
                    code = 1
        status = "cancelled" if cancelled.exists() and code != 0 else ("success" if code == 0 else "failed")
        job.update(status=status, exit_code=code, finished_at=now())
        write_json(path, job)
