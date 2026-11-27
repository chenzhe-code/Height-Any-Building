#!/usr/bin/env python3
"""Record completed pipeline stages and keep ATL03 downloads scoped to a request."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def file_stamp(path):
    logical = Path(path).absolute()
    target = logical.resolve()
    try:
        info = target.stat()
    except FileNotFoundError:
        return [str(logical), None]
    if not stat.S_ISREG(info.st_mode):
        return [str(logical), None]
    return [str(logical), info.st_size, info.st_mtime_ns, info.st_ctime_ns, str(target)]


def artifacts(files=(), groups=()):
    found = [file_stamp(path) for path in files]
    found.extend(
        [str(Path(folder).resolve()), pattern,
         [file_stamp(path) for path in sorted(Path(folder).glob(pattern)) if path.is_file()]]
        for folder, pattern in groups
    )
    return found


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def complete_artifacts(files, groups):
    result = artifacts(files, groups)
    for stamp in result[:len(files)]:
        if stamp[1] is None or stamp[1] <= 0:
            raise ValueError(f"Missing or empty output: {stamp[0]}")
    for folder, pattern, stamps in result[len(files):]:
        if not stamps or any(stamp[1] is None or stamp[1] <= 0 for stamp in stamps):
            raise ValueError(f"Missing or empty output matching {folder}/{pattern}")
    return result


def matches(record, key, files=(), groups=()):
    try:
        saved = json.loads(record.read_text())
        return saved == {"key": key, "outputs": complete_artifacts(files, groups)}
    except (OSError, ValueError):
        return False


def commit(record, key, files=(), groups=()):
    write_json(record, {"key": key, "outputs": complete_artifacts(files, groups)})


def write_json(record, value):
    record.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=record.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.close()
            os.replace(temporary, record)
        finally:
            temporary.unlink(missing_ok=True)


def harmony_job_completed(job_report):
    if not job_report.is_file():
        return False
    job = json.loads(job_report.read_text())
    status = str(job.get("status", "")).strip().lower()
    return status in {"successful", "success", "complete", "completed"} and not job.get("errors")


def atl03_result_status(directory, *, allow_legacy_local=False):
    job_report = directory / "atl03_harmony_job.json"
    for status in ("partial", "complete"):
        marker = directory / f"atl03_download_{status}.json"
        if not marker.is_file():
            continue
        payload = json.loads(marker.read_text())
        recorded_status = payload.get("status")
        if recorded_status is None and status == "complete" and harmony_job_completed(job_report):
            recorded_status = "complete"
        entries = payload.get("files", [])
        expected = {item["name"]: item["size_bytes"] for item in entries}
        actual = {str(path.relative_to(directory)): path.stat().st_size
                  for path in directory.glob("**/*.h5") if path.is_file()}
        if (recorded_status != status or not expected or len(entries) != len(expected)
                or expected != actual or any(size <= 0 for size in actual.values())):
            raise ValueError(f"Invalid or incomplete Harmony file inventory: {marker}. "
                             "Retry the download or use a new work directory.")
        return status
    if job_report.exists() and not (allow_legacy_local and harmony_job_completed(job_report)):
        raise ValueError(f"No verified Harmony result in {directory}; retry the download.")
    return None


def select_atl03(job_dir, fallback, explicit=None, *, allow_partial=False):
    record = job_dir / "atl03_selection.json"
    saved = json.loads(record.read_text()) if not explicit and record.is_file() else None
    directory = Path(explicit or (saved["directory"] if saved else fallback)).resolve()
    requests = sorted(path for path in directory.glob("request-*") if path.is_dir())
    if requests:
        if explicit or len(requests) != 1:
            raise ValueError(f"ATL03 cache root contains request directories: {directory}. "
                             "Set ICESAT_PATH to the specific request directory.")
        directory = requests[0].resolve()
    managed = bool(re.fullmatch(r"request-[0-9a-f]{20}", directory.name))
    status = atl03_result_status(directory, allow_legacy_local=not managed)
    if status is None and managed and not (directory / ".complete.json").is_file():
        raise ValueError(f"ATL03 download is incomplete: {directory}. Retry the download.")
    status = status or "local"
    if status == "partial":
        if not (allow_partial or (saved and saved.get("status") == "partial")):
            raise ValueError(f"Partial ATL03 result: {directory}. Set HARMONY_IGNORE_ERRORS=true to use it.")
        print(f"[WARN] Using partial ATL03 result: {directory}", file=sys.stderr)
    complete_artifacts([], [(directory, "**/*.h5")])
    selection = {"directory": str(directory), "status": status}
    if selection != saved:
        write_json(record, selection)
    return directory


def atlas_download(args):
    env = os.environ.copy()
    if env.get("DOWNLOAD", "true") != "true":
        raise ValueError("The pipeline needs ATL03 files. Use --local-atl03-dir for offline runs.")
    settings = {
        name: env.get(name) or default
        for name, default in {
            "START_DATE": "2019-01-01", "END_DATE": "2025-12-31",
            "SHORT_NAME": "ATL03", "VERSION": "007", "HARMONY_SUBSET": "true",
            "DOWNLOAD_FULL_GRANULE": "false", "HARMONY_IGNORE_ERRORS": "false",
        }.items()
    }
    if env.get("HARMONY_JOB_ID"):
        settings["HARMONY_JOB_ID"] = env["HARMONY_JOB_ID"]
    if args.tif:
        source = artifacts([args.tif])
        env.update(TIF_PATH=str(args.tif.resolve()), TIF_DIR="")
    else:
        patterns = ("*.[tT][iI][fF]", "*.[tT][iI][fF][fF]")
        source = artifacts(groups=[(args.tif_dir, pattern) for pattern in patterns])
        env.update(TIF_DIR=str(args.tif_dir.resolve()))
    key = digest(["atl03-v1", settings, source])
    active = args.cache_root.resolve() / f"request-{key[:20]}"
    active.mkdir(parents=True, exist_ok=True)
    record = active / ".complete.json"
    outputs = [(active, "**/*.h5")]
    harmony = settings["HARMONY_SUBSET"] == "true"
    with (active / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cached = env.get("OVERWRITE") != "true" and matches(record, key, groups=outputs)
        if cached and harmony:
            try:
                cached = atl03_result_status(active) == "complete"
            except (OSError, KeyError, TypeError, ValueError):
                cached = False
        if cached:
            print(f"[REUSE] ATL03 request: {active}", file=sys.stderr)
        else:
            record.unlink(missing_ok=True)
            env.update(ATL03_OUTPUT_DIR=str(active))
            subprocess.run(["bash", str(ROOT / "scripts/download_atl03.sh")],
                           env=env, stdout=sys.stderr, check=True)
            status = atl03_result_status(active) if harmony else "complete"
            if status == "complete":
                commit(record, key, groups=outputs)
            elif status == "partial" and settings["HARMONY_IGNORE_ERRORS"] == "true":
                print(f"[WARN] ATL03 request is partial: {active / 'atl03_download_partial.json'}",
                      file=sys.stderr)
            else:
                raise ValueError(f"No complete ATL03 result in {active}")
    print(active)


def add_artifacts(parser):
    parser.add_argument("--file", action="append", default=[])
    parser.add_argument("--files", nargs=2, action="append", default=[], metavar=("DIRECTORY", "PATTERN"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    key = commands.add_parser("key")
    key.add_argument("--value", action="append", default=[])
    add_artifacts(key)
    for action in ("matches", "commit"):
        command = commands.add_parser(action)
        command.add_argument("record", type=Path)
        command.add_argument("key")
        add_artifacts(command)
    download = commands.add_parser("download-atl03")
    download.add_argument("cache_root", type=Path)
    extent = download.add_mutually_exclusive_group(required=True)
    extent.add_argument("--tif", type=Path)
    extent.add_argument("--tif-dir", type=Path)
    select = commands.add_parser("select-atl03")
    select.add_argument("job_dir", type=Path)
    select.add_argument("fallback", type=Path)
    select.add_argument("--directory", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "key":
            print(digest([args.value, artifacts(args.file, args.files)]))
        elif args.action == "matches":
            return 0 if matches(args.record, args.key, args.file, args.files) else 1
        elif args.action == "commit":
            commit(args.record, args.key, args.file, args.files)
        elif args.action == "select-atl03":
            print(select_atl03(args.job_dir, args.fallback, args.directory,
                               allow_partial=os.environ.get("HARMONY_IGNORE_ERRORS") == "true"))
        else:
            atlas_download(args)
    except (OSError, KeyError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
