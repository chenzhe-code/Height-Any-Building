"""Record pipeline stage and batch tile status for logs and resume diagnostics."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

if __package__:
    from .pipeline_cache import write_json
else:
    from pipeline_cache import write_json


BATCH_FIELDS = [
    "tile", "status", "stage", "exit_code", "started_at", "finished_at",
    "model_name", "control_source", "compute_uncertainty", "fusion_mode",
    "apply_ground_filter", "gpu_id", "input_tif", "job_dir", "log",
]


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_status(path):
    return json.loads(path.read_text(encoding="utf-8"))


def set_stage(run, step, title, state, exit_code, elapsed):
    key = str(step)
    started = run["stages"].get(key, {}).get("started_at", now())
    run["stages"][key] = {
        "step": step, "title": title, "status": state, "exit_code": exit_code,
        "started_at": started, "finished_at": None if state == "running" else now(),
        "elapsed_seconds": None if state == "running" else elapsed,
    }


def failed_stage(path, run_id):
    try:
        run = read_status(path)
    except (OSError, ValueError):
        return "full_pipeline"
    if run.get("run_id") == run_id:
        for stage in reversed(list(run["stages"].values())):
            if stage["status"] in {"failed", "running"}:
                return f"[{stage['step']}/10] {stage['title']}"
    return "full_pipeline"


def write_tile_status(args):
    job_dir = Path(args.job_dir)
    write_json(args.path, {
        "tile": args.tile, "input_tif": args.input_tif, "job_dir": args.job_dir,
        "status": args.state, "stage": args.stage, "exit_code": args.exit_code,
        "started_at": args.started_at, "finished_at": args.finished_at or None,
        "model_name": args.model, "control_source": args.control_source,
        "compute_uncertainty": bool(args.compute_uncertainty), "fusion_mode": args.fusion_mode,
        "apply_ground_filter": bool(args.ground_filter), "gpu_id": args.gpu_id,
        "log": str(job_dir / "run.log"),
    })


def summarize_batch(root, inputs):
    paths = [root / source.stem / "status.json" for source in inputs]
    rows = [read_status(path) for path in paths if path.is_file()]
    rows.sort(key=lambda row: [int(part) if part.isdigit() else part
                               for part in re.split(r"(\d+)", row["tile"])])
    target = root / "batch_status.csv"
    temporary = target.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=BATCH_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)
    counts = Counter(row["status"] for row in rows)
    print(" ".join(f"{key}={counts[key]}" for key in sorted(counts)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("start", "stage", "finish", "failed-stage", "tile"):
        command = commands.add_parser(action)
        command.add_argument("path", type=Path)
    for action in ("start", "failed-stage"):
        commands.choices[action].add_argument("run_id")
    stage = commands.choices["stage"]
    stage.add_argument("step", type=int)
    stage.add_argument("title")
    stage.add_argument("state", choices=("running", "success", "reused", "skipped"))
    stage.add_argument("elapsed", type=int)
    finish = commands.choices["finish"]
    finish.add_argument("exit_code", type=int)
    finish.add_argument("--step", type=int)
    finish.add_argument("--title")
    finish.add_argument("--elapsed", type=int, default=0)
    tile = commands.choices["tile"]
    for name in ("tile", "input_tif", "job_dir", "state", "stage"):
        tile.add_argument(name)
    tile.add_argument("exit_code", type=int)
    tile.add_argument("started_at")
    tile.add_argument("finished_at")
    tile.add_argument("--gpu-id", type=int)
    tile.add_argument("--model", required=True)
    tile.add_argument("--control-source", required=True)
    tile.add_argument("--compute-uncertainty", type=int, choices=(0, 1), required=True)
    tile.add_argument("--fusion-mode", required=True)
    tile.add_argument("--ground-filter", type=int, choices=(0, 1), required=True)
    summary = commands.add_parser("summary")
    summary.add_argument("root", type=Path)
    summary.add_argument("inputs", type=Path, nargs="+")

    args = parser.parse_args()
    if args.action == "start":
        write_json(args.path, {"run_id": args.run_id, "status": "running",
                               "started_at": now(), "stages": {}})
    elif args.action == "failed-stage":
        print(failed_stage(args.path, args.run_id))
    elif args.action == "tile":
        write_tile_status(args)
    elif args.action == "summary":
        summarize_batch(args.root, args.inputs)
    else:
        run = read_status(args.path)
        if args.action == "stage":
            set_stage(run, args.step, args.title, args.state,
                      None if args.state == "running" else 0, args.elapsed)
        else:
            run.update(status="failed" if args.exit_code else "success",
                       exit_code=args.exit_code, finished_at=now())
            if args.exit_code and args.step is not None:
                set_stage(run, args.step, args.title, "failed", args.exit_code, args.elapsed)
        write_json(args.path, run)


if __name__ == "__main__":
    main()
