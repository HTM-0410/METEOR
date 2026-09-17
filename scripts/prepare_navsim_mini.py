#!/usr/bin/env python3
"""End-to-end NAVSIM raw -> METEOR mini -> DataLoader pipeline.

The pipeline is resumable and intentionally reproduces the currently verified
camera/map/box/ego/command/agent-trajectory dataset.  LiDAR BEV and sparse
depth are not materialized by this standard recipe.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Dict, List


# ---------------------------------------------------------------------------
# EDIT THIS PATH, THEN RUN:
#   py -3.12 scripts\prepare_navsim_mini.py
#
# Expected raw layout below DATASET_ROOT:
#   navsim_logs/<split>/*.pkl
#   sensor_blobs/<split>/...
#   maps/...                         (unless --skip-map-gt is used)
#
# Converted METEOR data is written next to the raw dataset by default, e.g.
# D:\navsim_workspace\dataset -> D:\navsim_workspace\meteor_mini.
# Command-line flags remain available and override these defaults.
# ---------------------------------------------------------------------------
DATASET_ROOT = Path(r"D:\navsim_workspace\dataset")
OUTPUT_ROOT = DATASET_ROOT.parent / "meteor_mini"
DATASET_SPLIT = "mini"

CONVERSION_WORKERS = 4
# A single NAVSIM sample contains eight 432x768 float32 images (~30.4 MiB)
# before the remaining targets and collate copies.  Multiple Windows spawn
# workers can exhaust RAM even for a short smoke test, so the zero-argument
# recipe deliberately verifies in-process.  Override these through the CLI
# only after measuring available memory on the target machine.
DATALOADER_WORKERS = 0
DATALOADER_BATCH_SIZE = 1
DATALOADER_PREFETCH_FACTOR = 1
DATALOADER_SMOKE_FRAMES = 16


def _require_directory(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_dir():
        raise SystemExit(f"Missing {label}: {path}")
    return path


def preflight(data_root: Path, output_root: Path, split: str,
              require_maps: bool = True) -> Dict[str, object]:
    data_root = _require_directory(data_root, "NAVSIM data root")
    logs = _require_directory(data_root / "navsim_logs" / split,
                              f"navsim_logs/{split}")
    sensors = _require_directory(data_root / "sensor_blobs" / split,
                                 f"sensor_blobs/{split}")
    maps_path = data_root / "maps"
    maps = (_require_directory(maps_path, "nuPlan maps")
            if require_maps else maps_path.resolve())
    pickle_logs = sorted(logs.glob("*.pkl"))
    if not pickle_logs:
        raise SystemExit(f"No .pkl logs under {logs}")

    output_root = output_root.resolve()
    if output_root == data_root or output_root == sensors:
        raise SystemExit("Output root must differ from raw data/sensor roots")
    if output_root.exists() and not output_root.is_dir():
        raise SystemExit(f"Output root is not a directory: {output_root}")
    return {
        "data_root": data_root,
        "output_root": output_root,
        "logs": logs,
        "sensors": sensors,
        "maps": maps,
        "pickle_logs": len(pickle_logs),
    }


def run_step(name: str, command: List[str], repo_root: Path,
             dry_run: bool = False) -> None:
    print(json.dumps({"event": "step", "name": name, "command": command},
                     ensure_ascii=False), flush=True)
    if not dry_run:
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        subprocess.run(command, cwd=repo_root, check=True, env=environment)


def cleanup_empty_gt(output_root: Path) -> int:
    """Remove only verified-empty direct ``<scene>/gt`` directories."""
    output_root = output_root.resolve()
    removed = 0
    for manifest in sorted(output_root.glob("*/manifest.json")):
        scene = manifest.parent.resolve()
        if scene.parent != output_root:
            raise RuntimeError(f"Manifest outside output root: {manifest}")
        target = scene / "gt"
        if not target.exists():
            continue
        if not target.is_dir() or target.is_symlink():
            raise RuntimeError(f"Refusing unexpected gt target: {target}")
        if any(target.iterdir()):
            raise RuntimeError(f"Refusing non-empty gt directory: {target}")
        target.rmdir()
        removed += 1
    return removed


def _scene_file(scene: Path, relative: str) -> Path:
    path = PurePosixPath(str(relative).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe manifest path {relative!r} in {scene}")
    return scene.joinpath(*path.parts)


def verify_processed(output_root: Path, expected_image_root: Path,
                     require_map_gt: bool = True) -> Dict[str, int]:
    """Verify every processed reference without decoding camera JPEGs."""
    manifests = sorted(output_root.resolve().glob("*/manifest.json"))
    if not manifests:
        raise RuntimeError(f"No manifests under {output_root}")
    frames = 0
    missing = 0
    command_frames = 0
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        image_root = Path(manifest.get("image_root", "")).resolve()
        if image_root != expected_image_root.resolve():
            raise RuntimeError(
                f"Wrong image_root in {manifest_path}: {image_root}"
            )
        scene = manifest_path.parent
        records = manifest.get("frames", [])
        metadata = manifest.get("navsim_agent_traj", {})
        if metadata.get("frames") != len(records):
            raise RuntimeError(f"Incomplete agent trajectory metadata: {manifest_path}")
        for record in records:
            frames += 1
            command_frames += int("driving_command" in record)
            if require_map_gt and record.get("gt") != record.get("gt_map"):
                raise RuntimeError(
                    f"Default gt is not promoted at {manifest_path}, "
                    f"frame {record.get('frame')}"
                )
            for key in ("gt", "bev_box_p", "agent_traj"):
                relative = record.get(key)
                if not relative or not _scene_file(scene, relative).is_file():
                    missing += 1
    if missing:
        raise RuntimeError(f"Missing {missing} processed frame artifacts")
    if command_frames != frames:
        raise RuntimeError(
            f"driving_command coverage {command_frames}/{frames}"
        )
    return {
        "scenes": len(manifests), "frames": frames,
        "commands": command_frames, "missing": missing,
    }


def map_gt_already_promoted(output_root: Path) -> bool:
    """Cheap resume check; the first promotion still performs its deep PNG audit."""
    manifests = sorted(output_root.resolve().glob("*/manifest.json"))
    if not manifests:
        return False
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest.get("frames", [])
        metadata = manifest.get("map_gt", {})
        if (not records
                or metadata.get("frames_rasterized") != len(records)
                or metadata.get("promoted_to_default_gt") is not True
                or any(record.get("gt") != record.get("gt_map")
                       for record in records)):
            return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=DATASET_ROOT,
        help=f"raw NAVSIM root (default from DATASET_ROOT: {DATASET_ROOT})",
    )
    parser.add_argument(
        "--out", type=Path, default=OUTPUT_ROOT,
        help=f"converted METEOR root (default from OUTPUT_ROOT: {OUTPUT_ROOT})",
    )
    parser.add_argument("--split", default=DATASET_SPLIT)
    parser.add_argument("--workers", type=int, default=CONVERSION_WORKERS,
                        help="workers for conversion/map/trajectory stages")
    parser.add_argument("--io-workers", type=int, default=DATALOADER_WORKERS,
                        help="DataLoader verification workers")
    parser.add_argument("--io-batch-size", type=int,
                        default=DATALOADER_BATCH_SIZE)
    parser.add_argument("--io-prefetch-factor", type=int,
                        default=DATALOADER_PREFETCH_FACTOR)
    parser.add_argument("--smoke-frames", type=int,
                        default=DATALOADER_SMOKE_FRAMES,
                        help="DataLoader smoke size; ignored by --full-io")
    parser.add_argument("--full-io", action="store_true",
                        help="read all converted frames through DataLoader")
    parser.add_argument("--skip-map-gt", action="store_true",
                        help="run without nuPlan maps; keep all-255 gt/ignore.png "
                             "and provide no lane/road segmentation supervision")
    parser.add_argument("--dry-run", action="store_true",
                        help="preflight and print subprocess commands only")
    return parser


def main() -> None:
    # Windows often inherits CP1252 even when repository/data paths contain
    # Vietnamese characters. Keep orchestration and every JSON status line
    # printable without requiring callers to preconfigure their shell.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    if (args.workers < 1 or args.io_workers < 0 or args.io_batch_size < 1
            or args.io_prefetch_factor < 1 or args.smoke_frames < 1):
        raise SystemExit("Invalid worker/batch/prefetch/smoke arguments")

    repo_root = Path(__file__).resolve().parents[1]
    state = preflight(args.data_root, args.out, args.split,
                      require_maps=not args.skip_map_gt)
    data_root = state["data_root"]
    output_root = state["output_root"]
    python = sys.executable
    print(json.dumps({
        "event": "preflight", "repo_root": str(repo_root),
        "data_root": str(data_root), "output_root": str(output_root),
        "split": args.split, "pickle_logs": state["pickle_logs"],
        "lidar_bev": False, "sparse_depth": False,
        "map_gt": not args.skip_map_gt,
    }, ensure_ascii=False), flush=True)

    steps = [
        ("ingest", [
            python, "-m", "bevlane.ingest_navsim",
            "--data-root", str(data_root), "--out", str(output_root),
            "--split", args.split, "--workers", str(args.workers), "--verify",
        ]),
    ]
    if args.skip_map_gt:
        print(json.dumps({
            "event": "skip", "name": "map_gt_and_promotion",
            "reason": "--skip-map-gt; gt remains all-255 ignore",
        }), flush=True)
    else:
        steps.append(("map_gt", [
            python, "-m", "bevlane.rasterize_navsim_map",
            "--data-root", str(data_root), "--out", str(output_root),
            "--split", args.split, "--workers", str(args.workers),
        ]))
        if not map_gt_already_promoted(output_root) or args.dry_run:
            steps.extend([
                ("promote_preflight", [
                    python, "-m", "bevlane.promote_navsim_map_gt",
                    "--root", str(output_root),
                ]),
                ("promote_apply", [
                    python, "-m", "bevlane.promote_navsim_map_gt",
                    "--root", str(output_root), "--apply",
                ]),
            ])
        else:
            print(json.dumps({"event": "skip", "name": "promotion",
                              "reason": "all manifests already promoted"}), flush=True)
    steps.append(("agent_trajectory", [
            python, "-m", "bevlane.extract_navsim_agent_traj",
            "--data-root", str(data_root), "--out", str(output_root),
            "--split", args.split, "--workers", str(args.workers),
        ]))
    for name, command in steps:
        run_step(name, command, repo_root, args.dry_run)

    if args.dry_run:
        print(json.dumps({"event": "dry_run_complete"}), flush=True)
        return

    # Without map GT, gt/ignore.png is a required DataLoader placeholder and
    # must not be removed. With maps, promotion deletes it and leaves gt empty.
    removed = 0 if args.skip_map_gt else cleanup_empty_gt(output_root)
    summary = verify_processed(output_root, state["sensors"],
                               require_map_gt=not args.skip_map_gt)
    print(json.dumps({"event": "processed_verify", **summary,
                      "empty_gt_removed": removed}), flush=True)

    io_command = [
        python, "-m", "scripts.navsim_full_io",
        "--root", str(output_root),
        "--workers", str(args.io_workers),
        "--batch-size", str(args.io_batch_size),
        "--prefetch-factor", str(args.io_prefetch_factor),
        "--with-agenttraj", "--with-command",
    ]
    if not args.full_io:
        io_command.extend(("--limit", str(args.smoke_frames)))
    run_step("dataloader_full_io" if args.full_io else "dataloader_smoke",
             io_command, repo_root)
    print(json.dumps({"event": "pipeline_complete", **summary,
                      "full_io": bool(args.full_io),
                      "map_gt": not args.skip_map_gt}), flush=True)


if __name__ == "__main__":
    main()
