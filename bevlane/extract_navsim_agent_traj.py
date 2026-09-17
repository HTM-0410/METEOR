#!/usr/bin/env python3
"""Materialize METEOR agent-trajectory targets from NAVSIM track tokens.

For every converted manifest frame, the current Vehicle/VRU boxes use the same
filter/order as ``ingest_navsim.meteor_boxes``.  The same ``track_token`` is
looked up at +0.5 .. +3.0 seconds without crossing a NAVSIM scene boundary.
Future local box centres are transformed through the future ego pose and back
into the *current* ego frame before offsets are stored.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from bevlane.ingest_navsim import (
    MAX_BOXES,
    _atomic_json,
    meteor_boxes_with_tracks,
    quaternion_yaw,
)


HORIZON = 6
DT_SECONDS = 0.5
TIME_TOLERANCE_SECONDS = 0.15
BACKUP_NAME = "manifest.before_agent_traj.json"


def _scene_key(frame: Dict) -> str:
    return str(frame.get("scene_token") or frame.get("scene_name") or "__log__")


def _track_centers(frame: Dict) -> Dict[str, np.ndarray]:
    annotations = frame.get("anns", {})
    boxes = np.asarray(annotations.get("gt_boxes", []), dtype=np.float64).reshape(-1, 7)
    tracks = list(annotations.get("track_tokens", []))
    if len(tracks) != len(boxes):
        return {}
    return {
        str(track): box[:2].copy()
        for track, box in zip(tracks, boxes)
        if track is not None and np.isfinite(box[:2]).all()
    }


class SceneTrackIndex:
    """Timestamp/track lookup for one NAVSIM scene, never another scene."""

    def __init__(self, frames: Sequence[Dict],
                 tolerance_seconds: float = TIME_TOLERANCE_SECONDS):
        self.frames = sorted(frames, key=lambda frame: int(frame["timestamp"]))
        self.timestamps = np.asarray(
            [int(frame["timestamp"]) for frame in self.frames], dtype=np.int64
        )
        if len(self.timestamps) > 1 and np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("scene timestamps must be strictly increasing")
        self.tolerance_us = int(round(float(tolerance_seconds) * 1e6))
        self.track_centers = [_track_centers(frame) for frame in self.frames]
        self.token_to_index = {
            str(frame["token"]): index for index, frame in enumerate(self.frames)
        }

    def _future_index(self, current_index: int, target_us: int) -> int | None:
        insertion = int(np.searchsorted(self.timestamps, target_us))
        candidates = [index for index in (insertion - 1, insertion)
                      if current_index < index < len(self.frames)]
        if not candidates:
            return None
        index = min(candidates, key=lambda value: abs(int(self.timestamps[value]) - target_us))
        return index if abs(int(self.timestamps[index]) - target_us) <= self.tolerance_us else None

    @staticmethod
    def _future_in_current_ego(current: Dict, future: Dict,
                               future_xy: np.ndarray) -> np.ndarray:
        current_translation = np.asarray(current["ego2global_translation"],
                                         dtype=np.float64)
        future_translation = np.asarray(future["ego2global_translation"],
                                        dtype=np.float64)
        current_yaw = quaternion_yaw(current["ego2global_rotation"])
        future_yaw = quaternion_yaw(future["ego2global_rotation"])

        cf, sf = np.cos(future_yaw), np.sin(future_yaw)
        global_x = future_translation[0] + cf * future_xy[0] - sf * future_xy[1]
        global_y = future_translation[1] + sf * future_xy[0] + cf * future_xy[1]
        dx, dy = global_x - current_translation[0], global_y - current_translation[1]
        cc, sc = np.cos(current_yaw), np.sin(current_yaw)
        return np.asarray([cc * dx + sc * dy, -sc * dx + cc * dy],
                          dtype=np.float32)

    def target(self, current_index: int) -> Tuple[np.ndarray, np.int64,
                                                   np.ndarray, np.ndarray]:
        current = self.frames[current_index]
        boxes, tracks = meteor_boxes_with_tracks(current)
        count = min(len(boxes), MAX_BOXES)
        padded_boxes = np.zeros((MAX_BOXES, 6), dtype=np.float32)
        trajectories = np.zeros((MAX_BOXES, HORIZON, 2), dtype=np.float32)
        valid = np.zeros((MAX_BOXES, HORIZON), dtype=np.float32)
        padded_boxes[:count] = boxes[:count]

        current_us = int(current["timestamp"])
        for horizon in range(HORIZON):
            target_us = current_us + int(round((horizon + 1) * DT_SECONDS * 1e6))
            future_index = self._future_index(current_index, target_us)
            if future_index is None:
                continue
            future = self.frames[future_index]
            centers = self.track_centers[future_index]
            for box_index, track in enumerate(tracks[:count]):
                if track is None or track not in centers:
                    continue
                future_xy = self._future_in_current_ego(
                    current, future, centers[track]
                )
                trajectories[box_index, horizon] = (
                    future_xy - padded_boxes[box_index, 1:3]
                )
                valid[box_index, horizon] = 1.0
        return padded_boxes, np.int64(count), trajectories, valid


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w+b", prefix=path.name + ".", suffix=".tmp",
        dir=path.parent, delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _valid_target_file(path: Path) -> bool:
    try:
        with np.load(path) as target:
            return (target["boxes"].shape == (MAX_BOXES, 6)
                    and np.asarray(target["count"]).shape == ()
                    and target["traj"].shape == (MAX_BOXES, HORIZON, 2)
                    and target["tvalid"].shape == (MAX_BOXES, HORIZON))
    except Exception:
        return False


def _process_log(config: Dict) -> Dict:
    output_root = Path(config["output_root"])
    scene = config["scene"]
    scene_root = output_root / scene
    manifest_path = scene_root / "manifest.json"
    log_path = (Path(config["data_root"]) / "navsim_logs" / config["split"]
                / f"{scene}.pkl")
    if not manifest_path.is_file() or not log_path.is_file():
        raise FileNotFoundError(f"Missing manifest/log for {scene}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("frames", [])
    metadata = manifest.get("navsim_agent_traj", {})
    if (not config["force"] and metadata.get("version") == 1
            and metadata.get("frames") == len(records)
            and all(record.get("agent_traj")
                    and _valid_target_file(scene_root / record["agent_traj"])
                    for record in records)):
        return {"status": "skip", "scene": scene, "frames": len(records)}

    with log_path.open("rb") as stream:
        raw_frames = pickle.load(stream)
    by_scene: Dict[str, List[Dict]] = {}
    for frame in raw_frames:
        by_scene.setdefault(_scene_key(frame), []).append(frame)

    lookup: Dict[str, Tuple[SceneTrackIndex, int]] = {}
    for frames in by_scene.values():
        index = SceneTrackIndex(frames, config["time_tolerance"])
        for token, position in index.token_to_index.items():
            if token in lookup:
                raise ValueError(f"Duplicate NAVSIM token {token} in {scene}")
            lookup[token] = (index, position)

    output_dir = scene_root / "agent_traj"
    output_dir.mkdir(exist_ok=True)
    boxes_total = 0
    valid_total = 0
    for record in records:
        token = str(record["token"])
        if token not in lookup:
            raise KeyError(f"Manifest token {token} not present in {log_path}")
        index, position = lookup[token]
        boxes, count, trajectory, valid = index.target(position)
        relative = f"agent_traj/{int(record['frame']):06d}.npz"
        target_path = scene_root / relative
        if config["force"] or not _valid_target_file(target_path):
            _atomic_npz(target_path, boxes=boxes, count=count,
                        traj=trajectory, tvalid=valid)
        record["agent_traj"] = relative
        boxes_total += int(count)
        valid_total += int(valid.sum())

    backup = scene_root / BACKUP_NAME
    if not backup.exists():
        with backup.open("xb") as stream:
            stream.write(manifest_path.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
    manifest["navsim_agent_traj"] = {
        "version": 1,
        "source": "NAVSIM anns.track_tokens",
        "coordinate_frame": "current_ego",
        "horizon": HORIZON,
        "dt_seconds": DT_SECONDS,
        "time_tolerance_seconds": config["time_tolerance"],
        "max_boxes": MAX_BOXES,
        "frames": len(records),
    }
    _atomic_json(manifest_path, manifest)
    return {
        "status": "ok", "scene": scene, "frames": len(records),
        "boxes": boxes_total, "valid_waypoints": valid_total,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="NAVSIM root containing navsim_logs/<split>")
    parser.add_argument("--out", type=Path, required=True,
                        help="existing METEOR NAVSIM conversion root")
    parser.add_argument("--split", default="mini")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-logs", type=int, default=None)
    parser.add_argument("--time-tolerance", type=float,
                        default=TIME_TOLERANCE_SECONDS)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers < 1 or not 0.0 <= args.time_tolerance < DT_SECONDS:
        raise SystemExit("--workers must be positive and time tolerance in [0, 0.5)")
    scenes_file = args.out / "scenes.txt"
    scenes = (scenes_file.read_text(encoding="utf-8").split()
              if scenes_file.is_file()
              else sorted(path.parent.name for path in args.out.glob("*/manifest.json")))
    if args.max_logs is not None:
        scenes = scenes[:args.max_logs]
    configs = [{
        "data_root": str(args.data_root), "output_root": str(args.out),
        "split": args.split, "scene": scene, "force": args.force,
        "time_tolerance": args.time_tolerance,
    } for scene in scenes]

    results: List[Dict] = []
    iterator: Iterable[Dict]
    if args.workers == 1:
        iterator = map(_process_log, configs)
        for result in iterator:
            results.append(result)
            print(json.dumps(result), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for result in executor.map(_process_log, configs):
                results.append(result)
                print(json.dumps(result), flush=True)
    print(json.dumps({
        "event": "complete", "scenes": len(results),
        "frames": sum(result.get("frames", 0) for result in results),
        "boxes": sum(result.get("boxes", 0) for result in results),
        "valid_waypoints": sum(result.get("valid_waypoints", 0) for result in results),
    }), flush=True)


if __name__ == "__main__":
    main()
