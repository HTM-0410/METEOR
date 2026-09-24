#!/usr/bin/env python3
"""Create gated PNK agent-forecast targets from source 3D track IDs."""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.prepare_pnk_comet import atomic_json


DEFAULT_ROOT = Path(r"D:\Backup_rosbag\PNKData_layer2_samples_v1")
DEFAULT_HANDOFF = Path(r"D:\Backup_rosbag\PNKData_comet_handoff_v2")
HORIZONS_S = np.arange(0.5, 3.01, 0.5, dtype=np.float64)
KMAX = 320
BRACKET_TOLERANCE_NS = 150_000_000
MAX_GAP_NS = 300_000_000
MAX_SPEED_MPS = {"Car": 60.0, "Rider": 35.0, "Pedestrian": 15.0}


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz",
                                     dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def ego_to_global(xy: np.ndarray, pose: np.ndarray) -> np.ndarray:
    c, s = math.cos(float(pose[2])), math.sin(float(pose[2]))
    return np.asarray([pose[0] + c * xy[0] - s * xy[1],
                       pose[1] + s * xy[0] + c * xy[1]], np.float64)


def global_to_ego(xy: np.ndarray, pose: np.ndarray) -> np.ndarray:
    dx, dy = float(xy[0] - pose[0]), float(xy[1] - pose[1])
    c, s = math.cos(float(pose[2])), math.sin(float(pose[2]))
    return np.asarray([c * dx + s * dy, -s * dx + c * dy], np.float64)


def load_frame_boxes(frame: dict, path: Path | None = None,
                     kmax: int = KMAX) -> dict:
    source_path = path if path is not None else Path(frame["boxes_3d_npz"])
    with np.load(source_path, allow_pickle=False) as value:
        boxes = value["meteor_boxes"].astype(np.float32)
        track_ids = value["track_id"].astype(np.int64)
        classes = value["source_class"].astype(str)
    if len(boxes) != len(track_ids) or len(boxes) > kmax:
        raise ValueError(f"Invalid source boxes: {source_path}")
    frequencies = Counter(track_ids.tolist())
    duplicate = {track_id for track_id, count in frequencies.items() if count > 1}
    by_track = {int(track_id): (str(cls), box)
                for track_id, cls, box in zip(track_ids, classes, boxes)
                if int(track_id) not in duplicate}
    return {"boxes": boxes, "track_ids": track_ids, "classes": classes,
            "by_track": by_track, "duplicate_ids": duplicate}


def track_global(frame_data: list[dict], index: int, track_id: int,
                 expected_class: str, poses: np.ndarray,
                 pose_valid: np.ndarray) -> np.ndarray | None:
    if not pose_valid[index]:
        return None
    item = frame_data[index]["by_track"].get(track_id)
    if item is None or item[0] != expected_class:
        return None
    return ego_to_global(item[1][1:3], poses[index])


def future_position(frame_data: list[dict], timestamps: np.ndarray,
                    poses: np.ndarray, pose_valid: np.ndarray,
                    current_index: int, track_id: int, expected_class: str,
                    target_ns: int, counters: Counter) -> np.ndarray | None:
    hi = int(np.searchsorted(timestamps, target_ns, side="left"))
    if hi >= len(timestamps):
        counters["drop_scene_end"] += 1
        return None
    if timestamps[hi] == target_ns:
        lo = hi
    else:
        lo = hi - 1
    left_error = target_ns - timestamps[lo]
    right_error = timestamps[hi] - target_ns
    # At nominal 5 Hz, a requested 1.0 s target can lie a few microseconds
    # before the fifth frame because sensor timestamps jitter. Requiring BOTH
    # bracket endpoints to be within 150 ms wrongly rejects that target: the
    # lower endpoint is then ~200 ms away while the upper is nearly exact.
    # Require a close nearest frame and a bounded interpolation interval.
    if lo < current_index or min(left_error, right_error) > BRACKET_TOLERANCE_NS \
            or timestamps[hi] - timestamps[lo] > MAX_GAP_NS:
        counters["drop_timestamp_bracket"] += 1
        return None
    if np.any(np.diff(timestamps[current_index:hi + 1]) > MAX_GAP_NS):
        counters["drop_frame_gap"] += 1
        return None
    positions = []
    previous = None
    previous_time = None
    speed_limit = MAX_SPEED_MPS.get(expected_class, 60.0)
    for frame_index in range(current_index, hi + 1):
        position = track_global(frame_data, frame_index, track_id,
                                expected_class, poses, pose_valid)
        if position is None:
            counters["drop_track_gap_or_class_switch"] += 1
            return None
        if previous is not None:
            dt = (timestamps[frame_index] - previous_time) / 1e9
            if dt <= 0 or np.linalg.norm(position - previous) / dt > speed_limit:
                counters["drop_implausible_speed"] += 1
                return None
        positions.append(position)
        previous, previous_time = position, timestamps[frame_index]
    if lo == hi:
        return positions[-1]
    lower = positions[lo - current_index]
    upper = positions[hi - current_index]
    weight = (target_ns - timestamps[lo]) / float(timestamps[hi] - timestamps[lo])
    return lower * (1.0 - weight) + upper * weight


def materialize(root: Path, handoff: Path, box_source: str = "handoff",
                kmax: int = KMAX) -> dict:
    root, handoff = root.resolve(), handoff.resolve()
    dataset_path = root / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if dataset["schema"] not in {"pnk-layer2-v1", "pnk-layer2-9head-v1",
                                 "pnk-layer2-9head-v2", "pnk-layer2-9head-v3",
                                 "pnk-layer2-9head-v4"}:
        raise ValueError("Expected PNK Layer-2 dataset")
    if box_source not in {"handoff", "root"}:
        raise ValueError("box_source must be 'handoff' or 'root'")
    totals: Counter = Counter()
    horizon_counts = np.zeros(len(HORIZONS_S), np.int64)
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_path = handoff / "scenes" / manifest["scene"] / "manifest.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source_by_frame = {int(frame["frame"]): frame for frame in source["frames"]}
        with np.load(manifest_path.parent / manifest["nav_ego_candidate"],
                     allow_pickle=False) as nav:
            timestamps_nav = nav["timestamp_ns"].astype(np.int64)
            raw_pose = nav["nav_reference_pose_enu"].astype(np.float64)
            pose_valid = nav["nav_reference_valid"].astype(bool)
        ordered_source = [source_by_frame[int(frame["frame"])]
                          for frame in manifest["frames"]]
        timestamps = np.asarray([int(frame["timestamp_ns"])
                                 for frame in ordered_source], np.int64)
        if not np.array_equal(timestamps, timestamps_nav):
            raise ValueError(f"NAV/source timestamp mismatch: {manifest['scene']}")
        poses = raw_pose[:, [0, 1, 3]]
        # Full source tracks remain the temporal interpolation reference.  A
        # filtered profile can restrict CURRENT supervised agents while still
        # following that trusted track through a temporarily sparse future box.
        frame_data = [load_frame_boxes(frame, kmax=KMAX) for frame in ordered_source]
        if box_source == "root":
            current_data = [load_frame_boxes(
                source_frame,
                path=manifest_path.parent / frame["bev_box_p"], kmax=kmax)
                for frame, source_frame in zip(manifest["frames"], ordered_source)]
        else:
            current_data = frame_data
        scene_valid = 0
        for current_index, (frame, source_frame, current) in enumerate(
                zip(manifest["frames"], ordered_source, current_data)):
            fi = int(frame["frame"])
            count = len(current["boxes"])
            boxes = np.zeros((kmax, 6), np.float32)
            track_ids = np.full(kmax, -1, np.int64)
            traj = np.zeros((kmax, len(HORIZONS_S), 2), np.float32)
            tvalid = np.zeros((kmax, len(HORIZONS_S)), np.float32)
            boxes[:count] = current["boxes"]
            track_ids[:count] = current["track_ids"]
            if pose_valid[current_index]:
                current_pose = poses[current_index]
                for box_index, (track_id, expected_class, box) in enumerate(zip(
                        current["track_ids"], current["classes"], current["boxes"])):
                    if int(track_id) in current["duplicate_ids"]:
                        totals["drop_duplicate_track_id"] += len(HORIZONS_S)
                        continue
                    for horizon_index, horizon_s in enumerate(HORIZONS_S):
                        target_ns = int(timestamps[current_index]
                                        + round(float(horizon_s) * 1e9))
                        global_position = future_position(
                            frame_data, timestamps, poses, pose_valid,
                            current_index, int(track_id), str(expected_class),
                            target_ns, totals)
                        if global_position is None:
                            continue
                        future_ego = global_to_ego(global_position, current_pose)
                        offset = future_ego - box[1:3]
                        if not np.isfinite(offset).all():
                            totals["drop_nonfinite"] += 1
                            continue
                        traj[box_index, horizon_index] = offset
                        tvalid[box_index, horizon_index] = 1.0
                        horizon_counts[horizon_index] += 1
            relative = Path("agent_traj") / f"{fi:06d}.npz"
            atomic_npz(manifest_path.parent / relative, boxes=boxes,
                       count=np.asarray(count, np.int64), traj=traj,
                       tvalid=tvalid, track_id=track_ids,
                       horizon_s=HORIZONS_S.astype(np.float32),
                       provenance=np.asarray(
                           "PNK_tierA_current_box_full_track_NAV_ego_compensated"
                           if box_source == "root" else
                           "PNK_candidate_track_id_NAV_ego_compensated"))
            frame["agent_traj"] = relative.as_posix()
            has_target = bool(tvalid.any())
            frame["supervision_valid"]["agent_forecast"] = has_target
            if has_target:
                scene_valid += 1
                totals["frames_with_forecast"] += 1
            totals["boxes"] += count
            totals["valid_future_points"] += int(tvalid.sum())
            totals["frames"] += 1
        manifest["agent_forecast"] = {
            "kind": "candidate_track_ID_NAV_ego_compensated",
            "shape": [kmax, len(HORIZONS_S), 2],
            "horizons_s": HORIZONS_S.tolist(),
            "continuous_track_required": True,
            "max_bracket_error_ms": BRACKET_TOLERANCE_NS / 1e6,
            "track_schema_status": "candidate_validated_statistically_not_source_confirmed",
            "current_box_source": box_source,
        }
        allowed = manifest["training_contract"]["allowed_now"]
        if "agent_forecast_candidate" not in allowed:
            allowed.append("agent_forecast_candidate")
        atomic_json(manifest_path, manifest)
        totals["scenes"] += 1
        print(f"[agent] {manifest['scene']}: {scene_valid}/{len(manifest['frames'])}", flush=True)
    report = {
        "status": "AGENT_FORECAST_CANDIDATE_MATERIALIZED",
        "counts": dict(totals),
        "valid_future_points_by_horizon": {
            f"{horizon:.1f}s": int(value)
            for horizon, value in zip(HORIZONS_S, horizon_counts)},
        "track_status": "candidate_statistical_QA_passed_source_schema_unconfirmed",
        "training_gate": "HOLD_TRACK_ID_REVIEW_AND_FORECAST_VISUAL_QA",
    }
    dataset["task_availability"]["10_agent_forecast"] = \
        "CANDIDATE_continuous_track_NAV_compensated_QA_pending"
    dataset["safe_loader_recipe_after_gate"]["with_agenttraj"] = True
    dataset["agent_forecast"] = report
    atomic_json(dataset_path, dataset)
    atomic_json(root / "agent_forecast_report.json", report)
    atomic_json(root / "task_availability.json", dataset["task_availability"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--box-source", choices=("handoff", "root"),
                        default="handoff")
    parser.add_argument("--kmax", type=int, default=KMAX)
    args = parser.parse_args()
    print(json.dumps(materialize(args.root, args.handoff,
                                 args.box_source, args.kmax), indent=2))


if __name__ == "__main__":
    main()
