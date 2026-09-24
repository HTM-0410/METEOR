#!/usr/bin/env python3
"""Materialize reviewable occupancy-flow targets from PNK agent futures."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from scripts.prepare_pnk_comet import atomic_json


DEFAULT_ROOT = Path(r"D:\Backup_rosbag\PNKData_layer2_samples_v1")
FLOW_HW = (200, 200)
RESOLUTION_M = 0.4


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


def build_flow_target(boxes: np.ndarray, count: int, traj: np.ndarray,
                      tvalid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = FLOW_HW
    flow = np.zeros((2, height, width), np.float32)
    valid = np.zeros((height, width), np.uint8)
    for index in range(int(count)):
        cls, xe, ye, length, box_width, yaw = boxes[index]
        # Missing future is unknown motion, not a zero-velocity target.
        if length <= 0 or abs(xe) > 42 or abs(ye) > 42 or tvalid[index, 0] <= 0.5:
            continue
        vx, vy = traj[index, 0] / 0.5
        c, s = np.cos(yaw), np.sin(yaw)
        polygon = []
        for local_x, local_y in ((length / 2, box_width / 2),
                                 (length / 2, -box_width / 2),
                                 (-length / 2, -box_width / 2),
                                 (-length / 2, box_width / 2)):
            px = xe + local_x * c - local_y * s
            py = ye + local_x * s + local_y * c
            polygon.append([int((40 - py) / RESOLUTION_M),
                            int((40 - px) / RESOLUTION_M)])
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [np.asarray(polygon, np.int32).reshape(-1, 1, 2)], 1)
        flow[0, mask > 0] = vx
        flow[1, mask > 0] = vy
        valid[mask > 0] = 1
    return flow, valid


def materialize(root: Path) -> dict:
    root = root.resolve()
    dataset_path = root / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    totals: Counter = Counter()
    speeds = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for frame in manifest["frames"]:
            with np.load(manifest_path.parent / frame["agent_traj"],
                         allow_pickle=False) as agent:
                boxes, count = agent["boxes"], int(agent["count"])
                traj, tvalid = agent["traj"], agent["tvalid"]
            flow, valid = build_flow_target(boxes, count, traj, tvalid)
            fi = int(frame["frame"])
            relative = Path("flow_target") / f"{fi:06d}.npz"
            atomic_npz(manifest_path.parent / relative,
                       flow=flow.astype(np.float16), valid=valid,
                       horizon_s=np.asarray(0.5, np.float32),
                       units=np.asarray("m/s"))
            frame["flow_target"] = relative.as_posix()
            frame["supervision_valid"]["occupancy_flow"] = bool(valid.any())
            totals["frames"] += 1
            totals["frames_with_flow"] += int(bool(valid.any()))
            totals["valid_cells"] += int(valid.sum())
            speed = np.linalg.norm(flow, axis=0)
            speeds.extend(speed[valid > 0].tolist())
        manifest["occupancy_flow"] = {
            "shape": [2, *FLOW_HW], "units": "m/s",
            "horizon_s": 0.5,
            "source": "agent_traj_first_horizon_box_footprint",
            "missing_future_policy": "ignore_not_zero_velocity",
        }
        allowed = manifest["training_contract"]["allowed_now"]
        if "occupancy_flow_candidate" not in allowed:
            allowed.append("occupancy_flow_candidate")
        atomic_json(manifest_path, manifest)
        totals["scenes"] += 1
    speed_array = np.asarray(speeds, np.float32)
    report = {
        "status": "FLOW_CANDIDATE_MATERIALIZED",
        "counts": dict(totals),
        "speed_mps_p50_p90_p99_max": (np.quantile(speed_array, [.5, .9, .99, 1]).tolist()
                                        if speed_array.size else None),
        "training_implementation": "flow_loss_builds_equivalent_target_from_agent_traj",
        "training_gate": "HOLD_TRACK_SCHEMA_AND_VISUAL_QA",
    }
    dataset["task_availability"]["9_occupancy_flow"] = \
        "CANDIDATE_from_valid_0.5s_agent_motion"
    dataset["occupancy_flow"] = report
    atomic_json(dataset_path, dataset)
    atomic_json(root / "flow_report.json", report)
    atomic_json(root / "task_availability.json", dataset["task_availability"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(materialize(args.root), indent=2))


if __name__ == "__main__":
    main()
