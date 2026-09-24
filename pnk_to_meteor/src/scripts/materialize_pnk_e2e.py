#!/usr/bin/env python3
"""Promote gated PNK NAV/CAN candidates into METEOR E2E targets."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.prepare_pnk_comet import atomic_json


DEFAULT_ROOT = Path(r"D:\Backup_rosbag\PNKData_layer2_samples_v1")
WHEELBASE_M = 2.8
V_MIN_STEER_MPS = 0.5


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz",
                                     dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def materialize(root: Path, wheelbase_m: float = WHEELBASE_M) -> dict:
    root = root.resolve()
    dataset_path = root / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if dataset["schema"] != "pnk-layer2-v1":
        raise ValueError("Expected PNK Layer-2 dataset")
    totals: Counter = Counter()
    steering_ratios = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidate_path = manifest_path.parent / manifest["nav_ego_candidate"]
        with np.load(candidate_path, allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}
        frame_count = 1 + max(int(frame["frame"]) for frame in manifest["frames"])
        if len(arrays["timestamp_ns"]) != len(manifest["frames"]):
            raise ValueError(f"NAV/frame count mismatch: {manifest['scene']}")
        wp = np.zeros((frame_count, 6, 2), np.float32)
        v0 = np.zeros(frame_count, np.float32)
        acc = np.zeros(frame_count, np.float32)
        steer = np.zeros(frame_count, np.float32)
        brake = np.zeros(frame_count, np.float32)
        valid = np.zeros(frame_count, np.float32)
        pose = np.zeros((frame_count, 3), np.float32)
        wheel = np.zeros(frame_count, np.float32)
        signal_valid = np.zeros(frame_count, np.uint8)
        for source_index, frame in enumerate(manifest["frames"]):
            fi = int(frame["frame"])
            if int(arrays["timestamp_ns"][source_index]) != int(frame["timestamp_ns"]):
                raise ValueError(f"NAV timestamp mismatch: {manifest['scene']}/{fi}")
            speed = float(arrays["speed_mps"][source_index])
            acceleration = float(arrays["longitudinal_accel_mps2"][source_index])
            yaw_rate = float(arrays["yaw_rate_radps"][source_index])
            finite = np.isfinite([speed, acceleration, yaw_rate]).all()
            sane = finite and 0.0 <= speed <= 60.0 and abs(acceleration) <= 10.0 \
                and abs(yaw_rate) <= 1.5
            signal_ok = bool(arrays["can_valid"][source_index]) and sane
            v0[fi], acc[fi] = speed, acceleration
            steer[fi] = (np.arctan(wheelbase_m * yaw_rate / speed)
                         if signal_ok and speed > V_MIN_STEER_MPS else 0.0)
            brake[fi] = float(signal_ok and acceleration < -0.5)
            wheel[fi] = arrays["steering_wheel_rad"][source_index]
            signal_valid[fi] = signal_ok
            nav_pose = arrays["nav_reference_pose_enu"][source_index]
            if arrays["nav_reference_valid"][source_index] and np.isfinite(nav_pose).all():
                pose[fi] = nav_pose[[0, 1, 3]].astype(np.float32)
            is_valid = bool(arrays["future_candidate_valid"][source_index]) and signal_ok
            if is_valid:
                wp[fi] = arrays["waypoints_nav_reference_m"][source_index]
                valid[fi] = 1.0
                totals["valid_frames"] += 1
            frame["supervision_valid"]["ego_motion"] = is_valid
            totals["frames"] += 1
            if signal_ok and speed > V_MIN_STEER_MPS and abs(steer[fi]) > 1e-3:
                steering_ratios.append(abs(float(wheel[fi] / steer[fi])))
        destination = manifest_path.parent / "ego_motion.npz"
        atomic_npz(destination, wp=wp, v0=v0, acc=acc, steer=steer,
                   brake=brake, valid=valid, pose=pose,
                   source_steering_wheel_rad=wheel,
                   signal_valid=signal_valid,
                   wheelbase_m=np.asarray(wheelbase_m, np.float32),
                   provenance=np.asarray("PNK_NAV_CAN_candidate_gated"))
        manifest["ego_motion"] = "ego_motion.npz"
        manifest["e2e_target"] = {
            "kind": "pseudo_target_NAV_CAN_gated",
            "waypoint_horizons_s": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            "steering_target": "road_wheel_from_bicycle_model_yaw_rate",
            "wheelbase_m": wheelbase_m,
            "source_steering_wheel": "stored_for_QA_not_training_target",
            "route_command": "unavailable",
            "lever_arm_and_NAV_status_schema": "unverified",
        }
        allowed = manifest["training_contract"]["allowed_now"]
        if "ego_motion_candidate" not in allowed:
            allowed.append("ego_motion_candidate")
        atomic_json(manifest_path, manifest)
        totals["scenes"] += 1
        print(f"[e2e] {manifest['scene']}: {int(valid.sum())}/{len(manifest['frames'])}", flush=True)
    ratio = np.asarray(steering_ratios, np.float64)
    report = {
        "status": "E2E_CANDIDATE_MATERIALIZED", "counts": dict(totals),
        "wheelbase_m": wheelbase_m,
        "steering_wheel_to_road_wheel_ratio_p10_p50_p90":
            np.quantile(ratio, [.1, .5, .9]).tolist() if ratio.size else None,
        "training_gate": "HOLD_LEVER_ARM_NAV_STATUS_AND_TARGET_QA",
    }
    dataset["task_availability"]["7_e2e"] = \
        "CANDIDATE_NAV_CAN_validity_gated_no_route_command"
    dataset["safe_loader_recipe_after_gate"]["with_ego"] = True
    dataset["e2e_target"] = report
    atomic_json(dataset_path, dataset)
    atomic_json(root / "e2e_report.json", report)
    atomic_json(root / "task_availability.json", dataset["task_availability"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--wheelbase-m", type=float, default=WHEELBASE_M)
    args = parser.parse_args()
    if args.wheelbase_m <= 0:
        parser.error("wheelbase must be positive")
    print(json.dumps(materialize(args.root, args.wheelbase_m), indent=2))


if __name__ == "__main__":
    main()
