#!/usr/bin/env python3
"""Verify every frame of a PNK CoMET handoff without needing PyTorch."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.prepare_pnk_comet import DEFAULT_OUTPUT, atomic_json


def verify(root: Path, require_lidar_bev: bool = True) -> dict:
    root = root.resolve()
    dataset = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    if dataset["schema"] != "pnk-comet-handoff-v1":
        raise ValueError("Wrong handoff schema")
    sensor_root = Path(dataset["source_sensor_root"]).resolve()
    if not sensor_root.is_dir():
        raise FileNotFoundError(sensor_root)
    counts: Counter = Counter()
    split_recordings = defaultdict(set)
    seen = set()
    max_boxes = 0
    for manifest_path in sorted((root / "scenes").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scene = manifest["scene"]
        frames = manifest["frames"]
        split_recordings[manifest["split"]].add(manifest["recording"])
        if len(manifest["cameras_raw"]) != 8 or len(frames) == 0:
            raise ValueError(f"Incomplete camera/frames: {scene}")
        with np.load(manifest_path.parent / manifest["nav_ego_candidate"]) as motion:
            if motion["timestamp_ns"].shape != (len(frames),):
                raise ValueError(f"NAV length mismatch: {scene}")
            if motion["waypoints_nav_reference_m"].shape != (len(frames), 6, 2):
                raise ValueError(f"NAV waypoint shape: {scene}")
            if not np.isfinite(motion["nav_reference_pose_enu"]).all():
                raise ValueError(f"Nonfinite NAV pose: {scene}")
            if not np.isfinite(motion["waypoints_nav_reference_m"]).all():
                raise ValueError(f"Nonfinite NAV future: {scene}")
            counts["nav_candidate_valid"] += int(np.sum(motion["nav_reference_valid"]))
            counts["future_candidate_valid"] += int(np.sum(motion["future_candidate_valid"]))
            if np.any(motion["waypoints_nav_reference_m"][motion["future_candidate_valid"] == 0]):
                raise ValueError(f"Invalid future has nonzero waypoint: {scene}")
            for i, frame in enumerate(frames):
                if frame["timestamp_ns"] != int(motion["timestamp_ns"][i]):
                    raise ValueError(f"NAV timestamp mismatch: {scene}/{i}")
        previous = -1
        for frame in frames:
            stamp = int(frame["timestamp_ns"])
            if stamp <= previous or stamp in seen:
                raise ValueError(f"Duplicate or unsorted frame: {scene}/{stamp}")
            previous = stamp
            seen.add(stamp)
            if len(frame["images_raw"]) != 8:
                raise ValueError(f"Missing camera: {scene}/{stamp}")
            for relative in list(frame["images_raw"].values()) + [frame["lidar_top"], frame["lidar_merged_ego"]]:
                path = (sensor_root / relative).resolve()
                if not path.is_relative_to(sensor_root) or not path.is_file():
                    raise FileNotFoundError(path)
            box_path = Path(frame["boxes_3d_npz"])
            if not box_path.is_file():
                raise FileNotFoundError(box_path)
            with np.load(box_path) as boxes:
                count = len(boxes["boxes"])
                if count != frame["boxes_3d_count"] or boxes["meteor_boxes"].shape != (count, 6):
                    raise ValueError(f"Box shape/count mismatch: {box_path}")
                if not np.isfinite(boxes["meteor_boxes"]).all() or np.any(boxes["boxes"][:, 3:6] <= 0):
                    raise ValueError(f"Invalid box: {box_path}")
                target_path = manifest_path.parent / frame["bev_box_p"]
                with np.load(target_path) as target:
                    if not np.array_equal(target["boxes"], boxes["meteor_boxes"]):
                        raise ValueError(f"METEOR 3D target differs: {target_path}")
                counts["bev_box_p"] += 1
            max_boxes = max(max_boxes, count)
            counts["frames_gt64_boxes"] += int(count > 64)
            counts["boxes"] += count
            if require_lidar_bev:
                relative = frame.get("lidar_bev")
                if relative is None:
                    raise ValueError(f"Missing LiDAR BEV reference: {scene}/{stamp}")
                with np.load(manifest_path.parent / relative) as raster:
                    value = raster["lb"]
                    if value.shape != (4, 400, 250) or not np.isfinite(value).all():
                        raise ValueError(f"Invalid LiDAR BEV: {scene}/{stamp}")
                counts["lidar_bev"] += 1
                span = frame.get("lidar_point_time_span_ms")
                if span is None or frame.get("lidar_single_sweep_duration_valid") != (span <= 110.0):
                    raise ValueError(f"Missing/inconsistent LiDAR duration audit: {scene}/{stamp}")
                counts["lidar_duration_over_110ms"] += int(span > 110.0)
            counts["frames"] += 1
            counts[f'{manifest["split"]}_frames'] += 1
        counts["scenes"] += 1
    if any(split_recordings[a] & split_recordings[b]
           for a in split_recordings for b in split_recordings if a < b):
        raise ValueError("Recording leakage across splits")
    for key, expected in dataset["counts"].items():
        if int(counts[key]) != int(expected):
            raise ValueError(f"Count mismatch {key}: verified={counts[key]} declared={expected}")
    loader_smoke_path = root / "loader_smoke.json"
    loader_smoke = (json.loads(loader_smoke_path.read_text(encoding="utf-8"))
                    if loader_smoke_path.is_file() else {})
    result = {
        "status": "PASS_Handoff_Integrity_Only",
        "counts": dict(counts), "max_boxes_per_frame": max_boxes,
        "splits_recording_disjoint": True,
        "comet_t4dataset_import": "NOT_RUN_NO_PNK_IMPORTER",
        "meteor_dataloader": ("PASS_IO_SHAPE_ONLY" if loader_smoke.get("status") == "PASS_IO_SHAPE_ONLY"
                             else "NOT_RUN"),
        "meteor_training_loss": "NOT_RUN_CAMERA_GEOMETRY_UNVERIFIED",
        "camera_projection": "NOT_VERIFIED",
        "nav_ego_ground_truth": "NOT_VERIFIED_LEVER_ARM_AND_STATUS_SCHEMA",
        "lidar_deskew": "NOT_VERIFIED",
    }
    atomic_json(root / "verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-lidar-bev", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, not args.skip_lidar_bev), indent=2))


if __name__ == "__main__":
    main()
