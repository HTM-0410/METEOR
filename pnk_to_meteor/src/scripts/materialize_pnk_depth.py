#!/usr/bin/env python3
"""Create sparse metric-depth targets for rectified PNK Layer-2 samples."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import laspy
import numpy as np

from scripts.materialize_pnk_layer2 import METEOR_CAMERAS
from scripts.prepare_pnk_comet import atomic_json


DEFAULT_ROOT = Path(r"D:\Backup_rosbag\PNKData_layer2_samples_v1")
DEFAULT_HANDOFF = Path(r"D:\Backup_rosbag\PNKData_comet_handoff_v2")
DEPTH_HW = (108, 192)
MAX_DEPTH_M = 80.0


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


def project_sparse_depth(points_ego: np.ndarray, camera: dict,
                         depth_hw: tuple[int, int] = DEPTH_HW) -> np.ndarray:
    height, width = depth_hw
    K = np.asarray(camera["K"], np.float64).copy()
    source_h, source_w = camera.get("source_hw", [432, 768])
    # Layer-2 rectified K is expressed in 768x432 pixels.
    K[0, :] *= width / float(source_w)
    K[1, :] *= height / float(source_h)
    T_cam_ego = np.linalg.inv(np.asarray(camera["T_ego_cam"], np.float64))
    points_cam = points_ego @ T_cam_ego[:3, :3].T + T_cam_ego[:3, 3]
    z = points_cam[:, 2]
    usable = np.isfinite(points_cam).all(axis=1) & (z > 0.5) & (z < MAX_DEPTH_M)
    points_cam, z = points_cam[usable], z[usable]
    depth = np.full(height * width, np.inf, np.float32)
    if len(z):
        u = np.floor(K[0, 0] * points_cam[:, 0] / z + K[0, 2]).astype(np.int32)
        v = np.floor(K[1, 1] * points_cam[:, 1] / z + K[1, 2]).astype(np.int32)
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        np.minimum.at(depth, v[inside] * width + u[inside], z[inside].astype(np.float32))
    depth[~np.isfinite(depth)] = 0.0
    return depth.reshape(height, width)


def validate_depth(path: Path, cameras: int) -> tuple[int, list[int]]:
    with np.load(path, allow_pickle=False) as value:
        depth = value["depth"]
        if depth.shape != (cameras, *DEPTH_HW) or not np.isfinite(depth).all():
            raise ValueError(f"Invalid depth target: {path}")
        if np.any(depth < 0) or np.any(depth > MAX_DEPTH_M):
            raise ValueError(f"Depth outside metric range: {path}")
        counts = np.count_nonzero(depth, axis=(1, 2)).astype(int).tolist()
        return int(np.sum(counts)), counts


def process_scene(job: tuple[str, str, str]) -> dict:
    layer2_manifest_raw, handoff_manifest_raw, sensor_root_raw = job
    layer2_path = Path(layer2_manifest_raw)
    handoff_path = Path(handoff_manifest_raw)
    sensor_root = Path(sensor_root_raw)
    manifest = json.loads(layer2_path.read_text(encoding="utf-8"))
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    source_by_frame = {int(frame["frame"]): frame for frame in handoff["frames"]}
    counts: Counter = Counter()
    per_camera = np.zeros(len(METEOR_CAMERAS), np.int64)
    for frame in manifest["frames"]:
        index = int(frame["frame"])
        source = source_by_frame[index]
        if int(source["timestamp_ns"]) != int(frame["timestamp_ns"]):
            raise ValueError(f"Timestamp mismatch: {manifest['scene']}/{index}")
        rel6 = Path("depth4") / f"{index:06d}.npz"
        rel2 = Path("depth4n") / f"{index:06d}.npz"
        path6, path2 = layer2_path.parent / rel6, layer2_path.parent / rel2
        if path6.exists() and path2.exists():
            valid6, count6 = validate_depth(path6, 6)
            valid2, count2 = validate_depth(path2, 2)
            counts["reused"] += 1
            valid_total = valid6 + valid2
            camera_counts = count6 + count2
        else:
            cloud = laspy.read(sensor_root / source["lidar_merged_ego"])
            points = np.column_stack((cloud.x, cloud.y, cloud.z)).astype(np.float64)
            points = points[np.isfinite(points).all(axis=1)]
            depth = np.stack([project_sparse_depth(points, manifest["cams"][camera])
                              for camera in METEOR_CAMERAS])
            camera_counts = np.count_nonzero(depth, axis=(1, 2)).astype(int).tolist()
            valid_total = int(np.sum(camera_counts))
            atomic_npz(path6, depth=depth[:6].astype(np.float16),
                       valid_count=np.asarray(camera_counts[:6], np.int32),
                       max_depth_m=np.asarray(MAX_DEPTH_M, np.float32))
            atomic_npz(path2, depth=depth[6:].astype(np.float16),
                       valid_count=np.asarray(camera_counts[6:], np.int32),
                       max_depth_m=np.asarray(MAX_DEPTH_M, np.float32))
            counts["created"] += 1
            counts["lidar_points_decoded"] += len(points)
        per_camera += np.asarray(camera_counts)
        frame["depth4"] = rel6.as_posix()
        frame["depth4n"] = rel2.as_posix()
        frame["depth_valid_pixels"] = valid_total
        frame["supervision_valid"]["metric_depth"] = valid_total > 0
        counts["frames"] += 1
        counts["valid_pixels"] += valid_total
    manifest["metric_depth"] = {
        "kind": "sparse_lidar_projection_candidate",
        "shape": [8, *DEPTH_HW], "units": "metres", "invalid_value": 0,
        "max_depth_m": MAX_DEPTH_M,
        "image_geometry": "rectified_pinhole",
        "lidar_source": "LIDAR_MERGED_EGO_single_sweep",
        "deskew_status": "unverified",
        "projection_status": "candidate_requires_visual_overlay_QA",
    }
    allowed = manifest["training_contract"]["allowed_now"]
    if "metric_depth_candidate" not in allowed:
        allowed.append("metric_depth_candidate")
    atomic_json(layer2_path, manifest)
    return {"scene": manifest["scene"], "counts": dict(counts),
            "valid_pixels_by_camera": per_camera.tolist()}


def materialize(root: Path, handoff: Path, workers: int = 4,
                scenes: set[str] | None = None) -> dict:
    root, handoff = root.resolve(), handoff.resolve()
    dataset_path = root / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    source_dataset = json.loads((handoff / "dataset.json").read_text(encoding="utf-8"))
    if dataset["schema"] != "pnk-layer2-v1" or source_dataset["schema"] != "pnk-comet-handoff-v1":
        raise ValueError("Expected PNK Layer-2 and handoff datasets")
    sensor_root = Path(source_dataset["source_sensor_root"])
    jobs = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        if scenes and manifest_path.parent.name not in scenes:
            continue
        jobs.append((str(manifest_path),
                     str(handoff / "scenes" / manifest_path.parent.name / "manifest.json"),
                     str(sensor_root)))
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(process_scene, jobs):
            results.append(result)
            print(f"[depth] {result['scene']}: {result['counts']['frames']} frames", flush=True)
    totals: Counter = Counter()
    per_camera = np.zeros(len(METEOR_CAMERAS), np.int64)
    for result in results:
        totals.update(result["counts"])
        per_camera += np.asarray(result["valid_pixels_by_camera"])
    task = dataset["task_availability"]
    task["2_metric_depth"] = "CANDIDATE_sparse_lidar_projection_overlay_QA_pending"
    dataset["safe_loader_recipe_after_gate"]["with_depth"] = True
    dataset["metric_depth"] = {
        "status": "CANDIDATE", "shape": [8, *DEPTH_HW],
        "valid_pixels_by_camera": per_camera.tolist(),
        "deskew_status": "unverified", "projection_QA": "pending",
    }
    atomic_json(dataset_path, dataset)
    report = {"status": "DEPTH_CANDIDATE_MATERIALIZED", "counts": dict(totals),
              "valid_pixels_by_camera": per_camera.tolist(),
              "training_gate": "HOLD_PROJECTION_OVERLAY_AND_DESKEW_QA"}
    atomic_json(root / "depth_report.json", report)
    atomic_json(root / "task_availability.json", task)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scenes", help="Comma-separated scene names")
    args = parser.parse_args()
    selected = set(args.scenes.split(",")) if args.scenes else None
    print(json.dumps(materialize(args.root, args.handoff, args.workers, selected), indent=2))


if __name__ == "__main__":
    main()
