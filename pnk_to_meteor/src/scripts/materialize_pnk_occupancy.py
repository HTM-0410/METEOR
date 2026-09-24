#!/usr/bin/env python3
"""Create conservative single-sweep semantic occupancy for PNK Layer 2."""
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
VOXEL_M = 0.4
X_HALF = Y_HALF = 40.0
Z_MIN, Z_MAX = -1.0, 5.4
GRID_Z, GRID_X, GRID_Y = 16, 200, 200
OCC_CLASSES = 10
RAY_DECIMATION = 20
RAY_STEPS = 128

# METEOR seg2d21 -> occupancy class; 0 means no semantic assignment.
SEG21_TO_OCC = np.asarray([0, 1, 2, 2, 2, 3, 3, 4, 5, 9, 9,
                           5, 6, 5, 5, 0, 8, 8, 7, 0, 9], np.uint8)
OCC_PRIORITY = np.asarray([0, 3, 5, 6, 7, 1, 1, 2, 2, 4], np.int8)


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


def camera_projection(camera: dict, target_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    target_h, target_w = target_hw
    source_h, source_w = camera.get("source_hw", [432, 768])
    K = np.asarray(camera["K"], np.float64).copy()
    K[0, :] *= target_w / float(source_w)
    K[1, :] *= target_h / float(source_h)
    T_cam_ego = np.linalg.inv(np.asarray(camera["T_ego_cam"], np.float64))
    return K, T_cam_ego


def label_points(points: np.ndarray, seg21: np.ndarray, depth: np.ndarray,
                 cameras: dict) -> np.ndarray:
    labels = np.zeros(len(points), np.uint8)
    height, width = seg21.shape[1:]
    for camera_index, camera_name in enumerate(METEOR_CAMERAS):
        K, transform = camera_projection(cameras[camera_name], (height, width))
        camera_points = points @ transform[:3, :3].T + transform[:3, 3]
        z = camera_points[:, 2]
        front = z > 0.5
        indices = np.flatnonzero(front)
        if not len(indices):
            continue
        visible_points = camera_points[indices]
        visible_z = z[indices]
        u = np.floor(K[0, 0] * visible_points[:, 0] / visible_z + K[0, 2]).astype(np.int32)
        v = np.floor(K[1, 1] * visible_points[:, 1] / visible_z + K[1, 2]).astype(np.int32)
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        indices, u, v, visible_z = indices[inside], u[inside], v[inside], visible_z[inside]
        reference = depth[camera_index, v, u]
        tolerance = np.maximum(0.8, 0.03 * visible_z)
        visible = (reference > 0) & (np.abs(visible_z - reference) <= tolerance)
        indices, u, v = indices[visible], u[visible], v[visible]
        semantic = seg21[camera_index, v, u]
        mapped = np.zeros(len(semantic), np.uint8)
        valid_semantic = semantic <= 20
        mapped[valid_semantic] = SEG21_TO_OCC[semantic[valid_semantic]]
        current = labels[indices]
        replace = OCC_PRIORITY[mapped] > OCC_PRIORITY[current]
        current[replace] = mapped[replace]
        labels[indices] = current
    return labels


def voxelize_single_sweep(points: np.ndarray, labels: np.ndarray) -> np.ndarray:
    occupancy = np.full((GRID_Z, GRID_X, GRID_Y), 255, np.uint8)
    rows = ((X_HALF - points[:, 0]) / VOXEL_M).astype(np.int32)
    cols = ((Y_HALF - points[:, 1]) / VOXEL_M).astype(np.int32)
    levels = ((points[:, 2] - Z_MIN) / VOXEL_M).astype(np.int32)
    inside = ((rows >= 0) & (rows < GRID_X) & (cols >= 0) & (cols < GRID_Y)
              & (levels >= 0) & (levels < GRID_Z))
    # Unmapped returns above the road plane remain a generic observed obstacle.
    endpoint_classes = labels.copy()
    endpoint_classes[(endpoint_classes == 0) & (points[:, 2] > 0.25)] = 1
    occupied = inside & (endpoint_classes > 0)
    linear = ((levels[occupied] * GRID_X + rows[occupied]) * GRID_Y
              + cols[occupied])
    vote = np.bincount(linear * OCC_CLASSES + endpoint_classes[occupied],
                       minlength=GRID_Z * GRID_X * GRID_Y * OCC_CLASSES)
    vote = vote.reshape(-1, OCC_CLASSES)
    hit = vote.sum(1) > 0
    occupancy.reshape(-1)[hit] = vote[hit].argmax(1).astype(np.uint8)

    # Single-sweep free-space carving. Decimation reduces duplicate neighboring
    # rays; unknown cells are changed to free but occupied endpoints are kept.
    ranges = np.linalg.norm(points, axis=1)
    ray_points = points[(ranges > 2.5) & (ranges < 75.0)
                        & (points[:, 2] > Z_MIN) & (points[:, 2] < Z_MAX + 2.0)]
    ray_points = ray_points[::RAY_DECIMATION]
    ray_ranges = np.linalg.norm(ray_points, axis=1)
    for fraction in np.linspace(0.01, 0.99, RAY_STEPS, dtype=np.float32):
        usable = fraction * ray_ranges < ray_ranges - 0.6
        sample = ray_points[usable] * fraction
        rr = ((X_HALF - sample[:, 0]) / VOXEL_M).astype(np.int32)
        cc = ((Y_HALF - sample[:, 1]) / VOXEL_M).astype(np.int32)
        zz = ((sample[:, 2] - Z_MIN) / VOXEL_M).astype(np.int32)
        valid = ((rr >= 0) & (rr < GRID_X) & (cc >= 0) & (cc < GRID_Y)
                 & (zz >= 0) & (zz < GRID_Z))
        rr, cc, zz = rr[valid], cc[valid], zz[valid]
        unknown = occupancy[zz, rr, cc] == 255
        occupancy[zz[unknown], rr[unknown], cc[unknown]] = 0
    return occupancy


def process_scene(job: tuple[str, str, str]) -> dict:
    manifest_path_raw, handoff_path_raw, sensor_root_raw = job
    manifest_path, handoff_path = Path(manifest_path_raw), Path(handoff_path_raw)
    sensor_root = Path(sensor_root_raw)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    source_by_frame = {int(frame["frame"]): frame for frame in handoff["frames"]}
    counts: Counter = Counter()
    for frame in manifest["frames"]:
        fi = int(frame["frame"])
        relative = Path("occ") / f"{fi:06d}.npz"
        destination = manifest_path.parent / relative
        if destination.exists():
            with np.load(destination, allow_pickle=False) as value:
                occupancy = value["occ"]
            if occupancy.shape != (GRID_Z, GRID_X, GRID_Y):
                raise ValueError(f"Invalid occupancy target: {destination}")
            counts["reused"] += 1
        else:
            source = source_by_frame[fi]
            cloud = laspy.read(sensor_root / source["lidar_merged_ego"])
            points = np.column_stack((cloud.x, cloud.y, cloud.z)).astype(np.float64)
            points = points[np.isfinite(points).all(axis=1)]
            with np.load(manifest_path.parent / frame["seg2d21"],
                         allow_pickle=False) as value:
                seg21 = value["seg"]
            with np.load(manifest_path.parent / frame["depth4"],
                         allow_pickle=False) as value:
                depth6 = value["depth"].astype(np.float32)
            with np.load(manifest_path.parent / frame["depth4n"],
                         allow_pickle=False) as value:
                depth2 = value["depth"].astype(np.float32)
            labels = label_points(points, seg21, np.concatenate((depth6, depth2)),
                                  manifest["cams"])
            occupancy = voxelize_single_sweep(points, labels)
            atomic_npz(destination, occ=occupancy,
                       source=np.asarray("single_sweep_semantic_lidar_candidate"))
            counts["created"] += 1
        values, frequencies = np.unique(occupancy, return_counts=True)
        for value, frequency in zip(values, frequencies):
            counts[f"voxel_class_{int(value)}"] += int(frequency)
        frame["occ"] = relative.as_posix()
        frame["supervision_valid"]["occupancy3d"] = bool(np.any(occupancy != 255))
        counts["frames"] += 1
    manifest["semantic_occupancy"] = {
        "shape": [GRID_Z, GRID_X, GRID_Y], "voxel_m": VOXEL_M,
        "extent_m": {"x": [-40, 40], "y": [-40, 40], "z": [Z_MIN, Z_MAX]},
        "source": "single_sweep_LiDAR_plus_partial_seg2d21",
        "free_space": "single_sweep_ray_carving",
        "unobserved_value": 255,
        "deskew_status": "unverified",
    }
    allowed = manifest["training_contract"]["allowed_now"]
    if "occupancy3d_candidate" not in allowed:
        allowed.append("occupancy3d_candidate")
    atomic_json(manifest_path, manifest)
    return {"scene": manifest["scene"], "counts": dict(counts)}


def materialize(root: Path, handoff: Path, workers: int = 4,
                scenes: set[str] | None = None) -> dict:
    root, handoff = root.resolve(), handoff.resolve()
    dataset_path = root / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    source_dataset = json.loads((handoff / "dataset.json").read_text(encoding="utf-8"))
    sensor_root = Path(source_dataset["source_sensor_root"])
    jobs = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        if scenes and manifest_path.parent.name not in scenes:
            continue
        jobs.append((str(manifest_path),
                     str(handoff / "scenes" / manifest_path.parent.name / "manifest.json"),
                     str(sensor_root)))
    totals: Counter = Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(process_scene, jobs):
            totals.update(result["counts"])
            print(f"[occupancy] {result['scene']}: {result['counts']['frames']} frames",
                  flush=True)
    report = {
        "status": "OCCUPANCY_CANDIDATE_MATERIALIZED", "counts": dict(totals),
        "mode": "single_sweep_partial_semantics",
        "training_gate": "HOLD_DESKEW_AND_STRATIFIED_SEMANTIC_QA",
    }
    dataset["task_availability"]["8_occupancy3d"] = \
        "CANDIDATE_single_sweep_partial_semantics"
    dataset["safe_loader_recipe_after_gate"]["with_occ"] = True
    dataset["semantic_occupancy"] = report
    atomic_json(dataset_path, dataset)
    atomic_json(root / "occupancy_report.json", report)
    atomic_json(root / "task_availability.json", dataset["task_availability"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scenes")
    args = parser.parse_args()
    selected = set(args.scenes.split(",")) if args.scenes else None
    print(json.dumps(materialize(args.root, args.handoff, args.workers, selected), indent=2))


if __name__ == "__main__":
    main()
