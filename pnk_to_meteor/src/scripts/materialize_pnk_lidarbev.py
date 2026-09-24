#!/usr/bin/env python3
"""Materialize the optional single-sweep LiDAR input for a PNK handoff.

This does not create occupancy GT and does not assert that source LAZ points
were deskewed. It is resumable and never changes source LAZ files.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path(r"D:\Backup_rosbag\PNKData_comet_handoff_v2")


def lidar_bev(points: np.ndarray) -> np.ndarray:
    """Match METEOR's 4x400x250 pillar raster, x forward/y left, 0.4 m."""
    height, width, resolution = 400, 250, 0.4
    output = np.zeros((4, height, width), np.float32)
    rows = ((80.0 - points[:, 0]) / resolution).astype(np.int32)
    cols = ((50.0 - points[:, 1]) / resolution).astype(np.int32)
    valid = ((rows >= 0) & (rows < height) & (cols >= 0) & (cols < width))
    rows, cols = rows[valid], cols[valid]
    z = np.clip(points[valid, 2], -1.0, 4.0)
    if not len(rows):
        return output
    flat = rows * width + cols
    count = np.bincount(flat, minlength=height * width).astype(np.float32)
    height_sum = np.bincount(flat, weights=z, minlength=height * width)
    height_max = np.full(height * width, -1.0, np.float32)
    np.maximum.at(height_max, flat, z)
    occupied = count > 0
    output[0] = np.log1p(count).reshape(height, width)
    output[1] = np.where(occupied, height_max, 0).reshape(height, width)
    output[2] = np.divide(height_sum, count, out=np.zeros_like(height_sum),
                          where=occupied).reshape(height, width)
    output[3] = occupied.reshape(height, width).astype(np.float32)
    return output


def atomic_npz(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, lb=array.astype(np.float16))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: object) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def materialize(root: Path, audit_timestamps: bool = False) -> dict:
    root = root.resolve()
    dataset_path = root / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if dataset["schema"] != "pnk-comet-handoff-v1":
        raise ValueError("Expected PNK CoMET handoff, not an arbitrary dataset")
    sensor_root = Path(dataset["source_sensor_root"]).resolve()
    created = 0
    reused = 0
    all_points = 0
    nonfinite_points = 0
    max_sweep_ms = 0.0
    spans_ms = []
    laspy_module = None
    for path in sorted((root / "scenes").glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for frame in manifest["frames"]:
            relative = Path("lidar_bev") / f'{frame["frame"]:06d}.npz'
            dest = path.parent / relative
            source = (sensor_root / frame["lidar_merged_ego"]).resolve()
            if not source.is_relative_to(sensor_root) or not source.is_file():
                raise ValueError(f"Invalid merged LiDAR path: {source}")
            if dest.exists():
                with np.load(dest) as saved:
                    if saved["lb"].shape != (4, 400, 250):
                        raise ValueError(f"Existing LiDAR BEV has wrong shape: {dest}")
                reused += 1
            if not dest.exists() or audit_timestamps:
                if laspy_module is None:
                    try:
                        import laspy as laspy_module
                    except ModuleNotFoundError as error:
                        raise ModuleNotFoundError(
                            "laspy is required only when creating LiDAR BEV files or "
                            "running --audit-timestamps; install laspy[lazrs]"
                        ) from error
                cloud = laspy_module.read(source)
                points = np.column_stack((cloud.x, cloud.y, cloud.z))
                finite = np.isfinite(points).all(axis=1)
                all_points += len(points)
                nonfinite_points += int(np.sum(~finite))
                frame["lidar_point_count"] = len(points)
                if "point_timestamp" in cloud.point_format.dimension_names:
                    stamps = np.asarray(cloud["point_timestamp"])
                    if len(stamps):
                        span_ms = (int(np.max(stamps)) - int(np.min(stamps))) / 1e6
                        frame["lidar_point_time_span_ms"] = span_ms
                if not dest.exists():
                    atomic_npz(dest, lidar_bev(points[finite]))
                    created += 1
            if "lidar_point_time_span_ms" in frame:
                span_ms = float(frame["lidar_point_time_span_ms"])
                spans_ms.append(span_ms)
                max_sweep_ms = max(max_sweep_ms, span_ms)
                # A nominal single sweep lasts about 100 ms. Longer files are
                # retained for provenance but withheld from temporal use.
                frame["lidar_single_sweep_duration_valid"] = span_ms <= 110.0
            frame["lidar_bev"] = relative.as_posix()
        manifest["lidar_bev"] = {
            "kind": "optional_model_input_not_ground_truth",
            "source": "LIDAR_MERGED_EGO_single_sweep",
            "shape": [4, 400, 250], "dtype": "float16",
            "deskew_status": "unverified",
        }
        atomic_json(path, manifest)
        print(f"[lidar-bev] {manifest['scene']}: {len(manifest['frames'])} frames", flush=True)
    report = {
        "frames_created": created, "frames_reused": reused,
        "points_decoded_this_run": all_points,
        "nonfinite_points_this_run": nonfinite_points,
        "max_recorded_point_time_span_ms": max_sweep_ms,
        "point_time_span_ms_p50_p95_p99": (np.quantile(spans_ms, [.5, .95, .99]).tolist()
                                           if spans_ms else None),
        "point_time_span_over_110ms": sum(span > 110 for span in spans_ms),
        "point_time_span_over_150ms": sum(span > 150 for span in spans_ms),
        "duration_gate_110ms": "longer files retained but flagged invalid for temporal use",
        "meaning": "optional single-sweep model input; no occupancy/depth GT generated",
    }
    atomic_json(root / "lidar_bev_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--audit-timestamps", action="store_true",
                        help="Decode existing LAZ again to record point-time span per frame")
    args = parser.parse_args()
    print(json.dumps(materialize(args.root, args.audit_timestamps), indent=2))


if __name__ == "__main__":
    main()
