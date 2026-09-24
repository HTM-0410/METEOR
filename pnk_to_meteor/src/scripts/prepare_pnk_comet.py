#!/usr/bin/env python3
"""Prepare a conservative, reproducible PNK -> CoMET/METEOR handoff bundle.

This is a staging adapter, not the t4dataset CoMET autolabel factory.  It keeps
source images/LAZ and clean-v1 box NPZs in place, and explicitly marks every
unverified target as unavailable for supervised training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CLEAN = Path(r"D:\Backup_rosbag\PNKData_clean_v1")
DEFAULT_OUTPUT = Path(r"D:\Backup_rosbag\PNKData_comet_handoff_v2")
GPS_EPOCH_UNIX_S = 315964800
GPS_UTC_LEAP_SECONDS = 18  # 2026 recordings; recorded as an explicit assumption.
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
HORIZONS_S = np.arange(0.5, 3.01, 0.5, dtype=np.float64)
NAV_FIELDS = ["GPSWeek", "GPSTime", "Latitude", "Longitude", "Altitude",
              "Ve", "Vn", "Vu", "V_2D", "Heading2", "Pitch", "Roll",
              "Latitude_std", "Longitude_std", "Altitude_std",
              "Heading2_std", "Age", "Status"]
VEHICLE_FIELDS = ["Timestamp", "vehicle_speed", "yaw_rate",
                  "acceleration_x", "acceleration_y"]
STEER_FIELDS = ["Timestamp", "steer_angle", "steer_speed", "steer_torque"]
PROVISIONAL_SLOT_MAP = {
    "CAM_FRONT_WIDE": "CAM_P_F", "CAM_FRONT_LEFT": "CAM_P_FL",
    "CAM_FRONT_RIGHT": "CAM_P_FR", "CAM_BACK_WIDE": "CAM_P_B",
    "CAM_BACK_LEFT": "CAM_P_LB", "CAM_BACK_RIGHT": "CAM_P_RB",
    "CAM_FRONT_NARROW": "CAM_P_L", "CAM_BACK_NARROW": "CAM_P_R",
}
TASKS = {
    "1_bev_lane": "blocked_missing_lane_semantics_map_or_panoptic",
    "2_metric_depth": "candidate_requires_camera_model_and_projection_qa",
    "3_3d_boxes": "available_clean_v1_source_labels_geometry_qa_pending",
    "4_unknown_objects": "blocked_missing_unknown_taxonomy_labels",
    "5_2d_semantic": "blocked_missing_pixel_semantics",
    "6_2d_detection": "candidate_partial_3d_projection_requires_camera_qa",
    "7_e2e": "candidate_nav_reference_trajectory_not_training_gt",
    "8_3d_occupancy": "candidate_binary_partial_requires_deskew_and_semantics",
    "9_occupancy_flow": "blocked_track_id_qa_and_ego_compensation",
    "10_agent_forecast": "blocked_track_id_qa_and_ego_compensation",
    "11_traffic_light": "blocked_missing_state_and_ego_association",
    "12_area_risk": "blocked_full_semantics_and_agent_futures",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def timestamp_ns(values: pd.Series) -> np.ndarray:
    parts = values.astype(str).str.split("-", n=1, expand=True)
    return parts[0].to_numpy(dtype=np.int64) * 1_000_000_000 + parts[1].to_numpy(dtype=np.int64)


def read_group(recording: Path, group: str, fields: list[str]) -> pd.DataFrame:
    paths = sorted((recording / "OTHERS" / group).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"Missing {group} CSVs: {recording}")
    frames = [pd.read_csv(path, usecols=fields) for path in paths]
    result = pd.concat(frames, ignore_index=True)
    if group == "NAV":
        gps_s = (result["GPSWeek"].to_numpy(np.float64) * 604800.0
                 + result["GPSTime"].to_numpy(np.float64))
        result["t_ns"] = np.rint((GPS_EPOCH_UNIX_S + gps_s - GPS_UTC_LEAP_SECONDS)
                                  * 1_000_000_000).astype(np.int64)
    else:
        result["t_ns"] = timestamp_ns(result["Timestamp"])
    result = result[np.isfinite(result["t_ns"])].sort_values("t_ns")
    return result.drop_duplicates("t_ns", keep="first").reset_index(drop=True)


def ecef(latitude_deg: np.ndarray, longitude_deg: np.ndarray,
         altitude_m: np.ndarray) -> np.ndarray:
    latitude = np.deg2rad(latitude_deg)
    longitude = np.deg2rad(longitude_deg)
    sin_lat, cos_lat = np.sin(latitude), np.cos(latitude)
    radius = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
    return np.stack(((radius + altitude_m) * cos_lat * np.cos(longitude),
                     (radius + altitude_m) * cos_lat * np.sin(longitude),
                     (radius * (1.0 - WGS84_E2) + altitude_m) * sin_lat), axis=-1)


def enu(latitude_deg: np.ndarray, longitude_deg: np.ndarray,
        altitude_m: np.ndarray, origin: tuple[float, float, float]) -> np.ndarray:
    lat0, lon0, alt0 = origin
    difference = ecef(latitude_deg, longitude_deg, altitude_m) - ecef(
        np.asarray(lat0), np.asarray(lon0), np.asarray(alt0))
    lat, lon = math.radians(lat0), math.radians(lon0)
    rotation = np.array([[-math.sin(lon), math.cos(lon), 0],
                         [-math.sin(lat)*math.cos(lon), -math.sin(lat)*math.sin(lon), math.cos(lat)],
                         [math.cos(lat)*math.cos(lon), math.cos(lat)*math.sin(lon), math.sin(lat)]])
    return difference @ rotation.T


def nav_at(nav: pd.DataFrame, times_ns: np.ndarray) -> dict[str, np.ndarray]:
    """Interpolate the NAV measurement clock; validity is separate from values."""
    t = nav["t_ns"].to_numpy(np.int64)
    query = np.asarray(times_ns, np.int64)
    right = np.searchsorted(t, query, side="left")
    lo = np.clip(right - 1, 0, len(t) - 1)
    hi = np.clip(right, 0, len(t) - 1)
    tl, th = t[lo], t[hi]
    span = th - tl
    weight = np.divide(query - tl, span, out=np.zeros(len(query), np.float64), where=span != 0)
    bracketed = (query >= t[0]) & (query <= t[-1]) & (span <= 30_000_000)
    nearest_ms = np.minimum(np.abs(query - tl), np.abs(query - th)) / 1e6
    out: dict[str, np.ndarray] = {"nearest_ms": nearest_ms,
                                  "bracketed": bracketed,
                                  "source_lo": lo, "source_hi": hi}
    for field in ("Latitude", "Longitude", "Altitude", "Ve", "Vn", "Vu", "V_2D",
                  "Pitch", "Roll"):
        values = nav[field].to_numpy(np.float64)
        out[field] = values[lo] * (1 - weight) + values[hi] * weight
    heading = np.unwrap(np.deg2rad(nav["Heading2"].to_numpy(np.float64)))
    out["yaw_enu"] = math.pi / 2 - (heading[lo] * (1 - weight) + heading[hi] * weight)
    for field in ("Latitude_std", "Longitude_std", "Altitude_std", "Heading2_std", "Age"):
        values = nav[field].to_numpy(np.float64)
        out[field] = np.maximum(values[lo], values[hi])
    out["Status"] = nav["Status"].to_numpy()[lo]
    finite = np.isfinite(np.stack([out[key] for key in ("Latitude", "Longitude", "Altitude",
                                                              "V_2D", "yaw_enu")], axis=1)).all(axis=1)
    out["candidate_valid"] = (bracketed & finite & (nearest_ms <= 10.0)
                               & (out["Latitude_std"] <= 0.5)
                               & (out["Longitude_std"] <= 0.5)
                               & (out["Heading2_std"] <= 1.0)
                               & (out["Age"] <= 1.0))
    return out


def nearest_at(table: pd.DataFrame, times_ns: np.ndarray, tolerance_ms: float) -> tuple[np.ndarray, np.ndarray]:
    t = table["t_ns"].to_numpy(np.int64)
    right = np.searchsorted(t, times_ns)
    lo = np.clip(right - 1, 0, len(t) - 1)
    hi = np.clip(right, 0, len(t) - 1)
    choose_hi = np.abs(t[hi] - times_ns) < np.abs(t[lo] - times_ns)
    index = np.where(choose_hi, hi, lo)
    delta_ms = np.abs(t[index] - times_ns) / 1e6
    return index, delta_ms <= tolerance_ms


def scene_motion(frames: list[dict], nav: pd.DataFrame, vehicle: pd.DataFrame,
                 steer: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict]:
    timestamps = np.asarray([frame["timestamp_ns"] for frame in frames], np.int64)
    state = nav_at(nav, timestamps)
    first_valid = np.flatnonzero(state["candidate_valid"])
    if not len(first_valid):
        raise ValueError("No candidate-valid NAV sample in scene")
    first = int(first_valid[0])
    origin = tuple(float(state[key][first]) for key in ("Latitude", "Longitude", "Altitude"))
    positions = enu(state["Latitude"], state["Longitude"], state["Altitude"], origin)
    yaw = state["yaw_enu"]
    v_idx, v_ok = nearest_at(vehicle, timestamps, 30.0)
    s_idx, s_ok = nearest_at(steer, timestamps, 30.0)
    speed = vehicle["vehicle_speed"].to_numpy(np.float64)[v_idx] / 3.6
    yaw_rate = np.deg2rad(vehicle["yaw_rate"].to_numpy(np.float64)[v_idx])
    acceleration = vehicle["acceleration_x"].to_numpy(np.float64)[v_idx]
    wheel_angle = np.deg2rad(steer["steer_angle"].to_numpy(np.float64)[s_idx])
    speed_residual = np.abs(speed - state["V_2D"])
    speed_ok = speed_residual <= np.maximum(1.0, 0.15 * state["V_2D"])
    # CoMET candidate is deliberately not named ego_motion.npz: the existing
    # METEOR loader would otherwise train on an unverified lever-arm pose.
    state_valid = state["candidate_valid"] & v_ok & speed_ok
    waypoints = np.zeros((len(frames), 6, 2), np.float32)
    future_valid = np.zeros(len(frames), np.uint8)
    for i, t0 in enumerate(timestamps):
        future_ns = t0 + np.rint(HORIZONS_S * 1e9).astype(np.int64)
        if future_ns[-1] > timestamps[-1] or not state_valid[i]:
            continue
        future = nav_at(nav, future_ns)
        if not np.all(future["candidate_valid"]):
            continue
        # Guard against a gap in accepted keyframes: this is a separate scene
        # continuity check from the 100 Hz NAV bracketing check.
        end = int(np.searchsorted(timestamps, future_ns[-1], side="right"))
        if np.any(np.diff(timestamps[i:end]) > 300_000_000):
            continue
        end_positions = enu(future["Latitude"], future["Longitude"], future["Altitude"], origin)
        delta = end_positions[:, :2] - positions[i, :2]
        c, s = math.cos(float(yaw[i])), math.sin(float(yaw[i]))
        waypoints[i, :, 0] = c * delta[:, 0] + s * delta[:, 1]
        waypoints[i, :, 1] = -s * delta[:, 0] + c * delta[:, 1]
        future_valid[i] = 1
    arrays = {
        "timestamp_ns": timestamps, "nav_reference_pose_enu": np.column_stack((positions, yaw)).astype(np.float64),
        "nav_reference_origin_wgs84": np.asarray(origin, np.float64),
        "waypoints_nav_reference_m": waypoints,
        "nav_reference_valid": state["candidate_valid"].astype(np.uint8),
        "future_candidate_valid": future_valid,
        "speed_mps": speed.astype(np.float32),
        "speed_nav_mps": state["V_2D"].astype(np.float32),
        "speed_residual_mps": speed_residual.astype(np.float32),
        "yaw_rate_radps": yaw_rate.astype(np.float32),
        "longitudinal_accel_mps2": acceleration.astype(np.float32),
        "steering_wheel_rad": wheel_angle.astype(np.float32),
        "can_valid": v_ok.astype(np.uint8), "steering_valid": s_ok.astype(np.uint8),
        "nav_nearest_ms": state["nearest_ms"].astype(np.float32),
        "nav_status_raw": state["Status"].astype(np.int16),
        "nav_age_raw": state["Age"].astype(np.float32),
        "horizontal_std_raw": np.hypot(state["Latitude_std"], state["Longitude_std"]).astype(np.float32),
        "heading_std_raw": state["Heading2_std"].astype(np.float32),
    }
    summary = {"frames": len(frames), "nav_candidate_valid": int(np.sum(state["candidate_valid"])),
               "can_matched": int(np.sum(v_ok)), "steer_matched": int(np.sum(s_ok)),
               "future_candidate_valid": int(np.sum(future_valid)),
               "speed_check_pass": int(np.sum(speed_ok & v_ok)),
               "nav_nearest_p95_ms": float(np.quantile(state["nearest_ms"], .95)),
               "speed_residual_p95_mps": float(np.quantile(speed_residual[v_ok], .95)) if np.any(v_ok) else None,
               "origin_wgs84": origin}
    return arrays, summary


def split_for_recording(recording: str) -> str:
    # Disjoint capture dates, not adjacent-frame random splitting.
    date = recording[:8]
    if date >= "20260313":
        return "test"
    if date >= "20260309":
        return "val"
    return "train"


def prepare(clean: Path, output: Path) -> dict:
    clean = clean.resolve()
    output = output.resolve()
    source = json.loads((clean / "dataset.json").read_text(encoding="utf-8"))
    sensor_root = Path(source["source_sensor_root"]).resolve()
    if not sensor_root.is_dir() or not (clean / "ready").is_dir():
        raise FileNotFoundError("Missing PNK source or clean-v1 ready root")
    if output == clean or output == sensor_root or output in (clean.parents):
        raise ValueError("Output cannot overwrite a source root")
    existing = output / "dataset.json"
    if existing.exists():
        raise FileExistsError(f"Output already exists; preserve it and choose a new path: {existing}")
    scene_paths = sorted((clean / "ready").glob("*/manifest.json"))
    if not scene_paths:
        raise ValueError("No clean-v1 scenes")
    by_recording: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for path in scene_paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        by_recording[manifest["recording"]].append((path, manifest))
    counters: Counter = Counter()
    scene_summaries = []
    catalog = []
    for recording_name, scenes in sorted(by_recording.items()):
        recording = sensor_root / recording_name
        nav = read_group(recording, "NAV", NAV_FIELDS)
        vehicle = read_group(recording, "VEHICLE_INFO", VEHICLE_FIELDS)
        steer = read_group(recording, "VEHICLE_STEER", STEER_FIELDS)
        print(f"[recording] {recording_name}: NAV={len(nav)} CAN={len(vehicle)} steering={len(steer)}", flush=True)
        for manifest_path, original in scenes:
            scene = original["scene"]
            frames = original["frames"]
            arrays, summary = scene_motion(frames, nav, vehicle, steer)
            scene_root = output / "scenes" / scene
            atomic_npz(scene_root / "nav_ego_candidate.npz", **arrays)
            records = []
            for i, frame in enumerate(frames):
                box_path = manifest_path.parent / frame["boxes"]
                if not box_path.is_file():
                    raise FileNotFoundError(box_path)
                with np.load(box_path) as boxes:
                    count = len(boxes["boxes"])
                    if count != frame["valid_box_count"]:
                        raise ValueError(f"Box count mismatch: {box_path}")
                records.append({
                    "frame": i, "timestamp_ns": frame["timestamp_ns"],
                    "images_raw": frame["imgs"],
                    "camera_delta_ms": frame["camera_delta_ms"],
                    "lidar_top": frame["lidar_top"],
                    "lidar_merged_ego": frame["lidar_merged_ego"],
                    "boxes_3d_npz": str(box_path), "boxes_3d_count": count,
                    "nav_ego_candidate_index": i,
                    "nav_ego_candidate_valid": bool(arrays["nav_reference_valid"][i]),
                    "e2e_future_candidate_valid": bool(arrays["future_candidate_valid"][i]),
                    "source_label": frame["source_label"],
                    "source_label_sha256": frame["source_label_sha256"],
                })
                counters["frames"] += 1
                counters["boxes"] += count
            split = split_for_recording(recording_name)
            scene_doc = {
                "schema": "pnk-comet-handoff-v1", "scene": scene,
                "recording": recording_name, "split": split,
                "sensor_root": str(sensor_root), "clean_v1_manifest": str(manifest_path),
                "calibration_period": original["calibration_period"],
                "cameras_raw": original["cameras"],
                "provisional_meteor_slot_map_not_validated": PROVISIONAL_SLOT_MAP,
                "lidar_top_calibration": original["lidar_top_calibration"],
                "nav_ego_candidate": "nav_ego_candidate.npz",
                "coordinate_contract": {
                    "world": "local_ENU_origin_at_first_candidate_valid_NAV_position",
                    "pose_origin": "NAV_reported_LatLon_origin_unverified_not_vehicle_ego_GT",
                    "yaw": "radians_ccw_from_East_derived_90deg_minus_Heading2",
                    "camera_geometry": "raw_distorted_model_and_meteor_slot_mapping_unverified_hold_for_projection",
                    "lidar_deskew": "unknown_hold_for_multisweep",
                },
                "task_status": TASKS, "frames": records,
            }
            atomic_json(scene_root / "manifest.json", scene_doc)
            row = {"scene": scene, "recording": recording_name, "split": split, **summary,
                   "max_boxes_per_frame": max(fr["boxes_3d_count"] for fr in records)}
            scene_summaries.append(row)
            catalog.extend({"scene": scene, "frame": rec["frame"], "split": split,
                            "timestamp_ns": rec["timestamp_ns"], "boxes": rec["boxes_3d_count"],
                            "nav_candidate_valid": int(rec["nav_ego_candidate_valid"]),
                            "future_candidate_valid": int(rec["e2e_future_candidate_valid"])}
                           for rec in records)
            counters["scenes"] += 1
            counters[f"{split}_frames"] += len(records)
            counters["nav_candidate_valid"] += summary["nav_candidate_valid"]
            counters["future_candidate_valid"] += summary["future_candidate_valid"]
            print(f"  [scene] {scene}: {len(records)} frames, {summary['future_candidate_valid']} future candidates", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "catalog.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(catalog[0]))
        writer.writeheader()
        writer.writerows(catalog)
    result = {
        "schema": "pnk-comet-handoff-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_clean_v1": str(clean), "source_clean_v1_dataset_sha256": sha256(clean / "dataset.json"),
        "source_sensor_root": str(sensor_root), "zero_copy_source_sensors_and_boxes": True,
        "not_a_t4dataset_export": True, "not_model_training_ready": True,
        "gps_utc_leap_seconds_assumed": GPS_UTC_LEAP_SECONDS,
        "nav_candidate_gate": {"bracketing_gap_max_ms": 30, "nearest_max_ms": 10,
                               "latitude_std_raw_max": .5, "longitude_std_raw_max": .5,
                               "heading_std_raw_max": 1.0, "age_raw_max": 1.0,
                               "status_code_not_interpreted": True},
        "split_rule": "capture date <=2026-03-07 train; 2026-03-09/10 val; >=2026-03-13 test",
        "counts": dict(counters), "task_status": TASKS, "scenes": scene_summaries,
    }
    atomic_json(output / "dataset.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = prepare(args.clean, args.output)
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
