#!/usr/bin/env python3
"""Build a non-destructive, zero-copy clean PNKData staging dataset.

The source trees are never modified.  The output materializes normalized 3D
labels and manifests while referring to the immutable JPEG/LAZ files in place.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(0)

CAMERAS = ("CAM_P_F", "CAM_P_FL", "CAM_P_FR", "CAM_P_B",
           "CAM_P_L", "CAM_P_R", "CAM_P_LB", "CAM_P_RB")
SIDE_LIDARS = ("LIDAR_E_F", "LIDAR_E_B", "LIDAR_E_L", "LIDAR_E_R")
CLASS_ID = {"Car": 1, "Rider": 2, "Pedestrian": 2}


def parse_ns(stem: str) -> int:
    sec, nsec = stem.split("-", 1)
    return int(sec) * 1_000_000_000 + int(nsec)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_segments(path: Path) -> tuple[list[dict], dict[str, dict]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    rows = [row for row in rows if row.get("idx", "").strip().isdigit()]
    by_folder = {row["folder_name"].strip(): row for row in rows}
    if len(by_folder) != len(rows):
        raise ValueError("Selection CSV contains duplicate folder_name values")
    return rows, by_folder


def nearest(paths: list[Path], timestamp_ns: int) -> tuple[Path, float]:
    if not paths:
        raise ValueError("Cannot select from an empty sensor stream")
    values = np.asarray([parse_ns(path.stem) for path in paths], dtype=np.int64)
    index = int(np.searchsorted(values, timestamp_ns))
    candidates = {max(0, min(len(paths) - 1, index - 1)),
                  max(0, min(len(paths) - 1, index))}
    selected = min(candidates, key=lambda i: abs(int(values[i]) - timestamp_ns))
    return paths[selected], abs(int(values[selected]) - timestamp_ns) / 1e6


def parse_label(path: Path) -> tuple[dict, list[dict]]:
    accepted = []
    rejected = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        reason = None
        if len(fields) != 11:
            reason = f"expected_11_fields_got_{len(fields)}"
        else:
            try:
                geometry = np.asarray([float(value) for value in fields[:7]], dtype=np.float32)
                source_flag = int(fields[8])
                point_count = int(fields[9])
                track_id = int(fields[10])
            except ValueError:
                reason = "parse_error"
        if reason is None and not np.isfinite(geometry).all():
            reason = "nonfinite_geometry"
        if reason is None and np.any(geometry[3:6] <= 0):
            reason = "nonpositive_dimension"
        if reason is None and fields[7] not in CLASS_ID:
            reason = "unsupported_class"
        if reason:
            rejected.append({
                "source_label": str(path),
                "line": line_number,
                "reason": reason,
                "raw": line,
            })
            continue
        accepted.append({
            "geometry": geometry,
            "source_class": fields[7],
            "class_id": CLASS_ID[fields[7]],
            "source_flag": source_flag,
            "point_count": point_count,
            "track_id": track_id,
        })
    return {
        "boxes": np.asarray([row["geometry"] for row in accepted], dtype=np.float32).reshape(-1, 7),
        "class_id": np.asarray([row["class_id"] for row in accepted], dtype=np.int64),
        "source_class": np.asarray([row["source_class"] for row in accepted], dtype="U16"),
        "source_flag": np.asarray([row["source_flag"] for row in accepted], dtype=np.int64),
        "point_count": np.asarray([row["point_count"] for row in accepted], dtype=np.int64),
        "track_id": np.asarray([row["track_id"] for row in accepted], dtype=np.int64),
    }, rejected


def collect_streams(recording: Path) -> dict[str, list[Path]]:
    result = {}
    for group, suffix in (("CAMERA", ".jpg"), ("LIDAR", ".laz"), ("RADAR", ".laz")):
        root = recording / group
        if not root.exists():
            continue
        for sensor in root.iterdir():
            if sensor.is_dir():
                result[sensor.name] = sorted(sensor.glob(f"*{suffix}"), key=lambda path: parse_ns(path.stem))
    return result


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_transform(name: str, matrix) -> None:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"Invalid 4x4 transform for {name}")
    if not np.allclose(value[-1], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"Invalid homogeneous row for {name}")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5):
        raise ValueError(f"Non-rigid rotation for {name}")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-5):
        raise ValueError(f"Improper rotation for {name}")


def build(args) -> Path:
    source_root = args.source_root.resolve()
    sensor_root = source_root / "PNKData"
    meta_root = source_root / "PNKData_meta"
    label_root = meta_root / "label" / "2026_08_14_vf6_01_02"
    selection_path = meta_root / "file_csv" / "2026_08_14_vf6_01_02.csv"
    intrinsics_path = meta_root / "Calib_2" / "VF6_01_Intrinsics.json"
    extrinsics_path = meta_root / "Calib_2" / "VF6_01_Extrinsics_By_Dates.json"
    required = (sensor_root, label_root, selection_path, intrinsics_path, extrinsics_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required PNK source: " + ", ".join(missing))
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    temporary = args.output.with_name(args.output.name + f".building-{os.getpid()}")
    temporary.mkdir(parents=True)
    (temporary / "ready").mkdir()
    (temporary / "catalog").mkdir()
    (temporary / "calibration").mkdir()
    (temporary / "quarantine").mkdir()

    rows, by_folder = read_segments(selection_path)
    intrinsics = json.loads(intrinsics_path.read_text())
    dated = json.loads(extrinsics_path.read_text())
    periods = dated["periods"]
    for camera in CAMERAS:
        if camera not in intrinsics:
            raise KeyError(f"No intrinsics for {camera}")
        calibration = intrinsics[camera]
        K = np.asarray(calibration["camera_matrix"], dtype=np.float64)
        D = np.asarray(calibration["distortion_coefficients"], dtype=np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all() or not np.isfinite(D).all():
            raise ValueError(f"Invalid intrinsics for {camera}")

    for period in periods:
        for sensor, transform in period["extrinsics"].items():
            validate_transform(f"{period['id']}/{sensor}", transform)

    shutil.copy2(intrinsics_path, temporary / "calibration" / intrinsics_path.name)
    shutil.copy2(extrinsics_path, temporary / "calibration" / extrinsics_path.name)

    annotated = [row for row in rows if row["3d_annotated"].strip() == "1"]
    missing_sensor_segments = []
    segment_records = []
    excluded_frames = []
    accepted_frames = []
    rejected_boxes = []
    conflicts = []

    label_locations = defaultdict(list)
    for segment_dir in label_root.iterdir():
        if not segment_dir.is_dir():
            continue
        for path in (segment_dir / "label").glob("*.txt"):
            label_locations[path.stem].append(path)
    for stem, locations in sorted(label_locations.items()):
        if len(locations) <= 1:
            continue
        hashes = [sha256(path) for path in locations]
        for path, digest in zip(locations, hashes):
            conflicts.append({
                "timestamp": stem,
                "segment": path.parents[1].name,
                "sha256": digest,
                "identical_group": len(set(hashes)) == 1,
                "source_label": str(path),
            })

    scene_names = []
    used_label_timestamps = set()
    for row in annotated:
        segment = row["folder_name"].strip()
        recording_name = row["name"].strip()
        recording = sensor_root / recording_name
        segment_source = label_root / segment / "label"
        labels = sorted(segment_source.glob("*.txt"), key=lambda path: parse_ns(path.stem))
        if not recording.exists():
            missing_sensor_segments.append({
                "segment": segment,
                "recording": recording_name,
                "label_frames": len(labels),
                "boxes_from_metadata": row["num_boxes"],
                "reason": "recording_directory_missing",
            })
            continue
        streams = collect_streams(recording)
        required_streams = (*CAMERAS, "LIDAR_TOP", "LIDAR_MERGED_EGO")
        absent = [sensor for sensor in required_streams if not streams.get(sensor)]
        if absent:
            missing_sensor_segments.append({
                "segment": segment,
                "recording": recording_name,
                "label_frames": len(labels),
                "boxes_from_metadata": row["num_boxes"],
                "reason": "missing_streams:" + "|".join(absent),
            })
            continue

        date = recording_name[:8]
        date_iso = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        matching_periods = [period for period in periods
                            if period["valid_from"] <= date_iso < period["valid_to"]]
        if len(matching_periods) != 1:
            raise ValueError(f"Expected one calibration period for {recording_name}, got {len(matching_periods)}")
        period = matching_periods[0]
        for camera in CAMERAS:
            if camera not in period["extrinsics"]:
                raise KeyError(f"No dated extrinsic for {camera} in {period['id']}")

        label_stems = {path.stem for path in labels}
        top_by_stem = {path.stem: path for path in streams["LIDAR_TOP"]}
        merged_by_stem = {path.stem: path for path in streams["LIDAR_MERGED_EGO"]}
        ready_dir = temporary / "ready" / segment
        boxes_dir = ready_dir / "boxes"
        boxes_dir.mkdir(parents=True)
        frames = []
        for source_label in labels:
            timestamp_ns = parse_ns(source_label.stem)
            reason = None
            if source_label.stem in used_label_timestamps:
                reason = "duplicate_label_timestamp_across_segments"
            elif len(label_locations[source_label.stem]) > 1:
                reason = "conflicting_label_timestamp"
            elif source_label.stem not in top_by_stem:
                reason = "no_exact_lidar_top"
            elif source_label.stem not in merged_by_stem:
                reason = "no_exact_lidar_merged_ego"

            images = {}
            camera_delta_ms = {}
            if reason is None:
                for camera in CAMERAS:
                    image_path, delta_ms = nearest(streams[camera], timestamp_ns)
                    images[camera] = relative(image_path, sensor_root)
                    camera_delta_ms[camera] = delta_ms
                if max(camera_delta_ms.values()) > args.camera_tolerance_ms:
                    reason = "camera_delta_exceeds_tolerance"

            if reason:
                excluded_frames.append({
                    "segment": segment,
                    "recording": recording_name,
                    "timestamp": source_label.stem,
                    "reason": reason,
                    "max_camera_delta_ms": (max(camera_delta_ms.values())
                                            if camera_delta_ms else ""),
                    "source_label": str(source_label),
                })
                continue

            clean, rejected = parse_label(source_label)
            for rejected_row in rejected:
                rejected_row.update({
                    "segment": segment,
                    "recording": recording_name,
                    "timestamp": source_label.stem,
                })
            rejected_boxes.extend(rejected)
            if len(clean["boxes"]) == 0:
                excluded_frames.append({
                    "segment": segment,
                    "recording": recording_name,
                    "timestamp": source_label.stem,
                    "reason": "no_valid_boxes_after_cleaning",
                    "max_camera_delta_ms": max(camera_delta_ms.values()),
                    "source_label": str(source_label),
                })
                continue

            output_index = len(frames)
            boxes_rel = f"boxes/{output_index:06d}.npz"
            meteor_boxes = np.column_stack((
                clean["class_id"].astype(np.float32),
                clean["boxes"][:, 0], clean["boxes"][:, 1],
                clean["boxes"][:, 3], clean["boxes"][:, 4], clean["boxes"][:, 6],
            )).astype(np.float32)
            np.savez_compressed(
                ready_dir / boxes_rel,
                boxes=clean["boxes"],
                class_id=clean["class_id"],
                source_class=clean["source_class"],
                source_flag=clean["source_flag"],
                point_count=clean["point_count"],
                track_id=clean["track_id"],
                meteor_boxes=meteor_boxes,
            )

            side_lidar = {}
            side_lidar_delta_ms = {}
            for sensor in SIDE_LIDARS:
                if streams.get(sensor):
                    path, delta = nearest(streams[sensor], timestamp_ns)
                    side_lidar[sensor] = relative(path, sensor_root)
                    side_lidar_delta_ms[sensor] = delta
            frame = {
                "frame": output_index,
                "timestamp": source_label.stem,
                "timestamp_ns": timestamp_ns,
                "imgs": images,
                "camera_delta_ms": camera_delta_ms,
                "lidar_top": relative(top_by_stem[source_label.stem], sensor_root),
                "lidar_merged_ego": relative(merged_by_stem[source_label.stem], sensor_root),
                "side_lidar": side_lidar,
                "side_lidar_delta_ms": side_lidar_delta_ms,
                "boxes": boxes_rel,
                "valid_box_count": int(len(clean["boxes"])),
                "rejected_box_count": len(rejected),
                "source_label": str(source_label.relative_to(meta_root)).replace("\\", "/"),
                "source_label_sha256": sha256(source_label),
            }
            frames.append(frame)
            used_label_timestamps.add(source_label.stem)
            accepted_frames.append({
                "segment": segment,
                "recording": recording_name,
                "frame": output_index,
                "timestamp": source_label.stem,
                "valid_boxes": len(clean["boxes"]),
                "rejected_boxes": len(rejected),
                "max_camera_delta_ms": max(camera_delta_ms.values()),
            })

        if not frames:
            shutil.rmtree(ready_dir)
            missing_sensor_segments.append({
                "segment": segment,
                "recording": recording_name,
                "label_frames": len(labels),
                "boxes_from_metadata": row["num_boxes"],
                "reason": "no_accepted_frames",
            })
            continue

        camera_calibration = {}
        for camera in CAMERAS:
            source = intrinsics[camera]
            camera_calibration[camera] = {
                "image_hw": [source["image_height"], source["image_width"]],
                "K_raw": source["camera_matrix"],
                "D_raw": source["distortion_coefficients"],
                "distortion_model": "unspecified_by_source",
                "T_cam_ego": period["extrinsics"][camera],
                "transform_direction_basis": "validated_by_diagnostic_projection_not_source_schema",
            }
        manifest = {
            "schema": "pnk-clean-v1",
            "scene": segment,
            "recording": recording_name,
            "source_sensor_root": str(sensor_root),
            "source_metadata_root": str(meta_root),
            "calibration_period": period["id"],
            "camera_tolerance_ms": args.camera_tolerance_ms,
            "cameras": camera_calibration,
            "lidar_top_calibration": {
                "T_ego_lidar_top": period["extrinsics"]["LIDAR_TOP"],
                "transform_direction_basis": "verified_against_merged_ego_sensor_id_11",
            },
            "track_id_status": "candidate_unverified",
            "frames": frames,
        }
        atomic_json(ready_dir / "manifest.json", manifest)
        scene_names.append(segment)
        segment_records.append({
            "segment": segment,
            "recording": recording_name,
            "source_label_frames": len(labels),
            "accepted_frames": len(frames),
            "excluded_frames": len(labels) - len(frames),
            "valid_boxes": sum(frame["valid_box_count"] for frame in frames),
            "rejected_boxes": sum(frame["rejected_box_count"] for frame in frames),
            "calibration_period": period["id"],
        })

    (temporary / "scenes.txt").write_text("\n".join(scene_names) + "\n", encoding="utf-8")
    write_csv(temporary / "catalog" / "ready_segments.csv", segment_records,
              ["segment", "recording", "source_label_frames", "accepted_frames",
               "excluded_frames", "valid_boxes", "rejected_boxes", "calibration_period"])
    write_csv(temporary / "catalog" / "accepted_frames.csv", accepted_frames,
              ["segment", "recording", "frame", "timestamp", "valid_boxes",
               "rejected_boxes", "max_camera_delta_ms"])
    write_csv(temporary / "quarantine" / "excluded_frames.csv", excluded_frames,
              ["segment", "recording", "timestamp", "reason", "max_camera_delta_ms",
               "source_label"])
    write_csv(temporary / "quarantine" / "missing_sensor_segments.csv", missing_sensor_segments,
              ["segment", "recording", "label_frames", "boxes_from_metadata", "reason"])
    write_csv(temporary / "quarantine" / "rejected_boxes.csv", rejected_boxes,
              ["segment", "recording", "timestamp", "source_label", "line", "reason", "raw"])
    write_csv(temporary / "quarantine" / "conflicting_label_timestamps.csv", conflicts,
              ["timestamp", "segment", "sha256", "identical_group", "source_label"])

    summary = {
        "schema": "pnk-clean-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "source_sensor_root": str(sensor_root),
        "source_metadata_root": str(meta_root),
        "zero_copy_sensor_files": True,
        "selection_csv": str(selection_path),
        "selection_csv_sha256": sha256(selection_path),
        "intrinsics_source": str(intrinsics_path),
        "intrinsics_sha256": sha256(intrinsics_path),
        "extrinsics_source": str(extrinsics_path),
        "extrinsics_sha256": sha256(extrinsics_path),
        "policy": {
            "require_annotated_segment": True,
            "require_exact_lidar_top_and_merged_timestamp": True,
            "require_all_8_CAM_P": True,
            "camera_tolerance_ms": args.camera_tolerance_ms,
            "reject_nonfinite_or_nonpositive_box_dimensions": True,
            "preserve_all_valid_boxes_without_64_box_truncation": True,
            "do_not_resample_or_undistort_images": True,
            "do_not_use_unverified_track_ids_for_trajectory": True,
        },
        "counts": {
            "metadata_annotated_segments": len(annotated),
            "ready_segments": len(segment_records),
            "missing_sensor_or_empty_segments": len(missing_sensor_segments),
            "accepted_frames": len(accepted_frames),
            "excluded_frames": len(excluded_frames),
            "accepted_boxes": sum(int(row["valid_boxes"]) for row in accepted_frames),
            "rejected_boxes_in_ready_source_frames": len(rejected_boxes),
            "conflicting_timestamp_rows": len(conflicts),
            "conflicting_timestamp_groups": len({row["timestamp"] for row in conflicts}),
        },
    }
    atomic_json(temporary / "dataset.json", summary)
    readme = f"""# PNKData clean staging v1

This directory was generated without modifying the source dataset.

- Ready scenes: {len(segment_records)}
- Accepted keyframes: {len(accepted_frames)}
- Valid 3D boxes: {summary['counts']['accepted_boxes']}
- Rejected source boxes in ready frames: {len(rejected_boxes)}
- Excluded source keyframes: {len(excluded_frames)}
- Sensor files are referenced in place under `{sensor_root}`; they are not copied.

`ready/*/manifest.json` is the canonical staging contract.  Images remain raw
and distorted because the source calibration does not name the distortion
model.  Rectification and METEOR camera-slot mapping are intentionally deferred
to the next conversion stage.  Candidate track IDs are retained but are not
accepted as clean motion tracks yet.

See `quarantine/` for excluded frames, invalid boxes, unavailable sensor
segments and conflicting labels at overlapping timestamps.
"""
    (temporary / "README.md").write_text(readme, encoding="utf-8")
    report = f"""# PNKData cleaning result

The source trees were not modified.  Sensor files remain zero-copy references.

| Metric | Count |
|---|---:|
| Ready scenes | {len(segment_records)} |
| Accepted keyframes | {len(accepted_frames)} |
| Valid 3D boxes | {summary['counts']['accepted_boxes']} |
| Quarantined boxes in ready source frames | {len(rejected_boxes)} |
| Excluded keyframes | {len(excluded_frames)} |
| Annotation segments without local sensor recordings | {len(missing_sensor_segments)} |
| Conflicting label timestamp groups | {summary['counts']['conflicting_timestamp_groups']} |

The next conversion stage must resolve the source distortion model, rectify and
resize images, normalize NAV/IMU/CAN clocks, validate candidate track IDs and
materialize only the METEOR targets whose supervision is available.
"""
    (temporary / "CLEANING_REPORT.md").write_text(report, encoding="utf-8")
    temporary.replace(args.output)
    return args.output


def verify(output: Path, workers: int) -> dict:
    dataset = json.loads((output / "dataset.json").read_text())
    sensor_root = Path(dataset["source_sensor_root"])
    scenes = (output / "scenes.txt").read_text().split()
    timestamps = set()
    image_jobs = []
    lidar_jobs = []
    counts = Counter()
    for scene in scenes:
        root = output / "ready" / scene
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["scene"] != scene or len(manifest["cameras"]) != 8:
            raise ValueError(f"Invalid manifest identity/calibration in {scene}")
        previous = -1
        for frame in manifest["frames"]:
            timestamp = int(frame["timestamp_ns"])
            if timestamp <= previous or timestamp in timestamps:
                raise ValueError(f"Non-increasing or duplicate accepted timestamp {timestamp}")
            previous = timestamp
            timestamps.add(timestamp)
            if len(frame["imgs"]) != 8 or max(frame["camera_delta_ms"].values()) > manifest["camera_tolerance_ms"]:
                raise ValueError(f"Camera contract failed for {scene}/{frame['frame']}")
            for value in frame["imgs"].values():
                image_jobs.append(sensor_root / value)
            for key in ("lidar_top", "lidar_merged_ego"):
                lidar_jobs.append(sensor_root / frame[key])
            with np.load(root / frame["boxes"]) as labels:
                boxes = labels["boxes"]
                if boxes.shape != (frame["valid_box_count"], 7):
                    raise ValueError(f"Box count mismatch in {scene}/{frame['frame']}")
                if not np.isfinite(boxes).all() or np.any(boxes[:, 3:6] <= 0):
                    raise ValueError(f"Unclean box in {scene}/{frame['frame']}")
                if len(labels["meteor_boxes"]) != len(boxes):
                    raise ValueError(f"METEOR projection count mismatch in {scene}/{frame['frame']}")
                counts["boxes"] += len(boxes)
            counts["frames"] += 1

    def check_image(path: Path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.shape != (1536, 1920, 3):
            return str(path), None if image is None else image.shape
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        failures = [result for result in pool.map(check_image, image_jobs) if result is not None]
    if failures:
        raise ValueError(f"Unreadable/unexpected images: {failures[:3]}")
    for path in lidar_jobs:
        if not path.exists() or path.stat().st_size <= 0:
            raise ValueError(f"Missing/empty LiDAR file: {path}")
        with path.open("rb") as stream:
            if stream.read(4) != b"LASF":
                raise ValueError(f"Invalid LAS/LAZ header: {path}")
    counts["images_decoded"] = len(image_jobs)
    counts["lidar_headers_checked"] = len(lidar_jobs)
    if counts["frames"] != dataset["counts"]["accepted_frames"] or counts["boxes"] != dataset["counts"]["accepted_boxes"]:
        raise ValueError("Verified counts do not match dataset.json")
    report = {"status": "PASS", **dict(counts), "scenes": len(scenes)}
    atomic_json(output / "verification.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path(r"D:\Backup_rosbag"))
    parser.add_argument("--output", type=Path, default=Path(r"D:\Backup_rosbag\PNKData_clean_v1"))
    parser.add_argument("--camera-tolerance-ms", type=float, default=20.0)
    parser.add_argument("--verify-workers", type=int, default=6)
    args = parser.parse_args()
    output = build(args)
    report = verify(output, args.verify_workers)
    print(json.dumps({"output": str(output), "verification": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
