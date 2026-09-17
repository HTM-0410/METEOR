#!/usr/bin/env python3
"""Prepare OpenScene/NAVSIM logs for METEOR's existing ``BevLaneDataset``.

The converter intentionally does not copy the raw camera tree.  It writes a
small METEOR manifest whose ``image_root`` points at the immutable NAVSIM
``sensor_blobs/<split>`` directory; ``BevLaneDataset`` loads and resizes those
JPEGs lazily.  Annotation products that do not exist as files in NAVSIM
(ego-motion and METEOR-format 3D boxes) are materialized beside the manifest.

NAVSIM mini does not include the nuPlan maps.  Therefore this converter writes
an all-255 BEV segmentation raster (explicit ignore, not background).  Camera,
3D-box, ego-trajectory and optional LiDAR/depth supervision remain usable, but
road/lane segmentation needs a separate map-backed target generation stage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


IMAGE_HEIGHT = 432
IMAGE_WIDTH = 768
DEPTH_HEIGHT = 108
DEPTH_WIDTH = 192
BEV_HEIGHT = 800
BEV_WIDTH = 500
BEV_HALF_X = 80.0
BEV_HALF_Y = 50.0
BEV_RESOLUTION = 0.2
LIDAR_BEV_HEIGHT = 400
LIDAR_BEV_WIDTH = 250
LIDAR_BEV_RESOLUTION = 0.4
WAYPOINT_COUNT = 6
WAYPOINT_DT = 0.5
WHEELBASE = 2.8
MAX_BOXES = 64

# The last two OpenScene side cameras occupy the two historical "narrow"
# slots in METEOR, matching navsim_meteor.geometry.METEOR_CAMERA_ORDER.
CAMERA_MAP: Tuple[Tuple[str, str], ...] = (
    ("CAM_FRONT_WIDE", "CAM_F0"),
    ("CAM_FRONT_LEFT", "CAM_L0"),
    ("CAM_FRONT_RIGHT", "CAM_R0"),
    ("CAM_BACK_WIDE", "CAM_B0"),
    ("CAM_BACK_LEFT", "CAM_L2"),
    ("CAM_BACK_RIGHT", "CAM_R2"),
    ("CAM_FRONT_NARROW", "CAM_L1"),
    ("CAM_BACK_NARROW", "CAM_R1"),
)

VEHICLE_NAMES = {"vehicle"}
VRU_NAMES = {"pedestrian", "bicycle"}


def _atomic_json(path: Path, value: Dict) -> None:
    """Write JSON without exposing a half-written manifest to dataloader workers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _safe_sensor_path(sensor_root: Path, relative_path: str) -> Path:
    """Resolve a dataset-owned relative path and reject path traversal."""
    relative = PurePosixPath(str(relative_path).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe sensor path in NAVSIM log: {relative_path!r}")
    path = sensor_root.joinpath(*relative.parts)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def quaternion_yaw(rotation: Sequence[float]) -> float:
    """Yaw from NAVSIM's scalar-first quaternion ``[w, x, y, z]``."""
    q = np.asarray(rotation, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"Invalid ego quaternion: {rotation!r}")
    w, x, y, z = q / norm
    return float(math.atan2(2.0 * (w * z + x * y),
                            1.0 - 2.0 * (y * y + z * z)))


def camera_calibration(frame: Dict, sensor_root: Path) -> Dict[str, Dict]:
    """Build METEOR scene calibration, scaled to 768x432 cached geometry."""
    result: Dict[str, Dict] = {}
    for meteor_name, navsim_name in CAMERA_MAP:
        camera = frame["cams"].get(navsim_name)
        if camera is None:
            raise KeyError(f"Frame {frame.get('token')} has no {navsim_name}")
        image_path = _safe_sensor_path(sensor_root, camera["data_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable camera image: {image_path}")
        source_height, source_width = image.shape[:2]

        intrinsic = np.asarray(camera["cam_intrinsic"], dtype=np.float64).copy()
        if intrinsic.shape != (3, 3):
            raise ValueError(f"{navsim_name} intrinsic has shape {intrinsic.shape}")
        intrinsic[0, :] *= IMAGE_WIDTH / source_width
        intrinsic[1, :] *= IMAGE_HEIGHT / source_height

        # NAVSIM/OpenScene stores camera -> merged-LiDAR.  The merged-LiDAR
        # frame is the local ego frame used by the NAVSIM agent adapter.
        rotation = np.asarray(camera["sensor2lidar_rotation"], dtype=np.float64)
        translation = np.asarray(camera["sensor2lidar_translation"], dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError(f"Invalid extrinsic for {navsim_name}")
        ego_from_camera = np.eye(4, dtype=np.float64)
        ego_from_camera[:3, :3] = rotation
        ego_from_camera[:3, 3] = translation
        result[meteor_name] = {
            "K": intrinsic.tolist(),
            "T_ego_cam": ego_from_camera.tolist(),
            "source_camera": navsim_name,
            "source_hw": [int(source_height), int(source_width)],
            "distortion": np.asarray(camera.get("distortion", []), dtype=np.float64).tolist(),
        }
    return result


def validate_static_calibration(frames: Sequence[Dict], reference: Dict[str, Dict]) -> None:
    """Fail if a log changes rig calibration while the manifest is scene-level."""
    if not frames:
        return
    probe_indices = sorted(set((0, len(frames) // 2, len(frames) - 1)))
    for index in probe_indices:
        frame = frames[index]
        for meteor_name, navsim_name in CAMERA_MAP:
            camera = frame["cams"].get(navsim_name)
            if camera is None:
                raise KeyError(f"Frame {frame.get('token')} has no {navsim_name}")
            ref = np.asarray(reference[meteor_name]["T_ego_cam"], dtype=np.float64)
            current = np.eye(4, dtype=np.float64)
            current[:3, :3] = np.asarray(camera["sensor2lidar_rotation"], dtype=np.float64)
            current[:3, 3] = np.asarray(camera["sensor2lidar_translation"], dtype=np.float64)
            if not np.allclose(ref, current, rtol=1e-6, atol=1e-6):
                raise ValueError(f"Calibration drift in {navsim_name} at frame {index}")


def pcd_xyz(path: Path) -> np.ndarray:
    """Read x/y/z from the binary PCD v0.7 files shipped by OpenScene."""
    with path.open("rb") as stream:
        header: Dict[str, List[str]] = {}
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PCD header ended before DATA: {path}")
            text = line.decode("ascii").strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            header[parts[0].upper()] = parts[1:]
            if parts[0].upper() == "DATA":
                break

        if header["DATA"][0].lower() != "binary":
            raise ValueError(f"Only binary PCD is supported, got {header['DATA'][0]!r}")
        fields = header["FIELDS"]
        sizes = [int(value) for value in header["SIZE"]]
        types = header["TYPE"]
        counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
        points = int(header.get("POINTS", header["WIDTH"])[0])
        if not (len(fields) == len(sizes) == len(types) == len(counts)):
            raise ValueError(f"Inconsistent PCD field metadata: {path}")

        dtype_fields = []
        for name, size, kind, count in zip(fields, sizes, types, counts):
            key = (kind.upper(), size)
            formats = {
                ("F", 4): "<f4", ("F", 8): "<f8",
                ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4",
                ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4",
            }
            if key not in formats:
                raise ValueError(f"Unsupported PCD field type {key} in {path}")
            dtype_fields.append((name, formats[key]) if count == 1
                                else (name, formats[key], (count,)))
        records = np.fromfile(stream, dtype=np.dtype(dtype_fields), count=points)
    if len(records) != points:
        raise ValueError(f"Truncated PCD {path}: expected {points}, read {len(records)}")
    return np.column_stack((records["x"], records["y"], records["z"])).astype(np.float32)


def lidar_bev(points: np.ndarray) -> np.ndarray:
    """Rasterize xyz into METEOR's 4x400x250 LiDAR teacher tensor."""
    output = np.zeros((4, LIDAR_BEV_HEIGHT, LIDAR_BEV_WIDTH), dtype=np.float32)
    rows = ((BEV_HALF_X - points[:, 0]) / LIDAR_BEV_RESOLUTION).astype(np.int32)
    cols = ((BEV_HALF_Y - points[:, 1]) / LIDAR_BEV_RESOLUTION).astype(np.int32)
    valid = ((rows >= 0) & (rows < LIDAR_BEV_HEIGHT) &
             (cols >= 0) & (cols < LIDAR_BEV_WIDTH) &
             np.isfinite(points).all(axis=1))
    rows, cols = rows[valid], cols[valid]
    heights = np.clip(points[valid, 2], -1.0, 4.0)
    if len(rows) == 0:
        return output
    flat = rows * LIDAR_BEV_WIDTH + cols
    count = np.bincount(flat, minlength=LIDAR_BEV_HEIGHT * LIDAR_BEV_WIDTH).astype(np.float32)
    height_sum = np.bincount(flat, weights=heights,
                             minlength=LIDAR_BEV_HEIGHT * LIDAR_BEV_WIDTH)
    height_max = np.full(LIDAR_BEV_HEIGHT * LIDAR_BEV_WIDTH, -1.0, dtype=np.float32)
    np.maximum.at(height_max, flat, heights)
    occupied = count > 0
    output[0] = np.log1p(count).reshape(LIDAR_BEV_HEIGHT, LIDAR_BEV_WIDTH)
    output[1] = np.where(occupied, height_max, 0.0).reshape(LIDAR_BEV_HEIGHT, LIDAR_BEV_WIDTH)
    output[2] = np.where(occupied, height_sum / np.maximum(count, 1), 0.0).reshape(
        LIDAR_BEV_HEIGHT, LIDAR_BEV_WIDTH)
    output[3] = occupied.reshape(LIDAR_BEV_HEIGHT, LIDAR_BEV_WIDTH).astype(np.float32)
    return output


def sparse_depth(points: np.ndarray, calibration: Dict[str, Dict]) -> np.ndarray:
    """Project one merged-LiDAR sweep to 8 camera-z depth maps at stride 4."""
    result = np.zeros((len(CAMERA_MAP), DEPTH_HEIGHT, DEPTH_WIDTH), dtype=np.float32)
    for camera_index, (meteor_name, _) in enumerate(CAMERA_MAP):
        camera = calibration[meteor_name]
        intrinsic = np.asarray(camera["K"], dtype=np.float64).copy()
        intrinsic[0, :] /= 4.0
        intrinsic[1, :] /= 4.0
        camera_from_ego = np.linalg.inv(np.asarray(camera["T_ego_cam"], dtype=np.float64))
        camera_points = points @ camera_from_ego[:3, :3].T + camera_from_ego[:3, 3]
        depth = camera_points[:, 2]
        valid = (depth > 0.5) & (depth < 79.0) & np.isfinite(camera_points).all(axis=1)
        if not valid.any():
            continue
        selected = camera_points[valid]
        selected_depth = depth[valid].astype(np.float32)
        u = (intrinsic[0, 0] * selected[:, 0] / selected[:, 2] + intrinsic[0, 2]).astype(np.int32)
        v = (intrinsic[1, 1] * selected[:, 1] / selected[:, 2] + intrinsic[1, 2]).astype(np.int32)
        inside = (u >= 0) & (u < DEPTH_WIDTH) & (v >= 0) & (v < DEPTH_HEIGHT)
        flat = np.full(DEPTH_HEIGHT * DEPTH_WIDTH, np.inf, dtype=np.float32)
        np.minimum.at(flat, v[inside] * DEPTH_WIDTH + u[inside], selected_depth[inside])
        flat[np.isinf(flat)] = 0.0
        result[camera_index] = flat.reshape(DEPTH_HEIGHT, DEPTH_WIDTH)
    return result


def meteor_boxes_with_tracks(frame: Dict) -> Tuple[np.ndarray, List[str | None]]:
    """Map NAVSIM boxes and retain aligned track IDs for future extraction."""
    annotations = frame.get("anns", {})
    boxes = np.asarray(annotations.get("gt_boxes", []), dtype=np.float32).reshape(-1, 7)
    names = list(map(str, annotations.get("gt_names", [])))
    raw_tracks = list(annotations.get("track_tokens", []))
    tracks = ([str(value) if value is not None else None for value in raw_tracks]
              if len(raw_tracks) == len(boxes) else [None] * len(boxes))
    converted = []
    for index, (box, name) in enumerate(zip(boxes, names)):
        class_id = 1 if name in VEHICLE_NAMES else 2 if name in VRU_NAMES else 0
        if not class_id or not np.isfinite(box).all():
            continue
        x, y = float(box[0]), float(box[1])
        if not (-BEV_HALF_X - 5.0 <= x <= BEV_HALF_X + 5.0 and
                -BEV_HALF_Y - 5.0 <= y <= BEV_HALF_Y + 5.0):
            continue
        converted.append((
            [class_id, x, y, float(box[3]), float(box[4]), float(box[6])],
            tracks[index],
        ))
    converted.sort(key=lambda value: value[0][1] ** 2 + value[0][2] ** 2)
    converted = converted[:MAX_BOXES]
    values = np.asarray([value for value, _ in converted],
                        dtype=np.float32).reshape(-1, 6)
    return values, [track for _, track in converted]


def meteor_boxes(frame: Dict) -> np.ndarray:
    """Map NAVSIM annotations to ``[class,x,y,length,width,yaw]``."""
    return meteor_boxes_with_tracks(frame)[0]


def box_raster(boxes: np.ndarray) -> np.ndarray:
    """Rasterize METEOR Vehicle/VRU footprints to 800x500 @ 0.2 m."""
    output = np.zeros((BEV_HEIGHT, BEV_WIDTH), dtype=np.uint8)
    for class_id, x, y, length, width, yaw in boxes:
        cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
        corners = []
        for forward, left in ((length / 2, width / 2), (length / 2, -width / 2),
                              (-length / 2, -width / 2), (-length / 2, width / 2)):
            px = x + forward * cosine - left * sine
            py = y + forward * sine + left * cosine
            row = (BEV_HALF_X - px) / BEV_RESOLUTION
            col = (BEV_HALF_Y - py) / BEV_RESOLUTION
            corners.append([col, row])
        polygon = np.round(np.asarray(corners)).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(output, [polygon], int(class_id))
    return output


def ego_motion(all_frames: Sequence[Dict], selected_indices: Sequence[int]) -> Dict[str, np.ndarray]:
    """Build METEOR's 6x0.5 s ego target and global pose cache."""
    timestamps = np.asarray([float(frame["timestamp"]) * 1e-6 for frame in all_frames], dtype=np.float64)
    positions = np.asarray([frame["ego2global_translation"][:2] for frame in all_frames], dtype=np.float64)
    yaws = np.unwrap(np.asarray([quaternion_yaw(frame["ego2global_rotation"])
                                 for frame in all_frames], dtype=np.float64))
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("NAVSIM log timestamps must be strictly increasing")
    yaw_rate = (np.gradient(yaws, timestamps) if len(timestamps) > 1
                else np.zeros_like(yaws))

    count = len(selected_indices)
    waypoints = np.zeros((count, WAYPOINT_COUNT, 2), dtype=np.float32)
    speed = np.zeros(count, dtype=np.float32)
    acceleration = np.zeros(count, dtype=np.float32)
    steering = np.zeros(count, dtype=np.float32)
    brake = np.zeros(count, dtype=np.float32)
    valid = np.zeros(count, dtype=np.float32)
    poses = np.zeros((count, 3), dtype=np.float32)

    for output_index, source_index in enumerate(selected_indices):
        frame = all_frames[source_index]
        dynamic = np.asarray(frame.get("ego_dynamic_state", [0, 0, 0, 0]), dtype=np.float64)
        speed[output_index] = float(np.linalg.norm(dynamic[:2])) if len(dynamic) >= 2 else 0.0
        acceleration[output_index] = float(dynamic[2]) if len(dynamic) >= 3 else 0.0
        steering[output_index] = (math.atan(WHEELBASE * yaw_rate[source_index] /
                                             speed[output_index])
                                  if speed[output_index] > 0.5 else 0.0)
        brake[output_index] = float(acceleration[output_index] < -0.5)
        poses[output_index] = (positions[source_index, 0], positions[source_index, 1],
                               yaws[source_index])

        query_times = timestamps[source_index] + WAYPOINT_DT * np.arange(1, WAYPOINT_COUNT + 1)
        if query_times[-1] > timestamps[-1]:
            continue
        future_x = np.interp(query_times, timestamps, positions[:, 0])
        future_y = np.interp(query_times, timestamps, positions[:, 1])
        dx = future_x - positions[source_index, 0]
        dy = future_y - positions[source_index, 1]
        cosine, sine = math.cos(yaws[source_index]), math.sin(yaws[source_index])
        waypoints[output_index, :, 0] = cosine * dx + sine * dy
        waypoints[output_index, :, 1] = -sine * dx + cosine * dy
        valid[output_index] = 1.0

    return {
        "wp": waypoints,
        "v0": speed,
        "acc": acceleration,
        "steer": steering,
        "brake": brake,
        "valid": valid,
        "pose": poses,
        "stamp": timestamps[np.asarray(selected_indices, dtype=np.int64)],
    }


def _process_log(config: Dict) -> Dict:
    log_path = Path(config["log_path"])
    sensor_root = Path(config["sensor_root"])
    output_root = Path(config["output_root"])
    scene_name = log_path.stem
    scene_root = output_root / scene_name
    manifest_path = scene_root / "manifest.json"
    if manifest_path.exists() and not config["force"]:
        return {"status": "skip", "scene": scene_name, "reason": "manifest exists"}

    with log_path.open("rb") as stream:
        raw_frames = pickle.load(stream)
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError(f"NAVSIM log is not a non-empty frame list: {log_path}")
    raw_frames = sorted(raw_frames, key=lambda frame: int(frame["timestamp"]))
    selected_indices = list(range(0, len(raw_frames), config["stride"]))
    if config["max_frames"] is not None:
        selected_indices = selected_indices[:config["max_frames"]]
    if not selected_indices:
        raise ValueError(f"No frames selected from {log_path}")

    calibration = camera_calibration(raw_frames[selected_indices[0]], sensor_root)
    validate_static_calibration(raw_frames, calibration)
    scene_root.mkdir(parents=True, exist_ok=True)
    (scene_root / "gt").mkdir(exist_ok=True)
    (scene_root / "bev_box").mkdir(exist_ok=True)
    if config["write_box_raster"]:
        (scene_root / "bev_box_raster").mkdir(exist_ok=True)
    if config["with_lidar_bev"]:
        (scene_root / "lidar_bev").mkdir(exist_ok=True)
    if config["with_depth"]:
        (scene_root / "depth4").mkdir(exist_ok=True)

    ignore_path = scene_root / "gt" / "ignore.png"
    if not cv2.imwrite(str(ignore_path), np.full((BEV_HEIGHT, BEV_WIDTH), 255, dtype=np.uint8)):
        raise OSError(f"Failed to write {ignore_path}")

    motion = ego_motion(raw_frames, selected_indices)
    np.savez_compressed(scene_root / "ego_motion.npz", **motion)

    manifest_frames = []
    for frame_index, source_index in enumerate(selected_indices):
        frame = raw_frames[source_index]
        images = {}
        for meteor_name, navsim_name in CAMERA_MAP:
            camera = frame["cams"].get(navsim_name)
            if camera is None:
                raise KeyError(f"Frame {frame.get('token')} has no {navsim_name}")
            _safe_sensor_path(sensor_root, camera["data_path"])
            images[meteor_name] = str(PurePosixPath(camera["data_path"]))

        boxes = meteor_boxes(frame)
        boxes_rel = f"bev_box/{frame_index:06d}.npz"
        np.savez_compressed(scene_root / boxes_rel, boxes=boxes)
        record = {
            "frame": frame_index,
            "source_frame_idx": int(frame.get("frame_idx", source_index)),
            "token": str(frame["token"]),
            "timestamp_us": int(frame["timestamp"]),
            "imgs": images,
            "gt": "gt/ignore.png",
            "bev_box_p": boxes_rel,
            "driving_command": np.asarray(frame.get("driving_command", []), dtype=np.int64).tolist(),
        }

        if config["write_box_raster"]:
            box_image_rel = f"bev_box_raster/{frame_index:06d}.png"
            if not cv2.imwrite(str(scene_root / box_image_rel), box_raster(boxes)):
                raise OSError(f"Failed to write {scene_root / box_image_rel}")
            record["bev_box"] = box_image_rel

        points = None
        if config["with_lidar_bev"] or config["with_depth"]:
            lidar_path = _safe_sensor_path(sensor_root, frame["lidar_path"])
            points = pcd_xyz(lidar_path)
            record["source_lidar"] = str(PurePosixPath(frame["lidar_path"]))
        if config["with_lidar_bev"]:
            lidar_rel = f"lidar_bev/{frame_index:06d}.npz"
            np.savez_compressed(scene_root / lidar_rel,
                                lb=lidar_bev(points).astype(np.float16))
            record["lidar_bev"] = lidar_rel
        if config["with_depth"]:
            depth = sparse_depth(points, calibration).astype(np.float16)
            depth_rel = f"depth4/{frame_index:06d}.npz"
            np.savez_compressed(scene_root / depth_rel, depth=depth[:6])
            depth_narrow_rel = f"depth4/{frame_index:06d}_narrow.npz"
            np.savez_compressed(scene_root / depth_narrow_rel, depth=depth[6:])
            record["depth4"] = depth_rel
            record["depth4n"] = depth_narrow_rel
        manifest_frames.append(record)

    manifest = {
        "scene": scene_name,
        "source": "NAVSIM/OpenScene",
        "split": config["split"],
        "image_root": str(sensor_root.resolve()),
        "img_hw": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "cams": calibration,
        "ego_motion": "ego_motion.npz",
        "segmentation_supervision": "ignore_255_no_maps",
        "frames": manifest_frames,
        "conversion": {
            "stride": config["stride"],
            "source_frames": len(raw_frames),
            "selected_frames": len(selected_indices),
            "with_lidar_bev": config["with_lidar_bev"],
            "with_depth": config["with_depth"],
            "write_box_raster": config["write_box_raster"],
        },
    }
    _atomic_json(manifest_path, manifest)
    return {
        "status": "ok",
        "scene": scene_name,
        "frames": len(selected_indices),
        "ego_valid": int(motion["valid"].sum()),
    }


def verify_output(output_root: Path, with_depth: bool, with_lidar_bev: bool) -> None:
    """Load one real sample through the unchanged METEOR tuple contract."""
    from torch.utils.data import DataLoader

    from bevlane.dataset import BevLaneDataset

    scenes = sorted(path.parent.name for path in output_root.glob("*/manifest.json"))
    if not scenes:
        raise RuntimeError(f"No converted manifests found under {output_root}")
    dataset = BevLaneDataset(
        str(output_root), scenes[:1], max_per_scene=2,
        with_depth=with_depth,
        with_boxdet=True,
        with_ego=True,
        with_lidarbev=with_lidar_bev,
        trim_start=0,
        trim_end=0,
    )
    if not dataset:
        raise RuntimeError("Converted dataset yielded zero METEOR samples")
    batch = next(iter(DataLoader(dataset, batch_size=min(2, len(dataset)),
                                 shuffle=False, num_workers=0)))
    shapes = [tuple(tensor.shape) for tensor in batch]
    expected_images = (min(2, len(dataset)), 8, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    if shapes[0] != expected_images:
        raise RuntimeError(f"Unexpected image batch shape {shapes[0]}, expected {expected_images}")
    print(f"VERIFY_OK scene={scenes[0]} samples={len(dataset)} shapes={shapes}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="NAVSIM dataset root containing navsim_logs/ and sensor_blobs/")
    parser.add_argument("--out", type=Path, required=True,
                        help="METEOR manifest/target output root (raw images are not copied)")
    parser.add_argument("--split", default="mini")
    parser.add_argument("--stride", type=int, default=1,
                        help="select every Nth 2 Hz NAVSIM frame")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel log converters; keep small when projecting LiDAR")
    parser.add_argument("--max-logs", type=int, default=None,
                        help="smoke-test limit; omitted means every log")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="per-log smoke-test limit after stride")
    parser.add_argument("--with-lidar-bev", action="store_true",
                        help="materialize 4x400x250 METEOR LiDAR teacher rasters")
    parser.add_argument("--with-depth", action="store_true",
                        help="materialize sparse 8-camera depth at 108x192")
    parser.add_argument("--write-box-raster", action="store_true",
                        help="also write 800x500 Vehicle/VRU occupancy PNGs")
    parser.add_argument("--force", action="store_true",
                        help="rewrite scenes whose manifest already exists")
    parser.add_argument("--verify", action="store_true",
                        help="load a real batch through BevLaneDataset after conversion")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stride < 1 or args.workers < 1:
        raise SystemExit("--stride and --workers must be positive")
    log_root = args.data_root / "navsim_logs" / args.split
    sensor_root = args.data_root / "sensor_blobs" / args.split
    if not log_root.is_dir() or not sensor_root.is_dir():
        raise SystemExit(f"Missing NAVSIM split paths: {log_root} and/or {sensor_root}")
    log_paths = sorted(log_root.glob("*.pkl"))
    if args.max_logs is not None:
        log_paths = log_paths[:args.max_logs]
    if not log_paths:
        raise SystemExit(f"No .pkl logs found under {log_root}")
    args.out.mkdir(parents=True, exist_ok=True)

    configs = [{
        "log_path": str(path),
        "sensor_root": str(sensor_root),
        "output_root": str(args.out),
        "split": args.split,
        "stride": args.stride,
        "max_frames": args.max_frames,
        "with_lidar_bev": args.with_lidar_bev,
        "with_depth": args.with_depth,
        "write_box_raster": args.write_box_raster,
        "force": args.force,
    } for path in log_paths]

    results: List[Dict] = []
    if args.workers == 1:
        iterator: Iterable[Dict] = map(_process_log, configs)
        for result in iterator:
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for result in executor.map(_process_log, configs):
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)

    converted_scenes = sorted(path.parent.name for path in args.out.glob("*/manifest.json"))
    (args.out / "scenes.txt").write_text("\n".join(converted_scenes) + "\n", encoding="utf-8")
    failures = [result for result in results if result["status"] not in {"ok", "skip"}]
    print(f"DONE scenes={len(converted_scenes)} failures={len(failures)} out={args.out}", flush=True)
    if failures:
        raise SystemExit(1)
    if args.verify:
        verify_output(args.out, args.with_depth, args.with_lidar_bev)


if __name__ == "__main__":
    main()
