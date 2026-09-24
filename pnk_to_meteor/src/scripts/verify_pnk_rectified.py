#!/usr/bin/env python3
"""Verify every rectified PNK image and its pinhole calibration contract."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from scripts.materialize_pnk_rectified import PNK_CAMERAS
from scripts.prepare_pnk_comet import atomic_json


def decode_shape(path: Path) -> tuple[int, int]:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable rectified image: {path}")
    return image.shape[:2]


def calibration_roundtrip_error(camera: dict) -> float:
    K_raw = np.asarray(camera["source_K_raw"], np.float64)
    D_raw = np.asarray(camera["source_D_raw"], np.float64)
    K_rect = np.asarray(camera["K"], np.float64)
    rays = np.asarray([[x, y, 1.0] for y in (-0.35, 0.0, 0.35)
                       for x in (-0.35, 0.0, 0.35)], np.float64)
    raw_uv = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), K_raw, D_raw)[0]
    rectified_from_raw = cv2.undistortPoints(raw_uv, K_raw, D_raw, P=K_rect).reshape(-1, 2)
    direct = (rays @ K_rect.T)
    direct = direct[:, :2] / direct[:, 2:3]
    return float(np.max(np.linalg.norm(rectified_from_raw - direct, axis=1)))


def verify(root: Path, workers: int = 8) -> dict:
    root = root.resolve()
    dataset = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    if dataset["schema"] != "pnk-rectified-v1" or not dataset["complete"]:
        raise ValueError("Expected a complete PNK rectified dataset")
    handoff = Path(dataset["source_handoff"])
    source_dataset = json.loads((handoff / "dataset.json").read_text(encoding="utf-8"))
    expected_hw = tuple(dataset["target_hw"])
    counts: Counter = Counter()
    paths = []
    min_valid_fraction = 1.0
    max_roundtrip_error = 0.0
    for manifest_path in sorted((root / "scenes").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_path = handoff / "scenes" / manifest["scene"] / "manifest.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if len(manifest["frames"]) != len(source["frames"]):
            raise ValueError(f"Frame count mismatch: {manifest['scene']}")
        image_root = Path(manifest["image_root"])
        for camera_name in PNK_CAMERAS:
            camera = manifest["cameras_rectified"][camera_name]
            if np.asarray(camera["K"]).shape != (3, 3) or any(camera["D"]):
                raise ValueError(f"Invalid pinhole contract: {manifest['scene']}/{camera_name}")
            if camera["distortion_model_status"] != "ASSUMED_REQUIRES_SOURCE_CONFIRMATION":
                raise ValueError("Unconfirmed distortion model was incorrectly promoted")
            min_valid_fraction = min(min_valid_fraction,
                                     float(camera["valid_pixel_fraction"]))
            max_roundtrip_error = max(max_roundtrip_error,
                                      calibration_roundtrip_error(camera))
            counts[camera["distortion_model"]] += 1
        for frame, raw_frame in zip(manifest["frames"], source["frames"]):
            if frame["timestamp_ns"] != raw_frame["timestamp_ns"]:
                raise ValueError(f"Timestamp mismatch: {manifest['scene']}")
            if set(frame["images_rectified"]) != set(PNK_CAMERAS):
                raise ValueError("Rectified camera set mismatch")
            paths.extend(image_root / frame["images_rectified"][camera]
                         for camera in PNK_CAMERAS)
            counts["frames"] += 1
        counts["scenes"] += 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for shape in pool.map(decode_shape, paths, chunksize=32):
            if shape != expected_hw:
                raise ValueError(f"Wrong rectified image shape: {shape}")
            counts["images"] += 1
    expected_frames = int(source_dataset["counts"]["frames"])
    if counts["frames"] != expected_frames or counts["images"] != expected_frames * 8:
        raise ValueError("Rectified dataset count mismatch")
    # OpenCV's inverse distortion is iterative; sub-0.05 px agreement is well
    # below the 768x432 sampling grid and catches wrong K/D/model wiring.
    if min_valid_fraction < 0.99 or max_roundtrip_error > 0.05:
        raise ValueError("Rectification numerical QA failed")
    result = {
        "status": "PASS_RECTIFICATION_CONTRACT_AND_IO",
        "counts": dict(counts),
        "target_hw": list(expected_hw),
        "minimum_valid_pixel_fraction": min_valid_fraction,
        "max_calibration_roundtrip_error_px": max_roundtrip_error,
        "distortion_model_status": "ASSUMED_REQUIRES_SOURCE_CONFIRMATION",
        "visual_projection_qa": "PASS_SAMPLE_APPEARANCE_ONLY",
        "training_gate": "HOLD_SOURCE_MODEL_CONFIRMATION_AND_LIDAR_BOX_OVERLAY_QA",
    }
    atomic_json(root / "verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(r"D:\Backup_rosbag\PNKData_rectified_v1"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    print(json.dumps(verify(args.root, args.workers), indent=2))


if __name__ == "__main__":
    main()
