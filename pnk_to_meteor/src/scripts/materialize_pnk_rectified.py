#!/usr/bin/env python3
"""Rectify PNK camera images into one pinhole contract for METEOR.

The source provides distortion coefficients but no model name.  This stage
uses the OpenCV convention implied by coefficient count: Brown for D5 and
rational-polynomial for D8.  That assumption is recorded in every manifest;
it is not silently promoted to source-confirmed calibration.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from scripts.prepare_pnk_comet import DEFAULT_OUTPUT as DEFAULT_HANDOFF, atomic_json, sha256


DEFAULT_OUTPUT = Path(r"D:\Backup_rosbag\PNKData_rectified_v1")
TARGET_HW = (432, 768)
PNK_CAMERAS = ("CAM_P_F", "CAM_P_FL", "CAM_P_FR", "CAM_P_L", "CAM_P_R",
               "CAM_P_B", "CAM_P_LB", "CAM_P_RB")


def distortion_model(coefficients: np.ndarray) -> str:
    size = int(np.asarray(coefficients).size)
    if size == 5:
        return "opencv_brown5_assumed_by_coefficient_count"
    if size == 8:
        return "opencv_rational8_assumed_by_coefficient_count"
    raise ValueError(f"Unsupported PNK distortion vector length: {size}")


def rectification_contract(camera: dict, target_hw: tuple[int, int],
                           alpha: float) -> tuple[dict, np.ndarray, np.ndarray]:
    source_h, source_w = map(int, camera["image_hw"])
    target_h, target_w = map(int, target_hw)
    K_raw = np.asarray(camera["K_raw"], dtype=np.float64)
    D_raw = np.asarray(camera["D_raw"], dtype=np.float64).reshape(-1)
    model = distortion_model(D_raw)
    K_rect, roi = cv2.getOptimalNewCameraMatrix(
        K_raw, D_raw, (source_w, source_h), alpha, (target_w, target_h),
        centerPrincipalPoint=False)
    map1, map2 = cv2.initUndistortRectifyMap(
        K_raw, D_raw, np.eye(3), K_rect, (target_w, target_h), cv2.CV_16SC2)
    valid_source = np.full((source_h, source_w), 255, np.uint8)
    valid = cv2.remap(valid_source, map1, map2, cv2.INTER_NEAREST,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    contract = {
        "K": K_rect.tolist(),
        "T_cam_ego": camera["T_cam_ego"],
        "D": [0.0] * len(D_raw),
        "source_K_raw": K_raw.tolist(),
        "source_D_raw": D_raw.tolist(),
        "source_image_hw": [source_h, source_w],
        "image_hw": [target_h, target_w],
        "distortion_model": model,
        "distortion_model_status": "ASSUMED_REQUIRES_SOURCE_CONFIRMATION",
        "rectification": "opencv_initUndistortRectifyMap",
        "alpha": float(alpha),
        "valid_pixel_fraction": float(np.count_nonzero(valid) / valid.size),
        "valid_roi_xywh": [int(value) for value in roi],
        "geometry_status": "RECTIFIED_WITH_UNCONFIRMED_SOURCE_MODEL",
    }
    return contract, map1, map2


def read_image(path: Path) -> np.ndarray:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as error:
        raise FileNotFoundError(path) from error
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable source image: {path}")
    return image


def atomic_jpg(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image,
                               [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise OSError(f"JPEG encode failed: {path}")
    fd, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=".jpg",
                                     dir=path.parent)
    os.close(fd)
    try:
        Path(temporary).write_bytes(encoded.tobytes())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_image(path: Path, target_hw: tuple[int, int]) -> None:
    image = read_image(path)
    if image.shape[:2] != tuple(target_hw):
        raise ValueError(f"Wrong rectified image shape {image.shape[:2]}: {path}")


def materialize(handoff: Path, output: Path, target_hw: tuple[int, int] = TARGET_HW,
                alpha: float = 0.0, jpeg_quality: int = 95,
                max_frames: int | None = None) -> dict:
    handoff, output = handoff.resolve(), output.resolve()
    dataset_path = handoff / "dataset.json"
    source_dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if source_dataset["schema"] != "pnk-comet-handoff-v1":
        raise ValueError("Expected the verified PNK handoff")
    if output == handoff or output.is_relative_to(handoff):
        raise ValueError("Rectified output must be separate from the handoff")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg quality must be between 1 and 100")
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "schema": "pnk-rectification-config-v1",
        "source_handoff": str(handoff),
        "source_handoff_dataset_sha256": sha256(dataset_path),
        "target_hw": list(target_hw), "alpha": float(alpha),
        "jpeg_quality": int(jpeg_quality),
        "model_rule": "D5=OpenCV_Brown; D8=OpenCV_rational",
        "model_status": "ASSUMED_REQUIRES_SOURCE_CONFIRMATION",
    }
    config_path = output / "rectification_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError("Rectification output already uses a different config")
    else:
        atomic_json(config_path, config)

    sensor_root = Path(source_dataset["source_sensor_root"])
    counts: Counter = Counter()
    scene_records = []
    selected_frames = 0
    stop = False
    for source_path in sorted((handoff / "scenes").glob("*/manifest.json")):
        source = json.loads(source_path.read_text(encoding="utf-8"))
        scene = source["scene"]
        scene_root = output / "scenes" / scene
        rectified_cameras, maps = {}, {}
        for camera_name in PNK_CAMERAS:
            contract, map1, map2 = rectification_contract(
                source["cameras_raw"][camera_name], target_hw, alpha)
            rectified_cameras[camera_name] = contract
            maps[camera_name] = (map1, map2)
        frames = []
        for frame in source["frames"]:
            if max_frames is not None and selected_frames >= max_frames:
                stop = True
                break
            images = {}
            for camera_name in PNK_CAMERAS:
                relative = Path("images") / camera_name / f'{int(frame["frame"]):06d}.jpg'
                destination = scene_root / relative
                if destination.exists():
                    validate_image(destination, target_hw)
                    counts["images_reused"] += 1
                else:
                    raw_path = sensor_root / frame["images_raw"][camera_name]
                    raw = read_image(raw_path)
                    expected_hw = tuple(source["cameras_raw"][camera_name]["image_hw"])
                    if raw.shape[:2] != expected_hw:
                        raise ValueError(f"Source image/calibration shape mismatch: {raw_path}")
                    map1, map2 = maps[camera_name]
                    rectified = cv2.remap(raw, map1, map2, cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=(0, 0, 0))
                    atomic_jpg(destination, rectified, jpeg_quality)
                    counts["images_created"] += 1
                images[camera_name] = relative.as_posix()
            record = dict(frame)
            record["images_rectified"] = images
            record["distortion_status"] = "rectified_assumed_model"
            frames.append(record)
            selected_frames += 1
            counts["frames"] += 1
            if counts["frames"] % 25 == 0:
                print(f'[rectify] {counts["frames"]} frames', flush=True)
        if frames:
            document = dict(source)
            document.update({
                "schema": "pnk-rectified-scene-v1",
                "source_handoff_manifest": str(source_path.resolve()),
                "image_root": str(scene_root.resolve()),
                "img_hw": list(target_hw),
                "cameras_rectified": rectified_cameras,
                "frames": frames,
                "distortion_contract": config,
            })
            atomic_json(scene_root / "manifest.json", document)
            scene_records.append({"scene": scene, "frames": len(frames),
                                  "split": source["split"]})
            counts["scenes"] += 1
        if stop:
            break
    result = {
        "schema": "pnk-rectified-v1",
        "source_handoff": str(handoff),
        "source_handoff_dataset_sha256": sha256(dataset_path),
        "complete": max_frames is None and counts["frames"] == source_dataset["counts"]["frames"],
        "counts": dict(counts), "scenes": scene_records,
        "target_hw": list(target_hw), "alpha": float(alpha),
        "jpeg_quality": int(jpeg_quality),
        "distortion_model_status": "ASSUMED_REQUIRES_SOURCE_CONFIRMATION",
        "camera_geometry_gate": "HOLD_SOURCE_MODEL_CONFIRMATION_AND_OVERLAY_QA",
    }
    atomic_json(output / "dataset.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--height", type=int, default=TARGET_HW[0])
    parser.add_argument("--width", type=int, default=TARGET_HW[1])
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if args.height < 1 or args.width < 1:
        parser.error("height and width must be positive")
    result = materialize(args.handoff, args.output, (args.height, args.width),
                         args.alpha, args.jpeg_quality, args.max_frames)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
