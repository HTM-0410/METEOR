import json
from pathlib import Path

import cv2
import numpy as np

from bevlane.dataset import (CAMS, BevLaneDataset,
                             navsim_driving_command_to_meteor)
from bevlane.ingest_navsim import ego_motion, meteor_boxes, pcd_xyz


def test_binary_pcd_reader_preserves_xyz(tmp_path: Path):
    path = tmp_path / "sample.pcd"
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("intensity", "u1"), ("lidar_info", "u1"), ("ring", "u1"),
    ])
    points = np.zeros(2, dtype=dtype)
    points["x"] = [1.0, -2.0]
    points["y"] = [3.0, 4.0]
    points["z"] = [5.0, 6.0]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity lidar_info ring\n"
        "SIZE 4 4 4 1 1 1\n"
        "TYPE F F F U U U\n"
        "COUNT 1 1 1 1 1 1\n"
        "WIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA binary\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        points.tofile(stream)

    np.testing.assert_allclose(pcd_xyz(path), [[1, 3, 5], [-2, 4, 6]])


def test_ego_motion_uses_half_second_future_in_current_ego_frame():
    frames = []
    for index in range(8):
        frames.append({
            "timestamp": int(index * 500_000),
            "ego2global_translation": np.array([float(index), 0.0, 0.0]),
            "ego2global_rotation": np.array([1.0, 0.0, 0.0, 0.0]),
            "ego_dynamic_state": [2.0, 0.0, 0.0, 0.0],
        })

    motion = ego_motion(frames, [0, 1])

    np.testing.assert_allclose(motion["wp"][0, :, 0], np.arange(1, 7))
    np.testing.assert_allclose(motion["wp"][0, :, 1], 0.0)
    np.testing.assert_allclose(motion["v0"], 2.0)
    np.testing.assert_allclose(motion["valid"], 1.0)


def test_box_mapping_keeps_vehicle_and_vru_only():
    frame = {
        "anns": {
            "gt_boxes": np.array([
                [10, 0, 0, 4.0, 2.0, 1.5, 0.1],
                [5, 1, 0, 0.5, 0.5, 1.7, -0.2],
                [3, -1, 0, 1.8, 0.6, 1.4, 0.3],
                [2, 2, 0, 0.4, 0.4, 1.0, 0.0],
            ], dtype=np.float32),
            "gt_names": np.array(["vehicle", "pedestrian", "bicycle", "traffic_cone"]),
        }
    }

    boxes = meteor_boxes(frame)

    assert boxes.shape == (3, 6)
    assert boxes[:, 0].tolist() == [2.0, 2.0, 1.0]


def test_navsim_command_mapping_preserves_unknown_as_no_command():
    np.testing.assert_array_equal(
        navsim_driving_command_to_meteor([1, 0, 0, 0]), [0, 1, 0])
    np.testing.assert_array_equal(
        navsim_driving_command_to_meteor([0, 1, 0, 0]), [1, 0, 0])
    np.testing.assert_array_equal(
        navsim_driving_command_to_meteor([0, 0, 1, 0]), [0, 0, 1])
    np.testing.assert_array_equal(
        navsim_driving_command_to_meteor([0, 0, 0, 1]), [0, 0, 0])


def test_bevlane_dataset_reads_external_images_and_resizes(tmp_path: Path):
    raw_root = tmp_path / "raw"
    scene_root = tmp_path / "converted" / "scene-a"
    raw_root.mkdir()
    (scene_root / "gt").mkdir(parents=True)
    source = np.full((100, 200, 3), 127, dtype=np.uint8)
    images = {}
    for index, camera in enumerate(CAMS):
        relative = f"camera_{index}.jpg"
        assert cv2.imwrite(str(raw_root / relative), source)
        images[camera] = relative
    assert cv2.imwrite(str(scene_root / "gt" / "ignore.png"),
                       np.full((800, 500), 255, dtype=np.uint8))

    calibration = {
        camera: {"K": np.eye(3).tolist(), "T_ego_cam": np.eye(4).tolist()}
        for camera in CAMS
    }
    manifest = {
        "scene": "scene-a",
        "image_root": str(raw_root),
        "img_hw": [432, 768],
        "cams": calibration,
        "frames": [{"frame": 0, "imgs": images, "gt": "gt/ignore.png",
                    "gt_map": "gt/ignore.png",
                    "driving_command": [1, 0, 0, 0]}],
    }
    (scene_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    dataset = BevLaneDataset(str(tmp_path / "converted"), ["scene-a"])
    image_tensor, intrinsics, transforms, ground_truth = dataset[0]

    assert tuple(image_tensor.shape) == (8, 3, 432, 768)
    assert tuple(intrinsics.shape) == (8, 3, 3)
    assert tuple(transforms.shape) == (8, 4, 4)
    assert tuple(ground_truth.shape) == (800, 500)
    assert int(ground_truth.min()) == 255

    map_dataset = BevLaneDataset(str(tmp_path / "converted"), ["scene-a"],
                                 gt_key="gt_map", yaw_fix_deg=5.0)
    map_ground_truth = map_dataset[0][3]
    assert int(map_ground_truth.min()) == 255

    command_dataset = BevLaneDataset(str(tmp_path / "converted"), ["scene-a"],
                                     with_command=True)
    np.testing.assert_array_equal(command_dataset[0][-1].numpy(), [0, 1, 0])
