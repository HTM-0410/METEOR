from types import SimpleNamespace

import numpy as np

from navsim_meteor.geometry import (
    METEOR_CAMERA_ORDER,
    build_meteor_inputs,
    meteor_ego_to_navsim_poses,
    navsim_command_to_meteor,
    resize_rgb_and_intrinsics,
    sensor_to_lidar_inverse,
)


def test_sensor_to_lidar_inverse_round_trip():
    angle = np.deg2rad(30.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]],
        dtype=np.float32,
    )
    translation = np.array([1.2, -0.4, 1.8], dtype=np.float32)
    camera_from_lidar = sensor_to_lidar_inverse(rotation, translation)
    lidar_from_camera = np.eye(4, dtype=np.float32)
    lidar_from_camera[:3, :3] = rotation
    lidar_from_camera[:3, 3] = translation
    np.testing.assert_allclose(camera_from_lidar @ lidar_from_camera, np.eye(4), atol=1e-6)


def test_resize_scales_intrinsics_with_image():
    image = np.zeros((900, 1600, 3), dtype=np.uint8)
    intrinsic = np.array([[1000, 0, 800], [0, 900, 450], [0, 0, 1]], dtype=np.float32)
    resized, scaled = resize_rgb_and_intrinsics(image, intrinsic)
    assert resized.shape == (432, 768, 3)
    np.testing.assert_allclose(scaled, [[480, 0, 384], [0, 432, 216], [0, 0, 1]])


def test_command_order_is_navsim_to_meteor():
    np.testing.assert_array_equal(navsim_command_to_meteor([1, 0, 0, 0]), [[0, 1, 0]])
    np.testing.assert_array_equal(navsim_command_to_meteor([0, 1, 0, 0]), [[1, 0, 0]])
    np.testing.assert_array_equal(navsim_command_to_meteor([0, 0, 1, 0]), [[0, 0, 1]])


def test_build_inputs_uses_all_eight_cameras_and_current_speed():
    camera = SimpleNamespace(
        image=np.zeros((900, 1600, 3), dtype=np.uint8),
        intrinsics=np.eye(3, dtype=np.float32),
        sensor2lidar_rotation=np.eye(3, dtype=np.float32),
        sensor2lidar_translation=np.zeros(3, dtype=np.float32),
    )
    cameras = SimpleNamespace(**{navsim_name: camera for _, navsim_name in METEOR_CAMERA_ORDER})
    status = SimpleNamespace(
        ego_velocity=np.array([3.0, 4.0], dtype=np.float32),
        driving_command=np.array([0, 1, 0, 0], dtype=np.float32),
    )
    agent_input = SimpleNamespace(cameras=[cameras], ego_statuses=[status])
    feed = build_meteor_inputs(agent_input, ("imgs", "K", "T_cam_ego", "v0", "intent"))
    assert feed["imgs"].shape == (1, 8, 3, 432, 768)
    assert feed["K"].shape == (1, 8, 3, 3)
    assert feed["T_cam_ego"].shape == (1, 8, 4, 4)
    np.testing.assert_allclose(feed["v0"], [5.0])
    np.testing.assert_array_equal(feed["intent"], [[1, 0, 0]])


def test_selects_best_mode_extends_to_four_seconds_and_builds_heading():
    ego = np.zeros(42, dtype=np.float32)
    straight = np.array([[i, 0] for i in range(1, 7)], dtype=np.float32)
    left = np.array([[i, 0.1 * i * i] for i in range(1, 7)], dtype=np.float32)
    right = left * np.array([1, -1], dtype=np.float32)
    ego[:36] = np.stack([straight, left, right]).reshape(-1)
    ego[36:39] = [0, 4, 1]

    poses = meteor_ego_to_navsim_poses(ego, output_count=8, extension="linear")
    assert poses.shape == (8, 3)
    np.testing.assert_allclose(poses[:6, :2], left)
    np.testing.assert_allclose(poses[6, :2], left[-1] + (left[-1] - left[-2]))
    np.testing.assert_allclose(poses[7, :2], left[-1] + 2 * (left[-1] - left[-2]))
    assert np.all(poses[:, 2] >= 0)


def test_hold_extension_keeps_endpoint():
    ego = np.zeros(42, dtype=np.float32)
    ego[:12] = np.array([[i, 0] for i in range(1, 7)], dtype=np.float32).reshape(-1)
    ego[36] = 1
    poses = meteor_ego_to_navsim_poses(ego, output_count=8, extension="hold")
    np.testing.assert_allclose(poses[5:, :2], [[6, 0], [6, 0], [6, 0]])
