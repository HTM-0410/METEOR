"""Pure NumPy/OpenCV preprocessing shared by the NAVSIM agent and tests.

METEOR and NAVSIM both use x-forward, y-left local trajectories.  The camera
calibration differs: NAVSIM exposes camera-to-LiDAR calibration while METEOR
expects an ego/LiDAR-to-camera homogeneous transform.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import cv2
import numpy as np


IMAGE_WIDTH = 768
IMAGE_HEIGHT = 432
METEOR_WAYPOINT_COUNT = 6
METEOR_WAYPOINT_DT = 0.5
METEOR_MODE_COUNT = 3

# The image backbone and geometric lift are shared across cameras, so every
# NAVSIM camera can be used as long as image/K/T stay in the same slot.  The
# first six entries preserve the closest semantic match to the Co-MLOps rig;
# the two remaining side cameras occupy METEOR's narrow-camera slots.
METEOR_CAMERA_ORDER: Tuple[Tuple[str, str], ...] = (
    ("CAM_FRONT_WIDE", "cam_f0"),
    ("CAM_FRONT_LEFT", "cam_l0"),
    ("CAM_FRONT_RIGHT", "cam_r0"),
    ("CAM_BACK_WIDE", "cam_b0"),
    ("CAM_BACK_LEFT", "cam_l2"),
    ("CAM_BACK_RIGHT", "cam_r2"),
    ("CAM_LEFT_SIDE", "cam_l1"),
    ("CAM_RIGHT_SIDE", "cam_r1"),
)


def sensor_to_lidar_inverse(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Return the conventional column-vector LiDAR-to-camera transform.

    NAVSIM stores ``p_lidar = R_sensor_to_lidar @ p_sensor + t``.  METEOR
    projects ego-frame points with ``p_camera = T_cam_ego @ p_ego``.  OpenScene
    defines its merged-LiDAR coordinates as the local vehicle frame used by
    sensor annotations, so the inverse below is the required projection
    transform.
    """

    rotation = np.asarray(rotation, dtype=np.float32)
    translation = np.asarray(translation, dtype=np.float32).reshape(3)
    if rotation.shape != (3, 3):
        raise ValueError(f"sensor2lidar rotation must be 3x3, got {rotation.shape}")
    camera_from_lidar = np.eye(4, dtype=np.float32)
    camera_from_lidar[:3, :3] = rotation.T
    camera_from_lidar[:3, 3] = -(rotation.T @ translation)
    return camera_from_lidar


def resize_rgb_and_intrinsics(
    image: np.ndarray,
    intrinsics: np.ndarray,
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize an RGB image and scale its pinhole intrinsics consistently."""

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"camera image must be HxWx3 RGB, got {image.shape}")
    source_height, source_width = image.shape[:2]
    if source_height <= 0 or source_width <= 0:
        raise ValueError("camera image has an empty dimension")

    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"camera intrinsics must be 3x3, got {intrinsics.shape}")

    if (source_width, source_height) != (width, height):
        interpolation = cv2.INTER_AREA if source_width >= width and source_height >= height else cv2.INTER_LINEAR
        image = cv2.resize(image, (width, height), interpolation=interpolation)

    scaled = intrinsics.copy()
    scaled[0, :] *= width / source_width
    scaled[1, :] *= height / source_height
    return np.ascontiguousarray(image), scaled


def _camera_fields(camera: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = ("image", "intrinsics", "sensor2lidar_rotation", "sensor2lidar_translation")
    missing = [name for name in required if getattr(camera, name, None) is None]
    if missing:
        raise ValueError(f"NAVSIM camera is missing required fields: {', '.join(missing)}")
    return (
        camera.image,
        camera.intrinsics,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
    )


def navsim_command_to_meteor(command: np.ndarray) -> np.ndarray:
    """Map NAVSIM [left, straight, right, unknown] to METEOR [straight,left,right]."""

    command = np.asarray(command, dtype=np.float32).reshape(-1)
    if command.size < 3:
        raise ValueError(f"NAVSIM driving command needs at least 3 entries, got {command.size}")
    intent = command[[1, 0, 2]].astype(np.float32, copy=True)
    if not np.isfinite(intent).all() or float(intent.sum()) <= 0:
        intent.fill(0.0)
    return intent[None]


def build_meteor_inputs(
    agent_input: Any,
    model_input_names: Iterable[str] = ("imgs", "K", "T_cam_ego", "v0"),
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT,
) -> Dict[str, np.ndarray]:
    """Convert the current NAVSIM AgentInput into METEOR ONNX inputs."""

    if not agent_input.cameras or not agent_input.ego_statuses:
        raise ValueError("NAVSIM AgentInput has no current cameras or ego status")
    cameras = agent_input.cameras[-1]
    names = set(model_input_names)

    images = []
    intrinsics = []
    transforms = []
    for _, navsim_name in METEOR_CAMERA_ORDER:
        camera = getattr(cameras, navsim_name)
        image, intrinsic, rotation, translation = _camera_fields(camera)
        image, intrinsic = resize_rgb_and_intrinsics(image, intrinsic, width, height)
        images.append(image.transpose(2, 0, 1))
        intrinsics.append(intrinsic)
        transforms.append(sensor_to_lidar_inverse(rotation, translation))

    velocity = np.asarray(agent_input.ego_statuses[-1].ego_velocity, dtype=np.float32).reshape(-1)
    if velocity.size < 2 or not np.isfinite(velocity[:2]).all():
        raise ValueError("NAVSIM ego velocity must contain two finite components")

    feed: Dict[str, np.ndarray] = {
        "imgs": np.stack(images)[None].astype(np.uint8, copy=False),
        "K": np.stack(intrinsics)[None].astype(np.float32, copy=False),
        "T_cam_ego": np.stack(transforms)[None].astype(np.float32, copy=False),
        "v0": np.array([np.linalg.norm(velocity[:2])], dtype=np.float32),
    }
    if "intent" in names:
        feed["intent"] = navsim_command_to_meteor(agent_input.ego_statuses[-1].driving_command)
    return {name: value for name, value in feed.items() if name in names}


def _headings_from_xy(points: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    previous = np.vstack([np.zeros((1, 2), dtype=np.float32), points[:-1]])
    delta = points - previous
    headings = np.zeros(len(points), dtype=np.float32)
    last_heading = 0.0
    for index, (dx, dy) in enumerate(delta):
        if float(np.hypot(dx, dy)) > epsilon:
            last_heading = float(np.arctan2(dy, dx))
        headings[index] = last_heading
    return np.unwrap(headings).astype(np.float32)


def meteor_ego_to_navsim_poses(
    ego_output: np.ndarray,
    output_count: int = 8,
    extension: str = "linear",
) -> np.ndarray:
    """Select METEOR's highest-confidence path and return NAVSIM SE(2) poses.

    The released model predicts six 0.5-second waypoints (3 seconds). NAVSIM's
    standard agent contract uses eight at the same interval (4 seconds).  The
    final two positions are either constant-velocity extrapolations from the
    last learned segment or held at the 3-second endpoint.
    """

    ego = np.asarray(ego_output, dtype=np.float32).reshape(-1)
    waypoint_values = METEOR_MODE_COUNT * METEOR_WAYPOINT_COUNT * 2
    logits_end = waypoint_values + METEOR_MODE_COUNT
    if ego.size < logits_end:
        raise ValueError(f"METEOR ego output needs at least {logits_end} values, got {ego.size}")
    if output_count < 1:
        raise ValueError("output_count must be positive")
    if not np.isfinite(ego[:logits_end]).all():
        raise ValueError("METEOR ego output contains NaN or infinity")

    modes = ego[:waypoint_values].reshape(METEOR_MODE_COUNT, METEOR_WAYPOINT_COUNT, 2)
    mode = int(np.argmax(ego[waypoint_values:logits_end]))
    points = modes[mode].copy()

    if output_count <= len(points):
        points = points[:output_count]
    else:
        extra_count = output_count - len(points)
        if extension == "linear":
            step = points[-1] - points[-2]
            extra = np.stack([points[-1] + step * (index + 1) for index in range(extra_count)])
        elif extension == "hold":
            extra = np.repeat(points[-1][None], extra_count, axis=0)
        else:
            raise ValueError(f"unknown trajectory extension policy: {extension}")
        points = np.vstack([points, extra])

    poses = np.column_stack([points, _headings_from_xy(points)]).astype(np.float32)
    if not np.isfinite(poses).all():
        raise ValueError("converted NAVSIM trajectory contains NaN or infinity")
    return poses
