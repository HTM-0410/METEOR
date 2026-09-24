#!/usr/bin/env python3
"""Pilot sensor-derived BEV lane labels for a short PNK sequence.

This deliberately does not call GPS an HD map.  GNSS/INS pose is used only to
move observations between frames.  Road/sidewalk comes from the existing
camera-semantic + LiDAR target, road paint requires temporal camera evidence
and LiDAR-intensity support, and road-edge requires a road/sidewalk interface.
Every generated thin class remains a review candidate with explicit support.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import laspy
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.improve_pnk_head1_head2 import pixel_warp, read_gray  # noqa: E402
from scripts.materialize_pnk_layer2 import METEOR_CAMERAS  # noqa: E402
from scripts.prepare_pnk_comet import atomic_json  # noqa: E402


ROOT_DEFAULT = Path(r"D:\Backup_rosbag\PNKData_layer2_9head_v3")
BASE_DEFAULT = Path(r"D:\Backup_rosbag\PNKData_layer2_9head_v2")
HANDOFF_DEFAULT = Path(r"D:\Backup_rosbag\PNKData_comet_handoff_v2")
SCENE_DEFAULT = "290_1772350499_1772350516"
TARGET_DEFAULT = (36, 38, 40, 42, 44)
BEV_H, BEV_W, BEV_RES = 800, 500, 0.2
MAPILLARY_MODEL = "facebook/maskformer-resnet50-vistas"
MAPILLARY_REVISION = "ae4b8c2590c0a090fc32d5c217d78738a2dd4b19"
MAPILLARY_CROSSWALK_IDS = (8, 23)  # Crosswalk - Plain, Lane Marking - Crosswalk
MAPILLARY_GENERAL_MARKING_ID = 24
MAPILLARY_PARKING_ID = 10

COLORS = np.asarray([
    [30, 30, 30],       # 0 background (not emitted by this pilot)
    [70, 170, 70],      # 1 road
    [190, 100, 40],     # 2 sidewalk
    [210, 80, 180],     # 3 crosswalk
    [255, 255, 0],      # 4 laneline
    [30, 30, 230],      # 5 stopline
    [0, 150, 255],      # 6 road edge
    [0, 230, 255],      # 7 generic marking
    [180, 100, 180],    # 8 parking
], np.uint8)


def read_bgr(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None
    if image is None:
        raise FileNotFoundError(path)
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(path)
    encoded.tofile(str(path))


def write_image(path: Path, image: np.ndarray) -> None:
    """Unicode-safe OpenCV write for PNG/JPEG QA artifacts on Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise OSError(path)
    encoded.tofile(str(path))


def to_target(points_xy: np.ndarray, source_pose: np.ndarray,
              target_pose: np.ndarray) -> np.ndarray:
    if not len(points_xy):
        return points_xy.copy()
    cs, ss = np.cos(source_pose[2]), np.sin(source_pose[2])
    ct, st = np.cos(target_pose[2]), np.sin(target_pose[2])
    gx = source_pose[0] + cs * points_xy[:, 0] - ss * points_xy[:, 1]
    gy = source_pose[1] + ss * points_xy[:, 0] + cs * points_xy[:, 1]
    dx, dy = gx - target_pose[0], gy - target_pose[1]
    return np.column_stack((ct * dx + st * dy,
                            -st * dx + ct * dy)).astype(np.float32)


def raster_points(points_xy: np.ndarray) -> np.ndarray:
    raster = np.zeros((BEV_H, BEV_W), np.uint8)
    if not len(points_xy):
        return raster
    rows = np.floor((80.0 - points_xy[:, 0]) / BEV_RES).astype(np.int32)
    cols = np.floor((50.0 - points_xy[:, 1]) / BEV_RES).astype(np.int32)
    valid = ((rows >= 0) & (rows < BEV_H)
             & (cols >= 0) & (cols < BEV_W))
    raster[rows[valid], cols[valid]] = 1
    return raster


def backproject_camera_paint(scene_dir: Path, manifest: dict,
                             frame: dict) -> np.ndarray:
    with np.load(scene_dir / frame["seg2d21"], allow_pickle=False) as value:
        semantic = value["seg"]
    with np.load(scene_dir / frame["depth4"], allow_pickle=False) as value:
        depth6 = value["depth"].astype(np.float32)
        conf6 = value["confidence"] if "confidence" in value.files \
            else (depth6 > 0).astype(np.uint8)
    with np.load(scene_dir / frame["depth4n"], allow_pickle=False) as value:
        depth2 = value["depth"].astype(np.float32)
        conf2 = value["confidence"] if "confidence" in value.files \
            else (depth2 > 0).astype(np.uint8)
    depth = np.concatenate((depth6, depth2))
    confidence = np.concatenate((conf6, conf2))
    image_root = Path(manifest["image_root"])
    points = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    for camera_index, camera_name in enumerate(METEOR_CAMERAS):
        image = read_bgr(image_root / frame["imgs"][camera_name])
        small = cv2.resize(image, (192, 108), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        white = ((hsv[..., 1] < 75) & (hsv[..., 2] > 145)
                 & (top_hat > 11))
        yellow = ((hsv[..., 0] >= 12) & (hsv[..., 0] <= 42)
                  & (hsv[..., 1] > 70) & (hsv[..., 2] > 105)
                  & (top_hat > 7))
        candidate = ((white | yellow) & (semantic[camera_index] == 11)
                     & (depth[camera_index] > 0)
                     & (confidence[camera_index] > 0))
        # Remove isolated image noise before geometry projection.
        candidate = cv2.morphologyEx(candidate.astype(np.uint8),
                                     cv2.MORPH_OPEN,
                                     np.ones((2, 2), np.uint8)) > 0
        vv, uu = np.nonzero(candidate)
        if not len(uu):
            continue
        z = depth[camera_index, vv, uu].astype(np.float64)
        camera = manifest["cams"][camera_name]
        source_h, source_w = camera.get("source_hw", [432, 768])
        K = np.asarray(camera["K"], np.float64).copy()
        K[0, :] *= 192.0 / source_w
        K[1, :] *= 108.0 / source_h
        x = (uu + 0.5 - K[0, 2]) * z / K[0, 0]
        y = (vv + 0.5 - K[1, 2]) * z / K[1, 1]
        camera_xyz = np.column_stack((x, y, z))
        T_ego_cam = np.asarray(camera["T_ego_cam"], np.float64)
        ego_xyz = camera_xyz @ T_ego_cam[:3, :3].T + T_ego_cam[:3, 3]
        usable = ((ego_xyz[:, 2] > -0.55) & (ego_xyz[:, 2] < 0.45)
                  & (ego_xyz[:, 0] > -20) & (ego_xyz[:, 0] < 80)
                  & (np.abs(ego_xyz[:, 1]) < 50))
        points.append(ego_xyz[usable, :2].astype(np.float32))
    return np.concatenate(points) if points else np.zeros((0, 2), np.float32)


def infer_mapillary_surfaces(image_root: Path, frame_by_id: dict,
                             frame_ids: list[int], cache_dir: Path) -> dict:
    """Cache crosswalk, general road-marking and parking teacher masks."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation

    cache_dir.mkdir(parents=True, exist_ok=True)
    masks = {}
    missing = []
    for frame_id in frame_ids:
        cache = cache_dir / f"{frame_id:06d}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as value:
                required = {"crosswalk", "general_marking", "parking"}
                if required.issubset(value.files):
                    masks[frame_id] = {
                        key: value[key].astype(bool) for key in required}
                else:
                    missing.append(frame_id)
        else:
            missing.append(frame_id)
    if not missing:
        return masks
    processor = AutoImageProcessor.from_pretrained(
        MAPILLARY_MODEL, revision=MAPILLARY_REVISION)
    model = MaskFormerForInstanceSegmentation.from_pretrained(
        MAPILLARY_MODEL, revision=MAPILLARY_REVISION).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for order, frame_id in enumerate(missing, 1):
        cache = cache_dir / f"{frame_id:06d}.npz"
        image_path = image_root / frame_by_id[frame_id]["imgs"]["CAM_FRONT_WIDE"]
        image = Image.open(image_path).convert("RGB")
        inputs = {key: value.to(device) for key, value in
                  processor(images=image, return_tensors="pt").items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        semantic = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]])[0].cpu().numpy()
        frame_masks = {
            "crosswalk": np.isin(semantic, MAPILLARY_CROSSWALK_IDS),
            "general_marking": semantic == MAPILLARY_GENERAL_MARKING_ID,
            "parking": semantic == MAPILLARY_PARKING_ID,
        }
        masks[frame_id] = frame_masks
        np.savez_compressed(cache,
                            crosswalk=frame_masks["crosswalk"].astype(np.uint8),
                            general_marking=frame_masks["general_marking"].astype(np.uint8),
                            parking=frame_masks["parking"].astype(np.uint8),
                            model=np.asarray(MAPILLARY_MODEL),
                            revision=np.asarray(MAPILLARY_REVISION))
        print(f"[lane-pilot] mapillary {order}/{len(missing)} frame={frame_id} "
              f"crosswalk={int(frame_masks['crosswalk'].sum())} "
              f"marking={int(frame_masks['general_marking'].sum())} "
              f"parking={int(frame_masks['parking'].sum())}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return masks


def backproject_front_mask(scene_dir: Path, manifest: dict, frame: dict,
                           full_mask: np.ndarray) -> np.ndarray:
    """Back-project a full-resolution front-camera mask through depth to ego XY."""
    with np.load(scene_dir / frame["depth4"], allow_pickle=False) as value:
        depth = value["depth"][0].astype(np.float32)
        confidence = (value["confidence"][0] if "confidence" in value.files
                      else (depth > 0).astype(np.uint8))
    mask = cv2.resize(full_mask.astype(np.uint8), (192, 108),
                      interpolation=cv2.INTER_NEAREST) > 0
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((3, 3), np.uint8)) > 0
    vv, uu = np.nonzero(mask & (depth > 0) & (confidence > 0))
    if not len(uu):
        return np.zeros((0, 2), np.float32)
    z = depth[vv, uu].astype(np.float64)
    camera = manifest["cams"]["CAM_FRONT_WIDE"]
    source_h, source_w = camera.get("source_hw", [432, 768])
    K = np.asarray(camera["K"], np.float64).copy()
    K[0, :] *= 192.0 / source_w
    K[1, :] *= 108.0 / source_h
    x = (uu + 0.5 - K[0, 2]) * z / K[0, 0]
    y = (vv + 0.5 - K[1, 2]) * z / K[1, 1]
    camera_xyz = np.column_stack((x, y, z))
    T_ego_cam = np.asarray(camera["T_ego_cam"], np.float64)
    ego_xyz = camera_xyz @ T_ego_cam[:3, :3].T + T_ego_cam[:3, 3]
    usable = ((ego_xyz[:, 2] > -0.6) & (ego_xyz[:, 2] < 0.55)
              & (ego_xyz[:, 0] > 1.0) & (ego_xyz[:, 0] < 55.0)
              & (np.abs(ego_xyz[:, 1]) < 35.0))
    return ego_xyz[usable, :2].astype(np.float32)


def lidar_paint_points(sensor_root: Path, source_frame: dict,
                       road_map: np.ndarray) -> np.ndarray:
    cloud = laspy.read(sensor_root / source_frame["lidar_merged_ego"])
    xyz = np.column_stack((cloud.x, cloud.y, cloud.z)).astype(np.float32)
    intensity = np.asarray(cloud.intensity).astype(np.float32)
    rows = np.floor((80.0 - xyz[:, 0]) / BEV_RES).astype(np.int32)
    cols = np.floor((50.0 - xyz[:, 1]) / BEV_RES).astype(np.int32)
    inside = ((rows >= 0) & (rows < BEV_H)
              & (cols >= 0) & (cols < BEV_W))
    on_road = np.zeros(len(xyz), bool)
    on_road[inside] = road_map[rows[inside], cols[inside]] == 1
    radius = np.hypot(xyz[:, 0], xyz[:, 1])
    ground = (xyz[:, 2] > -0.5) & (xyz[:, 2] < 0.4)
    selected = np.zeros(len(xyz), bool)
    # Intensity scale changes strongly with range. Select only the bright tail
    # inside each range band instead of applying one global threshold.
    for lo, hi, floor in ((2, 10, 13), (10, 20, 24), (20, 30, 7),
                          (30, 45, 4), (45, 65, 3)):
        band = on_road & ground & (radius >= lo) & (radius < hi)
        if np.count_nonzero(band) < 50:
            continue
        threshold = max(float(np.quantile(intensity[band], 0.985)), float(floor))
        selected |= band & (intensity >= threshold)
    return xyz[selected, :2]


def remove_small(mask: np.ndarray, minimum: int = 3) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask, bool)
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] >= minimum:
            keep |= labels == component
    return keep


def longitudinal_lines(marking: np.ndarray) -> np.ndarray:
    """Retain only thin, Hough-supported paint aligned with the road travel axis."""
    lane = np.zeros_like(marking, bool)
    lines = cv2.HoughLinesP(marking.astype(np.uint8) * 255, 1, np.pi / 180,
                            threshold=8, minLineLength=8, maxLineGap=3)
    if lines is None:
        return lane
    support = np.zeros_like(marking, np.uint8)
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = abs(int(x2) - int(x1)), abs(int(y2) - int(y1))
        # BEV row is longitudinal distance. Keep a strict longitudinal bias;
        # transverse crosswalk/stop paint stays generic class 7.
        if dy >= 2 * max(dx, 1):
            cv2.line(support, (int(x1), int(y1)), (int(x2), int(y2)), 1, 1)
    support = cv2.dilate(support, np.ones((3, 3), np.uint8))
    lane = marking & (support > 0)
    return lane


def transverse_stoplines(marking: np.ndarray, crosswalk: np.ndarray) -> np.ndarray:
    """Select long lateral paint bars near a detected crossing."""
    stopline = np.zeros_like(marking, bool)
    if not marking.any() or not crosswalk.any():
        return stopline
    distance = cv2.distanceTransform((~crosswalk).astype(np.uint8),
                                     cv2.DIST_L2, 3)
    near_crosswalk = distance <= 40.0  # 8 m at 0.2 m/cell
    candidate = marking & near_crosswalk & ~crosswalk
    lines = cv2.HoughLinesP(candidate.astype(np.uint8) * 255, 1, np.pi / 180,
                            threshold=4, minLineLength=6, maxLineGap=5)
    if lines is None:
        return stopline
    support = np.zeros_like(marking, np.uint8)
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = abs(int(x2) - int(x1)), abs(int(y2) - int(y1))
        length = float(np.hypot(dx, dy))
        if dx >= 2 * max(dy, 1) and 6 <= length <= 90:
            cv2.line(support, (int(x1), int(y1)), (int(x2), int(y2)), 1, 2)
    support = cv2.dilate(support, np.ones((3, 3), np.uint8))
    return remove_small(candidate & (support > 0), minimum=10)


def crosswalk_stripes(marking: np.ndarray) -> np.ndarray:
    """Find a local group of at least three parallel transverse paint stripes."""
    lines = cv2.HoughLinesP(marking.astype(np.uint8) * 255, 1, np.pi / 180,
                            threshold=5, minLineLength=4, maxLineGap=3)
    crosswalk = np.zeros_like(marking, bool)
    if lines is None:
        return crosswalk
    records = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = float(x2-x1), float(y2-y1)
        length = float(np.hypot(dx, dy))
        # Stripe direction is lateral in this ego-local pilot. Curved/diagonal
        # crossings are still accepted through the broad 25 degree grouping.
        if length < 5 or abs(dx) < 1.4 * abs(dy):
            continue
        records.append({
            "line": (int(x1), int(y1), int(x2), int(y2)),
            "mid": np.asarray([(x1+x2)/2, (y1+y2)/2], np.float32),
            "angle": math.atan2(dy, dx) % math.pi,
        })
    support = np.zeros_like(marking, np.uint8)
    accepted = set()
    for seed_index, seed in enumerate(records):
        group = []
        for index, record in enumerate(records):
            delta = abs(seed["angle"] - record["angle"])
            delta = min(delta, math.pi-delta)
            if delta <= math.radians(25) \
                    and np.linalg.norm(seed["mid"]-record["mid"]) <= 24:
                group.append(index)
        if len(group) < 3:
            continue
        # Estimate the common stripe direction with a circular mean modulo pi.
        sine = sum(math.sin(2*records[index]["angle"]) for index in group)
        cosine = sum(math.cos(2*records[index]["angle"]) for index in group)
        angle = 0.5 * math.atan2(sine, cosine)
        normal = np.asarray([-math.sin(angle), math.cos(angle)], np.float32)
        offsets = np.asarray([records[index]["mid"] @ normal for index in group])
        distinct_stripes = len(np.unique(np.round(offsets / 1.5)))
        # A single stopline/arrow may generate several Hough fragments but not
        # three separated parallel offsets. Crosswalk paint does.
        if distinct_stripes < 3 or np.ptp(offsets) < 3 or np.ptp(offsets) > 32:
            continue
        accepted.update(group)
    for index in accepted:
        x1, y1, x2, y2 = records[index]["line"]
        cv2.line(support, (x1, y1), (x2, y2), 1, 2)
    support = cv2.dilate(support, np.ones((3, 3), np.uint8))
    crosswalk = marking & (support > 0)
    return remove_small(crosswalk, minimum=8)


def colorize(label: np.ndarray) -> np.ndarray:
    image = np.zeros((*label.shape, 3), np.uint8)
    image[label == 255] = (12, 12, 12)
    for value in range(1, len(COLORS)):
        image[label == value] = COLORS[value]
    cv2.circle(image, (250, 400), 5, (255, 255, 255), -1)
    return image


def heatmap(votes: np.ndarray) -> np.ndarray:
    maximum = max(int(votes.max()), 1)
    gray = np.clip(votes.astype(np.float32) / maximum * 255, 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    out[votes == 0] = 0
    cv2.circle(out, (250, 400), 5, (255, 255, 255), -1)
    return out


def panel(image: np.ndarray, title: str, size=(500, 400)) -> np.ndarray:
    output = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(output, (0, 0), (size[0], 30), (0, 0, 0), -1)
    cv2.putText(output, title, (10, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def run(root: Path, base: Path, handoff: Path, scene_name: str,
        target_frames: tuple[int, ...], radius: int, output: Path,
        label_prefix: str = "gt_map_sensor_pilot",
        visual_every: int = 1) -> dict:
    scene_dir, base_scene = root / scene_name, base / scene_name
    manifest = json.loads((scene_dir / "manifest.json").read_text(encoding="utf-8"))
    base_manifest = json.loads((base_scene / "manifest.json").read_text(encoding="utf-8"))
    handoff_manifest = json.loads(
        (handoff / "scenes" / scene_name / "manifest.json").read_text(encoding="utf-8"))
    handoff_dataset = json.loads((handoff / "dataset.json").read_text(encoding="utf-8"))
    sensor_root = Path(handoff_dataset["source_sensor_root"])
    frames = manifest["frames"]
    source_frames = {int(frame["frame"]): frame for frame in handoff_manifest["frames"]}
    base_frames = {int(frame["frame"]): frame for frame in base_manifest["frames"]}
    frame_by_id = {int(frame["frame"]): frame for frame in frames}
    with np.load(scene_dir / manifest["nav_ego_candidate"], allow_pickle=False) as nav:
        pose4 = nav["nav_reference_pose_enu"].astype(np.float64)
        poses = pose4[:, [0, 1, 3]]
        pose_valid = nav["nav_reference_valid"].astype(bool)
    all_ids = sorted(frame_by_id)
    index_by_id = {frame_id: index for index, frame_id in enumerate(all_ids)}
    needed = set()
    for target in target_frames:
        index = index_by_id[target]
        needed.update(all_ids[max(0, index-radius):min(len(all_ids), index+radius+1)])

    output.mkdir(parents=True, exist_ok=True)
    mapillary_masks = infer_mapillary_surfaces(
        Path(manifest["image_root"]), frame_by_id, sorted(needed),
        output / "teacher_mapillary_crosswalk")
    camera_cache, lidar_cache = {}, {}
    crosswalk_cache, general_marking_cache, parking_cache = {}, {}, {}
    v3_maps, base_maps = {}, {}
    for order, frame_id in enumerate(sorted(needed), 1):
        frame = frame_by_id[frame_id]
        v3_maps[frame_id] = read_gray(scene_dir / frame["gt_map"])
        base_maps[frame_id] = read_gray(base_scene / base_frames[frame_id]["gt_map"])
        camera_cache[frame_id] = backproject_camera_paint(scene_dir, manifest, frame)
        crosswalk_cache[frame_id] = backproject_front_mask(
            scene_dir, manifest, frame, mapillary_masks[frame_id]["crosswalk"])
        general_marking_cache[frame_id] = backproject_front_mask(
            scene_dir, manifest, frame,
            mapillary_masks[frame_id]["general_marking"])
        parking_cache[frame_id] = backproject_front_mask(
            scene_dir, manifest, frame, mapillary_masks[frame_id]["parking"])
        lidar_cache[frame_id] = lidar_paint_points(
            sensor_root, source_frames[frame_id], v3_maps[frame_id])
        print(f"[lane-pilot] evidence {order}/{len(needed)} frame={frame_id} "
              f"camera={len(camera_cache[frame_id])} lidar={len(lidar_cache[frame_id])}",
              flush=True)

    result_frames = []
    generated = {}
    for target_order, target in enumerate(target_frames):
        target_index = index_by_id[target]
        target_pose = poses[target_index]
        road_votes = np.zeros((BEV_H, BEV_W), np.uint8)
        sidewalk_votes = np.zeros_like(road_votes)
        camera_votes = np.zeros_like(road_votes)
        lidar_votes = np.zeros_like(road_votes)
        crosswalk_votes = np.zeros_like(road_votes)
        general_marking_votes = np.zeros_like(road_votes)
        parking_votes = np.zeros_like(road_votes)
        source_ids = all_ids[max(0, target_index-radius):
                             min(len(all_ids), target_index+radius+1)]
        for source in source_ids:
            source_index = index_by_id[source]
            if not pose_valid[target_index] or not pose_valid[source_index]:
                continue
            warped = cv2.warpAffine(
                base_maps[source], pixel_warp(poses[source_index], target_pose),
                (BEV_W, BEV_H), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            road_votes += (warped == 1).astype(np.uint8)
            sidewalk_votes += (warped == 2).astype(np.uint8)
            camera_votes += raster_points(to_target(
                camera_cache[source], poses[source_index], target_pose))
            lidar_votes += raster_points(to_target(
                lidar_cache[source], poses[source_index], target_pose))
            crosswalk_votes += raster_points(to_target(
                crosswalk_cache[source], poses[source_index], target_pose))
            general_marking_votes += raster_points(to_target(
                general_marking_cache[source], poses[source_index], target_pose))
            parking_votes += raster_points(to_target(
                parking_cache[source], poses[source_index], target_pose))

        before = v3_maps[target].copy()
        label = before.copy()
        unknown = label == 255
        add_road = unknown & (road_votes >= 3) & (road_votes > sidewalk_votes)
        add_walk = unknown & (sidewalk_votes >= 3) & (sidewalk_votes > road_votes)
        label[add_road] = 1
        label[add_walk] = 2

        road = label == 1
        sidewalk = label == 2
        road_near = cv2.dilate(road.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        camera_near = cv2.dilate((camera_votes > 0).astype(np.uint8),
                                 np.ones((3, 3), np.uint8)) > 0
        lidar_near = cv2.dilate((lidar_votes > 0).astype(np.uint8),
                                np.ones((3, 3), np.uint8)) > 0
        # High-precision pilot: a candidate needs repeat camera evidence and
        # repeat LiDAR-intensity evidence within one 0.2 m cell.  The first
        # version admitted strong single-modality texture and visibly flooded
        # the near-road area; that output is intentionally not retained.
        lidar_repeat_near = cv2.dilate((lidar_votes >= 2).astype(np.uint8),
                                       np.ones((3, 3), np.uint8)) > 0
        crossmodal_marking = road_near & (camera_votes >= 2) & lidar_repeat_near
        rows = np.indices(road.shape)[0]
        teacher_marking = road_near & (general_marking_votes >= 4)
        marking = crossmodal_marking | teacher_marking
        marking &= (rows >= 175) & (rows <= 450)  # +45 m to -10 m
        marking = remove_small(marking, minimum=3)
        # High-precision crosswalk gate. Four pose-aligned teacher observations
        # and a sizeable connected BEV region rejected a measured night-time
        # failure where lamp reflection was predicted as one solid crosswalk.
        teacher_crosswalk = (crosswalk_votes >= 4) & road_near
        teacher_crosswalk = cv2.morphologyEx(
            teacher_crosswalk.astype(np.uint8), cv2.MORPH_CLOSE,
            np.ones((5, 5), np.uint8)) > 0
        crosswalk = remove_small(teacher_crosswalk, minimum=100)
        parking = ((parking_votes >= 4) & (road_votes >= 2)
                   & (rows >= 175) & (rows <= 500))
        parking = cv2.morphologyEx(parking.astype(np.uint8), cv2.MORPH_CLOSE,
                                   np.ones((5, 5), np.uint8)) > 0
        parking = remove_small(parking, minimum=100)
        stopline = transverse_stoplines(marking & ~crosswalk, crosswalk)
        lane = longitudinal_lines(marking & ~crosswalk & ~stopline)
        generic_marking = marking & ~lane & ~stopline & ~crosswalk

        edge = ((road & (cv2.dilate(sidewalk.astype(np.uint8),
                                     np.ones((3, 3), np.uint8)) > 0))
                | (sidewalk & (cv2.dilate(road.astype(np.uint8),
                                         np.ones((3, 3), np.uint8)) > 0)))
        edge = remove_small(edge, minimum=4) & ~marking & ~parking & ~crosswalk
        label[edge] = 6
        label[parking] = 8
        label[generic_marking] = 7
        label[lane] = 4
        label[stopline] = 5
        label[crosswalk] = 3

        confidence = np.zeros((BEV_H, BEV_W), np.uint8)
        confidence[before != 255] = 1
        confidence[add_road | add_walk] = 2
        confidence[edge] = 3
        confidence[marking] = 4
        confidence[crosswalk] = 5
        confidence[stopline] = 6
        confidence[parking] = 7
        rel_label = Path(label_prefix) / f"{target:06d}.png"
        rel_support = Path(label_prefix + "_support") / f"{target:06d}.npz"
        write_png(output / rel_label, label)
        (output / rel_support).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / rel_support,
                            confidence=confidence,
                            road_votes=road_votes,
                            sidewalk_votes=sidewalk_votes,
                            camera_paint_votes=camera_votes,
                            lidar_intensity_votes=lidar_votes,
                            mapillary_crosswalk_votes=crosswalk_votes,
                            mapillary_general_marking_votes=general_marking_votes,
                            mapillary_parking_votes=parking_votes)

        visual_path = detail_path = None
        if visual_every > 0 and target_order % visual_every == 0:
            front = read_bgr(Path(manifest["image_root"])
                             / frame_by_id[target]["imgs"]["CAM_FRONT_WIDE"])
            visual = np.vstack((
                np.hstack((panel(front, f"Front camera · frame {target}"),
                           panel(colorize(before), "BEV v3 before"),
                           panel(colorize(label), "Sensor-fused candidate"))),
                np.hstack((panel(heatmap(road_votes), "Temporal road votes"),
                           panel(heatmap(general_marking_votes), "Teacher marking votes"),
                           panel(heatmap(crosswalk_votes), "Crosswalk votes"))),
            ))
            cv2.putText(
                visual,
                f"road+={int(add_road.sum())} walk+={int(add_walk.sum())} "
                f"edge={int(edge.sum())} cross={int(crosswalk.sum())} "
                f"lane={int(lane.sum())} stop={int(stopline.sum())} "
                f"mark={int(generic_marking.sum())} park={int(parking.sum())}",
                (15, visual.shape[0]-14), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (255, 255, 255), 2, cv2.LINE_AA)
            visual_path = output / f"frame_{target:06d}_comparison.jpg"
            write_image(visual_path, visual)
            crop = np.s_[250:450, 150:350]
            thin_only = np.zeros((200, 200, 3), np.uint8)
            for class_id in (3, 4, 5, 6, 7, 8):
                thin_only[label[crop] == class_id] = COLORS[class_id]
            detail = np.hstack((
                panel(front, f"Front camera · frame {target}", size=(600, 440)),
                panel(colorize(label)[crop], "BEV near ego · all classes",
                      size=(560, 440)),
                panel(thin_only, "METEOR thin/map classes", size=(560, 440)),
            ))
            detail_path = output / f"frame_{target:06d}_lane_detail.jpg"
            write_image(detail_path, detail)
        record = {
            "frame": target, "sources": source_ids,
            "before_valid_pixels": int(np.count_nonzero(before != 255)),
            "after_valid_pixels": int(np.count_nonzero(label != 255)),
            "before_coverage": float(np.mean(before != 255)),
            "after_coverage": float(np.mean(label != 255)),
            "added_road_pixels": int(add_road.sum()),
            "added_sidewalk_pixels": int(add_walk.sum()),
            "road_edge_pixels": int(edge.sum()),
            "crosswalk_candidate_pixels": int(crosswalk.sum()),
            "mapillary_crosswalk_supported_cells": int(
                np.count_nonzero(crosswalk_votes)),
            "laneline_candidate_pixels": int(lane.sum()),
            "stopline_candidate_pixels": int(stopline.sum()),
            "generic_marking_pixels": int(generic_marking.sum()),
            "parking_candidate_pixels": int(parking.sum()),
            "camera_paint_supported_cells": int(np.count_nonzero(camera_votes)),
            "lidar_intensity_supported_cells": int(np.count_nonzero(lidar_votes)),
            "cross_modal_marking_pixels": int(
                np.count_nonzero(marking & camera_near & lidar_near)),
            "label": rel_label.as_posix(), "support": rel_support.as_posix(),
            "visualization": visual_path.name if visual_path else None,
            "lane_visualization": detail_path.name if detail_path else None,
        }
        result_frames.append(record)
        generated[target] = (label, poses[target_index])

    # Distance-tolerant temporal agreement between generated thin labels.
    agreement_matches = agreement_total = 0
    agreement_by_class = {}
    for first, second in zip(target_frames[:-1], target_frames[1:]):
        first_label, first_pose = generated[first]
        second_label, second_pose = generated[second]
        for class_id in (3, 4, 5, 6, 7, 8):
            warped = cv2.warpAffine(
                (first_label == class_id).astype(np.uint8),
                pixel_warp(first_pose, second_pose), (BEV_W, BEV_H),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                borderValue=0)
            current = second_label == class_id
            near = cv2.dilate(current.astype(np.uint8),
                              np.ones((5, 5), np.uint8)) > 0
            agreement_matches += int(np.count_nonzero((warped > 0) & near))
            agreement_total += int(np.count_nonzero(warped))
            stats = agreement_by_class.setdefault(
                str(class_id), {"matches": 0, "comparable": 0})
            stats["matches"] += int(np.count_nonzero((warped > 0) & near))
            stats["comparable"] += int(np.count_nonzero(warped))

    for stats in agreement_by_class.values():
        stats["agreement_within_0.4m"] = (
            stats["matches"] / stats["comparable"]
            if stats["comparable"] else None)

    report = {
        "status": "PNK_BEV_LANE_SENSOR_FUSION_PILOT_REVIEW_REQUIRED",
        "scene": scene_name, "target_frames": list(target_frames),
        "temporal_radius_frames": radius,
        "sources": {
            "road_sidewalk": "v2 single-sweep semantic occupancy warped by NAV pose",
            "paint_camera": "rectified RGB color/local-contrast inside panoptic road",
            "paint_lidar": "range-adaptive bright intensity tail on ground road points",
            "crosswalk": (f"{MAPILLARY_MODEL}@{MAPILLARY_REVISION} front-camera "
                          "semantic mask projected by depth and fused by NAV pose"),
            "general_marking_parking": "Mapillary front-camera semantics plus depth and NAV fusion",
            "pose": "nav_reference_pose_enu candidate",
        },
        "class_policy": {
            "1": "road", "2": "sidewalk",
            "3": "Mapillary crosswalk semantics with depth, >=4 temporal votes and >=100 BEV pixels",
            "4": "longitudinal paint candidate",
            "5": "transverse paint near crosswalk candidate",
            "6": "observed road-sidewalk interface", "7": "generic road paint",
            "8": "Mapillary parking semantics with depth and temporal consensus",
            "255": "ignore",
        },
        "confidence": {
            "1": "existing v3", "2": "extended temporal consensus",
            "3": "road-sidewalk interface", "4": "cross-sensor/strong-temporal paint",
            "5": "Mapillary crosswalk with depth and temporal consensus",
            "6": "stopline geometry near crossing",
            "7": "parking semantics with temporal consensus",
        },
        "frames": result_frames,
        "thin_label_temporal_agreement_within_0.4m": (
            agreement_matches / agreement_total if agreement_total else None),
        "thin_label_temporal_comparable_pixels": agreement_total,
        "temporal_agreement_by_class": agreement_by_class,
        "limits": [
            "LiDAR deskew and GNSS lever arm remain unverified",
            "paint class is pseudo-label and needs visual review",
            "longitudinal paint is not guaranteed to be a legal lane boundary",
            "OSM or GPS geometry is not treated as GT",
        ],
    }
    atomic_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--base", type=Path, default=BASE_DEFAULT)
    parser.add_argument("--handoff", type=Path, default=HANDOFF_DEFAULT)
    parser.add_argument("--scene", default=SCENE_DEFAULT)
    parser.add_argument("--frames", default=",".join(map(str, TARGET_DEFAULT)))
    parser.add_argument("--radius", type=int, default=8)
    parser.add_argument("--label-prefix", default="gt_map_sensor_pilot")
    parser.add_argument("--visual-every", type=int, default=1,
                        help="write QA visual every N target frames; 0 disables")
    parser.add_argument("--output", type=Path,
                        default=REPO / "out" / "pnk_bev_lane_sensor_pilot")
    args = parser.parse_args()
    targets = tuple(int(value) for value in args.frames.split(","))
    report = run(args.root.resolve(), args.base.resolve(), args.handoff.resolve(),
                 args.scene, targets, args.radius, args.output.resolve(),
                 label_prefix=args.label_prefix,
                 visual_every=args.visual_every)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
