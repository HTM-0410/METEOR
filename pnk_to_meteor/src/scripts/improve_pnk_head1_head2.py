#!/usr/bin/env python3
"""Build PNK 9-head v3 with improved Head-1 drivable and Head-2 depth GT.

Head 1 accumulates neighboring road/sidewalk labels after ego-motion warping.
Only unknown current cells with at least two agreeing temporal votes are added.

Head 2 preserves every measured LiDAR pixel exactly and fills small holes only
on static semantic surfaces when at least two same-class neighbors have locally
consistent metric depth.  A confidence raster records measured vs interpolated.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.prepare_pnk_comet import atomic_json, sha256  # noqa: E402
from scripts.verify_pnk_9head_profile import verify as verify_profile  # noqa: E402


SOURCE_DEFAULT = Path(r"D:\Backup_rosbag\PNKData_layer2_9head_v2")
OUTPUT_DEFAULT = Path(r"D:\Backup_rosbag\PNKData_layer2_9head_v3")
BEV_H, BEV_W, BEV_RES = 800, 500, 0.2
TEMPORAL_RADIUS = 2
TEMPORAL_MAX_DELTA_NS = 600_000_000
STATIC_SEG21 = np.asarray([9, 10, 11, 12, 16, 17, 18, 19], np.int64)
NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1),
             (-1, -1), (-1, 1), (1, -1), (1, 1))


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz",
                                     dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(path)
    fd, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=".png",
                                     dir=path.parent)
    os.close(fd)
    try:
        Path(temporary).write_bytes(encoded.tobytes())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_gray(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE) if raw.size else None
    if image is None:
        raise FileNotFoundError(path)
    return image


def clone_with_hardlinks(source: Path, output: Path) -> None:
    if output.exists():
        if any(output.iterdir()):
            raise FileExistsError(f"Output must be empty: {output}")
        output.rmdir()
    shutil.copytree(source, output, copy_function=os.link)


def pixel_warp(source_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    """Affine transform mapping source BEV pixels into target BEV pixels."""
    source_pixels = np.asarray([[0, 0], [BEV_W - 1, 0],
                                [0, BEV_H - 1]], np.float32)
    c_s, s_s = np.cos(source_pose[2]), np.sin(source_pose[2])
    c_t, s_t = np.cos(target_pose[2]), np.sin(target_pose[2])
    target_pixels = []
    for col, row in source_pixels:
        x_s, y_s = 80.0 - row * BEV_RES, 50.0 - col * BEV_RES
        x_g = source_pose[0] + c_s * x_s - s_s * y_s
        y_g = source_pose[1] + s_s * x_s + c_s * y_s
        dx, dy = x_g - target_pose[0], y_g - target_pose[1]
        x_t = c_t * dx + s_t * dy
        y_t = -s_t * dx + c_t * dy
        target_pixels.append([(50.0 - y_t) / BEV_RES,
                              (80.0 - x_t) / BEV_RES])
    return cv2.getAffineTransform(source_pixels,
                                  np.asarray(target_pixels, np.float32))


def temporal_drivable(maps: list[np.ndarray], timestamps: np.ndarray,
                      poses: np.ndarray, pose_valid: np.ndarray,
                      target_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current = maps[target_index]
    road_votes = np.zeros(current.shape, np.uint8)
    sidewalk_votes = np.zeros(current.shape, np.uint8)
    if pose_valid[target_index]:
        lo = max(0, target_index - TEMPORAL_RADIUS)
        hi = min(len(maps), target_index + TEMPORAL_RADIUS + 1)
        for source_index in range(lo, hi):
            if source_index == target_index or not pose_valid[source_index]:
                continue
            if abs(int(timestamps[source_index]) - int(timestamps[target_index])) \
                    > TEMPORAL_MAX_DELTA_NS:
                continue
            warped = cv2.warpAffine(
                maps[source_index], pixel_warp(poses[source_index], poses[target_index]),
                (BEV_W, BEV_H), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            road_votes += (warped == 1).astype(np.uint8)
            sidewalk_votes += (warped == 2).astype(np.uint8)
    output = current.copy()
    unknown = output == 255
    add_road = unknown & (road_votes >= 2) & (road_votes > sidewalk_votes)
    add_sidewalk = unknown & (sidewalk_votes >= 2) & (sidewalk_votes > road_votes)
    output[add_road] = 1
    output[add_sidewalk] = 2
    support = np.maximum(road_votes, sidewalk_votes)
    confidence = np.zeros(current.shape, np.uint8)
    confidence[current != 255] = 1
    confidence[add_road | add_sidewalk] = 2
    return output, confidence, support


def shift_pair(depth: np.ndarray, semantic: np.ndarray,
               dy: int, dx: int) -> tuple[np.ndarray, np.ndarray]:
    shifted_depth = np.zeros_like(depth)
    shifted_semantic = np.full_like(semantic, 255)
    y0, y1 = max(0, dy), depth.shape[0] + min(0, dy)
    x0, x1 = max(0, dx), depth.shape[1] + min(0, dx)
    shifted_depth[y0:y1, x0:x1] = depth[y0-dy:y1-dy, x0-dx:x1-dx]
    shifted_semantic[y0:y1, x0:x1] = semantic[y0-dy:y1-dy, x0-dx:x1-dx]
    return shifted_depth, shifted_semantic


def guided_depth_fill(depth: np.ndarray, semantic: np.ndarray,
                      iterations: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Conservative same-class local interpolation; measured pixels stay exact."""
    output = depth.astype(np.float32, copy=True)
    confidence = (output > 0).astype(np.uint8)  # 1=measured, 2=interpolated
    for _ in range(iterations):
        neighbors = []
        for dy, dx in NEIGHBORS:
            shifted_depth, shifted_semantic = shift_pair(output, semantic, dy, dx)
            neighbors.append(np.where(
                (shifted_depth > 0) & (shifted_semantic == semantic),
                shifted_depth, np.nan))
        values = np.stack(neighbors)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(values, axis=0)
            minimum = np.nanmin(values, axis=0)
            maximum = np.nanmax(values, axis=0)
        count = np.isfinite(values).sum(axis=0)
        consistent = (maximum - minimum) <= np.maximum(1.5, 0.08 * median)
        fill = ((output == 0) & np.isin(semantic, STATIC_SEG21)
                & (count >= 2) & consistent)
        output[fill] = median[fill]
        confidence[fill] = 2
    return output, confidence


def load_depth(scene_dir: Path, frame: dict) -> np.ndarray:
    with np.load(scene_dir / frame["depth4"], allow_pickle=False) as z:
        depth6 = z["depth"].astype(np.float32)
    with np.load(scene_dir / frame["depth4n"], allow_pickle=False) as z:
        depth2 = z["depth"].astype(np.float32)
    return np.concatenate((depth6, depth2))


def process_scene(job: tuple[str, str]) -> dict:
    source_scene_raw, output_scene_raw = job
    source_scene, output_scene = Path(source_scene_raw), Path(output_scene_raw)
    manifest_path = output_scene / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_manifest = json.loads((source_scene / "manifest.json").read_text(encoding="utf-8"))
    frames = manifest["frames"]
    source_frames = source_manifest["frames"]
    with np.load(output_scene / manifest["nav_ego_candidate"], allow_pickle=False) as nav:
        raw_pose = nav["nav_reference_pose_enu"].astype(np.float64)
        poses = raw_pose[:, [0, 1, 3]]
        pose_valid = nav["nav_reference_valid"].astype(bool)
        timestamps = nav["timestamp_ns"].astype(np.int64)
    maps = [read_gray(source_scene / frame["gt_map"]) for frame in source_frames]
    counts: Counter = Counter()
    depth_by_camera_measured = np.zeros(8, np.int64)
    depth_by_camera_total = np.zeros(8, np.int64)
    validation_errors = []
    validation_eligible = validation_recovered = 0
    for index, (frame, source_frame) in enumerate(zip(frames, source_frames)):
        fi = int(frame["frame"])
        # Head 1: temporally fused road/sidewalk.
        drivable, drive_conf, drive_support = temporal_drivable(
            maps, timestamps, poses, pose_valid, index)
        drive_rel = Path("gt_drivable_temporal") / f"{fi:06d}.png"
        support_rel = Path("gt_drivable_temporal_support") / f"{fi:06d}.npz"
        atomic_png(output_scene / drive_rel, drivable)
        atomic_npz(output_scene / support_rel, confidence=drive_conf,
                   temporal_votes=drive_support)
        original_valid = maps[index] != 255
        valid = drivable != 255
        added = valid & ~original_valid
        frame["gt_map"] = drive_rel.as_posix()
        frame["gt_map_support"] = support_rel.as_posix()
        frame["drivable_original_pixels"] = int(original_valid.sum())
        frame["drivable_temporal_added_pixels"] = int(added.sum())
        counts["drivable_original_pixels"] += int(original_valid.sum())
        counts["drivable_added_pixels"] += int(added.sum())
        counts["drivable_valid_pixels"] += int(valid.sum())
        counts["drivable_road_pixels"] += int(np.sum(drivable == 1))
        counts["drivable_sidewalk_pixels"] += int(np.sum(drivable == 2))
        # Cross-check temporal votes where the current sweep itself has a label.
        vote_prediction = np.where(drive_support >= 2,
                                   np.where(drive_support == 0, 255,
                                            np.where(drivable == 255, 255, drivable)), 255)
        comparable = original_valid & (drive_support >= 2)
        counts["drivable_temporal_comparable_pixels"] += int(comparable.sum())
        counts["drivable_temporal_agree_pixels"] += int(
            np.sum(comparable & (drivable == maps[index])))

        # Head 2: static same-class local interpolation.
        depth = load_depth(source_scene, source_frame)
        with np.load(source_scene / source_frame["seg2d21"], allow_pickle=False) as z:
            semantic = z["seg"].astype(np.int64)
        filled = np.zeros_like(depth, np.float32)
        confidence = np.zeros_like(depth, np.uint8)
        for camera in range(8):
            filled[camera], confidence[camera] = guided_depth_fill(
                depth[camera], semantic[camera])
            measured = depth[camera] > 0
            if not np.array_equal(filled[camera][measured], depth[camera][measured]):
                raise AssertionError("Measured depth changed during interpolation")
        rel6 = Path("depth4_guided") / f"{fi:06d}.npz"
        rel2 = Path("depth4n_guided") / f"{fi:06d}.npz"
        camera_counts = np.count_nonzero(filled, axis=(1, 2)).astype(np.int32)
        measured_counts = np.count_nonzero(depth, axis=(1, 2)).astype(np.int32)
        interpolated_counts = np.sum(confidence == 2, axis=(1, 2)).astype(np.int32)
        atomic_npz(output_scene / rel6, depth=filled[:6].astype(np.float16),
                   confidence=confidence[:6], valid_count=camera_counts[:6],
                   measured_count=measured_counts[:6],
                   interpolated_count=interpolated_counts[:6],
                   max_depth_m=np.asarray(80.0, np.float32))
        atomic_npz(output_scene / rel2, depth=filled[6:].astype(np.float16),
                   confidence=confidence[6:], valid_count=camera_counts[6:],
                   measured_count=measured_counts[6:],
                   interpolated_count=interpolated_counts[6:],
                   max_depth_m=np.asarray(80.0, np.float32))
        frame["depth4"] = rel6.as_posix(); frame["depth4n"] = rel2.as_posix()
        frame["depth_measured_pixels"] = int(measured_counts.sum())
        frame["depth_interpolated_pixels"] = int(interpolated_counts.sum())
        frame["depth_valid_pixels"] = int(camera_counts.sum())
        depth_by_camera_measured += measured_counts
        depth_by_camera_total += camera_counts
        counts["depth_measured_pixels"] += int(measured_counts.sum())
        counts["depth_interpolated_pixels"] += int(interpolated_counts.sum())
        counts["depth_valid_pixels"] += int(camera_counts.sum())

        # Deterministic holdout every 40th frame estimates interpolation error.
        if index % 40 == 0:
            yy, xx = np.indices(depth.shape[1:])
            for camera in range(8):
                hold = ((depth[camera] > 0) & np.isin(semantic[camera], STATIC_SEG21)
                        & (((yy * depth.shape[2] + xx + camera) % 10) == 0))
                validation_eligible += int(hold.sum())
                train = depth[camera].copy(); train[hold] = 0
                predicted, predicted_conf = guided_depth_fill(train, semantic[camera])
                recovered = hold & (predicted_conf == 2)
                validation_recovered += int(recovered.sum())
                if recovered.any():
                    validation_errors.append(
                        np.abs(predicted[recovered] - depth[camera][recovered]))
        counts["frames"] += 1

    # Correct temporal agreement: derive the neighbor-only winner explicitly
    # for a deterministic sample, avoiding a claim based on retained current GT.
    agree = comparable_n = 0
    for index in range(0, len(maps), 10):
        road = np.zeros(maps[index].shape, np.uint8)
        walk = np.zeros(maps[index].shape, np.uint8)
        if pose_valid[index]:
            for j in range(max(0, index-TEMPORAL_RADIUS),
                           min(len(maps), index+TEMPORAL_RADIUS+1)):
                if j == index or not pose_valid[j] or abs(int(timestamps[j])-int(timestamps[index])) > TEMPORAL_MAX_DELTA_NS:
                    continue
                w = cv2.warpAffine(maps[j], pixel_warp(poses[j], poses[index]),
                                   (BEV_W, BEV_H), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=255)
                road += (w == 1).astype(np.uint8); walk += (w == 2).astype(np.uint8)
        pred = np.full(maps[index].shape, 255, np.uint8)
        pred[(road >= 2) & (road > walk)] = 1
        pred[(walk >= 2) & (walk > road)] = 2
        comp = (maps[index] != 255) & (pred != 255)
        comparable_n += int(comp.sum()); agree += int(np.sum(comp & (pred == maps[index])))

    errors = np.concatenate(validation_errors) if validation_errors else np.zeros(0, np.float32)
    manifest["partial_bev_drivable"] = {
        "shape": [BEV_H, BEV_W], "classes": {"1": "road", "2": "sidewalk", "255": "ignore"},
        "source": "single_sweep_semantic_occupancy_plus_neighbor_pose_warp",
        "temporal_radius_frames": TEMPORAL_RADIUS,
        "fill_rule": "unknown_only_at_least_2_agreeing_neighbor_votes",
        "training_gt_key": "gt_map",
    }
    manifest["metric_depth"] = {
        "kind": "measured_lidar_plus_static_same_class_guided_fill",
        "shape": [8, 108, 192], "units": "metres", "invalid_value": 0,
        "confidence": {"0": "invalid", "1": "measured_lidar", "2": "guided_interpolation"},
        "fill_rule": "2_iterations_8_neighbor_same_semantic_count_ge2_spread_le_max_1.5m_8pct",
        "measured_anchor_policy": "preserved_exactly",
        "deskew_status": "unverified",
    }
    atomic_json(manifest_path, manifest)
    return {
        "scene": manifest["scene"], "counts": dict(counts),
        "depth_measured_by_camera": depth_by_camera_measured.tolist(),
        "depth_total_by_camera": depth_by_camera_total.tolist(),
        "depth_validation": {
            "eligible": validation_eligible, "recovered": validation_recovered,
            "errors": errors.tolist(),
        },
        "drivable_validation": {"comparable": comparable_n, "agree": agree},
    }


def summarize(results: list[dict], root: Path, source: Path) -> dict:
    totals: Counter = Counter()
    measured_cam = np.zeros(8, np.int64); total_cam = np.zeros(8, np.int64)
    depth_errors = []; depth_eligible = depth_recovered = 0
    drv_comp = drv_agree = 0
    for result in results:
        totals.update(result["counts"])
        measured_cam += np.asarray(result["depth_measured_by_camera"])
        total_cam += np.asarray(result["depth_total_by_camera"])
        validation = result["depth_validation"]
        depth_eligible += validation["eligible"]; depth_recovered += validation["recovered"]
        if validation["errors"]:
            depth_errors.append(np.asarray(validation["errors"], np.float32))
        drv_comp += result["drivable_validation"]["comparable"]
        drv_agree += result["drivable_validation"]["agree"]
    errors = np.concatenate(depth_errors) if depth_errors else np.zeros(0, np.float32)
    frame_count = int(totals["frames"])
    depth_den = frame_count * 8 * 108 * 192
    drive_den = frame_count * BEV_H * BEV_W
    report = {
        "status": "HEAD1_HEAD2_IMPROVEMENT_MATERIALIZED",
        "source_profile": str(source), "output_profile": str(root),
        "head1_drivable": {
            "original_valid_pixels": int(totals["drivable_original_pixels"]),
            "added_temporal_pixels": int(totals["drivable_added_pixels"]),
            "total_valid_pixels": int(totals["drivable_valid_pixels"]),
            "original_coverage": totals["drivable_original_pixels"] / drive_den,
            "new_coverage": totals["drivable_valid_pixels"] / drive_den,
            "relative_coverage_gain": totals["drivable_valid_pixels"] / max(totals["drivable_original_pixels"], 1) - 1,
            "sampled_temporal_agreement": drv_agree / max(drv_comp, 1),
            "sampled_temporal_comparable_pixels": drv_comp,
            "policy": "current labels immutable; unknown filled only by >=2 agreeing warped neighbors",
        },
        "head2_depth": {
            "measured_pixels": int(totals["depth_measured_pixels"]),
            "interpolated_pixels": int(totals["depth_interpolated_pixels"]),
            "total_valid_pixels": int(totals["depth_valid_pixels"]),
            "measured_coverage": totals["depth_measured_pixels"] / depth_den,
            "new_coverage": totals["depth_valid_pixels"] / depth_den,
            "relative_coverage_gain": totals["depth_valid_pixels"] / max(totals["depth_measured_pixels"], 1) - 1,
            "measured_by_camera": measured_cam.tolist(), "total_by_camera": total_cam.tolist(),
            "measured_anchors_preserved_exactly": True,
            "holdout_validation": {
                "eligible": depth_eligible, "recovered": depth_recovered,
                "recovery_rate": depth_recovered / max(depth_eligible, 1),
                "mae_m": float(errors.mean()) if errors.size else None,
                "median_ae_m": float(np.median(errors)) if errors.size else None,
                "p90_ae_m": float(np.percentile(errors, 90)) if errors.size else None,
                "p99_ae_m": float(np.percentile(errors, 99)) if errors.size else None,
                "within_1m": float(np.mean(errors <= 1.0)) if errors.size else None,
                "within_2m": float(np.mean(errors <= 2.0)) if errors.size else None,
            },
            "policy": "fill static same-class local holes only; confidence=2 marks interpolation",
            "training_recipe": {
                "loader_flag": "with_depth_confidence=True",
                "trainer_argument": "--depth-guided-w 0.35",
                "measured_weight": 1.0,
                "guided_weight": 0.35,
            },
        },
        "remaining_limits": [
            "LiDAR deskew status remains unverified",
            "drivable still supervises road and sidewalk only",
            "interpolated depth is not an independent sensor measurement",
        ],
    }
    atomic_json(root / "head1_head2_improvement_report.json", report)
    atomic_json(root / "drivable_report.json", {
        "status": "BEV_DRIVABLE_TEMPORAL_CONSENSUS_MATERIALIZED",
        "counts": {k: int(v) for k, v in totals.items() if k.startswith("drivable_") or k == "frames"},
        "coverage": report["head1_drivable"],
        "training_gate": "PARTIAL_ROAD_SIDEWALK_ONLY_KEEP_255_IGNORE",
    })
    atomic_json(root / "depth_report.json", {
        "status": "DEPTH_GUIDED_STATIC_FILL_MATERIALIZED",
        "counts": {k: int(v) for k, v in totals.items() if k.startswith("depth_") or k == "frames"},
        "valid_pixels_by_camera": total_cam.tolist(),
        "measured_pixels_by_camera": measured_cam.tolist(),
        "holdout_validation": report["head2_depth"]["holdout_validation"],
        "training_gate": "HOLD_DESKEW_QA_CONFIDENCE_AWARE_RECIPE_READY",
    })
    return report


def build(source: Path, output: Path, workers: int) -> dict:
    source, output = source.resolve(), output.resolve()
    clone_with_hardlinks(source, output)
    dataset_path = output / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["schema"] = "pnk-layer2-9head-v3"
    dataset["complete"] = False
    dataset["source_profile"] = str(source)
    dataset["source_profile_dataset_sha256"] = sha256(source / "dataset.json")
    dataset["task_availability"]["1_bev_lane"] = "CANDIDATE_temporal_consensus_road_sidewalk"
    dataset["task_availability"]["2_metric_depth"] = "CANDIDATE_measured_plus_confidence_marked_guided_static_fill"
    dataset["sample_contract"]["confidence_aware_loader_order"] = [
        "images", "K", "T_cam_ego", "gt_map", "depth", "depth_confidence",
        "seg2d21", "agent_boxes", "agent_count", "agent_traj", "agent_valid",
        "ego", "occupancy", "risk", "lidar_bev",
    ]
    dataset["sample_contract"]["depth_confidence_codes"] = {
        "0": "invalid", "1": "measured_lidar", "2": "guided_interpolation",
    }
    dataset["safe_loader_recipe_after_gate"]["with_depth_confidence"] = True
    dataset["safe_loader_recipe_after_gate"]["depth_guided_weight"] = 0.35
    atomic_json(dataset_path, dataset)
    jobs = [(str(source / p.parent.name), str(p.parent))
            for p in sorted(output.glob("*/manifest.json"))]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(process_scene, jobs):
            results.append(result)
            print(f"[head1/2] {result['scene']}: {result['counts']['frames']} frames", flush=True)
    report = summarize(results, output, source)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["complete"] = True
    dataset["head1_head2_improvement"] = report
    dataset["trainer_gate"] = "HOLD_deskew_track_and_other_head_QA"
    atomic_json(dataset_path, dataset)
    verification = verify_profile(output)
    verification["head1_head2_improvement"] = {
        "report": "head1_head2_improvement_report.json",
        "measured_depth_anchors_preserved": True,
        "drivable_current_labels_preserved": True,
    }
    atomic_json(output / "nine_head_verification.json", verification)
    return {"improvement": report, "verification": verification}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    ap.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    result = build(args.source, args.output, args.workers)
    print(json.dumps({
        "status": result["verification"]["status"],
        "head1": result["improvement"]["head1_drivable"],
        "head2": result["improvement"]["head2_depth"],
        "output": str(args.output.resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
