#!/usr/bin/env python3
"""Densify PNK semantic occupancy with conservative temporal consensus.

The source profile is cloned using hard links for immutable assets.  For every
target frame, neighbouring occupancy grids are warped with NAV poses.  Existing
single-sweep voxels are immutable.  Temporal evidence can only fill unknown
(``255``) voxels and follows these policies:

* free: at least two votes and >=2/3 of all static evidence;
* obstacle / vegetation: at least four unanimous votes;
* road / sidewalk: at least two ground votes with >=2/3 consensus, then the
  current-frame BEV lane map selects road versus sidewalk;
* dynamic classes (vehicle, two-wheeler, pedestrian) are never accumulated;
* current 3D-box footprints are protected from every temporal fill.

This stage assumes the source LAZ coordinates are already deskewed, as requested
for this profile.  The assumption is recorded in every manifest and report.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from scripts.materialize_pnk_route_command import clone_profile
from scripts.materialize_pnk_risk import materialize as materialize_risk
from scripts.prepare_pnk_comet import atomic_json, sha256


DEFAULT_SOURCE = Path(r"D:\Backup_rosbag\PNKData_layer2_9head_v5")
DEFAULT_OUTPUT = Path(r"D:\Backup_rosbag\PNKData_layer2_9head_v7")
RADIUS = 3
GRID = 200
VOXEL_M = 0.4
HALF_M = 40.0
STATIC_CLASSES = (0, 1, 5, 6, 7, 8, 9)
DYNAMIC_CLASSES = (2, 3, 4)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
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


GRID_COORD = HALF_M - (np.arange(GRID, dtype=np.float32) + 0.5) * VOXEL_M
GRID_X, GRID_Y = np.meshgrid(GRID_COORD, GRID_COORD, indexing="ij")
LANE_ROWS = 201 + 2 * np.arange(GRID)
LANE_COLS = 51 + 2 * np.arange(GRID)


def neighbor_remap(current_pose: np.ndarray,
                   neighbor_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map target-grid centers to source-neighbour pixel coordinates."""
    xc, yc, yawc = (float(x) for x in current_pose)
    xn, yn, yawn = (float(x) for x in neighbor_pose)
    cc, sc = np.cos(yawc), np.sin(yawc)
    world_x = xc + cc * GRID_X - sc * GRID_Y
    world_y = yc + sc * GRID_X + cc * GRID_Y
    cn, sn = np.cos(yawn), np.sin(yawn)
    dx, dy = world_x - xn, world_y - yn
    source_x = cn * dx + sn * dy
    source_y = -sn * dx + cn * dy
    map_x = ((HALF_M - source_y) / VOXEL_M - 0.5).astype(np.float32)
    map_y = ((HALF_M - source_x) / VOXEL_M - 0.5).astype(np.float32)
    return map_x, map_y


def temporal_prediction(votes: np.ndarray, lane_crop: np.ndarray) -> np.ndarray:
    """Return conservative temporal class proposal [Z,X,Y], 255 = reject."""
    total = votes.sum(0)
    proposal = np.full(total.shape, 255, np.uint8)
    enough_static = np.maximum(total, 1)

    free = (votes[0] >= 2) & (3 * votes[0] >= 2 * enough_static)
    proposal[free] = 0
    for class_id in (1, 7):
        certain = (votes[class_id] >= 4) & (votes[class_id] == total)
        proposal[certain] = class_id

    ground_votes = votes[5].astype(np.uint16) + votes[6]
    ground = (ground_votes >= 2) & (3 * ground_votes >= 2 * enough_static)
    lane3d = np.broadcast_to(lane_crop, total.shape)
    proposal[ground & (lane3d == 1)] = 5
    proposal[ground & (lane3d == 2)] = 6
    return proposal


def warp_occupancy(source: np.ndarray, map_x: np.ndarray,
                   map_y: np.ndarray) -> np.ndarray:
    """Nearest-neighbour warp with an explicit 255 border for all 16 z levels.

    OpenCV's scalar ``borderValue`` repeats in groups of four for arrays with
    more than four channels, which would silently turn most out-of-grid levels
    into class 0.  Override every out-of-grid target cell after remapping.
    """
    warped = cv2.remap(
        source.transpose(1, 2, 0), map_x, map_y, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0).transpose(2, 0, 1)
    inside = ((map_x >= -0.5) & (map_x < GRID - 0.5)
              & (map_y >= -0.5) & (map_y < GRID - 0.5))
    warped[:, ~inside] = 255
    return warped


def dynamic_footprint(boxes: np.ndarray) -> np.ndarray:
    """Conservative XY protection mask around current Vehicle/VRU boxes."""
    mask = np.zeros((GRID, GRID), np.uint8)
    for box in boxes:
        _, xe, ye, length, width, yaw = (float(x) for x in box)
        if length <= 0 or width <= 0:
            continue
        c, s = np.cos(yaw), np.sin(yaw)
        polygon = []
        for local_x, local_y in ((length / 2, width / 2),
                                 (length / 2, -width / 2),
                                 (-length / 2, -width / 2),
                                 (-length / 2, width / 2)):
            x = xe + c * local_x - s * local_y
            y = ye + s * local_x + c * local_y
            polygon.append([(HALF_M - y) / VOXEL_M,
                            (HALF_M - x) / VOXEL_M])
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
    return cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0


def read_gray(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(path)
    return image


def process_scene(job: tuple[str, int]) -> dict:
    manifest_path = Path(job[0]); radius = int(job[1])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene_dir = manifest_path.parent
    frames = manifest["frames"]
    with np.load(scene_dir / manifest["nav_ego_candidate"], allow_pickle=False) as nav:
        poses = nav["nav_reference_pose_enu"][:, [0, 1, 3]].astype(np.float64)
        pose_valid = nav["nav_reference_valid"].astype(bool)
    occupancy = []
    for frame in frames:
        with np.load(scene_dir / frame["occ"], allow_pickle=False) as value:
            occupancy.append(value["occ"].astype(np.uint8))

    counts: Counter = Counter()
    holdout_compared = holdout_correct = 0
    for index, (frame, current) in enumerate(zip(frames, occupancy)):
        fi = int(frame["frame"])
        votes = np.zeros((10, *current.shape), np.uint8)
        if fi < len(pose_valid) and pose_valid[fi]:
            for neighbor_index in range(max(0, index - radius),
                                        min(len(frames), index + radius + 1)):
                if neighbor_index == index:
                    continue
                neighbor_frame = frames[neighbor_index]
                nfi = int(neighbor_frame["frame"])
                if nfi >= len(pose_valid) or not pose_valid[nfi]:
                    continue
                map_x, map_y = neighbor_remap(poses[fi], poses[nfi])
                source = occupancy[neighbor_index].copy()
                source[np.isin(source, DYNAMIC_CLASSES)] = 255
                warped = warp_occupancy(source, map_x, map_y)
                for class_id in STATIC_CLASSES:
                    votes[class_id] += warped == class_id
                counts["neighbor_sweeps_used"] += 1

        lane = read_gray(scene_dir / frame["gt_map"])
        lane_crop = lane[np.ix_(LANE_ROWS, LANE_COLS)]
        proposal = temporal_prediction(votes, lane_crop)

        # Agreement on currently observed static voxels is a holdout proxy.
        comparable = (proposal != 255) & np.isin(current, STATIC_CLASSES)
        holdout_compared += int(comparable.sum())
        holdout_correct += int(np.sum(comparable & (proposal == current)))

        with np.load(scene_dir / frame["bev_box_p"], allow_pickle=False) as box_file:
            protect = dynamic_footprint(box_file["boxes"])
        fill = (current == 255) & (proposal != 255)
        fill[:, protect] = False
        fused = current.copy()
        fused[fill] = proposal[fill]
        confidence = np.zeros(current.shape, np.uint8)
        confidence[current != 255] = 1
        confidence[fill] = 2
        temporal_votes = votes.max(0)

        destination_rel = Path("occ_multisweep") / f"{fi:06d}.npz"
        atomic_npz(scene_dir / destination_rel, occ=fused,
                   confidence=confidence, temporal_votes=temporal_votes,
                   source=np.asarray("single_sweep_immutable_plus_NAV_temporal_consensus"))
        before = int(np.sum(current != 255)); after = int(np.sum(fused != 255))
        frame["occ"] = destination_rel.as_posix()
        frame["occupancy_original_observed_voxels"] = before
        frame["occupancy_temporal_added_voxels"] = after - before
        frame["supervision_valid"]["occupancy3d"] = after > 0
        counts["frames"] += 1
        counts["observed_before"] += before
        counts["observed_after"] += after
        counts["temporal_added"] += after - before
        for class_id in range(10):
            counts[f"class_{class_id}"] += int(np.sum(fused == class_id))
            counts[f"added_class_{class_id}"] += int(np.sum(fill & (proposal == class_id)))
        counts["class_255"] += int(np.sum(fused == 255))

    manifest["schema"] = "pnk-layer2-9head-scene-v7"
    manifest["semantic_occupancy"] = {
        "shape": [16, 200, 200], "voxel_m": VOXEL_M,
        "extent_m": {"x": [-40, 40], "y": [-40, 40], "z": [-1.0, 5.4]},
        "source": "single_sweep_immutable_plus_NAV_temporal_consensus",
        "temporal_radius_keyframes": radius,
        "deskew_assumption": "assumed_already_deskewed_by_user_direction",
        "current_voxel_policy": "immutable",
        "dynamic_policy": "neighbor_dynamic_classes_excluded_and_current_box_footprints_protected",
        "free_policy": "at_least_2_votes_and_2_over_3_consensus",
        "obstacle_vegetation_policy": "at_least_4_unanimous_votes",
        "ground_policy": "temporal_ground_consensus_plus_current_BEV_road_sidewalk",
        "confidence_codes": {"0": "unknown", "1": "current_single_sweep",
                             "2": "temporal_consensus"},
        "unobserved_value": 255,
    }
    values = manifest["training_contract"]["allowed_now"]
    if "occupancy3d_multisweep_candidate" not in values:
        values.append("occupancy3d_multisweep_candidate")
    atomic_json(manifest_path, manifest)
    counts["scenes"] = 1
    return {"scene": manifest["scene"], "counts": dict(counts),
            "holdout_compared": holdout_compared,
            "holdout_correct": holdout_correct}


def verify(root: Path, source: Path) -> dict:
    counts: Counter = Counter()
    source_manifests = {p.parent.name: json.loads(p.read_text(encoding="utf-8"))
                        for p in source.glob("*/manifest.json")}
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        original = source_manifests[manifest["scene"]]
        original_by_frame = {int(f["frame"]): f for f in original["frames"]}
        for frame in manifest["frames"]:
            fi = int(frame["frame"])
            with np.load(source / manifest["scene"] /
                         original_by_frame[fi]["occ"], allow_pickle=False) as value:
                before = value["occ"]
            with np.load(manifest_path.parent / frame["occ"],
                         allow_pickle=False) as value:
                after = value["occ"]; confidence = value["confidence"]
                temporal_votes = value["temporal_votes"]
            if after.shape != (16, 200, 200) or confidence.shape != after.shape \
                    or temporal_votes.shape != after.shape:
                raise ValueError("Invalid multisweep occupancy shape")
            if not set(np.unique(after)).issubset(set(range(10)) | {255}):
                raise ValueError("Invalid multisweep occupancy class")
            if not set(np.unique(confidence)).issubset({0, 1, 2}):
                raise ValueError("Invalid occupancy confidence code")
            if not np.array_equal(after[before != 255], before[before != 255]):
                raise ValueError("Current single-sweep occupancy was overwritten")
            if np.any((confidence == 1) != (before != 255)) \
                    or np.any((confidence == 2) != ((before == 255) & (after != 255))):
                raise ValueError("Occupancy confidence disagrees with source")
            counts["frames"] += 1
            counts["observed_before"] += int(np.sum(before != 255))
            counts["observed_after"] += int(np.sum(after != 255))
        counts["scenes"] += 1
    total = counts["frames"] * 16 * 200 * 200
    result = {
        "status": "PASS_PNK_MULTISWEEP_OCCUPANCY_CONTRACT_AND_SOURCE_IMMUTABILITY",
        "counts": dict(counts),
        "coverage_before": counts["observed_before"] / total,
        "coverage_after": counts["observed_after"] / total,
    }
    atomic_json(root / "occupancy_multisweep_verification.json", result)
    return result


def materialize(source: Path, output: Path, radius: int = RADIUS,
                workers: int = 4) -> dict:
    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError("Output must differ from source")
    source_dataset_path = source / "dataset.json"
    dataset = json.loads(source_dataset_path.read_text(encoding="utf-8"))
    if dataset.get("schema") != "pnk-layer2-9head-v5" or not dataset.get("complete"):
        raise ValueError("Expected complete PNK 9-head v5 source profile")
    clone_counts = clone_profile(source, output)
    original_report = output / "occupancy_report.json"
    if original_report.is_file():
        os.replace(original_report, output / "occupancy_single_sweep_report.json")

    totals: Counter = Counter(); holdout_compared = holdout_correct = 0
    jobs = [(str(path), radius) for path in sorted(output.glob("*/manifest.json"))]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(process_scene, jobs):
            totals.update(result["counts"])
            holdout_compared += result["holdout_compared"]
            holdout_correct += result["holdout_correct"]
            print(f"[occupancy-v7] {result['scene']}: "
                  f"+{result['counts'].get('temporal_added', 0):,} voxels", flush=True)

    total_voxels = int(totals["frames"] * 16 * 200 * 200)
    report = {
        "status": "PNK_MULTISWEEP_OCCUPANCY_MATERIALIZED",
        "source_profile": str(source), "output_profile": str(output),
        "clone_counts": dict(clone_counts), "counts": dict(totals),
        "coverage_before": totals["observed_before"] / total_voxels,
        "coverage_after": totals["observed_after"] / total_voxels,
        "relative_coverage_gain": ((totals["observed_after"] - totals["observed_before"])
                                   / max(totals["observed_before"], 1)),
        "temporal_holdout_agreement": holdout_correct / max(holdout_compared, 1),
        "temporal_holdout_compared": holdout_compared,
        "radius_keyframes": radius,
        "deskew_assumption": "assumed_already_deskewed_by_user_direction",
        "policies": {
            "current_single_sweep": "immutable",
            "dynamic": "neighbor dynamic excluded; current box footprints protected",
            "free": "at least 2 votes and >=2/3 consensus",
            "obstacle_vegetation": "at least 4 unanimous votes",
            "road_sidewalk": "ground consensus plus current BEV lane class",
            "structure_pole": "current sweep only",
        },
        "training_gate": "CANDIDATE_STRATIFIED_VISUAL_QA_AND_AB_TRAINING_REQUIRED",
    }
    dataset_path = output / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["schema"] = "pnk-layer2-9head-v7"
    dataset["source_profile"] = str(source)
    dataset["source_profile_dataset_sha256"] = sha256(source_dataset_path)
    dataset["task_availability"]["8_occupancy3d"] = \
        "CANDIDATE_multisweep_static_consensus_dynamic_protected"
    dataset["semantic_occupancy"] = report
    atomic_json(dataset_path, dataset)
    atomic_json(output / "task_availability.json", dataset["task_availability"])
    atomic_json(output / "occupancy_report.json", report)

    # Head 12 depends on occupancy, so regenerate it against the v6 targets.
    risk_report = materialize_risk(output)
    verification = verify(output, source)
    report["risk_regenerated"] = risk_report["status"]
    report["verification"] = verification["status"]
    atomic_json(output / "occupancy_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--radius", type=int, default=RADIUS)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.radius < 1:
        parser.error("--radius must be >=1")
    print(json.dumps(materialize(args.source, args.output, args.radius, args.workers),
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
