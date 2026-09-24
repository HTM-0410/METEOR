#!/usr/bin/env python3
"""Materialize full METEOR-class BEV lane pseudo-GT for all PNK frames."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bevlane.dataset import BevLaneDataset  # noqa: E402
from scripts.pilot_pnk_bev_lane_sensor_fusion import run as run_scene  # noqa: E402
from scripts.prepare_pnk_comet import atomic_json, sha256  # noqa: E402
from scripts.verify_pnk_9head_profile import verify as verify_profile  # noqa: E402


SOURCE_DEFAULT = Path(r"D:\Backup_rosbag\PNKData_layer2_9head_v3")
BASE_DEFAULT = Path(r"D:\Backup_rosbag\PNKData_layer2_9head_v2")
HANDOFF_DEFAULT = Path(r"D:\Backup_rosbag\PNKData_comet_handoff_v2")
OUTPUT_DEFAULT = Path(r"D:\Backup_rosbag\PNKData_layer2_9head_v4")
ALLOWED_LABELS = {1, 2, 3, 4, 5, 6, 7, 8, 255}
CLASS_NAMES = {
    1: "road", 2: "sidewalk", 3: "crosswalk", 4: "laneline",
    5: "stopline", 6: "road_edge", 7: "marking", 8: "parking",
    255: "ignore",
}


def clone_profile(source: Path, output: Path) -> None:
    if output.exists():
        return
    shutil.copytree(source, output, copy_function=os.link)


def read_gray(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE) if raw.size else None
    if image is None:
        raise FileNotFoundError(path)
    return image


def scene_complete(scene_dir: Path, manifest: dict) -> bool:
    contract = manifest.get("bev_lane_sensor_full", {})
    if not contract.get("complete"):
        return False
    for frame in manifest["frames"]:
        for key in ("gt_map", "gt_map_support"):
            relative = frame.get(key)
            if not relative or not (scene_dir / relative).exists():
                return False
    return True


def install_scene(source: Path, base: Path, handoff: Path, output: Path,
                  scene_name: str, radius: int, visual_every: int) -> dict:
    source_manifest_path = source / scene_name / "manifest.json"
    output_manifest_path = output / scene_name / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    output_manifest = json.loads(output_manifest_path.read_text(encoding="utf-8"))
    if scene_complete(output / scene_name, output_manifest):
        report_path = output / scene_name / "bev_lane_full_report.json"
        if report_path.exists():
            return json.loads(report_path.read_text(encoding="utf-8"))
    target_frames = tuple(int(frame["frame"]) for frame in source_manifest["frames"])
    report = run_scene(
        source, base, handoff, scene_name, target_frames, radius,
        output / scene_name, label_prefix="gt_map_sensor_full",
        visual_every=visual_every)
    records = {int(record["frame"]): record for record in report["frames"]}
    for frame in output_manifest["frames"]:
        frame_id = int(frame["frame"])
        record = records[frame_id]
        frame["gt_map"] = record["label"]
        frame["gt_map_support"] = record["support"]
        frame["bev_lane_class_pixels"] = {
            "crosswalk": record["crosswalk_candidate_pixels"],
            "laneline": record["laneline_candidate_pixels"],
            "stopline": record["stopline_candidate_pixels"],
            "road_edge": record["road_edge_pixels"],
            "marking": record["generic_marking_pixels"],
            "parking": record["parking_candidate_pixels"],
        }
        frame["supervision_valid"]["bev_lane"] = True
    output_manifest["schema"] = "pnk-layer2-9head-scene-v4"
    output_manifest["bev_lane_sensor_full"] = {
        "complete": True,
        "shape": [800, 500],
        "classes": {str(key): value for key, value in CLASS_NAMES.items()},
        "source": "camera_semantics_metric_depth_lidar_intensity_NAV_temporal_fusion",
        "training_gt_key": "gt_map",
        "support_key": "gt_map_support",
        "temporal_radius_frames": radius,
        "teacher": report["sources"].get("crosswalk"),
        "status": "candidate_requires_stratified_visual_QA",
    }
    atomic_json(output_manifest_path, output_manifest)
    report["status"] = "PNK_BEV_LANE_FULL_SCENE_MATERIALIZED"
    atomic_json(output / scene_name / "bev_lane_full_report.json", report)
    return report


def aggregate(reports: list[dict], source: Path, output: Path) -> dict:
    totals: Counter = Counter()
    agreements: Counter = Counter()
    for report in reports:
        totals["scenes"] += 1
        for frame in report["frames"]:
            totals["frames"] += 1
            for key in (
                "before_valid_pixels", "after_valid_pixels", "added_road_pixels",
                "added_sidewalk_pixels", "road_edge_pixels",
                "crosswalk_candidate_pixels", "laneline_candidate_pixels",
                "stopline_candidate_pixels", "generic_marking_pixels",
                "parking_candidate_pixels",
            ):
                totals[key] += int(frame[key])
        for class_id, stats in report.get("temporal_agreement_by_class", {}).items():
            agreements[f"{class_id}_matches"] += int(stats["matches"])
            agreements[f"{class_id}_comparable"] += int(stats["comparable"])
    class_pixels = {
        "1_road_plus_original": None,
        "2_sidewalk_plus_original": None,
        "3_crosswalk": totals["crosswalk_candidate_pixels"],
        "4_laneline": totals["laneline_candidate_pixels"],
        "5_stopline": totals["stopline_candidate_pixels"],
        "6_road_edge": totals["road_edge_pixels"],
        "7_marking": totals["generic_marking_pixels"],
        "8_parking": totals["parking_candidate_pixels"],
    }
    agreement_by_class = {}
    for class_id in (3, 4, 5, 6, 7, 8):
        matches = agreements[f"{class_id}_matches"]
        comparable = agreements[f"{class_id}_comparable"]
        agreement_by_class[str(class_id)] = {
            "matches": matches, "comparable": comparable,
            "agreement_within_0.4m": matches / comparable if comparable else None,
        }
    frame_denominator = max(totals["frames"] * 800 * 500, 1)
    return {
        "status": "PNK_BEV_LANE_FULL_MATERIALIZED",
        "source_profile": str(source), "output_profile": str(output),
        "schema": "pnk-layer2-9head-v4",
        "counts": dict(totals),
        "class_pixels": class_pixels,
        "coverage_before": totals["before_valid_pixels"] / frame_denominator,
        "coverage_after": totals["after_valid_pixels"] / frame_denominator,
        "temporal_agreement_by_class": agreement_by_class,
        "class_contract": {str(key): value for key, value in CLASS_NAMES.items()},
        "training_gate": "CANDIDATE_CONFIDENCE_AWARE_STRATIFIED_QA_REQUIRED",
        "limits": [
            "LiDAR deskew and GNSS lever arm remain unverified",
            "classes 3-8 are pseudo-labels rather than human polygon GT",
            "stopline and parking are conservative and may have low recall",
        ],
    }


def verify_full(source: Path, output: Path) -> dict:
    counts: Counter = Counter()
    for manifest_path in sorted(output.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("bev_lane_sensor_full", {}).get("complete"):
            raise ValueError(f"Incomplete scene: {manifest_path}")
        source_manifest = json.loads(
            (source / manifest_path.parent.name / "manifest.json").read_text(
                encoding="utf-8"))
        source_frames = {int(frame["frame"]): frame for frame in source_manifest["frames"]}
        for frame in manifest["frames"]:
            frame_id = int(frame["frame"])
            target = read_gray(manifest_path.parent / frame["gt_map"])
            original = read_gray(source / manifest_path.parent.name
                                 / source_frames[frame_id]["gt_map"])
            if target.shape != (800, 500):
                raise ValueError("Invalid BEV lane shape")
            values, frequency = np.unique(target, return_counts=True)
            if not set(int(value) for value in values).issubset(ALLOWED_LABELS):
                raise ValueError(f"Invalid class in {manifest_path}/{frame_id}")
            unchanged_area = np.isin(target, (1, 2)) & (original != 255)
            if not np.array_equal(target[unchanged_area], original[unchanged_area]):
                raise ValueError("Original road/sidewalk label changed class")
            with np.load(manifest_path.parent / frame["gt_map_support"],
                         allow_pickle=False) as support:
                confidence = support["confidence"]
                required = {"road_votes", "sidewalk_votes", "camera_paint_votes",
                            "lidar_intensity_votes", "mapillary_crosswalk_votes",
                            "mapillary_general_marking_votes",
                            "mapillary_parking_votes"}
                if confidence.shape != target.shape or not required.issubset(support.files):
                    raise ValueError("Invalid lane support contract")
                if not set(np.unique(confidence).tolist()).issubset(set(range(8))):
                    raise ValueError("Invalid lane confidence code")
            for value, number in zip(values, frequency):
                counts[f"class_{int(value)}"] += int(number)
            counts["frames"] += 1
        counts["scenes"] += 1
    expected_dataset = json.loads((source / "dataset.json").read_text(encoding="utf-8"))
    expected_scenes = len(expected_dataset["scenes"])
    expected_frames = sum(len(json.loads((source / item["scene"] / "manifest.json").read_text(encoding="utf-8"))["frames"])
                          for item in expected_dataset["scenes"])
    if counts["frames"] != expected_frames or counts["scenes"] != expected_scenes:
        raise ValueError(f"Full frame/scene count mismatch: {dict(counts)}")
    dataset = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    scenes = [item["scene"] for item in dataset["scenes"]]
    sample = BevLaneDataset(str(output), [scenes[0]], max_per_scene=1,
                            gt_key="gt_map", with_depth=True,
                            with_depth_confidence=True, depth_hw=(108, 192),
                            with_seg2d=True, seg2d_key="seg2d21",
                            with_agenttraj=True, with_ego=True, with_occ=True,
                            with_risk=True, with_lidarbev=True)[0]
    if len(sample) != 15 or sample[3].shape != (800, 500):
        raise ValueError("Full profile DataLoader contract mismatch")
    return {
        "status": "PASS_PNK_BEV_LANE_FULL_CONTRACT_AND_IO",
        "counts": dict(counts), "loader_tensor_count": len(sample),
        "loader_gt_shape": list(sample[3].shape),
    }


def materialize(source: Path, base: Path, handoff: Path, output: Path,
                radius: int, visual_every: int) -> dict:
    source, base = source.resolve(), base.resolve()
    handoff, output = handoff.resolve(), output.resolve()
    clone_profile(source, output)
    dataset_path = output / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["schema"] = "pnk-layer2-9head-v4"
    dataset["complete"] = False
    dataset["source_profile"] = str(source)
    dataset["source_profile_dataset_sha256"] = sha256(source / "dataset.json")
    dataset["task_availability"]["1_bev_lane"] = \
        "CANDIDATE_METEOR_1_TO_8_SENSOR_FUSION_CONFIDENCE_MARKED"
    atomic_json(dataset_path, dataset)
    reports = []
    scene_paths = sorted(output.glob("*/manifest.json"))
    for index, manifest_path in enumerate(scene_paths, 1):
        scene_name = manifest_path.parent.name
        report = install_scene(source, base, handoff, output, scene_name,
                               radius, visual_every)
        reports.append(report)
        progress = {
            "status": "RUNNING", "completed_scenes": index,
            "total_scenes": len(scene_paths), "last_scene": scene_name,
            "completed_frames": sum(len(item["frames"]) for item in reports),
        }
        atomic_json(output / "bev_lane_full_progress.json", progress)
        print(f"[lane-full] {index}/{len(scene_paths)} {scene_name}: "
              f"{len(report['frames'])} frames", flush=True)
    report = aggregate(reports, source, output)
    atomic_json(output / "bev_lane_full_report.json", report)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["complete"] = True
    dataset["bev_lane_sensor_full"] = report
    dataset["trainer_gate"] = "HOLD_STRATIFIED_LANE_PSEUDO_GT_QA"
    atomic_json(dataset_path, dataset)
    verification = verify_full(source, output)
    atomic_json(output / "bev_lane_full_verification.json", verification)
    profile_verification = verify_profile(output)
    return {"report": report, "verification": verification,
            "profile_verification": profile_verification}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--base", type=Path, default=BASE_DEFAULT)
    parser.add_argument("--handoff", type=Path, default=HANDOFF_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--radius", type=int, default=8)
    parser.add_argument("--visual-every", type=int, default=25)
    args = parser.parse_args()
    result = materialize(args.source, args.base, args.handoff, args.output,
                         args.radius, args.visual_every)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
