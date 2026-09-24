#!/usr/bin/env python3
"""Build an optimized PNK Layer-2 profile for heads 1,2,3,5,7,8,9,10,12.

The source Layer-2 tree is immutable.  Unchanged assets are hard-linked where
possible, Head-3 boxes are filtered into a confidence tier, and dependent
agent/flow/risk targets are regenerated from those filtered current boxes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.materialize_pnk_agent_traj import materialize as materialize_agent
from scripts.materialize_pnk_flow import materialize as materialize_flow
from scripts.materialize_pnk_risk import materialize as materialize_risk
from scripts.prepare_pnk_comet import atomic_json, sha256
from scripts.verify_pnk_9head_profile import verify as verify_nine_head


SOURCE_DEFAULT = Path(r"D:\Backup_rosbag\PNKData_layer2_samples_v1")
HANDOFF_DEFAULT = Path(r"D:\Backup_rosbag\PNKData_comet_handoff_v2")
OUTPUT_DEFAULT = Path(r"D:\Backup_rosbag\PNKData_layer2_9head_v2")
AGENT_KMAX = 256
ACTIVE_HEADS = [1, 2, 3, 5, 7, 8, 9, 10, 12]
EXCLUDED_HEADS = {
    "4_unknown": "excluded_by_project_scope_missing_unknown_semantics",
    "6_bbox2d": "excluded_by_project_scope_incomplete_image_annotations",
    "11_traffic_light": "excluded_by_project_scope_missing_state_and_ego_association",
}


def link_or_copy(source: Path, destination: Path, counters: Counter) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        counters["reused"] += 1
        return
    try:
        os.link(source, destination)
        counters["hardlinked"] += 1
    except OSError:
        shutil.copy2(source, destination)
        counters["copied"] += 1


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


def filter_boxes(source_path: Path, destination: Path) -> dict:
    with np.load(source_path, allow_pickle=False) as z:
        boxes = z["meteor_boxes"].astype(np.float32)
        boxes7 = z["boxes"].astype(np.float32)
        point_count = z["point_count"].astype(np.int32)
        track_id = z["track_id"].astype(np.int64)
        source_class = z["source_class"].astype(str)
        source_flag = z["source_flag"].astype(np.int16)
    in_bev = ((np.abs(boxes[:, 1]) <= 80.0)
              & (np.abs(boxes[:, 2]) <= 50.0))
    lidar_supported = point_count >= 5
    geometry_sane = np.isfinite(boxes7).all(1) & (np.abs(boxes7[:, 2]) <= 10.0)
    # Point count is evidence strength, not validity.  Zero/low-point objects
    # can be camera-visible or occluded source GT, so preserve every sane box
    # whose centre can be represented by the METEOR detection grid.
    keep = in_bev & geometry_sane
    indices = np.flatnonzero(keep).astype(np.int32)
    if len(indices) > AGENT_KMAX:
        raise ValueError(f"Filtered box count exceeds {AGENT_KMAX}: {source_path}")
    atomic_npz(
        destination,
        boxes=boxes[keep], meteor_boxes=boxes[keep],
        point_count=point_count[keep], track_id=track_id[keep],
        source_class=source_class[keep], source_flag=source_flag[keep],
        source_index=indices,
        confidence_tier=np.where(lidar_supported[keep], 1, 2).astype(np.uint8),
        filter_name=np.asarray("preserve_in_bev_sane_tier_points_no_delete"),
    )
    return {
        "source": len(boxes), "kept": len(indices),
        "tier_a_ge5_points": int(np.sum(keep & lidar_supported)),
        "tier_b_lt5_points": int(np.sum(keep & ~lidar_supported)),
        "drop_outside_bev": int(np.sum(~in_bev)),
        "drop_bad_geometry_in_bev": int(np.sum(in_bev & ~geometry_sane)),
    }


def build_base(source: Path, handoff: Path, output: Path) -> dict:
    source, handoff, output = source.resolve(), handoff.resolve(), output.resolve()
    if output == source or output == handoff:
        raise ValueError("Output must be a separate directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_dataset_path = source / "dataset.json"
    source_dataset = json.loads(source_dataset_path.read_text(encoding="utf-8"))
    counters: Counter = Counter()
    scene_rows = []
    max_boxes = 0
    for source_manifest_path in sorted(source.glob("*/manifest.json")):
        src = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        scene = src["scene"]
        handoff_manifest = json.loads(
            (handoff / "scenes" / scene / "manifest.json").read_text(encoding="utf-8"))
        handoff_by_frame = {int(x["frame"]): x for x in handoff_manifest["frames"]}
        dst_dir = output / scene
        dst_dir.mkdir(parents=True, exist_ok=True)
        for key in ("nav_ego_candidate", "ego_motion"):
            link_or_copy(source / scene / src[key], dst_dir / src[key], counters)
        frames = []
        scene_source = scene_kept = scene_tier_a = scene_tier_b = 0
        for old in src["frames"]:
            fi = int(old["frame"])
            source_box = Path(handoff_by_frame[fi]["boxes_3d_npz"])
            box_rel = Path("bev_box_p") / f"{fi:06d}.npz"
            stats = filter_boxes(source_box, dst_dir / box_rel)
            counters.update({f"boxes_{k}": v for k, v in stats.items()})
            scene_source += stats["source"]; scene_kept += stats["kept"]
            scene_tier_a += stats["tier_a_ge5_points"]
            scene_tier_b += stats["tier_b_lt5_points"]
            max_boxes = max(max_boxes, stats["kept"])
            keep_paths = {
                "gt_cons": old["gt_cons"], "seg2d21": old["seg2d21"],
                "lidar_bev": old["lidar_bev"], "depth4": old["depth4"],
                "depth4n": old["depth4n"], "occ": old["occ"],
                "gt_map": old["gt_map"],
            }
            for rel in dict.fromkeys(keep_paths.values()):
                link_or_copy(source / scene / rel, dst_dir / rel, counters)
            supervision = {
                "bev_lane": bool(old["supervision_valid"].get("bev_lane")),
                "metric_depth": bool(old["supervision_valid"].get("metric_depth")),
                "box3d": stats["kept"] > 0,
                "seg2d_partial": bool(old["supervision_valid"].get("seg2d_partial")),
                "ego_motion": bool(old["supervision_valid"].get("ego_motion")),
                "occupancy3d": bool(old["supervision_valid"].get("occupancy3d")),
                "occupancy_flow": False, "agent_forecast": False, "risk": False,
            }
            frames.append({
                "frame": fi, "timestamp_ns": int(old["timestamp_ns"]),
                "imgs": old["imgs"], **keep_paths,
                "bev_box_p": box_rel.as_posix(),
                "box3d_source_count": stats["source"],
                "box3d_profile_count": stats["kept"],
                "box3d_tier_a_count": stats["tier_a_ge5_points"],
                "box3d_tier_b_count": stats["tier_b_lt5_points"],
                "depth_valid_pixels": int(old["depth_valid_pixels"]),
                "supervision_valid": supervision,
            })
        manifest = {
            "schema": "pnk-layer2-9head-scene-v2", "scene": scene,
            "recording": src["recording"], "split": src["split"],
            "image_root": src["image_root"], "img_hw": src["img_hw"],
            "cams": src["cams"], "nav_ego_candidate": src["nav_ego_candidate"],
            "ego_motion": src["ego_motion"], "frames": frames,
            "active_heads": ACTIVE_HEADS, "excluded_heads": EXCLUDED_HEADS,
            "training_contract": {
                "gt_key": "gt_map", "agent_kmax": AGENT_KMAX,
                "active_targets": ["bev_drivable_partial", "metric_depth_candidate",
                                   "box3d_inbev_gt", "seg2d_partial",
                                   "ego_motion_candidate", "occupancy3d_candidate",
                                   "occupancy_flow_candidate", "agent_forecast_candidate",
                                   "area_risk_candidate"],
                "allowed_now": ["bev_drivable_partial", "metric_depth_candidate",
                                "box3d_inbev_gt", "seg2d_partial",
                                "ego_motion_candidate", "occupancy3d_candidate"],
                "excluded_targets": list(EXCLUDED_HEADS),
                "hold": ["metric_depth", "ego_motion", "occupancy3d",
                         "occupancy_flow", "agent_forecast", "risk"],
                "hold_reason": "remaining projection_deskew_track_and_derived_target_QA",
            },
            "box3d_filter": {
                "name": "preserve_in_bev_sane_tier_points_no_delete",
                "source_boxes": scene_source, "kept_boxes": scene_kept,
                "tier_a_ge5_points": scene_tier_a,
                "tier_b_lt5_points": scene_tier_b,
                "point_count_is_confidence_not_deletion": True,
                "raw_source_preserved": str(source),
            },
            "metric_depth": src["metric_depth"],
            "semantic_occupancy": src["semantic_occupancy"],
            "partial_bev_drivable": src["partial_bev_drivable"],
            "e2e_target": src["e2e_target"],
        }
        atomic_json(dst_dir / "manifest.json", manifest)
        scene_rows.append({"scene": scene, "frames": len(frames), "split": src["split"],
                           "source_boxes": scene_source, "profile_boxes": scene_kept,
                           "tier_a_boxes": scene_tier_a, "tier_b_boxes": scene_tier_b})
        counters["frames"] += len(frames); counters["scenes"] += 1
        print(f"[9head base] {scene}: boxes {scene_source}->{scene_kept}", flush=True)
    for report_name in ("depth_report.json", "e2e_report.json",
                        "occupancy_report.json", "drivable_report.json"):
        shutil.copy2(source / report_name, output / report_name)
    dataset = {
        "schema": "pnk-layer2-9head-v2", "complete": False,
        "source_layer2": str(source),
        "source_layer2_dataset_sha256": sha256(source_dataset_path),
        "active_heads": ACTIVE_HEADS, "excluded_heads": EXCLUDED_HEADS,
        "counts": dict(counters), "max_profile_boxes_per_frame": max_boxes,
        "scenes": scene_rows,
        "task_availability": {
            "1_bev_lane": "CANDIDATE_partial_road_sidewalk_ignore_elsewhere",
            "2_metric_depth": "CANDIDATE_sparse_lidar_projection",
            "3_box3d": "CANDIDATE_all_sane_inBEV_GT_pointcount_as_confidence",
            "4_unknown": "EXCLUDED",
            "5_seg2d": "CANDIDATE_partial_21class_teacher",
            "6_bbox2d": "EXCLUDED",
            "7_e2e": "CANDIDATE_validity_gated_no_route_command",
            "8_occupancy3d": "CANDIDATE_single_sweep_partial_semantics",
            "9_occupancy_flow": "PENDING_REGEN_FROM_FILTERED_AGENTS",
            "10_agent_forecast": "PENDING_REGEN_FROM_FILTERED_CURRENT_BOXES",
            "11_traffic_light": "EXCLUDED",
            "12_risk": "PENDING_REGEN_FROM_FILTERED_AGENTS_AND_OCCUPANCY",
        },
        "sample_contract": {
            "agent_kmax": AGENT_KMAX,
            "loader_order": ["images", "K", "T_cam_ego", "gt_map", "depth",
                             "seg2d21", "agent_boxes", "agent_count", "agent_traj",
                             "agent_valid", "ego", "occupancy", "risk", "lidar_bev"],
            "loader_flags": {"with_depth": True, "with_seg2d": True,
                             "with_boxdet": False, "with_bbox2d": False,
                             "with_agenttraj": True, "with_ego": True,
                             "with_occ": True, "with_tl": False,
                             "with_unknown": False, "with_risk": True,
                             "with_lidarbev": True},
            "boxdet_supervision_source": "agent_boxes_from_agent_traj",
        },
        "safe_loader_recipe_after_gate": {
            "gt_key": "gt_map", "with_depth": True,
            "with_seg2d": True, "seg2d_key": "seg2d21",
            "with_boxdet": False, "with_bbox2d": False,
            "with_agenttraj": True, "with_ego": True,
            "with_occ": True, "with_tl": False,
            "with_unknown": False, "with_risk": True,
            "with_lidarbev": True, "agent_kmax": AGENT_KMAX,
        },
        "trainer_gate": "HOLD_remaining_projection_deskew_track_and_visual_QA",
    }
    atomic_json(output / "dataset.json", dataset)
    atomic_json(output / "task_availability.json", dataset["task_availability"])
    box_report = {
        "status": "BOX3D_INBEV_GT_PRESERVED_WITH_CONFIDENCE_TIERS",
        "counts": dict(counters),
        "max_boxes_per_frame": max_boxes, "agent_kmax": AGENT_KMAX,
        "filter": "inside_METEOR_BEV_and_abs_z_le10; point_count_only_sets_tier",
        "tier_definition": {"1": "point_count_ge5", "2": "point_count_lt5_preserved"},
        "raw_source_preserved": str(source),
    }
    atomic_json(output / "box3d_filter_report.json", box_report)
    return dataset


def enforce_exclusions(root: Path) -> dict:
    checked = 0
    forbidden_frame_keys = {"bbox2d", "bbox2d_candidate", "layer1_panoptic",
                            "unknown", "unknown_v2", "tl", "traffic_light"}
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("active_heads") != ACTIVE_HEADS:
            raise ValueError(f"Wrong active heads: {manifest_path}")
        for frame in manifest["frames"]:
            bad = forbidden_frame_keys.intersection(frame)
            if bad:
                raise ValueError(f"Excluded target keys remain: {manifest_path}: {bad}")
            checked += 1
        for dirname in ("bbox2d", "bbox2d_candidate", "unknown", "traffic_light"):
            if (manifest_path.parent / dirname).exists():
                raise ValueError(f"Excluded target directory remains: {manifest_path.parent / dirname}")
    return {"status": "PASS_9HEAD_EXCLUSION_CONTRACT", "frames": checked,
            "excluded_heads": [4, 6, 11]}


def run(source: Path, handoff: Path, output: Path) -> dict:
    build_base(source, handoff, output)
    try:
        agent = materialize_agent(output, handoff, box_source="root", kmax=AGENT_KMAX)
        flow = materialize_flow(output)
        risk = materialize_risk(output)
        dataset_path = output / "dataset.json"
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        dataset["complete"] = True
        dataset["task_availability"]["3_box3d"] = \
            "CANDIDATE_all_sane_inBEV_GT_pointcount_as_confidence"
        dataset["profile_reports"] = {
            "agent_forecast": agent, "occupancy_flow": flow, "area_risk": risk}
        atomic_json(dataset_path, dataset)
        atomic_json(output / "task_availability.json", dataset["task_availability"])
        exclusion_verify = enforce_exclusions(output)
        verification = verify_nine_head(output)
        verification["exclusions"] = exclusion_verify
        atomic_json(output / "nine_head_verification.json", verification)
        return verification
    except Exception:
        dataset_path = output / "dataset.json"
        if dataset_path.exists():
            dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
            dataset["complete"] = False
            atomic_json(dataset_path, dataset)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    ap.add_argument("--handoff", type=Path, default=HANDOFF_DEFAULT)
    ap.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = ap.parse_args()
    print(json.dumps(run(args.source, args.handoff, args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
