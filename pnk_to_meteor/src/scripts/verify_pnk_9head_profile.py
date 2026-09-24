#!/usr/bin/env python3
"""Verify the optimized PNK 9-head sample contract end to end."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bevlane.dataset import BevLaneDataset  # noqa: E402
from scripts.prepare_pnk_comet import atomic_json  # noqa: E402
from scripts.materialize_pnk_route_command import derive_command  # noqa: E402
from scripts.verify_pnk_derived_targets import verify as verify_derived  # noqa: E402
from scripts.verify_pnk_task_targets import verify as verify_tasks  # noqa: E402


DEFAULT_ROOT = Path(r"D:\Backup_rosbag\PNKData_layer2_9head_v1")
ACTIVE = [1, 2, 3, 5, 7, 8, 9, 10, 12]
FORBIDDEN_KEYS = {"bbox2d", "bbox2d_candidate", "layer1_panoptic",
                  "unknown", "unknown_v2", "tl", "traffic_light"}


def verify(root: Path) -> dict:
    root = root.resolve()
    dataset_path = root / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if dataset.get("schema") not in {"pnk-layer2-9head-v1", "pnk-layer2-9head-v2",
                                     "pnk-layer2-9head-v3", "pnk-layer2-9head-v4",
                                     "pnk-layer2-9head-v5", "pnk-layer2-9head-v6",
                                     "pnk-layer2-9head-v7"} \
            or not dataset.get("complete"):
        raise ValueError("Expected complete PNK 9-head profile")
    preserve_low_point = dataset.get("schema") in {"pnk-layer2-9head-v2",
                                                    "pnk-layer2-9head-v3",
                                                     "pnk-layer2-9head-v4",
                                                     "pnk-layer2-9head-v5",
                                                     "pnk-layer2-9head-v6",
                                                     "pnk-layer2-9head-v7"}
    if dataset.get("active_heads") != ACTIVE:
        raise ValueError("Active head contract differs")
    if set(dataset.get("excluded_heads", {})) != {"4_unknown", "6_bbox2d", "11_traffic_light"}:
        raise ValueError("Excluded head contract differs")
    kmax = int(dataset["sample_contract"]["agent_kmax"])
    with_command = bool(dataset["sample_contract"].get("loader_flags", {})
                        .get("with_command", False))
    counts: Counter = Counter()
    scenes = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scenes.append(manifest["scene"])
        if manifest.get("active_heads") != ACTIVE:
            raise ValueError(f"Scene head contract differs: {manifest_path}")
        ego = None
        if with_command:
            ego = np.load(manifest_path.parent / manifest["ego_motion"],
                          allow_pickle=False)
        for frame in manifest["frames"]:
            bad = FORBIDDEN_KEYS.intersection(frame)
            if bad:
                raise ValueError(f"Excluded keys present: {manifest_path}: {bad}")
            with np.load(manifest_path.parent / frame["bev_box_p"],
                         allow_pickle=False) as box:
                boxes = box["boxes"]; points = box["point_count"]
                tiers = box["confidence_tier"]
                required = {"track_id", "source_class", "source_index",
                            "confidence_tier", "filter_name"}
                if not required.issubset(box.files):
                    raise ValueError("Filtered box provenance is incomplete")
            expected_count = int(frame.get("box3d_profile_count",
                                           frame["box3d_tier_a_count"]))
            if len(boxes) != expected_count or len(boxes) > kmax:
                raise ValueError("Filtered box count mismatch")
            if np.any(np.abs(boxes[:, 1]) > 80) or np.any(np.abs(boxes[:, 2]) > 50):
                raise ValueError("Out-of-grid box leaked into profile")
            if preserve_low_point:
                expected_tiers = np.where(points >= 5, 1, 2).astype(np.uint8)
                if not np.array_equal(tiers, expected_tiers):
                    raise ValueError("Box confidence tier differs from point evidence")
            elif np.any(points < 5):
                raise ValueError("Non-Tier-A box leaked into v1 profile")
            counts["profile_boxes"] += len(boxes)
            counts["tier_a_boxes"] += int(np.sum(points >= 5))
            counts["tier_b_boxes"] += int(np.sum(points < 5))
            counts["source_boxes"] += int(frame["box3d_source_count"])
            if with_command:
                command = frame.get("driving_command")
                if command is None:
                    raise ValueError(f"Missing driving_command: {manifest_path}")
                command_array = np.asarray(command, np.float32)
                if command_array.shape != (4,) or not np.isfinite(command_array).all() \
                        or not np.isin(command_array, [0, 1]).all() \
                        or float(command_array.sum()) != 1.0:
                    raise ValueError(f"Invalid driving_command: {manifest_path}")
                fi = int(frame["frame"])
                expected, label = derive_command(ego["wp"][fi], bool(ego["valid"][fi]))
                if command != expected or frame.get("driving_command_label") != label:
                    raise ValueError(f"Route command differs from METEOR rule: {manifest_path}/{fi}")
                counts[f"command_{label}"] += 1
            counts["frames"] += 1
        if ego is not None:
            ego.close()
        counts["scenes"] += 1
    if counts["frames"] != dataset["counts"]["frames"] \
            or counts["profile_boxes"] != dataset["counts"]["boxes_kept"]:
        raise ValueError("Dataset totals differ from frame assets")

    ds = BevLaneDataset(
        str(root), [scenes[0]], gt_key="gt_map",
        with_depth=True, depth_hw=(108, 192),
        with_seg2d=True, seg2d_key="seg2d21",
        with_agenttraj=True, with_ego=True, with_occ=True,
        with_risk=True, with_lidarbev=True, with_command=with_command)
    batch = next(iter(DataLoader(ds, batch_size=1, num_workers=0)))
    observed = [list(x.shape) for x in batch]
    expected = [
        [1, 8, 3, 432, 768], [1, 8, 3, 3], [1, 8, 4, 4],
        [1, 800, 500], [1, 8, 108, 192], [1, 8, 108, 192],
        [1, kmax, 6], [1], [1, kmax, 6, 2], [1, kmax, 6],
        [1, 17], [1, 16, 200, 200], [1, 400, 250],
        [1, 4, 400, 250],
    ]
    if with_command:
        expected.append([1, 3])
    if observed != expected:
        raise AssertionError({"observed": observed, "expected": expected})
    confidence_result = None
    if dataset.get("schema") in {"pnk-layer2-9head-v3",
                                 "pnk-layer2-9head-v4",
                                 "pnk-layer2-9head-v5",
                                 "pnk-layer2-9head-v6",
                                 "pnk-layer2-9head-v7"}:
        ds_conf = BevLaneDataset(
            str(root), [scenes[0]], gt_key="gt_map",
            with_depth=True, with_depth_confidence=True, depth_hw=(108, 192),
            with_seg2d=True, seg2d_key="seg2d21",
            with_agenttraj=True, with_ego=True, with_occ=True,
            with_risk=True, with_lidarbev=True, with_command=with_command)
        batch_conf = next(iter(DataLoader(ds_conf, batch_size=1, num_workers=0)))
        expected_conf = expected[:5] + [[1, 8, 108, 192]] + expected[5:]
        observed_conf = [list(x.shape) for x in batch_conf]
        if observed_conf != expected_conf:
            raise AssertionError({"confidence_observed": observed_conf,
                                  "confidence_expected": expected_conf})
        confidence = batch_conf[5]
        codes = sorted(int(v) for v in confidence.unique().tolist())
        if not set(codes).issubset({0, 1, 2}) or 1 not in codes or 2 not in codes:
            raise ValueError(f"Invalid or incomplete depth confidence codes: {codes}")
        first_manifest = json.loads(
            (root / scenes[0] / "manifest.json").read_text(encoding="utf-8"))
        first_frame = first_manifest["frames"][0]
        measured = int((confidence == 1).sum())
        guided = int((confidence == 2).sum())
        if measured != int(first_frame["depth_measured_pixels"]) \
                or guided != int(first_frame["depth_interpolated_pixels"]):
            raise ValueError("Depth confidence count differs from manifest")
        confidence_result = {
            "tensor_count": len(batch_conf),
            "loader_order": dataset["sample_contract"]["confidence_aware_loader_order"],
            "loader_shapes": observed_conf,
            "codes": codes,
            "sample_measured_pixels": measured,
            "sample_guided_pixels": guided,
        }
    task = verify_tasks(root)
    derived = verify_derived(root)
    result = {
        "status": "PASS_9HEAD_PROFILE_FULL_SAMPLE_CONTRACT_AND_IO",
        "active_heads": ACTIVE, "excluded_heads": [4, 6, 11],
        "counts": dict(counts), "loader_tensor_count": len(batch),
        "loader_order": dataset["sample_contract"]["loader_order"],
        "loader_shapes": observed,
        "component_verification": {
            "task_targets": task["status"], "derived_targets": derived["status"]},
    }
    if confidence_result is not None:
        result["confidence_aware_loader"] = confidence_result
    if with_command:
        result["route_command"] = {
            "status": "PASS_METEOR_ROUTE_COMMAND_CONTRACT_AND_IO",
            "stored_order": ["left", "straight", "right", "unknown"],
            "loader_order": ["straight", "left", "right"],
            "distribution": {name: counts[f"command_{name}"]
                             for name in ("left", "straight", "right", "unknown")},
        }
    atomic_json(root / "nine_head_verification.json", result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()
    print(json.dumps(verify(args.root), indent=2), flush=True)


if __name__ == "__main__":
    main()
