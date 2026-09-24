#!/usr/bin/env python3
"""Verify PNK flow, occupancy, risk and partial-drivable targets."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import DataLoader

from bevlane.dataset import BevLaneDataset, _imread
from scripts.materialize_pnk_flow import build_flow_target
from scripts.prepare_pnk_comet import atomic_json


def verify(root: Path) -> dict:
    root = root.resolve()
    dataset = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    if dataset.get("schema") not in {"pnk-layer2-v1", "pnk-layer2-9head-v1",
                                     "pnk-layer2-9head-v2", "pnk-layer2-9head-v3",
                                     "pnk-layer2-9head-v4", "pnk-layer2-9head-v5",
                                     "pnk-layer2-9head-v6", "pnk-layer2-9head-v7"}:
        raise ValueError("Expected PNK Layer-2 dataset")
    agent_kmax = int(dataset.get("sample_contract", {}).get("agent_kmax", 320))
    counts: Counter = Counter()
    scenes = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scenes.append(manifest["scene"])
        with np.load(manifest_path.parent / manifest["risk_map"],
                     allow_pickle=False) as value:
            risk_all = value["risk"]
        expected_frames = 1 + max(int(frame["frame"]) for frame in manifest["frames"])
        if risk_all.shape != (expected_frames, 400, 250):
            raise ValueError(f"Invalid risk shape: {manifest['scene']}")
        for frame in manifest["frames"]:
            fi = int(frame["frame"])
            with np.load(manifest_path.parent / frame["occ"],
                         allow_pickle=False) as value:
                occupancy = value["occ"]
            if occupancy.shape != (16, 200, 200):
                raise ValueError("Invalid occupancy shape")
            labels = np.unique(occupancy)
            if np.any((labels != 255) & (labels > 9)):
                raise ValueError("Invalid occupancy class")
            counts["observed_occ_voxels"] += int(np.sum(occupancy != 255))
            with np.load(manifest_path.parent / frame["agent_traj"],
                         allow_pickle=False) as agent:
                boxes, count = agent["boxes"], int(agent["count"])
                traj, tvalid = agent["traj"], agent["tvalid"]
            expected_flow, expected_valid = build_flow_target(
                boxes, count, traj, tvalid)
            with np.load(manifest_path.parent / frame["flow_target"],
                         allow_pickle=False) as flow_asset:
                flow = flow_asset["flow"].astype(np.float32)
                valid = flow_asset["valid"]
            if flow.shape != (2, 200, 200) or valid.shape != (200, 200):
                raise ValueError("Invalid flow shape")
            if not np.array_equal(valid, expected_valid) \
                    or not np.allclose(flow, expected_flow, atol=0.02):
                raise ValueError("Flow target differs from agent trajectory")
            if np.any(np.abs(flow[:, valid == 0]) > 1e-6):
                raise ValueError("Invalid flow cells carry velocity")
            counts["flow_valid_cells"] += int(valid.sum())
            risk = risk_all[fi]
            if risk.dtype != np.uint8:
                raise ValueError("Risk must use uint8 storage")
            counts["high_risk_pixels"] += int(np.sum(risk >= 128))
            drivable = _imread(manifest_path.parent / frame["gt_map"],
                               cv2.IMREAD_GRAYSCALE)
            allowed_drivable = ({1, 2, 3, 4, 5, 6, 7, 8, 255}
                                 if dataset.get("schema") in {
                                     "pnk-layer2-9head-v4", "pnk-layer2-9head-v5",
                                     "pnk-layer2-9head-v6",
                                     "pnk-layer2-9head-v7"}
                                 else {1, 2, 255})
            if drivable is None or drivable.shape != (800, 500) \
                    or not set(np.unique(drivable)).issubset(allowed_drivable):
                raise ValueError("Invalid partial drivable target")
            counts["drivable_valid_pixels"] += int(np.sum(drivable != 255))
            counts["frames"] += 1
        counts["scenes"] += 1
    if counts["frames"] != dataset["counts"]["frames"]:
        raise ValueError("Derived target frame count mismatch")
    loader = BevLaneDataset(str(root), [scenes[0]], gt_key="gt_map",
                            with_agenttraj=True, with_occ=True, with_risk=True)
    batch = next(iter(DataLoader(loader, batch_size=1, num_workers=0)))
    observed = [list(tensor.shape) for tensor in batch]
    expected = [[1, 8, 3, 432, 768], [1, 8, 3, 3], [1, 8, 4, 4],
                [1, 800, 500], [1, agent_kmax, 6], [1],
                [1, agent_kmax, 6, 2], [1, agent_kmax, 6],
                [1, 16, 200, 200], [1, 400, 250]]
    if observed != expected:
        raise AssertionError({"observed": observed, "expected": expected})
    result = {
        "status": "PASS_FLOW_OCC_RISK_DRIVABLE_CONTRACT_AND_IO",
        "counts": dict(counts), "loader_shapes": observed,
        "quality_status": {
            "occupancy_flow": "CANDIDATE_TRACK_SCHEMA_QA_PENDING",
            "occupancy3d": "CANDIDATE_SINGLE_SWEEP_SPARSE_DESKEW_UNVERIFIED",
            "area_risk": "DERIVED_CANDIDATE_DEPENDS_ON_OCC_AND_AGENT",
            "bev_drivable": (
                "CANDIDATE_METEOR_CLASSES_1_TO_8_CONFIDENCE_MARKED"
                if dataset.get("schema") in {
                    "pnk-layer2-9head-v4", "pnk-layer2-9head-v5",
                    "pnk-layer2-9head-v6", "pnk-layer2-9head-v7"}
                else "PARTIAL_ROAD_SIDEWALK_ONLY"),
        },
    }
    atomic_json(root / "derived_targets_verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(r"D:\Backup_rosbag\PNKData_layer2_samples_v1"))
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == "__main__":
    main()
