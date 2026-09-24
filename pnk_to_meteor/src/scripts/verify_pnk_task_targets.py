#!/usr/bin/env python3
"""Verify PNK depth, E2E and agent-forecast targets plus DataLoader I/O."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from bevlane.dataset import BevLaneDataset
from scripts.prepare_pnk_comet import atomic_json


def verify(root: Path) -> dict:
    root = root.resolve()
    dataset = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    if dataset["schema"] not in {"pnk-layer2-v1", "pnk-layer2-9head-v1",
                                 "pnk-layer2-9head-v2", "pnk-layer2-9head-v3",
                                 "pnk-layer2-9head-v4", "pnk-layer2-9head-v5",
                                 "pnk-layer2-9head-v6", "pnk-layer2-9head-v7"} \
            or not dataset["complete"]:
        raise ValueError("Expected complete PNK Layer-2 dataset")
    agent_kmax = int(dataset.get("sample_contract", {}).get("agent_kmax", 320))
    counts: Counter = Counter()
    horizon_counts = np.zeros(6, np.int64)
    scenes = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scenes.append(manifest["scene"])
        with np.load(manifest_path.parent / manifest["ego_motion"],
                     allow_pickle=False) as ego:
            required = {"wp", "v0", "acc", "steer", "brake", "valid", "pose"}
            if not required.issubset(ego.files):
                raise ValueError(f"Incomplete E2E target: {manifest['scene']}")
            frame_count = 1 + max(int(frame["frame"]) for frame in manifest["frames"])
            if ego["wp"].shape != (frame_count, 6, 2) \
                    or ego["pose"].shape != (frame_count, 3):
                raise ValueError(f"Wrong E2E target shape: {manifest['scene']}")
            if not np.isfinite(ego["wp"]).all() or not np.isfinite(ego["pose"]).all():
                raise ValueError(f"Non-finite E2E target: {manifest['scene']}")
            counts["e2e_valid_frames"] += int(np.sum(ego["valid"] > 0.5))
        for frame in manifest["frames"]:
            with np.load(manifest_path.parent / frame["depth4"], allow_pickle=False) as value:
                depth6 = value["depth"]
            with np.load(manifest_path.parent / frame["depth4n"], allow_pickle=False) as value:
                depth2 = value["depth"]
            depth = np.concatenate((depth6, depth2)).astype(np.float32)
            if depth.shape != (8, 108, 192) or not np.isfinite(depth).all() \
                    or np.any(depth < 0) or np.any(depth > 80):
                raise ValueError(f"Invalid depth: {manifest['scene']}/{frame['frame']}")
            valid_depth = int(np.count_nonzero(depth))
            if valid_depth != int(frame["depth_valid_pixels"]) or valid_depth == 0:
                raise ValueError("Depth validity count mismatch")
            counts["depth_valid_pixels"] += valid_depth
            with np.load(manifest_path.parent / frame["agent_traj"],
                         allow_pickle=False) as agent:
                boxes = agent["boxes"].astype(np.float32)
                count = int(agent["count"])
                traj = agent["traj"].astype(np.float32)
                tvalid = agent["tvalid"].astype(np.float32)
            if boxes.shape != (agent_kmax, 6) \
                    or traj.shape != (agent_kmax, 6, 2) \
                    or tvalid.shape != (agent_kmax, 6) \
                    or not 0 <= count <= agent_kmax:
                raise ValueError(f"Invalid forecast shape: {manifest['scene']}/{frame['frame']}")
            if not np.isfinite(traj).all() or not np.isin(tvalid, [0, 1]).all():
                raise ValueError("Invalid forecast values")
            if np.any(np.abs(traj[tvalid == 0]) > 1e-6):
                raise ValueError("Invalid forecast cells carry non-zero targets")
            with np.load(manifest_path.parent / frame["bev_box_p"],
                         allow_pickle=False) as detector:
                source_boxes = detector["boxes"].astype(np.float32)
            if count != len(source_boxes) or not np.allclose(boxes[:count], source_boxes):
                raise ValueError("Forecast/detector box mismatch")
            horizon_counts += np.sum(tvalid[:count], axis=0).astype(np.int64)
            counts["agent_valid_future_points"] += int(tvalid[:count].sum())
            counts["frames_with_agent_forecast"] += int(bool(tvalid[:count].any()))
            counts["frames"] += 1
        counts["scenes"] += 1
    if counts["frames"] != dataset["counts"]["frames"]:
        raise ValueError("Verified frame count mismatch")
    loader = BevLaneDataset(str(root), [scenes[0]], gt_key="gt_cons",
                            with_depth=True, depth_hw=(108, 192),
                            with_seg2d=True, seg2d_key="seg2d21",
                            with_agenttraj=True, with_ego=True,
                            with_lidarbev=True)
    batch = next(iter(DataLoader(loader, batch_size=1, num_workers=0)))
    observed = [list(tensor.shape) for tensor in batch]
    expected = [[1, 8, 3, 432, 768], [1, 8, 3, 3], [1, 8, 4, 4],
                [1, 800, 500], [1, 8, 108, 192], [1, 8, 108, 192],
                [1, agent_kmax, 6], [1], [1, agent_kmax, 6, 2],
                [1, agent_kmax, 6],
                [1, 17], [1, 4, 400, 250]]
    if observed != expected:
        raise AssertionError({"observed": observed, "expected": expected})
    result = {
        "status": "PASS_DEPTH_E2E_AGENT_CONTRACT_AND_IO",
        "counts": dict(counts),
        "agent_valid_future_points_by_horizon": horizon_counts.tolist(),
        "loader_shapes": observed,
        "quality_status": {
            "metric_depth": "PASS_IO_AND_SAMPLE_OVERLAY_QA_DESKEW_UNVERIFIED",
            "e2e": "CANDIDATE_LEVER_ARM_AND_NAV_STATUS_UNVERIFIED",
            "agent_forecast": "CANDIDATE_TRACK_SCHEMA_UNCONFIRMED",
        },
    }
    atomic_json(root / "task_targets_verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(r"D:\Backup_rosbag\PNKData_layer2_samples_v1"))
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == "__main__":
    main()
