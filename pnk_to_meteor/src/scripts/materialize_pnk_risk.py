#!/usr/bin/env python3
"""Create PNK area-risk targets from occupancy and agent trajectories."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from bevlane.risk_field import RH, RW, risk_field
from scripts.prepare_pnk_comet import atomic_json


DEFAULT_ROOT = Path(r"D:\Backup_rosbag\PNKData_layer2_samples_v1")


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz",
                                     dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def materialize(root: Path) -> dict:
    root = root.resolve()
    dataset_path = root / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    totals: Counter = Counter()
    risk_values = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frame_count = 1 + max(int(frame["frame"]) for frame in manifest["frames"])
        risk = np.zeros((frame_count, RH, RW), np.uint8)
        for frame in manifest["frames"]:
            fi = int(frame["frame"])
            with np.load(manifest_path.parent / frame["agent_traj"],
                         allow_pickle=False) as agent:
                boxes, count = agent["boxes"], int(agent["count"])
                traj, tvalid = agent["traj"], agent["tvalid"]
            with np.load(manifest_path.parent / frame["occ"],
                         allow_pickle=False) as occupancy:
                occ = occupancy["occ"]
            target = risk_field(boxes, count, traj, tvalid, occ)
            risk[fi] = np.rint(np.clip(target, 0, 1) * 255).astype(np.uint8)
            risk_values.append(target.reshape(-1))
            frame["supervision_valid"]["risk"] = True
            totals["frames"] += 1
            totals["high_risk_pixels"] += int(np.sum(target >= 0.5))
        destination = manifest_path.parent / "risk_map.npz"
        atomic_npz(destination, risk=risk,
                   provenance=np.asarray("METEOR_risk_field_occ_plus_agent_candidate"))
        manifest["risk_map"] = "risk_map.npz"
        manifest["area_risk"] = {
            "shape": [RH, RW], "range": [0, 1],
            "source": "semantic_occupancy_plus_agent_forecast",
            "definition": "bevlane.risk_field.risk_field",
        }
        allowed = manifest["training_contract"]["allowed_now"]
        if "area_risk_candidate" not in allowed:
            allowed.append("area_risk_candidate")
        atomic_json(manifest_path, manifest)
        totals["scenes"] += 1
    values = np.concatenate(risk_values) if risk_values else np.zeros(0, np.float32)
    report = {
        "status": "RISK_CANDIDATE_MATERIALIZED", "counts": dict(totals),
        "risk_p50_p90_p99_max": (np.quantile(values, [.5, .9, .99, 1]).tolist()
                                  if values.size else None),
        "training_gate": "HOLD_OCCUPANCY_AND_AGENT_LABEL_QA",
    }
    dataset["task_availability"]["12_risk"] = \
        "CANDIDATE_from_occupancy_and_agent_future"
    dataset["safe_loader_recipe_after_gate"]["with_risk"] = True
    dataset["area_risk"] = report
    atomic_json(dataset_path, dataset)
    atomic_json(root / "risk_report.json", report)
    atomic_json(root / "task_availability.json", dataset["task_availability"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(materialize(args.root), indent=2))


if __name__ == "__main__":
    main()
