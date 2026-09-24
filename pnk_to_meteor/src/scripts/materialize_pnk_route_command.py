#!/usr/bin/env python3
"""Add METEOR-compatible pseudo route commands to a PNK 9-head profile.

PNK does not provide an upstream route plan.  This stage therefore derives a
maneuver-intent pseudo label from the already materialized, validity-gated ego
future.  The rule intentionally matches the v43+ derived-intent rule in
``bevlane/train.py``:

* inspect lateral waypoints at 1.5--3.0 s;
* left when any value is above +2 m;
* right when no left value exists and any value is below -2 m;
* otherwise straight;
* invalid/non-finite ego futures are unknown.

Manifests store NAVSIM order ``[left, straight, right, unknown]``.  The dataset
loader converts it to METEOR order ``[straight, left, right]`` and maps unknown
to the all-zero vector.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.prepare_pnk_comet import atomic_json, sha256


DEFAULT_SOURCE = Path(r"D:\Backup_rosbag\PNKData_layer2_9head_v4")
DEFAULT_OUTPUT = Path(r"D:\Backup_rosbag\PNKData_layer2_9head_v5")
LATERAL_THRESHOLD_M = 2.0
COMMANDS = {
    "left": [1, 0, 0, 0],
    "straight": [0, 1, 0, 0],
    "right": [0, 0, 1, 0],
    "unknown": [0, 0, 0, 1],
}


def derive_command(waypoints: np.ndarray, valid: bool,
                   threshold_m: float = LATERAL_THRESHOLD_M) -> tuple[list[int], str]:
    """Return NAVSIM-order command and label using the METEOR v43+ rule."""
    wp = np.asarray(waypoints, dtype=np.float32)
    if not valid or wp.shape != (6, 2) or not np.isfinite(wp).all():
        return COMMANDS["unknown"].copy(), "unknown"
    lateral = wp[2:, 1]
    if float(lateral.max()) > threshold_m:
        label = "left"
    elif float(lateral.min()) < -threshold_m:
        label = "right"
    else:
        label = "straight"
    return COMMANDS[label].copy(), label


def clone_profile(source: Path, output: Path) -> Counter:
    """Copy mutable JSON and hard-link immutable assets into a new profile."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    for src in source.rglob("*"):
        rel = src.relative_to(source)
        dst = output / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix.lower() == ".json":
            shutil.copy2(src, dst)
            counts["json_copied"] += 1
            continue
        try:
            os.link(src, dst)
            counts["assets_hardlinked"] += 1
        except OSError:
            shutil.copy2(src, dst)
            counts["assets_copied"] += 1
    return counts


def materialize(source: Path, output: Path,
                threshold_m: float = LATERAL_THRESHOLD_M) -> dict:
    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError("Output must differ from source")
    source_dataset_path = source / "dataset.json"
    dataset = json.loads(source_dataset_path.read_text(encoding="utf-8"))
    if dataset.get("schema") != "pnk-layer2-9head-v4" or not dataset.get("complete"):
        raise ValueError("Expected complete PNK 9-head v4 source profile")
    counts = clone_profile(source, output)
    example_by_command = {}
    for manifest_path in sorted(output.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ego_path = manifest_path.parent / manifest["ego_motion"]
        with np.load(ego_path, allow_pickle=False) as ego:
            wp = ego["wp"].astype(np.float32)
            valid = ego["valid"].astype(bool)
        for frame in manifest["frames"]:
            fi = int(frame["frame"])
            in_range = fi < len(valid) and fi < len(wp)
            command, label = derive_command(
                wp[fi] if in_range else np.zeros((6, 2), np.float32),
                bool(valid[fi]) if in_range else False,
                threshold_m,
            )
            frame["driving_command"] = command
            frame["driving_command_label"] = label
            frame["driving_command_source"] = "realized_future_ego_trajectory_pseudo_intent"
            frame.setdefault("supervision_valid", {})["route_command"] = label != "unknown"
            counts[f"command_{label}"] += 1
            counts["frames"] += 1
            example_by_command.setdefault(label, {"scene": manifest["scene"], "frame": fi})
        manifest["schema"] = "pnk-layer2-9head-scene-v5"
        manifest.setdefault("e2e_target", {})["route_command"] = {
            "kind": "pseudo_intent_from_realized_future_trajectory",
            "storage_contract": "NAVSIM_[left,straight,right,unknown]",
            "loader_contract": "METEOR_[straight,left,right]; unknown_is_zero_vector",
            "source": "ego_motion.wp validity-gated NAV future",
            "horizons_used_s": [1.5, 2.0, 2.5, 3.0],
            "lateral_threshold_m": threshold_m,
            "left_priority_matches_train_v43_plus": True,
            "limitation": "pseudo maneuver label from realized future, not an upstream route instruction",
        }
        for key in ("active_targets", "allowed_now"):
            values = manifest.setdefault("training_contract", {}).setdefault(key, [])
            if "route_command_pseudo" not in values:
                values.append("route_command_pseudo")
        atomic_json(manifest_path, manifest)
        counts["scenes"] += 1

    report = {
        "status": "PNK_ROUTE_COMMAND_PSEUDO_LABELS_MATERIALIZED",
        "source_profile": str(source),
        "output_profile": str(output),
        "counts": dict(counts),
        "distribution": {name: int(counts[f"command_{name}"]) for name in COMMANDS},
        "examples": example_by_command,
        "contract": {
            "stored_order": ["left", "straight", "right", "unknown"],
            "loader_order": ["straight", "left", "right"],
            "unknown_loader_value": [0, 0, 0],
            "threshold_m": threshold_m,
            "waypoint_horizons_s": [1.5, 2.0, 2.5, 3.0],
        },
        "quality": {
            "valid_command_frames": int(counts["frames"] - counts["command_unknown"]),
            "unknown_frames": int(counts["command_unknown"]),
            "source": "realized future pseudo intent",
            "not_equivalent_to_navigation_route": True,
        },
    }
    dataset_path = output / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["schema"] = "pnk-layer2-9head-v5"
    dataset["source_profile"] = str(source)
    dataset["source_profile_dataset_sha256"] = sha256(source_dataset_path)
    dataset["task_availability"]["7_e2e"] = \
        "CANDIDATE_validity_gated_with_realized_future_pseudo_route_command"
    dataset["e2e_route_command"] = report
    contract = dataset["sample_contract"]
    contract.setdefault("loader_flags", {})["with_command"] = True
    if "driving_command" not in contract["loader_order"]:
        contract["loader_order"].append("driving_command")
    if "confidence_aware_loader_order" in contract \
            and "driving_command" not in contract["confidence_aware_loader_order"]:
        contract["confidence_aware_loader_order"].append("driving_command")
    recipe = dataset.setdefault("safe_loader_recipe_after_gate", {})
    recipe["with_command"] = True
    recipe["driving_command_source"] = "raw"
    atomic_json(dataset_path, dataset)
    atomic_json(output / "task_availability.json", dataset["task_availability"])
    atomic_json(output / "route_command_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold-m", type=float, default=LATERAL_THRESHOLD_M)
    args = parser.parse_args()
    if args.threshold_m <= 0:
        parser.error("--threshold-m must be positive")
    print(json.dumps(materialize(args.source, args.output, args.threshold_m),
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
