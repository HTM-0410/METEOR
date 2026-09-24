#!/usr/bin/env python3
"""Create partial BEV road/sidewalk targets from PNK semantic occupancy."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from scripts.prepare_pnk_comet import atomic_json


DEFAULT_ROOT = Path(r"D:\Backup_rosbag\PNKData_layer2_samples_v1")
TARGET_HW = (800, 500)


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


def partial_drivable(occupancy: np.ndarray) -> np.ndarray:
    if occupancy.shape != (16, 200, 200):
        raise ValueError("Expected occupancy [16,200,200]")
    # Occupancy class 5=road and 6=sidewalk. Preserve ambiguity as 255 rather
    # than making unobserved cells background negatives for the lane head.
    topdown = np.full((200, 200), 255, np.uint8)
    road = np.any(occupancy == 5, axis=0)
    sidewalk = np.any(occupancy == 6, axis=0)
    topdown[road] = 1
    topdown[sidewalk] = 2
    expanded = cv2.resize(topdown, (400, 400), interpolation=cv2.INTER_NEAREST)
    target = np.full(TARGET_HW, 255, np.uint8)
    target[200:600, 50:450] = expanded
    return target


def materialize(root: Path) -> dict:
    root = root.resolve()
    dataset_path = root / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    totals: Counter = Counter()
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for frame in manifest["frames"]:
            with np.load(manifest_path.parent / frame["occ"],
                         allow_pickle=False) as value:
                occupancy = value["occ"]
            target = partial_drivable(occupancy)
            fi = int(frame["frame"])
            relative = Path("gt_drivable") / f"{fi:06d}.png"
            atomic_png(manifest_path.parent / relative, target)
            frame["gt_map"] = relative.as_posix()
            valid = target != 255
            frame["supervision_valid"]["bev_lane"] = bool(valid.any())
            totals["frames"] += 1
            totals["valid_pixels"] += int(valid.sum())
            totals["road_pixels"] += int(np.sum(target == 1))
            totals["sidewalk_pixels"] += int(np.sum(target == 2))
        manifest["partial_bev_drivable"] = {
            "shape": list(TARGET_HW),
            "classes": {"1": "road", "2": "sidewalk", "255": "ignore"},
            "source": "single_sweep_semantic_occupancy",
            "missing_classes": ["crosswalk", "laneline", "stopline",
                                "road_edge", "marking", "parking"],
            "training_gt_key": "gt_map",
        }
        allowed = manifest["training_contract"]["allowed_now"]
        if "bev_drivable_partial" not in allowed:
            allowed.append("bev_drivable_partial")
        atomic_json(manifest_path, manifest)
        totals["scenes"] += 1
    report = {
        "status": "BEV_DRIVABLE_PARTIAL_MATERIALIZED", "counts": dict(totals),
        "classes": ["road", "sidewalk"],
        "training_recipe": {"gt_key": "gt_map",
                            "supervised_class_ids": [1, 2],
                            "all_other_pixels": "ignore_255"},
        "training_gate": "PARTIAL_ONLY_KEEP_OTHER_CLASSES_IGNORE_255",
    }
    dataset["task_availability"]["1_bev_lane"] = \
        "CANDIDATE_partial_road_sidewalk_only_other_classes_ignore"
    dataset["partial_bev_drivable"] = report
    atomic_json(dataset_path, dataset)
    atomic_json(root / "drivable_report.json", report)
    atomic_json(root / "task_availability.json", dataset["task_availability"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(materialize(args.root), indent=2))


if __name__ == "__main__":
    main()
