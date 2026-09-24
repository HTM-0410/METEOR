#!/usr/bin/env python3
"""Verify PNK Layer-1/Layer-2 artifacts and run a real loader smoke."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from bevlane.dataset import BevLaneDataset
from scripts.prepare_pnk_comet import atomic_json


def verify(layer1: Path, layer2: Path) -> dict:
    layer1, layer2 = layer1.resolve(), layer2.resolve()
    l1_dataset = json.loads((layer1 / "dataset.json").read_text(encoding="utf-8"))
    l2_dataset = json.loads((layer2 / "dataset.json").read_text(encoding="utf-8"))
    rectified = str(l1_dataset.get("image_geometry", "")).startswith("rectified_pinhole")
    counts: Counter = Counter()
    l2_scenes = []
    for l1_manifest_path in sorted((layer1 / "scenes").glob("*/manifest.json")):
        l1 = json.loads(l1_manifest_path.read_text(encoding="utf-8"))
        l2_manifest_path = layer2 / l1["scene"] / "manifest.json"
        if not l2_manifest_path.is_file():
            raise FileNotFoundError(l2_manifest_path)
        l2 = json.loads(l2_manifest_path.read_text(encoding="utf-8"))
        if rectified:
            for camera in l2["cams"].values():
                if any(abs(float(value)) > 1e-12 for value in camera["distortion"]):
                    raise ValueError("Rectified camera still carries non-zero distortion")
                if not camera["geometry_status"].startswith("RECTIFIED_PINHOLE"):
                    raise ValueError("Rectified camera contract was not promoted")
        if len(l1["frames"]) != len(l2["frames"]):
            raise ValueError(f"Layer frame mismatch: {l1['scene']}")
        l2_scenes.append(l1["scene"])
        for source, sample in zip(l1["frames"], l2["frames"]):
            if source["timestamp_ns"] != sample["timestamp_ns"]:
                raise ValueError("Layer timestamp mismatch")
            with np.load(l1_manifest_path.parent / source["layer1_panoptic"],
                         allow_pickle=False) as panoptic:
                sem = panoptic["semantic19"]
                seg = panoptic["segment_id"]
                inst = panoptic["instance_id"]
                meta = json.loads(str(panoptic["segments_json"]))
                if sem.shape != seg.shape or sem.shape != inst.shape or sem.shape[0] != 8:
                    raise ValueError("Invalid Layer-1 panoptic geometry")
                if len(meta) != 8 or np.any(inst[sem == 255]):
                    raise ValueError("Invalid Layer-1 metadata/void contract")
            with np.load(l2_manifest_path.parent / sample["seg2d21"],
                         allow_pickle=False) as target:
                if target["seg"].shape != (8, 108, 192) or target["valid"].shape != (8, 108, 192):
                    raise ValueError("Invalid Layer-2 seg2d target")
                if not np.array_equal(target["valid"], target["seg"] != 255):
                    raise ValueError("Layer-2 segmentation validity mismatch")
            with np.load(l2_manifest_path.parent / sample["bbox2d_candidate"],
                         allow_pickle=False) as candidate:
                if candidate["boxes"].shape != (8, 96, 5):
                    raise ValueError("Invalid 2D candidate shape")
                if np.any(candidate["image_annotation_complete"]):
                    raise ValueError("Candidate incorrectly marked complete")
            for key in ("bev_box_p", "lidar_bev"):
                if not (l2_manifest_path.parent / sample[key]).is_file():
                    raise FileNotFoundError(sample[key])
            if sample["supervision_valid"]["bbox2d_complete"]:
                raise ValueError("Incomplete 2D labels enabled")
            counts["frames"] += 1
        counts["scenes"] += 1
    if counts["frames"] != l1_dataset["counts"]["frames"]:
        raise ValueError("Verified frame count differs from Layer-1 declaration")
    if counts["frames"] != l2_dataset["counts"]["frames"]:
        raise ValueError("Verified frame count differs from Layer-2 declaration")
    dataset = BevLaneDataset(str(layer2), [l2_scenes[0]], gt_key="gt_cons",
                             with_seg2d=True, seg2d_key="seg2d21",
                             with_boxdet=True, boxdet_kmax=320,
                             with_lidarbev=True)
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    images, K, T, gt, seg2d, boxes, nbox, lidar = batch
    expected = {
        "images": [1, 8, 3, 432, 768], "K": [1, 8, 3, 3],
        "T_cam_ego": [1, 8, 4, 4], "gt": [1, 800, 500],
        "seg2d21": [1, 8, 108, 192], "boxes": [1, 320, 6],
        "lidar": [1, 4, 400, 250],
    }
    observed = {"images": list(images.shape), "K": list(K.shape),
                "T_cam_ego": list(T.shape), "gt": list(gt.shape),
                "seg2d21": list(seg2d.shape), "boxes": list(boxes.shape),
                "lidar": list(lidar.shape)}
    if observed != expected or not bool((gt == 255).all()):
        raise AssertionError({"observed": observed, "expected": expected,
                              "gt_is_all_ignore": bool((gt == 255).all())})
    result = {
        "status": "PASS_CONTRACT_AND_IO_ONLY", "counts": dict(counts),
        "loader_shapes": observed, "boxes_in_smoke_frame": int(nbox[0]),
        "lane_target": "ALL_IGNORE_255", "bbox2d_training": "DISABLED_CANDIDATE_ONLY",
        "distortion_status": ("RECTIFIED_ASSUMED_MODEL" if rectified
                              else "RAW_DISTORTED"),
        "training_readiness": ("HOLD_SOURCE_CAMERA_MODEL_CONFIRMATION_PROJECTION_AND_LABEL_QA"
                               if rectified else
                               "HOLD_CAMERA_GEOMETRY_AND_LABEL_QUALITY_QA"),
    }
    atomic_json(layer2 / "verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer1", type=Path,
                        default=Path(r"D:\Backup_rosbag\PNKData_layer1_v1"))
    parser.add_argument("--layer2", type=Path,
                        default=Path(r"D:\Backup_rosbag\PNKData_layer2_samples_v1"))
    args = parser.parse_args()
    print(json.dumps(verify(args.layer1, args.layer2), indent=2))


if __name__ == "__main__":
    main()
