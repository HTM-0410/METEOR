#!/usr/bin/env python3
"""Verify every PNK Layer-1 panoptic asset and summarize its label content."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.prepare_pnk_comet import atomic_json, sha256


def verify(root: Path) -> dict:
    root = root.resolve()
    dataset_path = root / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if dataset["schema"] != "pnk-layer1-v1" or not dataset["complete"]:
        raise ValueError("Expected a complete PNK Layer-1 dataset")
    source = Path(dataset["source_input"])
    source_dataset_path = source / "dataset.json"
    if sha256(source_dataset_path) != dataset["source_input_dataset_sha256"]:
        raise ValueError("Layer-1 source dataset changed after inference")
    source_dataset = json.loads(source_dataset_path.read_text(encoding="utf-8"))
    expected_shape = tuple(dataset["panoptic_shape"])
    counts: Counter = Counter()
    class_pixels: Counter = Counter()
    scores = []
    for manifest_path in sorted((root / "scenes").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_path = source / "scenes" / manifest["scene"] / "manifest.json"
        source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
        if len(manifest["frames"]) != len(source_manifest["frames"]):
            raise ValueError(f"Frame count mismatch: {manifest['scene']}")
        for frame, source_frame in zip(manifest["frames"], source_manifest["frames"]):
            if frame["timestamp_ns"] != source_frame["timestamp_ns"]:
                raise ValueError(f"Timestamp mismatch: {manifest['scene']}")
            asset_path = manifest_path.parent / frame["layer1_panoptic"]
            with np.load(asset_path, allow_pickle=False) as asset:
                semantic = asset["semantic19"]
                segment = asset["segment_id"]
                instance = asset["instance_id"]
                metadata = json.loads(str(asset["segments_json"]))
            if (semantic.shape != expected_shape or segment.shape != expected_shape
                    or instance.shape != expected_shape):
                raise ValueError(f"Panoptic shape mismatch: {asset_path}")
            if semantic.dtype != np.uint8 or len(metadata) != expected_shape[0]:
                raise ValueError(f"Panoptic dtype/metadata mismatch: {asset_path}")
            labels = np.unique(semantic)
            if np.any((labels != 255) & (labels > 18)):
                raise ValueError(f"Invalid Cityscapes train ID: {asset_path}")
            for camera_index, regions in enumerate(metadata):
                valid_ids = {int(region["id"]): int(region["label_id"])
                             for region in regions}
                observed_ids = set(np.unique(segment[camera_index][semantic[camera_index] != 255]))
                if not observed_ids.issubset(valid_ids):
                    raise ValueError(f"Segment metadata mismatch: {asset_path}")
                for region in regions:
                    mask = segment[camera_index] == int(region["id"])
                    if mask.any() and not np.all(semantic[camera_index][mask]
                                                 == int(region["label_id"])):
                        raise ValueError(f"Semantic/segment mismatch: {asset_path}")
                    if region["isthing"] and mask.any() and not np.all(
                            instance[camera_index][mask] == int(region["id"])):
                        raise ValueError(f"Thing instance mismatch: {asset_path}")
                    if not region["isthing"] and np.any(instance[camera_index][mask]):
                        raise ValueError(f"Stuff carries instance ID: {asset_path}")
                    scores.append(float(region["score"]))
                    counts["segments"] += 1
            values, frequencies = np.unique(semantic, return_counts=True)
            for label, frequency in zip(values, frequencies):
                class_pixels[str(int(label))] += int(frequency)
            counts["frames"] += 1
            counts["camera_images"] += expected_shape[0]
        counts["scenes"] += 1
    if counts["frames"] != source_dataset["counts"]["frames"]:
        raise ValueError("Layer-1 frame count differs from rectified source")
    total_pixels = sum(class_pixels.values())
    score_array = np.asarray(scores, dtype=np.float64)
    result = {
        "status": "PASS_LAYER1_CONTRACT_AND_IO",
        "counts": dict(counts),
        "panoptic_shape": list(expected_shape),
        "class_pixel_counts": dict(class_pixels),
        "void_pixel_fraction": float(class_pixels.get("255", 0) / total_pixels),
        "segment_score_p10_p50_p90": (np.quantile(score_array, [.1, .5, .9]).tolist()
                                       if score_array.size else None),
        "model": dataset["model"], "revision": dataset["revision"],
        "image_geometry": dataset["image_geometry"],
        "quality_status": "PSEUDO_LABELS_REQUIRE_STRATIFIED_HUMAN_QA",
    }
    atomic_json(root / "verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(r"D:\Backup_rosbag\PNKData_layer1_v1"))
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == "__main__":
    main()
