#!/usr/bin/env python3
"""Materialize loss-format 3D boxes without the legacy 64-object truncation."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.prepare_pnk_comet import DEFAULT_OUTPUT, atomic_json


def atomic_boxes(path: Path, boxes: np.ndarray, point_count: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, boxes=boxes.astype(np.float32),
                            point_count=point_count.astype(np.int32))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def materialize(root: Path) -> dict:
    root = root.resolve()
    dataset = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    if dataset["schema"] != "pnk-comet-handoff-v1":
        raise ValueError("Wrong handoff schema")
    stats: Counter = Counter()
    max_boxes = 0
    for path in sorted((root / "scenes").glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for frame in manifest["frames"]:
            source = Path(frame["boxes_3d_npz"])
            with np.load(source) as raw:
                boxes = raw["meteor_boxes"]
                points = raw["point_count"]
                source_class = raw["source_class"]
            if boxes.shape != (frame["boxes_3d_count"], 6) or not np.isfinite(boxes).all():
                raise ValueError(f"Invalid METEOR boxes: {source}")
            relative = Path("bev_box_p") / f'{frame["frame"]:06d}.npz'
            destination = path.parent / relative
            if destination.exists():
                with np.load(destination) as saved:
                    if saved["boxes"].shape != boxes.shape or not np.array_equal(saved["boxes"], boxes):
                        raise ValueError(f"Existing target differs from source: {destination}")
                stats["reused"] += 1
            else:
                atomic_boxes(destination, boxes, points)
                stats["created"] += 1
            frame["bev_box_p"] = relative.as_posix()
            # No occlusion/camera confirmation has been performed.  These are
            # clean source labels, not all-negative supervision beyond boxes.
            frame["bev_box_p_provenance"] = "source_3d_annotation_no_camera_confirmation"
            stats["frames"] += 1
            stats["boxes"] += len(boxes)
            stats["frames_over_loader_kmax64"] += int(len(boxes) > 64)
            max_boxes = max(max_boxes, len(boxes))
            for name, count in zip(*np.unique(source_class, return_counts=True)):
                stats[f"source_class_{name}"] += int(count)
            within = ((boxes[:, 1] >= -80) & (boxes[:, 1] <= 80)
                      & (boxes[:, 2] >= -50) & (boxes[:, 2] <= 50))
            stats["boxes_within_bev_extent"] += int(np.sum(within))
        atomic_json(path, manifest)
        print(f"[boxes] {manifest['scene']}: {len(manifest['frames'])} frames", flush=True)
    report = {"counts": dict(stats), "max_boxes_per_frame": max_boxes,
              "source_gt_status": "clean_source_labels_geometry_semantics_still_need_qa",
              "legacy_loader_kmax64_safe": max_boxes <= 64,
              "no_camera_visibility_filter": True}
    atomic_json(root / "boxdet_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(materialize(args.root), indent=2))


if __name__ == "__main__":
    main()
