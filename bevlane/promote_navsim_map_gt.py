#!/usr/bin/env python3
"""Promote verified NAVSIM map rasters to METEOR's default ``gt`` field.

The initial NAVSIM ingestion writes one all-255 ``gt/ignore.png`` per scene so
the base dataloader can read frames before map processing.  Once every frame
has a real ``gt_map`` target, this command backs up the manifest, redirects
``gt`` to ``gt_map``, verifies the published manifests, and only then removes
the 64 obsolete placeholder PNGs.  It never deletes a map raster or raw data.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path, PurePosixPath
from typing import Dict, List, Tuple

import cv2
import numpy as np

from bevlane.ingest_navsim import _atomic_json


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
EXPECTED_WIDTH = 500
EXPECTED_HEIGHT = 800
BACKUP_NAME = "manifest.before_gt_promotion.json"


def _inside_scene(scene: Path, relative: str) -> Path:
    path = PurePosixPath(str(relative).replace("\\", "/"))
    if (path.is_absolute() or ".." in path.parts or len(path.parts) != 2
            or path.parts[0] != "gt_map" or path.suffix.lower() != ".png"):
        raise ValueError(f"Unsafe manifest target: {relative!r}")
    if (scene / "gt_map").is_symlink():
        raise ValueError(f"Map target directory is a symlink: {scene / 'gt_map'}")
    full = scene.joinpath(*path.parts)
    return full


def _check_map_png(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Missing/nonregular map target: {path}")
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) < 24 or header[:8] != PNG_SIGNATURE:
        raise ValueError(f"Invalid PNG signature: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if (width, height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
        raise ValueError(f"Wrong map target size {width}x{height}: {path}")


def _check_placeholder(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Placeholder is not a regular file: {path}")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (EXPECTED_HEIGHT, EXPECTED_WIDTH):
        raise ValueError(f"Placeholder has wrong image shape: {path}")
    if not np.all(image == 255):
        raise ValueError(f"Refusing to remove non-placeholder image: {path}")


def preflight(root: Path) -> Tuple[List[Dict], int]:
    """Validate all scenes before modifying a single manifest or PNG."""
    root = root.resolve()
    manifests = sorted(root.glob("*/manifest.json"))
    if not manifests:
        raise RuntimeError(f"No scene manifests found under {root}")
    plans = []
    frames_total = 0
    for manifest_path in manifests:
        scene = manifest_path.parent
        if scene.parent.resolve() != root:
            raise ValueError(f"Manifest outside selected root: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source") != "NAVSIM/OpenScene":
            raise ValueError(f"Not a NAVSIM/OpenScene manifest: {manifest_path}")
        records = manifest.get("frames", [])
        map_metadata = manifest.get("map_gt", {})
        if not records or map_metadata.get("frames_rasterized") != len(records):
            raise ValueError(f"Incomplete map conversion: {manifest_path}")

        for record in records:
            relative = record.get("gt_map")
            if not relative:
                raise ValueError(f"Frame {record.get('frame')} has no gt_map: {manifest_path}")
            target = _inside_scene(scene, relative)
            _check_map_png(target)
            if record.get("gt") not in ("gt/ignore.png", relative):
                raise ValueError(f"Unexpected default gt in {manifest_path}: {record.get('gt')!r}")
            frames_total += 1

        placeholder = scene / "gt" / "ignore.png"
        _check_placeholder(placeholder)
        plans.append({
            "manifest_path": manifest_path,
            "backup_path": scene / BACKUP_NAME,
            "placeholder": placeholder,
            "manifest": manifest,
        })
    return plans, frames_total


def promote(root: Path, apply: bool = False) -> Dict[str, int]:
    plans, frames_total = preflight(root)
    summary = {
        "scenes": len(plans),
        "frames": frames_total,
        "placeholders": sum(plan["placeholder"].exists() for plan in plans),
    }
    if not apply:
        return summary

    # Create all backups before the first manifest edit. Exclusive creation
    # preserves an earlier recovery point if a previous run was interrupted.
    for plan in plans:
        backup = plan["backup_path"]
        if not backup.exists():
            with backup.open("xb") as stream:
                stream.write(plan["manifest_path"].read_bytes())
                stream.flush()
                os.fsync(stream.fileno())

    for plan in plans:
        manifest = plan["manifest"]
        for record in manifest["frames"]:
            record["gt"] = record["gt_map"]
        manifest["map_gt"]["promoted_to_default_gt"] = True
        _atomic_json(plan["manifest_path"], manifest)

    # Publish-then-delete ordering: if an edit fails, every placeholder still
    # exists and the user can rerun safely. Verify all manifests again before
    # unlinking the exact, preflighted all-255 files.
    for plan in plans:
        manifest = json.loads(plan["manifest_path"].read_text(encoding="utf-8"))
        if any(record.get("gt") != record.get("gt_map") for record in manifest["frames"]):
            raise RuntimeError(f"Manifest promotion verification failed: {plan['manifest_path']}")

    for plan in plans:
        placeholder = plan["placeholder"]
        if placeholder.exists():
            _check_placeholder(placeholder)
            placeholder.unlink()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="existing METEOR NAVSIM output root")
    parser.add_argument("--apply", action="store_true",
                        help="write manifest backups, switch gt, and delete verified placeholders")
    args = parser.parse_args()
    summary = promote(args.root, apply=args.apply)
    print(json.dumps({"mode": "applied" if args.apply else "dry_run", **summary}))


if __name__ == "__main__":
    main()
