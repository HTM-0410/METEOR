#!/usr/bin/env python3
"""Convert PNK Layer-1 assets into guarded METEOR Layer-2 samples.

Only targets with explicit provenance are installed.  Unsupported supervision
stays ignored (255) or is kept as a candidate without a training manifest key.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from scripts.materialize_pnk_layer1 import PNK_CAMERAS
from scripts.prepare_pnk_comet import PROVISIONAL_SLOT_MAP, atomic_json, sha256


DEFAULT_LAYER1 = Path(r"D:\Backup_rosbag\PNKData_layer1_v1")
DEFAULT_OUTPUT = Path(r"D:\Backup_rosbag\PNKData_layer2_samples_v1")
METEOR_CAMERAS = ("CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                  "CAM_BACK_WIDE", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
                  "CAM_FRONT_NARROW", "CAM_BACK_NARROW")

# Cityscapes train ID -> METEOR 21-class ID.  Ambiguous rider/train labels stay
# 255 rather than being silently coerced into a wrong training class.
CITYSCAPES_TO_METEOR21 = {
    0: 11, 1: 12, 2: 17, 3: 16, 4: 16, 5: 20, 6: 9, 7: 10,
    8: 18, 9: 18, 10: 19, 11: 7, 13: 2, 14: 3, 15: 4,
    17: 5, 18: 6,
}
CITYSCAPES_THING_TO_DET10 = {
    "person": 6, "car": 1, "truck": 2, "bus": 3,
    "motorcycle": 5, "bicycle": 4,
}


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "reused"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def camera_contract(layer1_manifest: dict) -> dict:
    output = {}
    rectified = "cameras_rectified" in layer1_manifest
    for meteor_name in METEOR_CAMERAS:
        pnk_name = PROVISIONAL_SLOT_MAP[meteor_name]
        source = layer1_manifest["cameras_rectified" if rectified else "cameras_raw"][pnk_name]
        height, width = source["image_hw"]
        if rectified:
            K = np.asarray(source["K"], np.float64).copy()
            distortion = source["D"]
            geometry_status = "RECTIFIED_PINHOLE_SOURCE_MODEL_CONFIRMATION_PENDING"
        else:
            K = np.asarray(source["K_raw"], np.float64).copy()
            K[0, :] *= 768.0 / width
            K[1, :] *= 432.0 / height
            distortion = source["D_raw"]
            geometry_status = "PROVISIONAL_distortion_and_transform_schema_pending_QA"
        T_ego_cam = np.linalg.inv(np.asarray(source["T_cam_ego"], np.float64))
        output[meteor_name] = {
            "K": K.tolist(), "T_ego_cam": T_ego_cam.tolist(),
            "source_camera": pnk_name, "source_hw": [height, width],
            "distortion": distortion,
            "geometry_status": geometry_status,
        }
    return output


def materialize_seg2d(semantic19: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if semantic19.shape[0] != len(PNK_CAMERAS):
        raise ValueError("Layer-1 camera count mismatch")
    lookup = np.full(256, 255, np.uint8)
    for source, target in CITYSCAPES_TO_METEOR21.items():
        lookup[source] = target
    mapped = lookup[semantic19]
    resized = np.stack([cv2.resize(channel, (192, 108), interpolation=cv2.INTER_NEAREST)
                        for channel in mapped])
    return resized.astype(np.uint8), (resized != 255).astype(np.uint8)


def materialize_bbox_candidates(instance_id: np.ndarray,
                                 segments_meta: list[list[dict]]) -> tuple[np.ndarray, np.ndarray]:
    boxes = np.zeros((len(PNK_CAMERAS), 96, 5), np.float32)
    counts = np.zeros(len(PNK_CAMERAS), np.uint8)
    height, width = instance_id.shape[1:]
    for camera_index, regions in enumerate(segments_meta):
        candidates = []
        for region in regions:
            target_class = CITYSCAPES_THING_TO_DET10.get(region["label"])
            if target_class is None or not region["isthing"]:
                continue
            ys, xs = np.nonzero(instance_id[camera_index] == int(region["id"]))
            if not len(xs):
                continue
            x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            cx = (x1 + x2) * .5 * 768 / width
            cy = (y1 + y2) * .5 * 432 / height
            bw, bh = (x2 - x1) * 768 / width, (y2 - y1) * 432 / height
            if bw < 2 or bh < 2:
                continue
            candidates.append((bw * bh, target_class, cx, cy, bw, bh))
        candidates.sort(reverse=True)
        counts[camera_index] = min(len(candidates), 96)
        for index, (_, cls, cx, cy, bw, bh) in enumerate(candidates[:96]):
            boxes[camera_index, index] = (cls, cx, cy, bw, bh)
    return boxes, counts


def materialize(layer1: Path, output: Path) -> dict:
    layer1, output = layer1.resolve(), output.resolve()
    dataset_path = layer1 / "dataset.json"
    source_dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if source_dataset["schema"] != "pnk-layer1-v1":
        raise ValueError("Expected PNK Layer-1 dataset")
    handoff = Path(source_dataset["source_handoff"])
    output.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    scene_records = []
    for layer1_manifest_path in sorted((layer1 / "scenes").glob("*/manifest.json")):
        manifest = json.loads(layer1_manifest_path.read_text(encoding="utf-8"))
        scene = manifest["scene"]
        scene_root = output / scene
        scene_root.mkdir(parents=True, exist_ok=True)
        ignore_path = scene_root / "gt" / "ignore.png"
        ignore_path.parent.mkdir(exist_ok=True)
        if not ignore_path.exists():
            success, encoded = cv2.imencode(".png", np.full((800, 500), 255, np.uint8))
            if not success:
                raise OSError(ignore_path)
            ignore_path.write_bytes(encoded.tobytes())
        frames = []
        for frame in manifest["frames"]:
            frame_index = int(frame["frame"])
            layer1_asset = layer1_manifest_path.parent / frame["layer1_panoptic"]
            with np.load(layer1_asset, allow_pickle=False) as panoptic:
                semantic19 = panoptic["semantic19"]
                instance_id = panoptic["instance_id"]
                segments_meta = json.loads(str(panoptic["segments_json"]))
            seg, seg_valid = materialize_seg2d(semantic19)
            # Reorder PNK camera slots into METEOR camera order.
            pnk_index = {name: index for index, name in enumerate(PNK_CAMERAS)}
            order = [pnk_index[PROVISIONAL_SLOT_MAP[name]] for name in METEOR_CAMERAS]
            seg, seg_valid = seg[order], seg_valid[order]
            bbox, bbox_counts = materialize_bbox_candidates(instance_id, segments_meta)
            bbox, bbox_counts = bbox[order], bbox_counts[order]
            seg_rel = Path("seg2d21") / f"{frame_index:06d}.npz"
            bbox_rel = Path("bbox2d_candidate") / f"{frame_index:06d}.npz"
            atomic_npz(scene_root / seg_rel, seg=seg, valid=seg_valid,
                       taxonomy=np.asarray("METEOR_21_partial_from_Cityscapes19"))
            atomic_npz(scene_root / bbox_rel, boxes=bbox, counts=bbox_counts,
                       image_annotation_complete=np.zeros(len(METEOR_CAMERAS), np.uint8),
                       provenance=np.asarray("external_panoptic_candidate_not_complete_GT"))
            source_scene = handoff / "scenes" / scene
            box_rel = Path("bev_box_p") / f"{frame_index:06d}.npz"
            lidar_rel = Path("lidar_bev") / f"{frame_index:06d}.npz"
            counts[f"box_{link_or_copy(source_scene / frame['bev_box_p'], scene_root / box_rel)}"] += 1
            counts[f"lidar_{link_or_copy(source_scene / frame['lidar_bev'], scene_root / lidar_rel)}"] += 1
            image_key = ("images_rectified" if "images_rectified" in frame
                         else "images_raw")
            images = {meteor: frame[image_key][PROVISIONAL_SLOT_MAP[meteor]]
                      for meteor in METEOR_CAMERAS}
            record = {
                "frame": frame_index, "timestamp_ns": frame["timestamp_ns"],
                "imgs": images, "gt_cons": "gt/ignore.png",
                "seg2d21": seg_rel.as_posix(),
                "bbox2d_candidate": bbox_rel.as_posix(),
                "bev_box_p": box_rel.as_posix(), "lidar_bev": lidar_rel.as_posix(),
                "layer1_panoptic": str(layer1_asset.resolve()),
                "supervision_valid": {
                    "bev_lane": False, "metric_depth": False,
                    "box3d": True, "unknown": False, "seg2d_partial": True,
                    "bbox2d_complete": False, "ego_motion": False,
                    "occupancy3d": False, "occupancy_flow": False,
                    "agent_forecast": False, "traffic_light_state": False,
                    "risk": False,
                },
            }
            frames.append(record)
            counts["frames"] += 1
            counts["seg2d_valid_pixels"] += int(seg_valid.sum())
            counts["bbox2d_candidates"] += int(bbox_counts.sum())
        nav_source = handoff / "scenes" / scene / manifest["nav_ego_candidate"]
        nav_dest = scene_root / "nav_ego_candidate.npz"
        counts[f"nav_{link_or_copy(nav_source, nav_dest)}"] += 1
        output_manifest = {
            "schema": "pnk-layer2-scene-v1", "scene": scene,
            "recording": manifest["recording"], "split": manifest["split"],
            "image_root": manifest.get("image_root", manifest["sensor_root"]),
            "img_hw": [432, 768],
            "cams": camera_contract(manifest), "frames": frames,
            "nav_ego_candidate": "nav_ego_candidate.npz",
            "training_contract": {
                "gt_key": "gt_cons", "boxdet_kmax": 320,
                "allowed_now": ["seg2d_partial", "box3d", "lidar_input"],
                "hold": ["bev_lane", "metric_depth", "bbox2d", "ego_motion",
                         "occupancy3d", "occupancy_flow", "agent_forecast",
                         "traffic_light_state", "unknown", "risk"],
                "hold_reason": "missing_complete_supervision_or_remaining_geometry_QA",
            },
        }
        atomic_json(scene_root / "manifest.json", output_manifest)
        scene_records.append({"scene": scene, "frames": len(frames),
                              "split": manifest["split"]})
        counts["scenes"] += 1
        print(f"[layer2] {scene}: {len(frames)} frames", flush=True)
    task_availability = {
        "1_bev_lane": "HOLD_missing_lane_semantics",
        "2_metric_depth": "HOLD_missing_geometry_verified_sparse_depth",
        "3_box3d": "CANDIDATE_available_geometry_QA_pending",
        "4_unknown": "HOLD_missing_unknown_semantics",
        "5_seg2d": "CANDIDATE_partial_Cityscapes_mapping_unmapped_pixels_ignored",
        "6_bbox2d": "HOLD_candidate_only_annotation_incomplete",
        "7_e2e": "HOLD_NAV_candidate_not_ego_GT",
        "8_occupancy3d": "HOLD_missing_deskew_pose_and_complete_semantics",
        "9_occupancy_flow": "HOLD_track_ID_QA",
        "10_agent_forecast": "HOLD_track_ID_QA",
        "11_traffic_light": "HOLD_missing_state_and_ego_association",
        "12_risk": "HOLD_missing_occupancy_and_agent_futures",
    }
    result = {
        "schema": "pnk-layer2-v1", "source_layer1": str(layer1),
        "source_layer1_dataset_sha256": sha256(dataset_path),
        "complete": source_dataset["complete"] and counts["frames"] == source_dataset["counts"]["frames"],
        "counts": dict(counts), "scenes": scene_records,
        "task_availability": task_availability,
        "trainer_gate": "HOLD_until_source_distortion_model_confirmation_projection_QA_and_recipe_selection",
        "safe_loader_recipe_after_gate": {
            "gt_key": "gt_cons", "with_seg2d": True, "seg2d_key": "seg2d21",
            "with_boxdet": True, "boxdet_kmax": 320, "with_lidarbev": True,
            "with_bbox2d": False, "with_ego": False, "with_occ": False,
            "with_agenttraj": False, "with_tl": False, "with_risk": False,
        },
    }
    atomic_json(output / "dataset.json", result)
    (output / "scenes.txt").write_text("\n".join(row["scene"] for row in scene_records) + "\n",
                                        encoding="utf-8")
    atomic_json(output / "task_availability.json", task_availability)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer1", type=Path, default=DEFAULT_LAYER1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = materialize(args.layer1, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
