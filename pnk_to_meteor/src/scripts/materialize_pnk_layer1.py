#!/usr/bin/env python3
"""Materialize PNK Layer-1 generic label assets.

Layer 1 keeps source 3D boxes/NAV/LiDAR by provenance and adds per-camera
panoptic predictions.  The public runner is an external Cityscapes
Mask2Former because an executable original CoMET teacher is not available.
Outputs are resumable, zero-copy for source sensors, and never overwrite the
input handoff.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from scripts.prepare_pnk_comet import atomic_json, sha256


DEFAULT_OUTPUT = Path(r"D:\Backup_rosbag\PNKData_layer1_v1")
DEFAULT_INPUT = Path(r"D:\Backup_rosbag\PNKData_rectified_v1")
DEFAULT_MODEL = "facebook/mask2former-swin-small-cityscapes-panoptic"
DEFAULT_REVISION = "a87607429f7474fd1e2d1d55d6a4ce18a893a526"
PNK_CAMERAS = ("CAM_P_F", "CAM_P_FL", "CAM_P_FR", "CAM_P_L", "CAM_P_R",
               "CAM_P_B", "CAM_P_LB", "CAM_P_RB")


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


def validate_asset(path: Path, cameras: int, height: int, width: int) -> None:
    with np.load(path, allow_pickle=False) as value:
        expected = (cameras, height, width)
        semantic = value["semantic19"]
        segment = value["segment_id"]
        instance = value["instance_id"]
        if (semantic.shape != expected or segment.shape != expected
                or instance.shape != expected):
            raise ValueError(f"Wrong Layer-1 panoptic shape: {path}")
        if semantic.dtype != np.uint8:
            raise ValueError(f"Wrong semantic dtype: {path}")
        metadata = json.loads(str(value["segments_json"]))
        if len(metadata) != cameras:
            raise ValueError(f"Wrong camera metadata count: {path}")
        for camera_index, regions in enumerate(metadata):
            ids = [int(region["id"]) for region in regions]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate panoptic segment IDs: {path}")
            for region in regions:
                mask = segment[camera_index] == int(region["id"])
                if mask.any() and not np.all(
                        semantic[camera_index][mask] == int(region["label_id"])):
                    raise ValueError(f"Semantic/segment mismatch: {path}")


def infer_frame(model, processor, device: str, image_paths: list[Path],
                output_hw: tuple[int, int], batch_size: int) -> tuple[np.ndarray, np.ndarray,
                                                                      np.ndarray, list[list[dict]]]:
    height, width = output_hw
    semantics, segments, instances, metadata = [], [], [], []
    for start in range(0, len(image_paths), batch_size):
        chunk_paths = image_paths[start:start + batch_size]
        images = [Image.open(path).convert("RGB") for path in chunk_paths]
        inputs = processor(images=images, return_tensors="pt",
                           size={"shortest_edge": 512, "longest_edge": 1024})
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        results = processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[output_hw] * len(images), threshold=0.5,
            mask_threshold=0.5, overlap_mask_area_threshold=0.8,
            # transformers 5.7.0 can reuse a fused stuff ID for a later class,
            # corrupting semantic lookup. Unique IDs are safer; semantic classes
            # can still contain several disconnected stuff segments.
            label_ids_to_fuse=set())
        for result in results:
            segment_map = result["segmentation"].cpu().numpy().astype(np.uint16)
            semantic = np.full((height, width), 255, np.uint8)
            instance = np.zeros((height, width), np.uint16)
            records = []
            for region in result["segments_info"]:
                region_id, label_id = int(region["id"]), int(region["label_id"])
                mask = segment_map == region_id
                if not mask.any():
                    continue
                semantic[mask] = label_id
                isthing = label_id >= 11
                if isthing:
                    instance[mask] = region_id
                records.append({"id": region_id, "label_id": label_id,
                                "label": model.config.id2label[label_id],
                                "isthing": isthing, "pixels": int(mask.sum()),
                                "score": float(region.get("score", 0.0))})
            semantics.append(semantic)
            segments.append(segment_map)
            instances.append(instance)
            metadata.append(records)
    return (np.stack(semantics), np.stack(segments), np.stack(instances), metadata)


def materialize(input_root: Path, output: Path, model_id: str, revision: str,
                stride: int = 4, batch_size: int = 2,
                max_frames: int | None = None) -> dict:
    input_root, output = input_root.resolve(), output.resolve()
    dataset_path = input_root / "dataset.json"
    source_dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if source_dataset["schema"] not in ("pnk-comet-handoff-v1", "pnk-rectified-v1"):
        raise ValueError("Expected a verified PNK handoff or rectified PNK dataset")
    rectified_input = source_dataset["schema"] == "pnk-rectified-v1"
    handoff = Path(source_dataset.get("source_handoff", input_root)).resolve()
    if output == input_root or output.is_relative_to(input_root):
        raise ValueError("Layer-1 output must be separate from its input")
    output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id, revision=revision)
    expected = {0: "road", 10: "sky", 11: "person", 13: "car", 18: "bicycle"}
    if any(model.config.id2label[index] != name for index, name in expected.items()):
        raise ValueError("Layer-1 mapper requires the Cityscapes train-ID taxonomy")
    model = model.to(device).eval()
    sensor_root = (None if rectified_input
                   else Path(source_dataset["source_sensor_root"]))
    counts: Counter = Counter()
    selected_frames = 0
    source_h, source_w = (source_dataset.get("target_hw", [1536, 1920])
                          if rectified_input else [1536, 1920])
    output_h, output_w = int(source_h) // stride, int(source_w) // stride
    scene_records = []
    stop = False
    for source_manifest_path in sorted((input_root / "scenes").glob("*/manifest.json")):
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        scene = source_manifest["scene"]
        scene_root = output / "scenes" / scene
        output_frames = []
        for frame in source_manifest["frames"]:
            if max_frames is not None and selected_frames >= max_frames:
                stop = True
                break
            relative = Path("panoptic") / f'{int(frame["frame"]):06d}.npz'
            destination = scene_root / relative
            if destination.exists():
                try:
                    validate_asset(destination, len(PNK_CAMERAS), output_h, output_w)
                    counts["panoptic_reused"] += 1
                except ValueError:
                    destination.unlink()
                    counts["panoptic_invalid_rebuilt"] += 1
            if not destination.exists():
                if rectified_input:
                    image_root = Path(source_manifest["image_root"])
                    image_paths = [(image_root / frame["images_rectified"][camera]).resolve()
                                   for camera in PNK_CAMERAS]
                else:
                    image_paths = [(sensor_root / frame["images_raw"][camera]).resolve()
                                   for camera in PNK_CAMERAS]
                if not all(path.is_file() for path in image_paths):
                    raise FileNotFoundError(next(path for path in image_paths if not path.is_file()))
                semantic, segment, instance, segments_meta = infer_frame(
                    model, processor, device, image_paths, (output_h, output_w), batch_size)
                atomic_npz(destination, semantic19=semantic,
                           segment_id=segment, instance_id=instance,
                           segments_json=np.asarray(json.dumps(segments_meta,
                                                               separators=(",", ":"))))
                counts["panoptic_created"] += 1
                counts["segments"] += sum(len(value) for value in segments_meta)
            record = dict(frame)
            record["layer1_panoptic"] = relative.as_posix()
            record["layer1_panoptic_valid"] = True
            output_frames.append(record)
            selected_frames += 1
            counts["frames"] += 1
            counts["camera_images"] += len(PNK_CAMERAS)
            if counts["frames"] % 25 == 0:
                print(f'[layer1] {counts["frames"]} frames', flush=True)
        if output_frames:
            scene_doc = dict(source_manifest)
            scene_doc["schema"] = "pnk-layer1-scene-v1"
            scene_doc["source_input_manifest"] = str(source_manifest_path.resolve())
            scene_doc["frames"] = output_frames
            scene_doc["layer1_panoptic"] = {
                "model": model_id, "revision": revision,
                "source": "external_teacher_not_original_comet",
                "image_geometry": ("rectified_pinhole_assumed_source_model"
                                   if rectified_input else "raw_distorted"),
                "taxonomy": {str(key): value for key, value in model.config.id2label.items()},
                "camera_order": list(PNK_CAMERAS),
                "shape": [len(PNK_CAMERAS), output_h, output_w],
                "source_image_hw": [int(source_h), int(source_w)], "mask_stride": stride,
                "void_id": 255, "instance_scope": "per_image",
                "missing_classes_for_meteor": ["lane_marking", "stopline", "crosswalk",
                                                "cone", "road_debris", "unknown"],
            }
            atomic_json(scene_root / "manifest.json", scene_doc)
            scene_records.append({"scene": scene, "frames": len(output_frames),
                                  "split": source_manifest["split"]})
            counts["scenes"] += 1
        if stop:
            break
    result = {
        "schema": "pnk-layer1-v1", "source_input": str(input_root),
        "source_input_dataset_sha256": sha256(dataset_path),
        "source_handoff": str(handoff),
        "complete": max_frames is None and counts["frames"] == source_dataset["counts"]["frames"],
        "counts": dict(counts), "model": model_id, "revision": revision,
        "model_provenance": "external_teacher_not_original_comet",
        "device": device, "camera_order": list(PNK_CAMERAS),
        "panoptic_shape": [len(PNK_CAMERAS), output_h, output_w],
        "image_geometry": ("rectified_pinhole_assumed_source_model"
                           if rectified_input else "raw_distorted"),
        "scenes": scene_records,
        "known_limits": ["Cityscapes taxonomy lacks METEOR lane/stopline/crosswalk/unknown classes",
                         ("source distortion model confirmation and projection overlay QA remain pending"
                          if rectified_input else
                          "camera distortion model and projection geometry remain pending QA"),
                         "panoptic predictions are pseudo-labels without PNK pixel ground truth"],
    }
    atomic_json(output / "dataset.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--handoff", dest="input_root", type=Path,
                        default=DEFAULT_INPUT,
                        help="Rectified PNK dataset (preferred); raw handoff remains accepted for diagnostics")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--mask-stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if args.mask_stride < 1 or args.batch_size < 1:
        parser.error("mask stride and batch size must be positive")
    result = materialize(args.input_root, args.output, args.model, args.revision,
                         args.mask_stride, args.batch_size, args.max_frames)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
