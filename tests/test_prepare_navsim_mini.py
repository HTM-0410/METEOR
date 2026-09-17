import json
from pathlib import Path

from scripts.prepare_navsim_mini import (
    DATASET_ROOT,
    OUTPUT_ROOT,
    build_parser,
    cleanup_empty_gt,
    map_gt_already_promoted,
    preflight,
    verify_processed,
)


def test_parser_can_run_without_dataset_arguments():
    args = build_parser().parse_args([])

    assert args.data_root == DATASET_ROOT
    assert args.out == OUTPUT_ROOT
    assert args.out == args.data_root.parent / "meteor_mini"


def test_preflight_accepts_complete_raw_layout_without_creating_output(tmp_path: Path):
    data = tmp_path / "raw"
    logs = data / "navsim_logs" / "mini"
    sensors = data / "sensor_blobs" / "mini"
    maps = data / "maps"
    logs.mkdir(parents=True)
    sensors.mkdir(parents=True)
    maps.mkdir(parents=True)
    (logs / "log.pkl").write_bytes(b"test")
    output = tmp_path / "converted"

    result = preflight(data, output, "mini")

    assert result["pickle_logs"] == 1
    assert result["sensors"] == sensors.resolve()
    assert not output.exists()


def test_preflight_can_explicitly_run_without_maps(tmp_path: Path):
    data = tmp_path / "raw"
    logs = data / "navsim_logs" / "mini"
    sensors = data / "sensor_blobs" / "mini"
    logs.mkdir(parents=True)
    sensors.mkdir(parents=True)
    (logs / "log.pkl").write_bytes(b"test")

    result = preflight(data, tmp_path / "converted", "mini",
                       require_maps=False)

    assert result["pickle_logs"] == 1
    assert not (data / "maps").exists()


def test_cleanup_and_processed_reference_verification(tmp_path: Path):
    output = tmp_path / "converted"
    sensors = tmp_path / "raw" / "sensor_blobs" / "mini"
    scene = output / "scene-a"
    sensors.mkdir(parents=True)
    (scene / "gt").mkdir(parents=True)
    for directory in ("gt_map", "bev_box", "agent_traj"):
        (scene / directory).mkdir()
        (scene / directory / "000000.npz").write_bytes(b"target")
    # GT is a PNG path by contract; reference verification checks existence,
    # while the map stage itself validates image signature and dimensions.
    (scene / "gt_map" / "000000.npz").replace(
        scene / "gt_map" / "000000.png"
    )
    manifest = {
        "image_root": str(sensors.resolve()),
        "map_gt": {"frames_rasterized": 1,
                   "promoted_to_default_gt": True},
        "navsim_agent_traj": {"frames": 1},
        "frames": [{
            "frame": 0,
            "gt": "gt_map/000000.png",
            "gt_map": "gt_map/000000.png",
            "bev_box_p": "bev_box/000000.npz",
            "agent_traj": "agent_traj/000000.npz",
            "driving_command": [0, 1, 0, 0],
        }],
    }
    (scene / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert cleanup_empty_gt(output) == 1
    assert not (scene / "gt").exists()
    assert verify_processed(output, sensors) == {
        "scenes": 1, "frames": 1, "commands": 1, "missing": 0,
    }
    assert map_gt_already_promoted(output)
    manifest["map_gt"]["promoted_to_default_gt"] = False
    (scene / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert not map_gt_already_promoted(output)


def test_no_map_verification_keeps_ignore_target(tmp_path: Path):
    output = tmp_path / "converted"
    sensors = tmp_path / "raw" / "sensor_blobs" / "mini"
    scene = output / "scene-a"
    sensors.mkdir(parents=True)
    for directory in ("gt", "bev_box", "agent_traj"):
        (scene / directory).mkdir(parents=True, exist_ok=True)
    (scene / "gt" / "ignore.png").write_bytes(b"placeholder")
    (scene / "bev_box" / "000000.npz").write_bytes(b"box")
    (scene / "agent_traj" / "000000.npz").write_bytes(b"trajectory")
    manifest = {
        "image_root": str(sensors.resolve()),
        "segmentation_supervision": "ignore_255_no_maps",
        "navsim_agent_traj": {"frames": 1},
        "frames": [{
            "frame": 0, "gt": "gt/ignore.png",
            "bev_box_p": "bev_box/000000.npz",
            "agent_traj": "agent_traj/000000.npz",
            "driving_command": [0, 1, 0, 0],
        }],
    }
    (scene / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert verify_processed(output, sensors, require_map_gt=False)["frames"] == 1
    assert (scene / "gt" / "ignore.png").is_file()
