import json
import pickle
from pathlib import Path

import numpy as np

from bevlane.extract_navsim_agent_traj import SceneTrackIndex, _process_log


def _frame(index: int, scene_token: str = "scene-a"):
    # Ego advances 1 m/step; tracked vehicle advances 2 m/step globally.
    # Its local x is therefore 10 + index, while from the CURRENT frame at
    # index zero its future offsets must be 2, 4, ... metres.
    return {
        "token": f"token-{index}",
        "timestamp": index * 500_000,
        "scene_token": scene_token,
        "ego2global_translation": np.array([float(index), 0.0, 0.0]),
        "ego2global_rotation": np.array([1.0, 0.0, 0.0, 0.0]),
        "anns": {
            "gt_boxes": np.array(
                [[10.0 + index, 1.0, 0.0, 4.0, 2.0, 1.5, 0.0]],
                dtype=np.float64,
            ),
            "gt_names": np.array(["vehicle"]),
            "track_tokens": ["vehicle-track"],
            "gt_velocity_3d": np.array([[2.0, 0.0, 0.0]]),
        },
    }


def test_track_trajectory_is_expressed_in_current_ego_frame():
    index = SceneTrackIndex([_frame(i) for i in range(7)])

    boxes, count, trajectory, valid = index.target(0)

    assert int(count) == 1
    np.testing.assert_allclose(boxes[0], [1, 10, 1, 4, 2, 0])
    np.testing.assert_allclose(trajectory[0, :, 0], [2, 4, 6, 8, 10, 12])
    np.testing.assert_allclose(trajectory[0, :, 1], 0)
    np.testing.assert_array_equal(valid[0], 1)
    assert int(valid[1:].sum()) == 0


def test_scene_end_does_not_invent_future_tracks():
    index = SceneTrackIndex([_frame(i) for i in range(3)])
    _, _, _, valid = index.target(2)
    assert int(valid.sum()) == 0


def test_process_log_writes_contract_and_atomic_manifest_backup(tmp_path: Path):
    data_root = tmp_path / "data"
    output_root = tmp_path / "out"
    scene = "sample-log"
    log_dir = data_root / "navsim_logs" / "mini"
    scene_dir = output_root / scene
    log_dir.mkdir(parents=True)
    scene_dir.mkdir(parents=True)
    frames = [_frame(i) for i in range(7)]
    with (log_dir / f"{scene}.pkl").open("wb") as stream:
        pickle.dump(frames, stream)
    manifest = {
        "scene": scene,
        "frames": [
            {"frame": i, "token": frame["token"],
             "timestamp_us": frame["timestamp"]}
            for i, frame in enumerate(frames)
        ],
    }
    (scene_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = _process_log({
        "data_root": str(data_root), "output_root": str(output_root),
        "split": "mini", "scene": scene, "force": False,
        "time_tolerance": 0.15,
    })

    assert result["status"] == "ok"
    assert result["frames"] == 7
    assert (scene_dir / "manifest.before_agent_traj.json").is_file()
    converted = json.loads((scene_dir / "manifest.json").read_text(encoding="utf-8"))
    assert converted["navsim_agent_traj"]["coordinate_frame"] == "current_ego"
    target_path = scene_dir / converted["frames"][0]["agent_traj"]
    with np.load(target_path) as target:
        assert target["boxes"].shape == (64, 6)
        assert target["traj"].shape == (64, 6, 2)
        assert target["tvalid"].shape == (64, 6)
        np.testing.assert_allclose(target["traj"][0, :, 0],
                                   [2, 4, 6, 8, 10, 12])
