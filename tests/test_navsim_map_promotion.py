import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bevlane.promote_navsim_map_gt import promote


def _scene(root: Path, name: str, complete: bool = True) -> Path:
    scene = root / name
    (scene / "gt").mkdir(parents=True)
    (scene / "gt_map").mkdir()
    assert cv2.imwrite(str(scene / "gt" / "ignore.png"),
                       np.full((800, 500), 255, dtype=np.uint8))
    if complete:
        assert cv2.imwrite(str(scene / "gt_map" / "000000.png"),
                           np.full((800, 500), 1, dtype=np.uint8))
    manifest = {
        "source": "NAVSIM/OpenScene",
        "map_gt": {"frames_rasterized": 1},
        "frames": [{
            "frame": 0,
            "gt": "gt/ignore.png",
            "gt_map": "gt_map/000000.png",
        }],
    }
    (scene / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return scene


def test_promote_backs_up_manifest_and_removes_only_placeholder(tmp_path: Path):
    scene = _scene(tmp_path, "scene-a")

    assert promote(tmp_path) == {"scenes": 1, "frames": 1, "placeholders": 1}
    assert (scene / "gt" / "ignore.png").is_file()  # dry-run made no change

    assert promote(tmp_path, apply=True) == {"scenes": 1, "frames": 1, "placeholders": 1}
    new = json.loads((scene / "manifest.json").read_text(encoding="utf-8"))
    old = json.loads((scene / "manifest.before_gt_promotion.json").read_text(encoding="utf-8"))
    assert new["frames"][0]["gt"] == "gt_map/000000.png"
    assert new["frames"][0]["gt_map"] == "gt_map/000000.png"
    assert new["map_gt"]["promoted_to_default_gt"] is True
    assert old["frames"][0]["gt"] == "gt/ignore.png"
    assert (scene / "gt_map" / "000000.png").is_file()
    assert not (scene / "gt" / "ignore.png").exists()

    # The operation is safe to rerun, and keeps the original recovery point.
    assert promote(tmp_path, apply=True)["placeholders"] == 0
    old_again = json.loads((scene / "manifest.before_gt_promotion.json").read_text(encoding="utf-8"))
    assert old_again == old


def test_incomplete_map_blocks_all_scene_mutation(tmp_path: Path):
    first = _scene(tmp_path, "scene-a")
    second = _scene(tmp_path, "scene-b", complete=False)

    with pytest.raises(FileNotFoundError):
        promote(tmp_path, apply=True)

    assert (first / "gt" / "ignore.png").is_file()
    assert (second / "gt" / "ignore.png").is_file()
    assert not (first / "manifest.before_gt_promotion.json").exists()
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["frames"][0]["gt"] == "gt/ignore.png"
