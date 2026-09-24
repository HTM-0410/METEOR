#!/usr/bin/env python3
"""Run the PNK raw-data to METEOR nine-head sample pipeline."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
STAGES = (
    "clean", "handoff", "box3d", "lidar_bev", "verify_handoff",
    "rectified", "verify_rectified", "layer1", "verify_layer1",
    "layer2", "depth", "e2e", "agent_trajectory", "verify_targets",
    "flow", "occupancy", "risk", "drivable", "verify_derived",
    "verify_layer2", "profile_v2", "profile_v3", "profile_v4",
    "profile_v5", "profile_v7", "verify_final",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def configuration(path: Path) -> dict:
    cfg = read_json(path)
    required = {"source_root", "work_root", "panoptic_model", "panoptic_revision",
                "panoptic_batch_size", "target_workers"}
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f"Missing configuration fields: {sorted(missing)}")
    cfg["source_root"] = str(Path(cfg["source_root"]).expanduser().resolve())
    cfg["work_root"] = str(Path(cfg["work_root"]).expanduser().resolve())
    if not 1 <= int(cfg["panoptic_batch_size"]) <= 64:
        raise ValueError("panoptic_batch_size must be 1..64")
    if int(cfg["target_workers"]) < 1:
        raise ValueError("target_workers must be positive")
    return cfg


def paths(cfg: dict) -> dict[str, Path]:
    base = Path(cfg["work_root"])
    return {
        "clean": base / "PNKData_clean_v1",
        "handoff": base / "PNKData_comet_handoff_v2",
        "rectified": base / "PNKData_rectified_v1",
        "layer1": base / "PNKData_layer1_v1",
        "layer2": base / "PNKData_layer2_samples_v1",
        "v2": base / "PNKData_layer2_9head_v2",
        "v3": base / "PNKData_layer2_9head_v3",
        "v4": base / "PNKData_layer2_9head_v4",
        "v5": base / "PNKData_layer2_9head_v5",
        "v7": base / "PNKData_layer2_9head_v7",
    }


def commands(cfg: dict, p: dict[str, Path]) -> dict[str, list[str]]:
    worker = str(cfg["target_workers"])
    m = lambda name, *args: ["-m", f"scripts.{name}", *map(str, args)]
    return {
        "clean": m("prepare_pnk_clean", "--source-root", cfg["source_root"], "--output", p["clean"]),
        "handoff": m("prepare_pnk_comet", "--clean", p["clean"], "--output", p["handoff"]),
        "box3d": m("materialize_pnk_boxdet", "--root", p["handoff"]),
        "lidar_bev": m("materialize_pnk_lidarbev", "--root", p["handoff"]),
        "verify_handoff": m("verify_pnk_comet", "--root", p["handoff"]),
        "rectified": m("materialize_pnk_rectified", "--handoff", p["handoff"], "--output", p["rectified"], "--alpha", cfg.get("rectification_alpha", 0), "--jpeg-quality", cfg.get("jpeg_quality", 95)),
        "verify_rectified": m("verify_pnk_rectified", "--root", p["rectified"], "--workers", worker),
        "layer1": m("materialize_pnk_layer1", "--input", p["rectified"], "--output", p["layer1"], "--model", cfg["panoptic_model"], "--revision", cfg["panoptic_revision"], "--batch-size", cfg["panoptic_batch_size"]),
        "verify_layer1": m("verify_pnk_layer1", "--root", p["layer1"]),
        "layer2": m("materialize_pnk_layer2", "--layer1", p["layer1"], "--output", p["layer2"]),
        "depth": m("materialize_pnk_depth", "--root", p["layer2"], "--handoff", p["handoff"], "--workers", worker),
        "e2e": m("materialize_pnk_e2e", "--root", p["layer2"]),
        "agent_trajectory": m("materialize_pnk_agent_traj", "--root", p["layer2"], "--handoff", p["handoff"]),
        "verify_targets": m("verify_pnk_task_targets", "--root", p["layer2"]),
        "flow": m("materialize_pnk_flow", "--root", p["layer2"]),
        "occupancy": m("materialize_pnk_occupancy", "--root", p["layer2"], "--handoff", p["handoff"], "--workers", worker),
        "risk": m("materialize_pnk_risk", "--root", p["layer2"]),
        "drivable": m("materialize_pnk_drivable", "--root", p["layer2"]),
        "verify_derived": m("verify_pnk_derived_targets", "--root", p["layer2"]),
        "verify_layer2": m("verify_pnk_layer_pipeline", "--layer1", p["layer1"], "--layer2", p["layer2"]),
        "profile_v2": m("build_pnk_9head_profile", "--source", p["layer2"], "--handoff", p["handoff"], "--output", p["v2"]),
        "profile_v3": m("improve_pnk_head1_head2", "--source", p["v2"], "--output", p["v3"], "--workers", worker),
        "profile_v4": m("materialize_pnk_bev_lane_full", "--source", p["v3"], "--base", p["v2"], "--handoff", p["handoff"], "--output", p["v4"], "--radius", cfg.get("lane_radius", 8), "--visual-every", cfg.get("lane_visual_every", 25)),
        "profile_v5": m("materialize_pnk_route_command", "--source", p["v4"], "--output", p["v5"]),
        "profile_v7": m("materialize_pnk_occupancy_multisweep", "--source", p["v5"], "--output", p["v7"], "--radius", cfg.get("occupancy_radius", 3), "--workers", worker),
        "verify_final": m("verify_pnk_9head_profile", "--root", p["v7"]),
    }


def preflight(cfg: dict) -> None:
    source = Path(cfg["source_root"])
    for relative in (
        "PNKData", "PNKData_meta/label/2026_08_14_vf6_01_02",
        "PNKData_meta/file_csv/2026_08_14_vf6_01_02.csv",
        "PNKData_meta/Calib_2/VF6_01_Intrinsics.json",
        "PNKData_meta/Calib_2/VF6_01_Extrinsics_By_Dates.json",
    ):
        if not (source / relative).exists():
            raise FileNotFoundError(source / relative)
    for module in ("cv2", "numpy", "pandas", "PIL", "scipy", "laspy", "lazrs", "torch", "torchvision", "transformers"):
        __import__(module)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON file with source_root and work_root")
    parser.add_argument("--dry-run", action="store_true", help="Print planned stages; no import, data read or write")
    parser.add_argument("--check", action="store_true", help="Check raw layout and Python dependencies without building")
    parser.add_argument("--verify-existing", type=Path, help="Verify a completed v7 profile")
    args = parser.parse_args()
    if args.verify_existing:
        profile = args.verify_existing.expanduser().resolve()
        if not (profile / "dataset.json").is_file():
            parser.error(f"Missing profile: {profile}")
        dataset = read_json(profile / "dataset.json")
        if dataset.get("schema") != "pnk-layer2-9head-v7" or not dataset.get("complete"):
            parser.error(f"Expected complete v7 profile: {profile}")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run([sys.executable, "-m", "scripts.verify_pnk_9head_profile", "--root", str(profile)], cwd=SRC, env=env, check=True)
        return
    if not args.config:
        parser.error("--config is required for a build")
    cfg = configuration(args.config)
    p = paths(cfg)
    plans = commands(cfg, p)
    if args.dry_run:
        for number, name in enumerate(STAGES, 1):
            print(f"{number:02d} {name}: {sys.executable} {' '.join(plans[name])}")
        print(f"Final output: {p['v7']}")
        return
    preflight(cfg)
    if args.check:
        print(json.dumps({"status": "PREFLIGHT_OK", "source_root": cfg["source_root"],
                          "work_root": cfg["work_root"], "final_profile": str(p["v7"])}, indent=2))
        return
    base = Path(cfg["work_root"])
    base.mkdir(parents=True, exist_ok=True)
    state_path = base / "pnk_pipeline_state.json"
    state = read_json(state_path) if state_path.exists() else {"config": cfg, "completed": []}
    if state["config"] != cfg:
        raise ValueError(f"Configuration differs from {state_path}. Use a new work_root or restore the original config.")
    completed = state["completed"]
    if completed != list(STAGES[:len(completed)]):
        raise ValueError(f"Invalid stage order in {state_path}; refusing to skip stages.")
    required_outputs = {"clean": p["clean"] / "dataset.json",
                        "handoff": p["handoff"] / "dataset.json",
                        "rectified": p["rectified"] / "dataset.json",
                        "layer1": p["layer1"] / "dataset.json",
                        "layer2": p["layer2"] / "dataset.json",
                        "profile_v2": p["v2"] / "dataset.json",
                        "profile_v3": p["v3"] / "dataset.json",
                        "profile_v4": p["v4"] / "dataset.json",
                        "profile_v5": p["v5"] / "dataset.json",
                        "profile_v7": p["v7"] / "dataset.json"}
    for stage in completed:
        marker = required_outputs.get(stage)
        if marker and not marker.is_file():
            raise FileNotFoundError(f"Checkpoint lists {stage} but output is missing: {marker}")
    if not state["completed"] and any(path.exists() for path in p.values()):
        raise ValueError("work_root contains pipeline outputs without a checkpoint. Choose an empty work_root to avoid overwriting data.")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    for number, name in enumerate(STAGES, 1):
        if name in state["completed"]:
            print(f"[{number:02d}/{len(STAGES)}] {name}: checkpoint OK", flush=True)
            continue
        print(f"[{number:02d}/{len(STAGES)}] {name}: running", flush=True)
        subprocess.run([sys.executable, *plans[name]], cwd=SRC, env=env, check=True)
        state["completed"].append(name)
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, state)
    result = read_json(p["v7"] / "nine_head_verification.json")
    print(json.dumps({"final_profile": str(p["v7"]), "status": result["status"], "counts": result["counts"]}, indent=2))


if __name__ == "__main__":
    main()
