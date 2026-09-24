#!/usr/bin/env python3
"""Render Layer-2 PNK targets for direct human quality inspection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ROOT_DEFAULT = Path(r"D:\Backup_rosbag\PNKData_layer2_samples_v1")
CAM = "CAM_FRONT_WIDE"
HORIZONS = np.asarray([.5, 1., 1.5, 2., 2.5, 3.], np.float32)

SEG_COLORS = np.asarray([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32], [110, 110, 110],
    [255, 255, 255]], np.uint8)
OCC_COLORS = np.asarray([
    [35, 35, 35], [0, 0, 255], [255, 0, 255], [255, 128, 0],
    [160, 32, 240], [60, 180, 75], [255, 225, 25], [0, 128, 0],
    [128, 128, 128], [0, 255, 255]], np.uint8)


def imread(path: Path, flags=cv2.IMREAD_COLOR):
    raw = np.fromfile(str(path), np.uint8)
    return cv2.imdecode(raw, flags) if raw.size else None


def imwrite(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, image,
                            [cv2.IMWRITE_JPEG_QUALITY, 95] if path.suffix.lower() == ".jpg" else [])
    if not ok:
        raise OSError(path)
    data.tofile(str(path))


def read_manifest(root: Path, scene: str):
    return json.loads((root / scene / "manifest.json").read_text(encoding="utf-8"))


def image_path(root: Path, scene: str, manifest: dict, rel: str):
    base = manifest.get("image_root")
    if base:
        base = Path(base)
        if not base.is_absolute():
            base = (root / scene / base).resolve()
    else:
        base = root / scene
    return base / rel


def sample_candidates(root: Path, stride: int):
    rows = []
    for scene_dir in sorted(p for p in root.iterdir() if (p / "manifest.json").is_file()):
        manifest = read_manifest(root, scene_dir.name)
        try:
            risk_all = np.load(scene_dir / "risk_map.npz", mmap_mode="r")["risk"]
        except Exception:
            risk_all = None
        try:
            ego_valid = np.load(scene_dir / "ego_motion.npz", mmap_mode="r")["valid"]
        except Exception:
            ego_valid = np.zeros(len(manifest["frames"]), np.uint8)
        frames = manifest["frames"]
        indices = sorted(set(range(0, len(frames), max(stride, 1))) | {len(frames) // 2})
        for pos in indices:
            f = frames[pos]
            fi = int(f["frame"])
            with np.load(scene_dir / f["agent_traj"], allow_pickle=False) as z:
                nbox = int(z["count"])
                tv = z["tvalid"][:nbox]
                future3 = int(np.sum(tv[:, -1] > .5)) if nbox else 0
                valid_future = int(np.sum(tv > .5)) if nbox else 0
            with np.load(scene_dir / f["flow_target"], allow_pickle=False) as z:
                fv = z["valid"].astype(bool)
                flow_cells = int(fv.sum())
                flow_speed = (float(np.linalg.norm(z["flow"][:, fv], axis=0).mean())
                              if flow_cells else 0.0)
            gt = imread(scene_dir / f["gt_map"], cv2.IMREAD_GRAYSCALE)
            drivable = int(np.sum(gt != 255))
            risk_high = (int(np.sum(risk_all[fi] >= 128))
                         if risk_all is not None and fi < len(risk_all) else 0)
            rows.append({
                "scene": scene_dir.name, "frame": fi, "manifest_pos": pos,
                "boxes": nbox, "future_3s_agents": future3,
                "valid_future_points": valid_future, "flow_cells": flow_cells,
                "flow_mean_speed_mps": flow_speed, "drivable_pixels": drivable,
                "high_risk_pixels": risk_high,
                "depth_valid_pixels": int(f.get("depth_valid_pixels", 0)),
                "e2e_valid": bool(fi < len(ego_valid) and ego_valid[fi] > .5),
            })
    return rows


def choose_samples(rows, count: int):
    criteria = [
        ("dense_agents", lambda r: r["boxes"] + 2 * r["future_3s_agents"]),
        ("high_motion", lambda r: r["flow_cells"] * r["flow_mean_speed_mps"]),
        ("high_risk", lambda r: r["high_risk_pixels"]),
        ("best_drivable_coverage", lambda r: r["drivable_pixels"]),
        ("low_drivable_coverage", lambda r: -r["drivable_pixels"]),
    ]
    chosen, used_scenes, used_keys = [], set(), set()
    for reason, score in criteria:
        ranked = sorted(rows, key=score, reverse=True)
        pick = next((r for r in ranked
                     if r["scene"] not in used_scenes and
                     (r["scene"], r["frame"]) not in used_keys), None)
        if pick is None:
            pick = next(r for r in ranked if (r["scene"], r["frame"]) not in used_keys)
        out = dict(pick); out["selection_reason"] = reason
        chosen.append(out); used_scenes.add(pick["scene"]); used_keys.add((pick["scene"], pick["frame"]))
        if len(chosen) >= count:
            return chosen
    return chosen


def box_corners(box):
    _, x, y, length, width, yaw = box
    local = np.asarray([[length/2, width/2], [length/2, -width/2],
                        [-length/2, -width/2], [-length/2, width/2]], np.float32)
    c, s = np.cos(yaw), np.sin(yaw)
    return local @ np.asarray([[c, s], [-s, c]], np.float32) + [x, y]


def bev_point(xy, res=.2, xhalf=80., yhalf=50.):
    return int((yhalf - float(xy[1])) / res), int((xhalf - float(xy[0])) / res)


def draw_grid(canvas, res=.2, xhalf=80., yhalf=50.):
    for x in range(-int(xhalf), int(xhalf) + 1, 10):
        y = bev_point((x, 0), res, xhalf, yhalf)[1]
        if 0 <= y < canvas.shape[0]: cv2.line(canvas, (0, y), (canvas.shape[1]-1, y), (45,45,45), 1)
    for yy in range(-int(yhalf), int(yhalf) + 1, 10):
        x = bev_point((0, yy), res, xhalf, yhalf)[0]
        if 0 <= x < canvas.shape[1]: cv2.line(canvas, (x, 0), (x, canvas.shape[0]-1), (45,45,45), 1)
    ex, ey = bev_point((0, 0), res, xhalf, yhalf)
    cv2.circle(canvas, (ex, ey), 5, (255, 255, 255), -1)


def title(panel, text, sub=""):
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 42 if sub else 27), (0, 0, 0), -1)
    cv2.putText(panel, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .53, (255,255,255), 1, cv2.LINE_AA)
    if sub: cv2.putText(panel, sub, (8, 37), cv2.FONT_HERSHEY_SIMPLEX, .38, (220,220,220), 1, cv2.LINE_AA)
    return panel


def fit(image, size=(420, 280), interpolation=cv2.INTER_NEAREST):
    return cv2.resize(image, size, interpolation=interpolation)


def render_sample(root: Path, row: dict, destination: Path):
    scene, fi = row["scene"], row["frame"]
    scene_dir = root / scene
    manifest = read_manifest(root, scene)
    f = next(x for x in manifest["frames"] if int(x["frame"]) == fi)
    raw = imread(image_path(root, scene, manifest, f["imgs"][CAM]))
    with np.load(scene_dir / f["seg2d21"], allow_pickle=False) as z: seg = z["seg"][0]
    with np.load(scene_dir / f["depth4"], allow_pickle=False) as z:
        depth = z["depth"][0].astype(np.float32)
        depth_conf = (z["confidence"][0].astype(np.uint8)
                      if "confidence" in z.files
                      else (depth > .5).astype(np.uint8))
    with np.load(scene_dir / f["bev_box_p"], allow_pickle=False) as z:
        boxes = z["boxes"].astype(np.float32)
        box_points = (z["point_count"].astype(np.int32)
                      if "point_count" in z.files else np.zeros(len(boxes), np.int32))
    with np.load(scene_dir / f["agent_traj"], allow_pickle=False) as z:
        nbox = int(z["count"]); abox = z["boxes"][:nbox]; traj = z["traj"][:nbox]; tv = z["tvalid"][:nbox]
    with np.load(scene_dir / f["flow_target"], allow_pickle=False) as z:
        flow = z["flow"].astype(np.float32); flow_valid = z["valid"].astype(bool)
    with np.load(scene_dir / f["occ"], allow_pickle=False) as z: occ = z["occ"]
    risk = np.load(scene_dir / "risk_map.npz", allow_pickle=False)["risk"][fi].astype(np.float32) / 255.
    egoz = np.load(scene_dir / "ego_motion.npz", allow_pickle=False)
    ego_wp, ego_valid = egoz["wp"][fi], bool(egoz["valid"][fi] > .5)
    gt = imread(scene_dir / f["gt_map"], cv2.IMREAD_GRAYSCALE)

    # Front semantic overlay.
    seg_up = cv2.resize(seg, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)
    seg_color = SEG_COLORS[np.clip(seg_up, 0, 20)][:, :, ::-1]
    overlay = raw.copy(); valid_seg = seg_up != 255
    overlay[valid_seg] = (0.45 * raw[valid_seg] + 0.55 * seg_color[valid_seg]).astype(np.uint8)

    # Sparse metric-depth overlay.
    dep_up = cv2.resize(depth, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)
    dep_color = cv2.applyColorMap((np.clip(dep_up / 80., 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    dep_overlay = (raw.astype(np.float32) * .35).astype(np.uint8)
    vd = dep_up > .5; dep_overlay[vd] = dep_color[vd]
    guided_up = cv2.resize((depth_conf == 2).astype(np.uint8),
                           (raw.shape[1], raw.shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
    dep_overlay[guided_up] = (
        0.72 * dep_color[guided_up] + 0.28 * raw[guided_up]).astype(np.uint8)

    # BEV 3D boxes.
    box_img = np.zeros((800, 500, 3), np.uint8); draw_grid(box_img)
    box_in_bev = (np.abs(boxes[:, 1]) <= 80) & (np.abs(boxes[:, 2]) <= 50)
    box_supported = box_in_bev & (box_points >= 5)
    box_source_total = int(f.get("box3d_source_count", len(boxes)))
    for box, is_in_bev, is_supported in zip(boxes, box_in_bev, box_supported):
        if not is_in_bev:
            continue
        pts = np.asarray([bev_point(p) for p in box_corners(box)], np.int32)
        color = ((0, 180, 255) if box[0] < 1.5 else (255, 0, 255)) \
            if is_supported else (100, 100, 100)
        cv2.polylines(box_img, [pts], True, color, 1, cv2.LINE_AA)

    # METEOR BEV lane map (v4 adds classes 3-8; older profiles have 1/2 only).
    drv = np.zeros((*gt.shape, 3), np.uint8)
    lane_colors = {
        1: (60, 180, 75), 2: (25, 225, 255), 3: (210, 80, 180),
        4: (255, 255, 0), 5: (30, 30, 230), 6: (0, 150, 255),
        7: (0, 230, 255), 8: (180, 100, 180),
    }
    for class_id, color in lane_colors.items():
        drv[gt == class_id] = color
    drive_conf = np.ones(gt.shape, np.uint8)
    if f.get("gt_map_support"):
        with np.load(scene_dir / f["gt_map_support"], allow_pickle=False) as z:
            drive_conf = z["confidence"].astype(np.uint8)
        added = drive_conf == 2
        drv[added & (gt == 1)] = (120, 230, 135)
        drv[added & (gt == 2)] = (120, 245, 255)
    draw_grid(drv)

    # Collapse semantic occupancy: prefer occupied endpoint classes over free.
    observed = (occ != 255).any(0)
    occ2 = np.zeros(occ.shape[1:], np.uint8)
    occ2[observed] = np.where(occ == 255, 0, occ).max(0)[observed]
    occ_img = np.ascontiguousarray(OCC_COLORS[occ2][:, :, ::-1]); occ_img[~observed] = 0
    draw_grid(occ_img, .4, 40., 40.)

    # Flow target.
    speed = np.linalg.norm(flow, axis=0)
    flow_img = cv2.applyColorMap((np.clip(speed / 12., 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    flow_img[~flow_valid] = 0; draw_grid(flow_img, .4, 40., 40.)
    for rr in range(5, 200, 10):
        for cc in range(5, 200, 10):
            if not flow_valid[rr, cc]: continue
            dx, dy = flow[:, rr, cc]
            cv2.arrowedLine(flow_img, (cc, rr),
                            (int(cc - dy * 1.3), int(rr - dx * 1.3)),
                            (255,255,255), 1, tipLength=.3)

    # Agent futures and ego waypoints in METEOR's 0.4 m grid.
    tr_img = np.zeros((400, 250, 3), np.uint8); draw_grid(tr_img, .4, 80., 50.)
    for i in range(nbox):
        if abox[i, 3] <= 0 or not (tv[i] > .5).any(): continue
        current = abox[i, 1:3]
        points = [bev_point(current, .4, 80., 50.)]
        points += [bev_point(current + traj[i, h], .4, 80., 50.)
                   for h in range(6) if tv[i, h] > .5]
        color = (0, 200, 255) if abox[i, 0] < 1.5 else (255, 0, 255)
        cv2.polylines(tr_img, [np.asarray(points, np.int32)], False, color, 1, cv2.LINE_AA)
    if ego_valid:
        ep = np.asarray([bev_point(p, .4, 80., 50.) for p in ego_wp.reshape(6,2)], np.int32)
        cv2.polylines(tr_img, [ep], False, (0,255,0), 3, cv2.LINE_AA)

    risk_img = cv2.applyColorMap((risk * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    draw_grid(risk_img, .4, 80., 50.)
    if ego_valid:
        ep = np.asarray([bev_point(p, .4, 80., 50.) for p in ego_wp.reshape(6,2)], np.int32)
        cv2.polylines(risk_img, [ep], False, (0,255,0), 2, cv2.LINE_AA)

    observed_ratio = float(np.mean(occ != 255))
    panels = [
        title(fit(raw, interpolation=cv2.INTER_AREA), "Raw CAM_FRONT_WIDE",
              f"{scene}  frame {fi}  selected: {row['selection_reason']}"),
        title(fit(overlay, interpolation=cv2.INTER_AREA), "Head 5: partial 2D semantic GT",
              f"valid {np.mean(seg != 255)*100:.1f}% of front pixels"),
        title(fit(dep_overlay, interpolation=cv2.INTER_AREA), "Head 2: metric-depth GT",
              f"measured {np.sum(depth_conf==1):,}; guided {np.sum(depth_conf==2):,}; 0-80 m"),
        title(fit(box_img), "Head 3: source 3D box candidates",
              f"source {box_source_total}; profile {len(boxes)}; supported {box_supported.sum()}; gray=weak"),
        title(fit(drv), "Head 1: METEOR BEV lane GT",
              f"valid {np.mean(gt != 255)*100:.1f}%; classes {','.join(map(str, sorted(set(np.unique(gt))-set([255]))))}"),
        title(fit(occ_img), "Head 8: semantic occupancy GT",
              f"single sweep; observed voxels {observed_ratio*100:.1f}%"),
        title(fit(flow_img), "Head 9: occupancy-flow GT",
              f"valid {flow_valid.sum():,} cells; mean {speed[flow_valid].mean() if flow_valid.any() else 0:.2f} m/s"),
        title(fit(tr_img), "Heads 7/10: ego + agent trajectory GT",
              f"agents {nbox}; 3s-valid {np.sum(tv[:,-1]>.5)}; ego-valid {ego_valid}"),
        title(fit(risk_img), "Head 12: derived risk GT",
              f"pixels >=0.5: {np.sum(risk >= .5):,}; green ego path"),
    ]
    canvas = np.vstack([np.hstack(panels[i:i+3]) for i in range(0, 9, 3)])
    imwrite(destination, canvas)
    details = dict(row)
    details.update({
        "seg_front_valid_ratio": float(np.mean(seg != 255)),
        "depth_front_valid_pixels": int(np.sum(depth > .5)),
        "depth_front_measured_pixels": int(np.sum(depth_conf == 1)),
        "depth_front_guided_pixels": int(np.sum(depth_conf == 2)),
        "drivable_temporal_added_pixels": int(np.sum(drive_conf == 2)),
        "box_source_total": box_source_total, "box_profile_total": int(len(boxes)),
        "box_within_bev": int(box_in_bev.sum()),
        "box_supported_ge5_points": int(box_supported.sum()),
        "occupancy_observed_ratio": observed_ratio,
        "risk_mean": float(risk.mean()), "risk_p99": float(np.percentile(risk, 99)),
        "output": str(destination.resolve()),
    })
    return details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--scan-stride", type=int, default=4)
    ap.add_argument("--output-dir", type=Path, default=Path("out/pnk_layer2_gt_qa"))
    args = ap.parse_args()
    rows = sample_candidates(args.root, args.scan_stride)
    selected = choose_samples(rows, args.count)
    outputs = []
    for i, row in enumerate(selected, 1):
        out = args.output_dir / f"sample_{i:02d}_{row['selection_reason']}_{row['scene']}_f{row['frame']:06d}.jpg"
        outputs.append(render_sample(args.root, row, out))
        print(f"[{i}/{len(selected)}] {out}", flush=True)
    report = {
        "status": "LAYER2_GT_SAMPLE_QA_RENDERED",
        "dataset": str(args.root.resolve()),
        "selection": "content-stratified across distinct scenes; split labels unused",
        "candidate_scan_stride": args.scan_stride,
        "samples": outputs,
        "limitations": [
            "metric depth is sparse LiDAR projection and deskew is unverified",
            "2D semantic labels are partial external-teacher labels",
            "drivable GT contains only road and sidewalk",
            "occupancy is single-sweep and unobserved voxels remain ignore=255",
            "risk and flow inherit errors from occupancy and agent tracking",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "samples": len(outputs),
                      "report": str(report_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
