#!/usr/bin/env python3
"""Rasterize nuPlan GeoPackage vectors into METEOR BEV map targets.

This is *map-derived* supervision, not camera-observed lane paint.  The
GeoPackage geometries are WGS84 lon/lat; NAVSIM ego poses use each map's
projected UTM CRS.  The projected CRS is read from the GeoPackage ``meta``
table instead of guessed from the city name.

Outputs ``gt_map/<frame>.png`` and adds ``gt_map`` to manifest frames only
after every target in a log has been written and its ego-coverage gate passes.
Raw NAVSIM logs/sensors and the default ``gt`` field are untouched by this
stage; promotion to default ``gt`` is a separate verified operation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from pyproj import Transformer
from shapely import wkb
from shapely.geometry import Polygon
from shapely.ops import transform
from shapely.strtree import STRtree

from bevlane.ingest_navsim import _atomic_json, quaternion_yaw


HEIGHT = 800
WIDTH = 500
RESOLUTION = 0.2
HALF_FORWARD = 80.0
HALF_LEFT = 50.0
IGNORE = 255

# METEOR's nine-class taxonomy; classes absent from the map stay ignored.
ROAD = 1
SIDEWALK = 2
CROSSWALK = 3
LANE_BOUNDARY_PROXY = 4
ROAD_EDGE_PROXY = 6

ROAD_LAYERS = (
    "lanes_polygons",
    "gen_lane_connectors_scaled_width_polygons",
    "intersections",
    "generic_drivable_areas",
)


def _gpkg_geometry(blob: bytes):
    """Decode GeoPackage geometry header followed by OGC WKB."""
    if blob is None or len(blob) < 8 or blob[:2] != b"GP":
        raise ValueError("Invalid GeoPackage geometry header")
    envelope_code = (blob[3] >> 1) & 0b111
    envelope_bytes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    if envelope_code not in envelope_bytes:
        raise ValueError(f"Unsupported GeoPackage envelope code {envelope_code}")
    return wkb.loads(blob[8 + envelope_bytes[envelope_code]:])


def locate_gpkg(maps_root: Path, map_location: str) -> Path:
    """Resolve exactly one map version for the NAVSIM map location."""
    if not map_location or Path(map_location).name != map_location:
        raise ValueError(f"Invalid map location {map_location!r}")
    matches = sorted((maps_root / map_location).glob("*/map.gpkg"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one map.gpkg for {map_location}, found {len(matches)} under {maps_root}")
    return matches[0]


class MapVectors:
    """One city's immutable vector layers, spatially indexed in UTM metres."""

    def __init__(self, gpkg: Path):
        self.gpkg = gpkg
        connection = sqlite3.connect(f"file:{gpkg.as_posix()}?mode=ro&immutable=1", uri=True)
        try:
            projected = connection.execute(
                "SELECT value FROM meta WHERE key='projectedCoordSystem'"
            ).fetchone()
            if projected is None:
                raise ValueError(f"No projectedCoordSystem metadata in {gpkg}")
            self.projected_crs = projected[0].lower()
            transformer = Transformer.from_crs("EPSG:4326", self.projected_crs, always_xy=True)
            self.layers: Dict[str, Tuple[List[Any], STRtree]] = {}
            for layer in (*ROAD_LAYERS, "walkways", "crosswalks"):
                self.layers[layer] = self._load_layer(connection, layer, transformer)

            # nuPlan lane polygons reference boundary IDs.  Shared boundaries
            # split two lanes; singly referenced boundaries form an outside
            # lane edge.  These are geometric proxies, not paint annotations.
            references: Dict[int, int] = {}
            for left, right in connection.execute(
                "SELECT left_boundary_fid,right_boundary_fid FROM lanes_polygons"
            ):
                for boundary_id in (left, right):
                    if boundary_id is not None:
                        references[int(boundary_id)] = references.get(int(boundary_id), 0) + 1
            inner: List[Any] = []
            outer: List[Any] = []
            for boundary_id, blob in connection.execute("SELECT fid,geom FROM boundaries"):
                count = references.get(int(boundary_id), 0)
                if count == 0 or blob is None:
                    continue
                geometry = transform(transformer.transform, _gpkg_geometry(blob))
                if geometry.is_empty:
                    continue
                (inner if count >= 2 else outer).append(geometry)
            self.layers["inner_boundaries"] = (inner, STRtree(inner))
            self.layers["outer_boundaries"] = (outer, STRtree(outer))
        finally:
            connection.close()

    @staticmethod
    def _load_layer(connection, table: str, transformer) -> Tuple[List[Any], STRtree]:
        geometries = []
        for (blob,) in connection.execute(f'SELECT geom FROM "{table}" WHERE geom IS NOT NULL'):
            geometry = transform(transformer.transform, _gpkg_geometry(blob))
            if not geometry.is_empty:
                geometries.append(geometry)
        return geometries, STRtree(geometries)

    def nearby(self, layer: str, window: Polygon) -> Iterable[Any]:
        geometries, tree = self.layers[layer]
        for index in tree.query(window):
            geometry = geometries[int(index)]
            if geometry.intersects(window):
                yield geometry


@lru_cache(maxsize=4)
def load_map(gpkg_path: str) -> MapVectors:
    return MapVectors(Path(gpkg_path))


def _window(global_x: float, global_y: float, yaw: float) -> Polygon:
    """Global UTM footprint of METEOR's 160-by-100 m ego raster."""
    cosine, sine = math.cos(yaw), math.sin(yaw)
    corners = []
    for forward, left in (
        (-HALF_FORWARD, -HALF_LEFT),
        (HALF_FORWARD, -HALF_LEFT),
        (HALF_FORWARD, HALF_LEFT),
        (-HALF_FORWARD, HALF_LEFT),
    ):
        corners.append((global_x + cosine * forward - sine * left,
                        global_y + sine * forward + cosine * left))
    return Polygon(corners)


def _pixel_ring(coords, global_x: float, global_y: float,
                cosine: float, sine: float) -> np.ndarray:
    array = np.asarray(coords, dtype=np.float64)[:, :2]
    dx, dy = array[:, 0] - global_x, array[:, 1] - global_y
    forward = cosine * dx + sine * dy
    left = -sine * dx + cosine * dy
    columns = (HALF_LEFT - left) / RESOLUTION
    rows = (HALF_FORWARD - forward) / RESOLUTION
    return np.round(np.column_stack([columns, rows])).astype(np.int32).reshape(-1, 1, 2)


def _draw_polygon(mask: np.ndarray, geometry, pose: Tuple[float, float, float]) -> None:
    if geometry.geom_type == "MultiPolygon":
        for part in geometry.geoms:
            _draw_polygon(mask, part, pose)
        return
    if geometry.geom_type != "Polygon":
        return
    x, y, yaw = pose
    cosine, sine = math.cos(yaw), math.sin(yaw)
    cv2.fillPoly(mask, [_pixel_ring(geometry.exterior.coords, x, y, cosine, sine)], 1)
    for interior in geometry.interiors:
        cv2.fillPoly(mask, [_pixel_ring(interior.coords, x, y, cosine, sine)], 0)


def _draw_line(mask: np.ndarray, geometry, pose: Tuple[float, float, float],
               thickness: int) -> None:
    if geometry.geom_type == "MultiLineString":
        for part in geometry.geoms:
            _draw_line(mask, part, pose, thickness)
        return
    if geometry.geom_type != "LineString":
        return
    x, y, yaw = pose
    cosine, sine = math.cos(yaw), math.sin(yaw)
    cv2.polylines(mask, [_pixel_ring(geometry.coords, x, y, cosine, sine)],
                  False, 1, thickness=thickness, lineType=cv2.LINE_8)


def rasterize(vectors: MapVectors, global_x: float, global_y: float,
              yaw: float) -> np.ndarray:
    """Map layers → METEOR classes; 255 outside mapped geometry."""
    pose = (float(global_x), float(global_y), float(yaw))
    window = _window(*pose)
    labels = np.full((HEIGHT, WIDTH), IGNORE, dtype=np.uint8)

    def paint(layers: Sequence[str], class_id: int, line_width: int = 0) -> None:
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        for layer in layers:
            for geometry in vectors.nearby(layer, window):
                if line_width:
                    _draw_line(mask, geometry, pose, line_width)
                else:
                    _draw_polygon(mask, geometry, pose)
        labels[mask != 0] = class_id

    paint(ROAD_LAYERS, ROAD)
    paint(("walkways",), SIDEWALK)
    paint(("crosswalks",), CROSSWALK)
    paint(("inner_boundaries",), LANE_BOUNDARY_PROXY, line_width=2)
    paint(("outer_boundaries",), ROAD_EDGE_PROXY, line_width=2)
    return labels


def _atomic_png(path: Path, labels: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", labels)
    if not success:
        raise OSError(f"Failed to encode map target {path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded.tobytes())
    os.replace(temporary, path)


def _process_log(config: Dict[str, Any]) -> Dict[str, Any]:
    manifest_path = Path(config["manifest_path"])
    scene_root = manifest_path.parent
    log_path = Path(config["data_root"]) / "navsim_logs" / config["split"] / (scene_root.name + ".pkl")
    if not log_path.is_file():
        raise FileNotFoundError(log_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("map_gt") and not config["force"]:
        complete = (manifest["map_gt"].get("frames_rasterized") == len(manifest["frames"])
                    and all(record.get("gt_map")
                            and (scene_root / record["gt_map"]).is_file()
                            for record in manifest["frames"]))
        if complete:
            return {"status": "skip", "scene": scene_root.name, "reason": "complete map_gt exists"}
    with log_path.open("rb") as stream:
        source_frames = pickle.load(stream)
    frames_by_token = {str(frame["token"]): frame for frame in source_frames}
    frames = manifest["frames"]
    if config["max_frames"] is not None:
        frames = frames[:config["max_frames"]]
    if not frames:
        raise ValueError(f"No manifest frames in {manifest_path}")
    map_locations = {frames_by_token[record["token"]]["map_location"] for record in frames}
    if len(map_locations) != 1:
        raise ValueError(f"Multiple map locations within scene {scene_root.name}: {map_locations}")
    map_location = map_locations.pop()
    gpkg = locate_gpkg(Path(config["data_root"]) / "maps", map_location)
    vectors = load_map(str(gpkg))

    target_dir = scene_root / "gt_map"
    target_dir.mkdir(exist_ok=True)
    ego_covered = 0
    labeled_cells = 0
    class_counts = np.zeros(9, dtype=np.int64)
    for record in frames:
        source = frames_by_token[record["token"]]
        if int(source["timestamp"]) != int(record["timestamp_us"]):
            raise ValueError(f"Timestamp mismatch for {record['token']} in {scene_root.name}")
        global_x, global_y = map(float, source["ego2global_translation"][:2])
        yaw = quaternion_yaw(source["ego2global_rotation"])
        target = rasterize(vectors, global_x, global_y, yaw)
        ego_covered += int(target[HEIGHT // 2, WIDTH // 2] != IGNORE)
        labeled_cells += int(np.count_nonzero(target != IGNORE))
        class_counts += np.bincount(target[target != IGNORE], minlength=9)[:9]
        relative = f"gt_map/{int(record['frame']):06d}.png"
        path = scene_root / relative
        if not path.exists() or config["force"]:
            _atomic_png(path, target)
        record["gt_map"] = relative

    coverage = ego_covered / len(frames)
    if coverage < config["min_ego_coverage"]:
        raise RuntimeError(
            f"Map/ego alignment gate failed in {scene_root.name}: "
            f"ego covered {coverage:.1%} < {config['min_ego_coverage']:.1%}; "
            "manifest was not promoted")
    if len(frames) != len(manifest["frames"]):
        return {
            "status": "preview",
            "scene": scene_root.name,
            "frames": len(frames),
            "ego_coverage": round(coverage, 4),
            "reason": "partial frame selection; manifest not promoted",
        }
    manifest["map_gt"] = {
        "source": "nuPlan vector GeoPackage",
        "map_location": map_location,
        "map_file": str(gpkg),
        "projected_crs": vectors.projected_crs,
        "classes": {
            "1": "road/drivable geometry",
            "2": "walkway polygon",
            "3": "crosswalk polygon",
            "4": "shared lane boundary geometry proxy, not observed paint",
            "6": "outer lane boundary geometry proxy",
            "255": "not covered by selected map layers",
        },
        "ego_coverage": coverage,
        "frames_rasterized": len(frames),
    }
    _atomic_json(manifest_path, manifest)
    return {
        "status": "ok",
        "scene": scene_root.name,
        "frames": len(frames),
        "ego_coverage": round(coverage, 4),
        "labeled_fraction": round(labeled_cells / (len(frames) * HEIGHT * WIDTH), 4),
        "class_pixels": {str(index): int(count) for index, count in enumerate(class_counts) if count},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="NAVSIM root containing maps and navsim_logs/<split>")
    parser.add_argument("--out", type=Path, required=True,
                        help="existing METEOR output root from ingest_navsim.py")
    parser.add_argument("--split", default="mini")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-logs", type=int, default=None,
                        help="process only first N logs for a smoke test")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="process only first N frames/log for a smoke test")
    parser.add_argument("--min-ego-coverage", type=float, default=0.75,
                        help="minimum fraction of ego-center pixels covered by map geometry")
    parser.add_argument("--force", action="store_true",
                        help="regenerate a scene's existing map targets")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers < 1 or not 0 <= args.min_ego_coverage <= 1:
        raise SystemExit("--workers must be positive; coverage threshold must be in [0,1]")
    manifests = sorted(args.out.glob("*/manifest.json"))
    if args.max_logs is not None:
        manifests = manifests[:args.max_logs]
    if not manifests:
        raise SystemExit(f"No converted scene manifests in {args.out}")
    configs = [{
        "manifest_path": str(path),
        "data_root": str(args.data_root),
        "split": args.split,
        "max_frames": args.max_frames,
        "min_ego_coverage": args.min_ego_coverage,
        "force": args.force,
    } for path in manifests]
    results = []
    if args.workers == 1:
        iterator = map(_process_log, configs)
        for result in iterator:
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for result in executor.map(_process_log, configs):
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
    total_frames = sum(result.get("frames", 0) for result in results)
    print(f"MAP_GT_READY scenes={len(results)} frames={total_frames} root={args.out}", flush=True)


if __name__ == "__main__":
    main()
