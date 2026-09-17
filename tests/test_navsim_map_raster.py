import numpy as np
from shapely.geometry import LineString, Polygon

from bevlane.rasterize_navsim_map import _gpkg_geometry, rasterize


class TinyMap:
    def __init__(self):
        road = Polygon([(-10, -4), (10, -4), (10, 4), (-10, 4)])
        self.layers = {
            "lanes_polygons": [road],
            "walkways": [Polygon([(-10, 5), (10, 5), (10, 8), (-10, 8)])],
            "crosswalks": [Polygon([(2, -4), (4, -4), (4, 4), (2, 4)])],
            "inner_boundaries": [LineString([(-10, 0), (10, 0)])],
            "outer_boundaries": [LineString([(-10, 4), (10, 4)])],
        }

    def nearby(self, layer, window):
        return (geometry for geometry in self.layers.get(layer, [])
                if geometry.intersects(window))


def _pixel(ego_x, ego_y):
    return int((80 - ego_x) / 0.2), int((50 - ego_y) / 0.2)


def test_map_raster_contract_and_layer_priority():
    target = rasterize(TinyMap(), 0.0, 0.0, 0.0)

    assert target.shape == (800, 500)
    assert target.dtype == np.uint8
    assert target[_pixel(-5, 2)] == 1     # lane polygon
    assert target[_pixel(-5, 6)] == 2     # walkway
    assert target[_pixel(3, 2)] == 3      # crosswalk overlays road
    assert target[_pixel(-5, 0)] == 4     # shared boundary proxy
    assert target[_pixel(-5, 4)] == 6     # outer edge proxy
    assert target[_pixel(30, 30)] == 255  # unmapped, not negative background


def test_map_raster_rotates_global_geometry_into_ego_frame():
    target = rasterize(TinyMap(), 0.0, 0.0, np.pi / 2)

    # Global +x appears to ego-right after a 90-degree left turn.
    assert target[_pixel(0, -5)] == 4
    assert target[_pixel(0, -30)] == 255


def test_gpkg_geometry_header_without_envelope():
    from shapely import wkb

    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gpkg_blob = b"GP\x00\x01" + b"\x00\x00\x00\x00" + wkb.dumps(polygon)
    decoded = _gpkg_geometry(gpkg_blob)

    assert decoded.equals(polygon)
