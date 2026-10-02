"""The explorer's 3D preview can be rebuilt from an epJSON file alone."""

import json

import pytest

pytest.importorskip("honeybee")

from beg.report.epjson_model import model_from_epjson  # noqa: E402


def _v(*points):
    return [{"vertex_x_coordinate": x, "vertex_y_coordinate": y, "vertex_z_coordinate": z}
            for x, y, z in points]


def test_zones_surfaces_windows_and_shades_come_across(tmp_path):
    epjson = {
        "GlobalGeometryRules": {"r": {"coordinate_system": "Relative"}},
        "Zone": {"Z1": {"x_origin": 10, "y_origin": 0, "z_origin": 0}, "Z2": {}},
        "BuildingSurface:Detailed": {
            "Z1_wall": {"zone_name": "Z1", "surface_type": "Wall",
                        "outside_boundary_condition": "Outdoors",
                        "vertices": _v((0, 0, 3), (0, 0, 0), (4, 0, 0), (4, 0, 3))},
            "Z1_ceiling": {"zone_name": "Z1", "surface_type": "Ceiling",
                           "outside_boundary_condition": "Surface",
                           "outside_boundary_condition_object": "Z2_floor",
                           "vertices": _v((0, 0, 3), (4, 0, 3), (4, 4, 3), (0, 4, 3))},
            "Z2_floor": {"zone_name": "Z2", "surface_type": "Floor",
                         "outside_boundary_condition": "Ground",
                         "vertices": _v((0, 0, 0), (0, 4, 0), (4, 4, 0), (4, 0, 0))},
        },
        "FenestrationSurface:Detailed": {
            "Z1_win": {"building_surface_name": "Z1_wall", "surface_type": "Window",
                       "vertex_1_x_coordinate": 1, "vertex_1_y_coordinate": 0, "vertex_1_z_coordinate": 2,
                       "vertex_2_x_coordinate": 1, "vertex_2_y_coordinate": 0, "vertex_2_z_coordinate": 1,
                       "vertex_3_x_coordinate": 3, "vertex_3_y_coordinate": 0, "vertex_3_z_coordinate": 1,
                       "vertex_4_x_coordinate": 3, "vertex_4_y_coordinate": 0, "vertex_4_z_coordinate": 2},
        },
        "Shading:Building:Detailed": {
            "neighbour": {"vertices": _v((20, 5, 9), (20, 5, 0), (30, 5, 0), (30, 5, 9))},
        },
    }
    path = tmp_path / "Test_0001.epJSON"
    path.write_text(json.dumps(epjson))

    model, report = model_from_epjson(path)

    assert (report.rooms, report.faces, report.apertures, report.shades) == (2, 3, 1, 1)
    assert report.skipped == []
    wall = next(f for r in model.rooms for f in r.faces if f.display_name == "Z1_wall")
    assert str(wall.type) == "Wall" and len(wall.apertures) == 1
    # Relative coordinates are shifted by the zone origin.
    assert min(v.x for v in wall.geometry.vertices) == pytest.approx(10)
    ceiling = next(f for r in model.rooms for f in r.faces if f.display_name == "Z1_ceiling")
    assert "Surface" in str(ceiling.boundary_condition)
