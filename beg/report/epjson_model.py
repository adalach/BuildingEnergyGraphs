"""Build a Honeybee ``Model`` from an EnergyPlus epJSON file, for display.

The graph explorer shows each building next to its graphs; this module rebuilds
the 3D geometry from the epJSON, so the preview needs only ``data/epjson/``. It
makes one Room per ``Zone``, one Face per ``BuildingSurface:Detailed``, apertures
and doors from ``FenestrationSurface:Detailed``, and shades from the three
``Shading:*:Detailed`` objects.

The result is for drawing, not for simulation. Constructions, schedules and
loads are not carried over, and interzone boundary conditions point at the
adjacent surface by name without checking that it matches.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

#: epJSON ``surface_type`` to Honeybee face type name.
_FACE_TYPES = {"wall": "Wall", "floor": "Floor", "roof": "RoofCeiling", "ceiling": "RoofCeiling"}

_SHADING_OBJECTS = ("Shading:Site:Detailed", "Shading:Building:Detailed", "Shading:Zone:Detailed")


@dataclass
class ConversionReport:
    """What was converted, and what had to be skipped."""

    rooms: int = 0
    faces: int = 0
    apertures: int = 0
    doors: int = 0
    shades: int = 0
    skipped: list[str] = field(default_factory=list)


def _vertices(obj: dict) -> list[tuple[float, float, float]]:
    """Vertices in either epJSON layout: a ``vertices`` list or numbered fields."""
    if "vertices" in obj:
        return [
            (float(v["vertex_x_coordinate"]), float(v["vertex_y_coordinate"]), float(v["vertex_z_coordinate"]))
            for v in obj["vertices"]
        ]
    points, i = [], 1
    while f"vertex_{i}_x_coordinate" in obj:
        points.append(tuple(float(obj[f"vertex_{i}_{axis}_coordinate"]) for axis in "xyz"))
        i += 1
    return points


def _zone_transform(zone: dict, relative: bool):
    """Map a zone's relative coordinates to building coordinates.

    With ``coordinate_system`` = Relative, surface vertices are given relative to
    the zone origin and rotated by the zone's relative north. With World
    coordinates the vertices are already absolute.
    """
    if not relative:
        return lambda p: p
    ox, oy, oz = (float(zone.get(f"{a}_origin", 0) or 0) for a in "xyz")
    theta = math.radians(float(zone.get("direction_of_relative_north", 0) or 0))
    c, s = math.cos(theta), math.sin(theta)
    return lambda p: (ox + c * p[0] - s * p[1], oy + s * p[0] + c * p[1], oz + p[2])


def model_from_epjson(path: Path | str, *, identifier: str | None = None):
    """Honeybee ``Model`` and a :class:`ConversionReport` for one epJSON file."""
    from honeybee.aperture import Aperture
    from honeybee.boundarycondition import Surface, boundary_conditions
    from honeybee.door import Door
    from honeybee.face import Face
    from honeybee.facetype import face_types
    from honeybee.model import Model
    from honeybee.room import Room
    from honeybee.shade import Shade
    from honeybee.typing import clean_string
    from ladybug_geometry.geometry3d import Face3D, Point3D

    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    report = ConversionReport()

    rules = next(iter((data.get("GlobalGeometryRules") or {}).values()), {})
    relative = str(rules.get("coordinate_system", "Relative")).lower() == "relative"
    zones = data.get("Zone") or {}
    transforms = {name: _zone_transform(z, relative) for name, z in zones.items()}
    identity = lambda p: p  # noqa: E731

    def face3d(name: str, points, transform) -> Face3D | None:
        pts = [Point3D(*transform(p)) for p in points]
        try:
            geometry = Face3D(pts)
            if geometry.area < 1e-6:
                raise ValueError("zero area")
            return geometry
        except Exception as exc:  # degenerate or collinear vertex lists
            report.skipped.append(f"{name}: {exc}")
            return None

    surfaces = data.get("BuildingSurface:Detailed") or {}
    faces: dict[str, Face] = {}
    faces_by_zone: dict[str, list[Face]] = {name: [] for name in zones}
    for name, surface in surfaces.items():
        zone = surface.get("zone_name", "")
        geometry = face3d(name, _vertices(surface), transforms.get(zone, identity))
        if geometry is None:
            continue
        kind = _FACE_TYPES.get(str(surface.get("surface_type", "")).lower(), "Wall")
        face = Face(clean_string(name), geometry, getattr(face_types, _snake(kind)))
        face.display_name = name
        bc = str(surface.get("outside_boundary_condition", "")).lower()
        other = surface.get("outside_boundary_condition_object")
        if bc in ("surface", "zone") and other:
            other_zone = surfaces.get(other, {}).get("zone_name", other)
            face.boundary_condition = Surface((clean_string(other), clean_string(other_zone)))
        elif bc in ("ground", "foundation") or bc.startswith("groundfcfactor") or bc.startswith("groundslab"):
            face.boundary_condition = boundary_conditions.ground
        else:
            face.boundary_condition = boundary_conditions.outdoors
        faces[name] = face
        faces_by_zone.setdefault(zone, []).append(face)

    for name, sub in (data.get("FenestrationSurface:Detailed") or {}).items():
        host_name = sub.get("building_surface_name", "")
        host = faces.get(host_name)
        if host is None:
            report.skipped.append(f"{name}: host surface {host_name!r} missing")
            continue
        zone = surfaces.get(host_name, {}).get("zone_name", "")
        geometry = face3d(name, _vertices(sub), transforms.get(zone, identity))
        if geometry is None:
            continue
        kind = str(sub.get("surface_type", "Window")).lower()
        if kind in ("door", "glassdoor"):
            door = Door(clean_string(name), geometry, is_glass=kind == "glassdoor")
            door.display_name = name
            host.add_door(door)
            report.doors += 1
        else:
            aperture = Aperture(clean_string(name), geometry)
            aperture.display_name = name
            host.add_aperture(aperture)
            report.apertures += 1

    rooms = []
    for zone, zone_faces in faces_by_zone.items():
        if not zone_faces:
            continue
        room = Room(clean_string(zone), zone_faces)
        room.display_name = zone
        rooms.append(room)
    report.rooms, report.faces = len(rooms), sum(len(f) for f in faces_by_zone.values())

    shades = []
    for obj_type in _SHADING_OBJECTS:
        for name, obj in (data.get(obj_type) or {}).items():
            # Site shading is always in world coordinates; building and zone
            # shading follow the building's coordinate system.
            transform = identity
            if obj_type == "Shading:Zone:Detailed":
                base = surfaces.get(obj.get("base_surface_name", ""), {})
                transform = transforms.get(base.get("zone_name", ""), identity)
            geometry = face3d(name, _vertices(obj), transform)
            if geometry is not None:
                shade = Shade(clean_string(name), geometry)
                shade.display_name = name
                shades.append(shade)
    report.shades = len(shades)

    model = Model(clean_string(identifier or path.stem), rooms=rooms, orphaned_shades=shades)
    model.display_name = identifier or path.stem
    return model, report


def _snake(name: str) -> str:
    """``RoofCeiling`` -> ``roof_ceiling``, the attribute name on ``face_types``."""
    return "".join(f"_{c.lower()}" if c.isupper() and i else c.lower() for i, c in enumerate(name))
