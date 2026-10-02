"""
Expectations derived from a configuration and domain rules for feature columns.

``ConfigSpec`` turns a configuration dictionary — as ``beg.config.loader``
resolves it, with ``extends`` already merged — into the *expected*
structure of every generated graph: node types, (src_type, rel, tgt_type)
triples (including reverse edges for ``bidirectional`` relations), which
relations are one-directional, the weather variant, etc.  All structural
checks compare observed data against a ``ConfigSpec``.

The ``COLUMN_RULES`` table encodes what we know about feature semantics
(areas are positive, transmittances are fractions, ...).  Rules are matched by
exact column name first and then by regex, so new SQL transforms that follow
the same naming get validated automatically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

Triple = Tuple[str, str, str]

# --------------------------------------------------------------------------
# Naming conventions
# --------------------------------------------------------------------------

LEVELS = ("zone", "component")
GRAPH_TYPES = ("hetero", "homo")
WEATHER_VARIANTS = ("no_weather", "building_weather", "external_weather", "graph_weather")
THREE_AXIS_RE = re.compile(
    r"^(?P<level>zone|component)_(?P<graph_type>hetero|homo)_(?P<weather>no_weather|building_weather|external_weather|graph_weather)$"
)

# Tables that are legitimately *not* produced by a transform in the config
# (they are raw / helper tables in the ``ep`` schema).
EXTERNAL_SOURCE_TABLES: FrozenSet[str] = frozenset({"weather_city"})

# Node tables that carry weather information.
WEATHER_NODE_TABLES: FrozenSet[str] = frozenset({"weather_city", "building_weather"})
WEATHER_RELATIONS: FrozenSet[str] = frozenset({"weather_for"})

# Columns that are join keys / identifiers rather than features.
IDENTIFIER_COLUMNS: FrozenSet[str] = frozenset(
    {"id", "idx", "index", "model_id", "node_id", "instance_name", "src_id", "tgt_id", "src_idx", "tgt_idx", "node_uid", "config_name", "__mv_chunk__"}
)
IDENTIFIER_SUFFIXES = ("_id", "_idx", "_key", "_name", "_uid")

# Attributes the NetworkX builder adds itself.
NX_RESERVED_NODE_ATTRS: FrozenSet[str] = frozenset({"node_type", "node_id"})
NX_RESERVED_EDGE_ATTRS: FrozenSet[str] = frozenset({"edge_type", "rel", "is_reverse"})
NX_MISSING_NODE_TYPE = "(missing_from_nodes)"


def is_identifier_column(col: str) -> bool:
    lc = str(col).lower()
    if lc in IDENTIFIER_COLUMNS or lc.startswith("__"):
        return True
    return lc.endswith(IDENTIFIER_SUFFIXES)


# --------------------------------------------------------------------------
# ConfigSpec
# --------------------------------------------------------------------------

@dataclass
class EdgeSpec:
    edge_src: str
    src_cat: str
    src_col: str
    tgt_cat: str
    tgt_col: str
    rel: str
    bidirectional: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def table(self) -> str:
        return self.edge_src.replace(".sql", "")

    @property
    def forward(self) -> Triple:
        return (self.src_cat, self.rel, self.tgt_cat)

    @property
    def reverse(self) -> Triple:
        return (self.tgt_cat, self.rel, self.src_cat)

    @property
    def is_self_type(self) -> bool:
        return self.src_cat == self.tgt_cat


@dataclass
class ConfigSpec:
    name: str                       # config_name / prefix (e.g. component_hetero_external_weather)
    config_id: Optional[str]        # e.g. "C10"
    graph_type_field: str           # "heterogeneous" | "homogeneous"
    node_tables: List[str]
    edges: List[EdgeSpec]
    sql_transforms: List[str]
    weather_field: Optional[str] = None
    description: str = ""
    path: Optional[Path] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- names
    @property
    def full_name(self) -> str:
        return f"{self.config_id}_{self.name}" if self.config_id else self.name

    @property
    def homogeneous(self) -> bool:
        return self.graph_type_field == "homogeneous"

    @property
    def axes(self) -> Optional[Dict[str, str]]:
        m = THREE_AXIS_RE.match(self.name)
        return m.groupdict() if m else None

    @property
    def level(self) -> Optional[str]:
        a = self.axes
        return a["level"] if a else None

    @property
    def weather_variant(self) -> Optional[str]:
        a = self.axes
        if a:
            return a["weather"]
        # fall back to the free-text field used in the root configs
        wf = (self.weather_field or "").lower()
        if "separate" in wf:
            return "external_weather"
        if "model node" in wf or "building" in wf:
            return "building_weather"
        if "not" in wf:
            return "no_weather"
        return None

    # ------------------------------------------------------------ structure
    @property
    def primary_node_table(self) -> str:
        return self.node_tables[0]

    @property
    def edge_tables(self) -> List[str]:
        return [e.table for e in self.edges]

    @property
    def expected_triples(self) -> Set[Triple]:
        out: Set[Triple] = set()
        for e in self.edges:
            out.add(e.forward)
            if e.bidirectional:
                out.add(e.reverse)
        return out

    @property
    def forward_triples(self) -> Set[Triple]:
        return {e.forward for e in self.edges}

    @property
    def one_directional_triples(self) -> Set[Triple]:
        """Triples whose reverse must NOT appear in the graph."""
        out: Set[Triple] = set()
        for e in self.edges:
            if not e.bidirectional and not e.is_self_type:
                out.add(e.forward)
        return out

    @property
    def relations(self) -> Set[str]:
        return {e.rel for e in self.edges}

    @property
    def has_weather_node(self) -> bool:
        return any(t in WEATHER_NODE_TABLES for t in self.node_tables)

    @property
    def weather_node_tables(self) -> List[str]:
        return [t for t in self.node_tables if t in WEATHER_NODE_TABLES]

    def edges_by_table(self) -> Dict[str, EdgeSpec]:
        return {e.table: e for e in self.edges}

    def edges_touching(self, node_table: str) -> List[EdgeSpec]:
        return [e for e in self.edges if node_table in (e.src_cat, e.tgt_cat)]

    # Node tables for which "exactly one row per model" is expected.
    @property
    def singleton_node_tables(self) -> Set[str]:
        return {t for t in self.node_tables if t in {"building", "building_weather", "weather_city"}}


def config_spec_from_dict(cfg: Dict[str, Any], *, path: Optional[Path] = None, name: Optional[str] = None) -> ConfigSpec:
    nodes = [n["table"] if isinstance(n, dict) else str(n) for n in cfg.get("nodes", [])]
    edges: List[EdgeSpec] = []
    for e in cfg.get("edges", []):
        if str(e.get("edge_src", "")).lower() == "fk":
            continue
        edges.append(
            EdgeSpec(
                edge_src=str(e.get("edge_src", "")),
                src_cat=str(e.get("src_cat", "")),
                src_col=str(e.get("src_col", "")),
                tgt_cat=str(e.get("tgt_cat", "")),
                tgt_col=str(e.get("tgt_col", "")),
                rel=str(e.get("rel", "") or e.get("edge_src", "")),
                bidirectional=bool(e.get("bidirectional", False)),
                raw=dict(e),
            )
        )
    cfg_name = name or str(cfg.get("prefix") or (path.stem if path else "config"))
    return ConfigSpec(
        name=cfg_name,
        config_id=(str(cfg["id"]).strip() if cfg.get("id") not in (None, "") else None),
        graph_type_field=str(cfg.get("type", "")),
        node_tables=nodes,
        edges=edges,
        sql_transforms=[str(s) for s in cfg.get("sql_transforms", [])],
        weather_field=cfg.get("weather"),
        description=str(cfg.get("description", "")),
        path=path,
        raw=dict(cfg),
    )


def load_config_spec(path: Path | str) -> ConfigSpec:
    """Load one configuration, with ``extends`` resolved.

    Reading the raw JSON is not enough: six of the eight configurations inherit
    from ``base/zone_hetero.json`` or ``base/component_hetero.json`` and state
    only their difference, as ``{"add": [...], "remove": [...]}`` per list key.
    Unresolved, ``edges`` is that directive dict rather than a list of edges.
    """
    from beg.config.loader import resolve_config

    path = Path(path)
    return config_spec_from_dict(resolve_config(path), path=path)


def load_config_specs(configs_dir: Path | str, names: Optional[Sequence[str]] = None) -> Dict[str, ConfigSpec]:
    """Load every configuration in a directory, keyed by config name.

    Files are located through :func:`beg.config.loader.discover_config_paths`,
    which resolves names via ``beg.config.naming``.
    """
    from beg.config.loader import discover_config_paths

    out: Dict[str, ConfigSpec] = {}
    for path in discover_config_paths(Path(configs_dir), names).values():
        spec = load_config_spec(path)
        out[spec.name] = spec
    return out


# --------------------------------------------------------------------------
# Column domain rules
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnRule:
    """Numeric domain of a feature column.

    ``lo``/``hi`` are inclusive bounds (None = unbounded); ``lo_exclusive`` makes
    ``lo`` a strict bound (e.g. area > 0).  ``integer`` requires integral values.
    ``severity`` is "FAIL" or "WARN".
    """

    lo: Optional[float] = None
    hi: Optional[float] = None
    lo_exclusive: bool = False
    integer: bool = False
    severity: str = "FAIL"
    note: str = ""


_FRACTION = ColumnRule(lo=0.0, hi=1.0, note="fraction in [0, 1]")
_NONNEG = ColumnRule(lo=0.0, note=">= 0")
_POSITIVE = ColumnRule(lo=0.0, lo_exclusive=True, note="> 0")
_COUNT = ColumnRule(lo=0.0, integer=True, note="non-negative integer")
_SINCOS = ColumnRule(lo=-1.0, hi=1.0, note="sine/cosine component in [-1, 1]")

# Exact column name → rule
COLUMN_RULES: Dict[str, ColumnRule] = {
    # geometry
    "area": _POSITIVE,
    "floor_area": _POSITIVE,
    "ceiling_height": ColumnRule(lo=1.0, hi=30.0, severity="WARN", note="plausible zone height [1, 30] m"),
    "floor_level": ColumnRule(lo=0.0, integer=True, note="non-negative integer floor index"),
    "wall_area_total": _NONNEG,
    "wall_area_ext": _NONNEG,
    "wall_area_int": _NONNEG,
    "window_area_ext": _NONNEG,
    "roof_area_total": _NONNEG,
    "room_boundary_area_total": _NONNEG,
    "shared_boundary_area": _POSITIVE,
    "surface_area": _POSITIVE,
    "aperture_area": _POSITIVE,
    "external_surface_area": _NONNEG,
    "n_external_surfaces": _COUNT,
    # ratios
    "wwr": _FRACTION,
    "ewr": ColumnRule(lo=0.0, severity="WARN", note="envelope ratio >= 0"),
    "iwr": ColumnRule(lo=0.0, severity="WARN", note="internal wall ratio >= 0"),
    "frr": ColumnRule(lo=0.0, severity="WARN", note="floor/roof ratio >= 0"),
    "rrr": ColumnRule(lo=0.0, severity="WARN", note="roof ratio >= 0"),
    # thermal / material
    "total_thickness": ColumnRule(lo=0.0, hi=5.0, note="layer thickness [0, 5] m"),
    "r_value_layers": _NONNEG,
    "u_value_total": ColumnRule(lo=0.0, hi=20.0, lo_exclusive=True, note="U-value (0, 20] W/m2K"),
    "r_ext_wall": _NONNEG,
    "r_ext_window": _NONNEG,
    "r_roof": _NONNEG,
    "r_floor": _NONNEG,
    "solar_transmittance": _FRACTION,
    "visible_transmittance": _FRACTION,
    "front_emissivity": _FRACTION,
    "back_emissivity": _FRACTION,
    "thermal_absorptance": _FRACTION,
    "solar_absorptance": _FRACTION,
    "visible_absorptance": _FRACTION,
    "density": ColumnRule(lo=0.0, hi=30000.0, note="density [0, 30000] kg/m3"),
    "specific_heat": ColumnRule(lo=0.0, hi=10000.0, note="specific heat [0, 10000] J/kgK"),
    "outside_roughness_category": ColumnRule(lo=0.0, hi=6.0, integer=True, note="EnergyPlus roughness class index"),
    # loads / internal gains
    "return_air_fraction": _FRACTION,
    "fraction_radiant": _FRACTION,
    "fraction_visible": _FRACTION,
    "people_per_floor_area": ColumnRule(lo=0.0, hi=2.0, note="people/m2 [0, 2]"),
    "outdoor_air_flow_per_person": ColumnRule(lo=0.0, hi=1.0, note="m3/s per person"),
    "outdoor_air_flow_per_zone_floor_area": ColumnRule(lo=0.0, hi=1.0),
    "outdoor_air_flow_air_changes_per_hour": ColumnRule(lo=0.0, hi=50.0),
    "peak_flow_rate": _NONNEG,
    "num_wateruse_objects": _COUNT,
    # shading metrics (surface_shading_metrics.sql)
    "shade_count": _COUNT,
    "shade_ray_count": _COUNT,
    "shade_min_dist": _NONNEG,
    "shade_mean_dist": _NONNEG,
    "shade_p90_dist": _NONNEG,
    "shade_ray_dir_mean_sin": _SINCOS,
    "shade_ray_dir_mean_cos": _SINCOS,
    "shade_ray_elevation_mean_rad": ColumnRule(lo=-1.6, hi=1.6, note="elevation angle in radians"),
    "shade_ray_elevation_mean_sin": _SINCOS,
    "shade_ray_elevation_mean_cos": _SINCOS,
    # weather (city level)
    "latitude": ColumnRule(lo=-90.0, hi=90.0),
    "longitude": ColumnRule(lo=-180.0, hi=180.0),
    "elevation_m": ColumnRule(lo=-500.0, hi=6000.0),
    "n_summer_design_days": _COUNT,
    "n_winter_design_days": _COUNT,
    "summer_daily_range_mean_c": ColumnRule(lo=0.0, hi=40.0, severity="WARN"),
    "winter_daily_range_mean_c": ColumnRule(lo=0.0, hi=40.0, severity="WARN"),
    "summer_wind_speed_mean_ms": ColumnRule(lo=0.0, hi=60.0),
    "winter_wind_speed_mean_ms": ColumnRule(lo=0.0, hi=60.0),
    "summer_wind_direction_mean_deg": ColumnRule(lo=0.0, hi=360.0),
    "winter_wind_direction_mean_deg": ColumnRule(lo=0.0, hi=360.0),
}

# Regex → rule (applied when no exact rule matched).  Order matters.
COLUMN_RULE_PATTERNS: List[Tuple[re.Pattern[str], ColumnRule]] = [
    (re.compile(r"^(is|has)_[a-z0-9_]+$"), ColumnRule(lo=0.0, hi=1.0, integer=True, note="boolean flag 0/1")),
    (re.compile(r"^(cdd|hdd)\d+_\d{2}$"), ColumnRule(lo=0.0, hi=1500.0, note="degree-days per month")),
    (re.compile(r"^daylight_hours_\d{2}$"), ColumnRule(lo=0.0, hi=744.0, note="hours per month")),
    (re.compile(r"^(precip_hours|rh80_hours|strong_wind_hours)_\d{2}$"), ColumnRule(lo=0.0, hi=744.0, note="hours per month")),
    (re.compile(r"^precip_mm_\d{2}$"), ColumnRule(lo=0.0, hi=3000.0)),
    (re.compile(r"^(dhi|dni|ghi)_kwh_m2_\d{2}$"), ColumnRule(lo=0.0, hi=400.0, note="monthly irradiation kWh/m2")),
    (re.compile(r"^rh_mean_pct_\d{2}$"), ColumnRule(lo=0.0, hi=100.0)),
    (re.compile(r"^t_(min|max|mean|day_mean|night_mean)_c_\d{2}$"), ColumnRule(lo=-70.0, hi=60.0, note="air temperature °C")),
    (re.compile(r"^wind_speed_mean_ms_\d{2}$"), ColumnRule(lo=0.0, hi=60.0)),
    (re.compile(r"^(summer|winter)_(min|max|mean_max)_drybulb_c$"), ColumnRule(lo=-70.0, hi=60.0)),
    (re.compile(r"^(summer|winter)_humidity_value_mean$"), ColumnRule(lo=-70.0, hi=100.0, severity="WARN", note="design-day humidity value; values ≫100 mean humidity_condition_type differs between cities (WetBulb °C vs Enthalpy J/kg) and are averaged as if comparable")),
    (re.compile(r"^(vintage|primary_function)_[a-z0-9_]+$"), ColumnRule(lo=0.0, hi=1.0, integer=True, note="one-hot indicator")),
    (re.compile(r".*_(area|count|hours|thickness|distance|dist)$"), _NONNEG),
]


def rule_for_column(col: str) -> Optional[ColumnRule]:
    if col in COLUMN_RULES:
        return COLUMN_RULES[col]
    for pat, rule in COLUMN_RULE_PATTERNS:
        if pat.match(col):
            return rule
    return None


# One-hot groups: a prefix whose indicator columns must sum to exactly one per row.
# ``optional=True`` allows all-zero rows (e.g. apertures that are none of the listed kinds).
@dataclass(frozen=True)
class OneHotGroup:
    name: str
    columns: Tuple[str, ...] = ()
    prefix: Optional[str] = None
    optional: bool = False
    node_tables: Tuple[str, ...] = ()  # empty = any table that has the columns

    def resolve(self, columns: Sequence[str]) -> List[str]:
        cols = [c for c in self.columns if c in columns]
        if self.prefix:
            cols += [c for c in columns if c.startswith(self.prefix) and c not in cols and c != self.prefix.rstrip("_")]
        return cols


ONE_HOT_GROUPS: List[OneHotGroup] = [
    OneHotGroup("vintage", prefix="vintage_"),
    OneHotGroup("primary_function", prefix="primary_function_"),
    OneHotGroup("surface_kind", columns=("is_wall", "is_floor", "is_roof", "is_ceiling")),
    OneHotGroup("boundary_condition", columns=("is_outdoors", "is_surface", "is_zone", "is_ground", "is_adiabatic")),
    OneHotGroup("aperture_kind", columns=("is_window", "is_door", "is_glass_door", "is_tubular_daylight"), optional=True),
    OneHotGroup("unified_relationship", columns=("is_cross_zone", "is_adjacency", "is_aperture_to_element"), optional=True),
]

# Columns that are string labels mirrored by a one-hot group (label value must match hot column).
ONE_HOT_LABEL_COLUMNS: Dict[str, str] = {"vintage": "vintage_", "primary_function": "primary_function_"}

# Columns that must be all-zero *only* for a minority of the dataset — pipeline rule:
# "never accept an all-zero shading dataset".
SHADING_COLUMNS: Tuple[str, ...] = ("shade_count", "shade_ray_count")

# Weather-exposure rule at component level (mirrors the WHERE clause of
# edge_comp_env_weather.sql): external walls, external roofs/ceilings and
# ground-contact floors receive a weather edge.
def element_is_weather_exposed(attrs: Dict[str, Any]) -> bool:
    def _f(k: str) -> int:
        v = attrs.get(k, 0)
        try:
            return int(v) if v is not None and v == v else 0  # NaN-safe
        except (TypeError, ValueError):
            return 0

    if _f("is_outdoors") == 1 and (_f("is_wall") == 1 or _f("is_roof") == 1 or _f("is_ceiling") == 1):
        return True
    return _f("is_floor") == 1 and _f("is_ground") == 1
