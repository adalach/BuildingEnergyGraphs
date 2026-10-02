"""
Static checks on configuration JSON files and the SQL transforms they reference.

These checks need no database and no generated data — they catch configuration
errors *before* an expensive graph build:

* the config passes the strict ``validate_config`` schema check
* ``prefix`` follows the 3-axis naming convention
  ``{level}_{hetero|homo}_{weather_variant}``, ``type`` agrees with it, ``id`` has
  the form ``C<n>``, and the file name (``C<n>.json``) names the same configuration
* every referenced SQL transform exists and is registered in ``PRIMARY_ORDER.txt``
* every node table / edge table has a producing transform in the config (or is a
  known external source table such as ``weather_city``)
* edge endpoints reference declared node tables; ``src_col`` / ``tgt_col`` appear
  in the producing SQL; no duplicated relations
* weather variant ⇔ presence of weather node tables / ``weather_for`` relations
* homogeneous ⇒ exactly one node table and only self-type edges; heterogeneous ⇒
  ≥ 2 node tables; every node table is reachable by at least one edge
* ``id`` values are unique across the directory
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from beg.validation.core import CheckResult, examples
from beg.validation.spec import (
    EXTERNAL_SOURCE_TABLES,
    WEATHER_RELATIONS,
    ConfigSpec,
    config_spec_from_dict,
    load_config_specs,
)

SCOPE = "config"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def read_primary_order(transforms_dir: Path) -> List[str]:
    p = Path(transforms_dir) / "PRIMARY_ORDER.txt"
    if not p.exists():
        return []
    out: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("--"):
            continue
        out.append(s)
    return out


_CREATE_RE = re.compile(
    r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:MATERIALIZED\s+)?(?:TABLE|VIEW)\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"(?:\"?[A-Za-z_][A-Za-z0-9_]*\"?\.)?\"?([A-Za-z_][A-Za-z0-9_]*)\"?",
    re.IGNORECASE,
)


def relations_created_by_sql(sql_text: str) -> Set[str]:
    """Relation names a SQL transform creates (best-effort regex; handles {schema} placeholders)."""
    txt = re.sub(r"\{[^}]*\}\.", "", sql_text)  # drop templated schema prefixes like {schema}.
    return {m.group(1).lower() for m in _CREATE_RE.finditer(txt)}


def _resolve_transform_path(transforms_dir: Path, transform: str) -> Optional[Path]:
    p = Path(transforms_dir) / transform
    if p.exists():
        return p
    # allow bare filename registered under a sub-folder
    hits = list(Path(transforms_dir).rglob(Path(transform).name))
    return hits[0] if len(hits) == 1 else None


# --------------------------------------------------------------------------
# single config
# --------------------------------------------------------------------------

def _config_name_from_path(config_path: Path) -> str:
    """The configuration a file declares.

    ``C5.json`` and ``zone_hetero_external_weather.json`` both give ``C5``.
    """
    stem = Path(config_path).stem
    try:
        from beg.config.naming import resolve

        return resolve(stem).id
    except KeyError:
        return stem


def check_config_file(
    config_path: Path,
    *,
    transforms_dir: Optional[Path] = None,
    strict_three_axis: bool = True,
) -> List[CheckResult]:
    """Run all static checks on one configuration file."""
    config_path = Path(config_path)
    results: List[CheckResult] = []
    # Only a label for failures raised before the spec exists. The spec's own
    # name comes from the config's `prefix`, because the three-axis check parses
    # it -- the file is called `C5.json`, which carries none of those axes.
    label = _config_name_from_path(config_path)

    # 1) JSON + schema validation (reuse the project's strict validator when importable)
    # `resolve_config`, not `json.loads`: configurations inherit through `extends`,
    # and the raw file states only its difference from the base it extends.
    try:
        from beg.config.loader import resolve_config

        cfg = resolve_config(config_path)
    except Exception as exc:
        return [CheckResult.fail("config.json_parse", f"cannot load config: {exc}", config=label, scope=SCOPE)]
    try:
        from beg.config.validator import validate_config  # type: ignore

        validate_config(cfg, label)
        results.append(CheckResult.ok("config.schema_valid", "validate_config passed", config=label, scope=SCOPE))
    except ImportError:  # pragma: no cover
        results.append(CheckResult.skip("config.schema_valid", "beg.config.validator not importable", config=label, scope=SCOPE))
    except Exception as exc:
        results.append(CheckResult.fail("config.schema_valid", str(exc), config=label, scope=SCOPE))

    # No `name=`: let the spec take its name from the config's `prefix` field.
    spec = config_spec_from_dict(cfg, path=config_path)
    results.extend(check_config_spec(spec, transforms_dir=transforms_dir, strict_three_axis=strict_three_axis))
    return results


def check_config_spec(
    spec: ConfigSpec,
    *,
    transforms_dir: Optional[Path] = None,
    strict_three_axis: bool = True,
) -> List[CheckResult]:
    r: List[CheckResult] = []
    name = spec.name
    kw = dict(config=name, scope=SCOPE)

    # ---- naming -----------------------------------------------------------
    prefix = str(spec.raw.get("prefix", ""))
    if prefix != prefix.lower():
        r.append(CheckResult.fail("config.prefix_lowercase", f"prefix '{prefix}' must be lowercase (Postgres relation names)", **kw))
    # The file name and the prefix must name the same configuration: files are
    # `C5.json` while the prefix is the full name `zone_hetero_external_weather`,
    # so both are resolved before comparing.
    if spec.path is not None and prefix:
        try:
            from beg.config.naming import resolve

            if resolve(spec.path.stem).id != resolve(prefix).id:
                r.append(
                    CheckResult.fail(
                        "config.filename_matches_prefix",
                        f"file '{spec.path.name}' and prefix '{prefix}' name different configurations",
                        **kw,
                    )
                )
        except KeyError as exc:
            r.append(
                CheckResult.fail(
                    "config.filename_matches_prefix",
                    f"file '{spec.path.name}' or prefix '{prefix}' is not a known configuration ({exc})",
                    **kw,
                )
            )

    axes = spec.axes
    if axes is None:
        sev = CheckResult.fail if strict_three_axis else CheckResult.warn
        r.append(sev("config.three_axis_name", f"'{name}' does not follow {{zone|component}}_{{hetero|homo}}_{{weather_variant}}", **kw))
    else:
        r.append(CheckResult.ok("config.three_axis_name", f"level={axes['level']} type={axes['graph_type']} weather={axes['weather']}", **kw))
        expected_type = "homogeneous" if axes["graph_type"] == "homo" else "heterogeneous"
        if spec.graph_type_field != expected_type:
            r.append(CheckResult.fail("config.type_matches_name", f"type='{spec.graph_type_field}' but name says '{axes['graph_type']}'", **kw))

    if spec.config_id is None:
        r.append(CheckResult.warn("config.has_id", "no 'id' (C-number) — output folder ordering falls back to load index", **kw))
    elif not re.fullmatch(r"C\d+", spec.config_id):
        r.append(CheckResult.warn("config.id_format", f"id '{spec.config_id}' is not of the form C<number>", **kw))

    # ---- node / edge structure -------------------------------------------
    node_set = set(spec.node_tables)
    if len(node_set) != len(spec.node_tables):
        r.append(CheckResult.fail("config.node_tables_unique", f"duplicate node tables: {examples([t for t in spec.node_tables if spec.node_tables.count(t) > 1])}", **kw))
    if not spec.node_tables:
        r.append(CheckResult.fail("config.has_nodes", "no node tables", **kw))

    for e in spec.edges:
        for side, cat in (("src_cat", e.src_cat), ("tgt_cat", e.tgt_cat)):
            if cat not in node_set:
                r.append(CheckResult.fail("config.edge_endpoint_declared", f"{e.table}.{side}='{cat}' is not a declared node table", **kw))
        if not e.src_col or not e.tgt_col:
            r.append(CheckResult.fail("config.edge_columns_declared", f"{e.table} lacks src_col/tgt_col", **kw))
        if e.is_self_type and e.src_col == e.tgt_col:
            r.append(CheckResult.fail("config.self_edge_distinct_columns", f"{e.table}: src_col == tgt_col == '{e.src_col}' for a self-type relation", **kw))
        if not e.rel:
            r.append(CheckResult.fail("config.edge_has_rel", f"{e.table} has empty rel", **kw))

    seen_triples: Dict[tuple, str] = {}
    for e in spec.edges:
        key = e.forward
        if key in seen_triples:
            r.append(CheckResult.fail("config.duplicate_relation", f"{e.table} duplicates {seen_triples[key]} ({key})", **kw))
        seen_triples[key] = e.table
    tables = spec.edge_tables
    if len(set(tables)) != len(tables):
        r.append(CheckResult.fail("config.edge_tables_unique", f"duplicate edge_src: {examples([t for t in tables if tables.count(t) > 1])}", **kw))

    # homogeneous / heterogeneous consistency
    if spec.homogeneous:
        if len(node_set) != 1:
            r.append(CheckResult.fail("config.homo_single_node_type", f"homogeneous config declares {len(node_set)} node tables", **kw))
        bad = [e.table for e in spec.edges if not e.is_self_type]
        if bad:
            r.append(CheckResult.fail("config.homo_self_type_edges", f"edges across node types in homogeneous config: {bad}", **kw))
    else:
        if len(node_set) < 2:
            r.append(CheckResult.fail("config.hetero_multiple_node_types", "heterogeneous config declares < 2 node tables", **kw))

    # reachability: every node table participates in at least one edge
    if len(node_set) > 1 or spec.edges:
        touched = {c for e in spec.edges for c in (e.src_cat, e.tgt_cat)}
        isolated = sorted(node_set - touched)
        if isolated:
            r.append(CheckResult.fail("config.node_tables_connected", f"node tables without any edge: {isolated}", **kw))
        else:
            r.append(CheckResult.ok("config.node_tables_connected", "every node table has an edge", **kw))
    # connectivity of the *type graph*
    if len(node_set) > 1:
        adj: Dict[str, Set[str]] = {t: set() for t in node_set}
        for e in spec.edges:
            adj.setdefault(e.src_cat, set()).add(e.tgt_cat)
            adj.setdefault(e.tgt_cat, set()).add(e.src_cat)
        seen: Set[str] = set()
        stack = [spec.primary_node_table]
        while stack:
            t = stack.pop()
            if t in seen:
                continue
            seen.add(t)
            stack.extend(adj.get(t, ()))
        unreachable = sorted(node_set - seen)
        if unreachable:
            r.append(CheckResult.fail("config.type_graph_connected", f"node types unreachable from '{spec.primary_node_table}': {unreachable}", **kw))

    # ---- weather variant --------------------------------------------------
    wv = spec.weather_variant
    weather_nodes = set(spec.weather_node_tables)
    weather_rels = [e for e in spec.edges if e.rel in WEATHER_RELATIONS]
    if wv == "no_weather":
        if weather_nodes or weather_rels:
            r.append(CheckResult.fail("config.weather_variant", f"no_weather config has weather nodes {sorted(weather_nodes)} / relations {[e.table for e in weather_rels]}", **kw))
    elif wv == "external_weather":
        if "weather_city" not in weather_nodes:
            r.append(CheckResult.fail("config.weather_variant", "external_weather config lacks 'weather_city' node table", **kw))
        if not weather_rels:
            r.append(CheckResult.fail("config.weather_variant", "external_weather config has no 'weather_for' relation", **kw))
        bidir = [e.table for e in weather_rels if e.bidirectional]
        if bidir:
            r.append(CheckResult.warn("config.weather_edge_direction", f"weather_for edges are bidirectional ({bidir}); C10-style configs use one-directional weather→component", **kw))
    elif wv == "building_weather":
        if "building_weather" not in weather_nodes:
            r.append(CheckResult.fail("config.weather_variant", "building_weather config lacks 'building_weather' node table", **kw))
        if "weather_city" in weather_nodes:
            r.append(CheckResult.fail("config.weather_variant", "building_weather config must not also have a separate 'weather_city' node", **kw))
    elif wv == "graph_weather":
        if weather_nodes:
            r.append(CheckResult.fail("config.weather_variant", "graph_weather (graph-level feature) config must not declare weather node tables", **kw))
    if wv is not None and not any(x.status.value == "FAIL" and x.name == "config.weather_variant" for x in r):
        r.append(CheckResult.ok("config.weather_variant", f"weather variant '{wv}' consistent with nodes/edges", **kw))

    # ---- SQL transforms ---------------------------------------------------
    if transforms_dir is None:
        r.append(CheckResult.skip("config.sql_transforms", "transforms_dir not provided", **kw))
        return r
    transforms_dir = Path(transforms_dir)
    order = read_primary_order(transforms_dir)
    order_names = set(order) | {Path(o).name for o in order}

    produced: Set[str] = set()
    sql_texts: Dict[str, str] = {}
    missing_files, unregistered = [], []
    for t in spec.sql_transforms:
        if t not in order_names and Path(t).name not in order_names:
            unregistered.append(t)
        p = _resolve_transform_path(transforms_dir, t)
        if p is None:
            missing_files.append(t)
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        sql_texts[t] = txt
        produced |= relations_created_by_sql(txt)
        produced.add(Path(t).stem.lower())  # convention: <name>.sql produces <name>
    if missing_files:
        r.append(CheckResult.fail("config.sql_transform_files_exist", f"missing transform files: {missing_files}", **kw))
    else:
        r.append(CheckResult.ok("config.sql_transform_files_exist", f"{len(spec.sql_transforms)} transform files found", **kw))
    if order and unregistered:
        r.append(CheckResult.fail("config.sql_transforms_in_primary_order", f"not in PRIMARY_ORDER.txt: {unregistered}", **kw))
    elif not order:
        r.append(CheckResult.warn("config.sql_transforms_in_primary_order", "PRIMARY_ORDER.txt not found/empty", **kw))
    dups = [t for t in spec.sql_transforms if spec.sql_transforms.count(t) > 1]
    if dups:
        r.append(CheckResult.warn("config.sql_transforms_unique", f"duplicated transforms: {examples(dups)}", **kw))

    # config order vs PRIMARY_ORDER (informational — PRIMARY_ORDER is authoritative)
    if order:
        pos = {Path(o).name: i for i, o in enumerate(order)}
        idxs = [pos[Path(t).name] for t in spec.sql_transforms if Path(t).name in pos]
        if idxs != sorted(idxs):
            r.append(CheckResult.warn("config.sql_transform_order", "sql_transforms listed in a different order than PRIMARY_ORDER.txt (harmless, but confusing)", **kw))

    # every node/edge table has a producer
    for table in spec.node_tables:
        if table.lower() in produced or table in EXTERNAL_SOURCE_TABLES:
            continue
        r.append(CheckResult.fail("config.node_table_has_producer", f"node table '{table}' is not produced by any listed transform", **kw))
    for e in spec.edges:
        if e.table.lower() not in produced:
            r.append(CheckResult.fail("config.edge_table_has_producer", f"edge table '{e.table}' is not produced by any listed transform", **kw))
            continue
        # src_col / tgt_col must appear in the producing SQL text
        txt = None
        for t, s in sql_texts.items():
            if Path(t).stem.lower() == e.table.lower() or e.table.lower() in relations_created_by_sql(s):
                txt = s
                break
        if txt is not None:
            for col in (e.src_col, e.tgt_col):
                if not re.search(rf"\b{re.escape(col)}\b", txt):
                    r.append(CheckResult.fail("config.edge_column_in_sql", f"{e.table}: column '{col}' not found in producing SQL", **kw))
    return r


# --------------------------------------------------------------------------
# directory-level
# --------------------------------------------------------------------------

def check_configs_dir(
    configs_dir: Path,
    *,
    transforms_dir: Optional[Path] = None,
    names: Optional[Sequence[str]] = None,
    strict_three_axis: bool = True,
) -> List[CheckResult]:
    configs_dir = Path(configs_dir)
    results: List[CheckResult] = []
    from beg.config.loader import discover_config_paths

    paths = list(discover_config_paths(configs_dir, names).values())
    if not paths:
        return [
            CheckResult.fail(
                "config.dir_has_configs",
                f"no configuration files in {configs_dir}; expected C1.json .. C10.json",
                scope=SCOPE,
            )
        ]
    for p in paths:
        results.extend(check_config_file(p, transforms_dir=transforms_dir, strict_three_axis=strict_three_axis))

    specs = load_config_specs(configs_dir, names)
    ids: Dict[str, List[str]] = {}
    prefixes: Dict[str, List[str]] = {}
    for s in specs.values():
        if s.config_id:
            ids.setdefault(s.config_id, []).append(s.name)
        prefixes.setdefault(str(s.raw.get("prefix", "")), []).append(s.name)
    dup_ids = {k: v for k, v in ids.items() if len(v) > 1}
    if dup_ids:
        results.append(CheckResult.fail("config.ids_unique", f"duplicate ids across configs: {dup_ids}", scope=SCOPE))
    else:
        results.append(CheckResult.ok("config.ids_unique", f"{len(ids)} unique ids", scope=SCOPE))
    dup_prefix = {k: v for k, v in prefixes.items() if len(v) > 1}
    if dup_prefix:
        results.append(CheckResult.fail("config.prefixes_unique", f"duplicate prefixes: {dup_prefix}", scope=SCOPE))
    return results


