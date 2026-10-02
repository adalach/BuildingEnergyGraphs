"""
Consistency of the *same building* across different graph configurations.

The whole point of the benchmark is a paired comparison of configurations, so
every configuration must describe the same set of buildings and agree on the
facts they share: number of zones, number of envelope components, building
attributes, total floor area, weather statistics of the city.

Two entry points:

* :func:`check_payloads_consistent` — DataFrame level (``out["data"]``)
* :func:`check_graphs_consistent`   — NetworkX level, graphs of one model_id
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from beg.validation.core import CheckResult, examples
from beg.validation.dataframe_checks import MODEL_ID, feature_columns, flatten_payload
from beg.validation.spec import ConfigSpec, is_identifier_column

SCOPE = "cross"

# node tables that represent the same physical concept at the same granularity
ZONE_TABLES = ("zones_zone", "comp_zone")
BUILDING_TABLES = ("building", "building_weather")
COMPONENT_TABLES = ("comp_element", "comp_aperture")
UNIFIED_COMPONENT_TABLE = "comp_base_element"
WEATHER_TABLES = ("weather_city", "building_weather")


def _counts_per_model(df: pd.DataFrame) -> pd.Series:
    return df.groupby(df[MODEL_ID].astype(str)).size()


def check_payloads_consistent(payloads: Mapping[str, Dict[str, Any]], specs: Mapping[str, ConfigSpec]) -> List[CheckResult]:
    r: List[CheckResult] = []
    flat = {name: flatten_payload(p) for name, p in payloads.items()}
    if len(flat) < 2:
        return r

    # 1) identical model id sets
    id_sets = {name: {str(m) for m in (p["model_ids"] or [])} for name, p in flat.items()}
    for name, p in flat.items():
        if not id_sets[name]:
            prim = specs[name].primary_node_table if name in specs else None
            if prim and prim in p["nodes"]:
                id_sets[name] = set(p["nodes"][prim][MODEL_ID].astype(str))
    ref_name = next(iter(id_sets))
    ref = id_sets[ref_name]
    for name, s in id_sets.items():
        if s != ref:
            r.append(CheckResult.fail("cross.model_ids_identical", f"{name} has {len(s)} models vs {ref_name} {len(ref)}; only_in_{name}={examples(s - ref)} only_in_{ref_name}={examples(ref - s)}", config=name, scope=SCOPE))
    if not any(x.name == "cross.model_ids_identical" for x in r):
        r.append(CheckResult.ok("cross.model_ids_identical", f"all {len(flat)} configs cover the same {len(ref)} models", scope=SCOPE))

    # 2) zone counts agree
    zone_counts: Dict[str, pd.Series] = {}
    for name, p in flat.items():
        for t in ZONE_TABLES:
            if t in p["nodes"] and not p["nodes"][t].empty:
                zone_counts[f"{name}:{t}"] = _counts_per_model(p["nodes"][t])
    r += _compare_series(zone_counts, "cross.zone_count_agrees", "zone count per model")

    # 3) building attributes agree
    bld: Dict[str, pd.DataFrame] = {}
    for name, p in flat.items():
        for t in BUILDING_TABLES:
            if t in p["nodes"] and not p["nodes"][t].empty:
                df = p["nodes"][t]
                bld[f"{name}:{t}"] = df.set_index(df[MODEL_ID].astype(str))
    if len(bld) > 1:
        ref_key = next(iter(bld))
        ref_df = bld[ref_key]
        common_cols = None
        for df in bld.values():
            cols = [c for c in feature_columns(df) if c in ref_df.columns and not c.startswith(("cdd", "hdd", "t_", "rh", "dni", "dhi", "ghi", "precip", "daylight", "wind", "strong", "summer", "winter", "n_summer", "n_winter", "latitude", "longitude", "elevation"))]
            common_cols = cols if common_cols is None else [c for c in common_cols if c in cols]
        for key, df in bld.items():
            if key == ref_key or not common_cols:
                continue
            idx = ref_df.index.intersection(df.index)
            a = ref_df.loc[idx, common_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            b = df.loc[idx, common_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            diff = ~np.isclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True)
            if diff.any():
                cols_bad = [common_cols[j] for j in np.where(diff.any(axis=0))[0]]
                r.append(CheckResult.fail("cross.building_attrs_agree", f"{key} vs {ref_key}: {int(diff.any(axis=1).sum())} models differ in building attributes {cols_bad[:8]}", config=key.split(":")[0], scope=SCOPE))
            else:
                r.append(CheckResult.ok("cross.building_attrs_agree", f"{key} vs {ref_key}: {len(common_cols)} building attributes identical for {len(idx)} models", scope=SCOPE))

    # 4) component counts: hetero (element + aperture) == homogeneous unified
    comp_counts: Dict[str, pd.Series] = {}
    for name, p in flat.items():
        parts = [p["nodes"][t] for t in COMPONENT_TABLES if t in p["nodes"] and not p["nodes"][t].empty]
        if parts and "comp_element" in p["nodes"]:
            comp_counts[f"{name}:element+aperture"] = sum((_counts_per_model(df) for df in parts), pd.Series(dtype=int)).fillna(0).astype(int)
        if UNIFIED_COMPONENT_TABLE in p["nodes"] and not p["nodes"][UNIFIED_COMPONENT_TABLE].empty:
            comp_counts[f"{name}:{UNIFIED_COMPONENT_TABLE}"] = _counts_per_model(p["nodes"][UNIFIED_COMPONENT_TABLE])
    r += _compare_series(comp_counts, "cross.component_count_agrees", "component count per model")

    # 5) total floor area agrees between zone tables
    fa: Dict[str, pd.Series] = {}
    for name, p in flat.items():
        for t in ZONE_TABLES:
            df = p["nodes"].get(t)
            if df is not None and not df.empty and "floor_area" in df.columns:
                fa[f"{name}:{t}"] = pd.to_numeric(df["floor_area"], errors="coerce").groupby(df[MODEL_ID].astype(str)).sum().round(2)
    r += _compare_series(fa, "cross.floor_area_agrees", "total floor area per model", rtol=1e-3)

    # 6) weather statistics agree between weather_city nodes and building_weather nodes
    wx: Dict[str, pd.DataFrame] = {}
    for name, p in flat.items():
        for t in WEATHER_TABLES:
            df = p["nodes"].get(t)
            if df is not None and not df.empty:
                wx[f"{name}:{t}"] = df.set_index(df[MODEL_ID].astype(str))
    if len(wx) > 1:
        ref_key = next(iter(wx))
        ref_df = wx[ref_key]
        for key, df in wx.items():
            if key == ref_key:
                continue
            cols = [c for c in feature_columns(df) if c in ref_df.columns and (c.startswith(("cdd", "hdd", "t_", "rh", "dni", "dhi", "ghi", "precip", "daylight", "wind", "strong", "summer", "winter", "n_summer", "n_winter")) or c in ("latitude", "longitude", "elevation_m"))]
            if not cols:
                continue
            idx = ref_df.index.intersection(df.index)
            a = ref_df.loc[idx, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            b = df.loc[idx, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            diff = ~np.isclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True)
            if diff.any():
                r.append(CheckResult.fail("cross.weather_attrs_agree", f"{key} vs {ref_key}: weather features differ for {int(diff.any(axis=1).sum())} models in {[cols[j] for j in np.where(diff.any(axis=0))[0]][:8]}", scope=SCOPE))
            else:
                r.append(CheckResult.ok("cross.weather_attrs_agree", f"{key} vs {ref_key}: {len(cols)} weather features identical", scope=SCOPE))
    return r


def _compare_series(series: Dict[str, pd.Series], name: str, what: str, *, rtol: float = 0.0) -> List[CheckResult]:
    r: List[CheckResult] = []
    if len(series) < 2:
        return r
    ref_key = next(iter(series))
    ref = series[ref_key]
    for key, s in series.items():
        if key == ref_key:
            continue
        idx = ref.index.intersection(s.index)
        a, b = ref.loc[idx].astype(float), s.loc[idx].astype(float)
        bad = ~np.isclose(a.to_numpy(), b.to_numpy(), rtol=rtol, atol=1e-9)
        k = int(bad.sum())
        if k:
            ex = [(m, float(a[m]), float(b[m])) for m in idx[bad][:5]]
            r.append(CheckResult.fail(name, f"{key} vs {ref_key}: {what} differs for {k}/{len(idx)} models e.g. (model, {ref_key.split(':')[0]}, {key.split(':')[0]}) {ex}", config=key.split(":")[0], scope=SCOPE))
        else:
            r.append(CheckResult.ok(name, f"{key} vs {ref_key}: {what} identical for {len(idx)} models", scope=SCOPE))
    return r


# --------------------------------------------------------------------------
# NetworkX graphs of one model across configs
# --------------------------------------------------------------------------

def _nodes_of(G, types: Sequence[str]) -> list:
    return [n for n, d in G.nodes(data=True) if str(d.get("node_type")) in types]


def check_graphs_consistent(graphs: Mapping[str, Any], *, model_id: Optional[str] = None) -> List[CheckResult]:
    """``graphs`` maps config (full) name → NetworkX graph of the same building."""
    r: List[CheckResult] = []
    if len(graphs) < 2:
        return r
    mids = {str(G.graph.get("model_id")) for G in graphs.values()}
    kw = dict(model_id=model_id or (next(iter(mids)) if len(mids) == 1 else None), scope=SCOPE)
    if len(mids) > 1:
        return [CheckResult.fail("cross.same_model", f"graphs are from different models: {sorted(mids)}", **kw)]

    zone_n = {c: len(_nodes_of(G, ZONE_TABLES)) for c, G in graphs.items() if _nodes_of(G, ZONE_TABLES)}
    if len(set(zone_n.values())) > 1:
        r.append(CheckResult.fail("cross.zone_count_agrees", f"zone count differs across configs: {zone_n}", **kw))
    elif zone_n:
        r.append(CheckResult.ok("cross.zone_count_agrees", f"{next(iter(zone_n.values()))} zones in all {len(zone_n)} zone-bearing configs", **kw))

    comp_n = {}
    for c, G in graphs.items():
        het = len(_nodes_of(G, COMPONENT_TABLES))
        uni = len(_nodes_of(G, (UNIFIED_COMPONENT_TABLE,)))
        if het:
            comp_n[c] = het
        elif uni:
            comp_n[c] = uni
    if len(set(comp_n.values())) > 1:
        r.append(CheckResult.fail("cross.component_count_agrees", f"component count differs: {comp_n}", **kw))
    elif comp_n:
        r.append(CheckResult.ok("cross.component_count_agrees", f"{next(iter(comp_n.values()))} components in all {len(comp_n)} component configs", **kw))

    fa = {}
    for c, G in graphs.items():
        zs = _nodes_of(G, ZONE_TABLES)
        if zs:
            fa[c] = round(sum(float(G.nodes[z].get("floor_area", 0) or 0) for z in zs), 2)
    if fa and (max(fa.values()) - min(fa.values())) > 1e-3 * max(1.0, max(fa.values())):
        r.append(CheckResult.fail("cross.floor_area_agrees", f"total floor area differs: {fa}", **kw))

    bld = {}
    for c, G in graphs.items():
        bs = _nodes_of(G, BUILDING_TABLES)
        if len(bs) == 1:
            d = G.nodes[bs[0]]
            bld[c] = {k: v for k, v in d.items() if not is_identifier_column(k) and k not in ("node_type", "node_id") and isinstance(v, (int, float)) and not isinstance(v, bool) and not k.startswith(("cdd", "hdd", "t_", "rh", "dni", "dhi", "ghi", "precip", "daylight", "wind", "strong", "summer", "winter", "n_summer", "n_winter", "latitude", "longitude", "elevation"))}
    if len(bld) > 1:
        ref_c = next(iter(bld))
        for c, attrs in bld.items():
            if c == ref_c:
                continue
            common = set(attrs) & set(bld[ref_c])
            diff = [k for k in common if not np.isclose(attrs[k], bld[ref_c][k], rtol=1e-6, atol=1e-9)]
            if diff:
                r.append(CheckResult.fail("cross.building_attrs_agree", f"{c} vs {ref_c}: building attributes differ {diff[:8]}", **kw))
        if not any(x.name == "cross.building_attrs_agree" for x in r):
            r.append(CheckResult.ok("cross.building_attrs_agree", f"building attributes identical across {len(bld)} configs", **kw))

    # weather: separate node / building_weather node / graph-level dict must agree
    wx: Dict[str, Dict[str, float]] = {}
    for c, G in graphs.items():
        for _, d in G.nodes(data=True):
            if str(d.get("node_type")) in WEATHER_TABLES:
                wx[c] = {k: v for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and (k.startswith(("cdd", "hdd", "t_", "rh", "dni", "dhi", "ghi", "precip", "daylight", "wind", "strong", "summer", "winter", "n_summer", "n_winter")) or k in ("latitude", "longitude", "elevation_m"))}
        gv = G.graph.get("graph_feature_values") or G.graph.get("weather_city")
        if isinstance(gv, dict):
            wx[c] = {k: v for k, v in gv.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if len(wx) > 1:
        ref_c = next(iter(wx))
        for c, attrs in wx.items():
            if c == ref_c:
                continue
            common = set(attrs) & set(wx[ref_c])
            diff = [k for k in common if not np.isclose(attrs[k], wx[ref_c][k], rtol=1e-6, atol=1e-6)]
            if diff:
                r.append(CheckResult.fail("cross.weather_attrs_agree", f"{c} vs {ref_c}: weather features differ {diff[:8]}", **kw))
            elif not common:
                r.append(CheckResult.warn("cross.weather_attrs_agree", f"{c} vs {ref_c}: no common weather feature names", **kw))
        if not any(x.name == "cross.weather_attrs_agree" for x in r):
            r.append(CheckResult.ok("cross.weather_attrs_agree", f"weather features identical across {len(wx)} weather-bearing configs", **kw))
    return r
