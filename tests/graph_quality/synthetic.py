"""
A tiny synthetic component-level dataset shaped like ``C10_component_hetero_external_weather``.

Used by ``test_selfcheck_synthetic.py`` to prove that every check detects the
defect it is meant to detect (and stays quiet on clean data).  It follows the
real config JSON (node tables, edge tables, src/tgt columns) so the
NetworkX builder in ``beg.build.networkx_graphs`` can consume it unchanged.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from beg.validation.spec import ConfigSpec, config_spec_from_dict

C10_CONFIG: Dict[str, Any] = {
    "prefix": "component_hetero_external_weather",
    "id": "C10",
    "weather": "separate node",
    "type": "heterogeneous",
    "sql_transforms": ["building.sql", "comp_element.sql", "comp_zone.sql", "comp_aperture.sql",
                       "edge_comp_zone_building.sql", "edge_comp_element_zone.sql", "edge_comp_aperture_element.sql",
                       "edge_comp_element_adjacency.sql", "edge_comp_env_weather.sql", "edge_comp_aperture_env_weather.sql"],
    "nodes": [{"table": "building"}, {"table": "comp_zone"}, {"table": "comp_element"}, {"table": "comp_aperture"}, {"table": "weather_city"}],
    "edges": [
        {"edge_src": "edge_comp_zone_building", "src_cat": "comp_zone", "src_col": "zone_instance_name", "tgt_cat": "building", "tgt_col": "building_instance_name", "rel": "belongs_to", "bidirectional": True},
        {"edge_src": "edge_comp_element_zone", "src_cat": "comp_element", "src_col": "comp_element_name", "tgt_cat": "comp_zone", "tgt_col": "zone_name", "rel": "belongs_to", "bidirectional": True},
        {"edge_src": "edge_comp_aperture_element", "src_cat": "comp_aperture", "src_col": "aperture_name", "tgt_cat": "comp_element", "tgt_col": "element_name", "rel": "part_of", "bidirectional": True},
        {"edge_src": "edge_comp_element_adjacency", "src_cat": "comp_element", "src_col": "element_1_name", "tgt_cat": "comp_element", "tgt_col": "element_2_name", "rel": "adjacent_surface", "bidirectional": True},
        {"edge_src": "edge_comp_env_weather", "src_cat": "weather_city", "src_col": "city_key", "tgt_cat": "comp_element", "tgt_col": "element_name", "rel": "weather_for", "bidirectional": False},
        {"edge_src": "edge_comp_aperture_env_weather", "src_cat": "weather_city", "src_col": "city_key", "tgt_cat": "comp_aperture", "tgt_col": "aperture_name", "rel": "weather_for", "bidirectional": False},
    ],
}


def c10_spec() -> ConfigSpec:
    return config_spec_from_dict(copy.deepcopy(C10_CONFIG))


VINTAGES = ["pre_1980", "1980_2004", "2007"]
FUNCTIONS = ["largeoffice", "retail", "hospital"]
SHADE_COLS = ["shade_count", "shade_ray_count", "shade_min_dist", "shade_mean_dist", "shade_ray_dir_mean_sin", "shade_ray_dir_mean_cos"]


def _shade(rng: np.random.Generator, active: bool) -> Dict[str, Any]:
    if not active:
        return {c: 0 for c in SHADE_COLS}
    return {"shade_count": int(rng.integers(1, 5)), "shade_ray_count": int(rng.integers(5, 50)), "shade_min_dist": float(rng.uniform(1, 5)),
            "shade_mean_dist": float(rng.uniform(5, 20)), "shade_ray_dir_mean_sin": float(rng.uniform(-1, 1)), "shade_ray_dir_mean_cos": float(rng.uniform(-1, 1))}


def make_synthetic(n_models: int = 3, *, seed: int = 0, city: str = "testcity") -> Tuple[ConfigSpec, Dict[str, Any]]:
    """Return (spec, payload) with ``payload = {"model_ids", "nodes", "edges"}``."""
    rng = np.random.default_rng(seed)
    B, Z, E, A, W = [], [], [], [], []
    ezb, eez, eae, eadj, eew, eaw = [], [], [], [], [], []
    model_ids = [f"{city}_{i:04d}" for i in range(1, n_models + 1)]
    for mi, mid in enumerate(model_ids):
        vint, func = VINTAGES[mi % 3], FUNCTIONS[mi % 3]
        b = {"model_id": mid, "instance_name": "Building", "id": 1000 + mi, "return_air_fraction": 0.0, "fraction_radiant": 0.6, "fraction_visible": 0.2,
             "people_per_floor_area": 0.05, "vintage": vint, "primary_function": func}
        b.update({f"vintage_{v}": int(v == vint) for v in VINTAGES})
        b.update({f"primary_function_{f}": int(f == func) for f in FUNCTIONS})
        b.update(_shade(rng, mi % 2 == 0))
        B.append(b)
        W.append({"model_id": mid, "instance_name": city, "city_key": city, "latitude": 48.1, "longitude": 11.5, "elevation_m": 520.0,
                  "hdd18_01": 400.0, "cdd26_07": 12.0, "t_mean_c_01": -1.2, "rh_mean_pct_07": 65.0, "daylight_hours_06": 500.0})
        n_zones = 2
        zone_names = [f"{mid}_zone{z}_L{z}" for z in range(n_zones)]
        for z, zn in enumerate(zone_names):
            Z.append({"model_id": mid, "instance_name": zn, "id": 5000 + 10 * mi + z, "floor_area": 80.0 + 5 * z, "ceiling_height": 3.0, "floor_level": z,
                      "is_ground_floor": int(z == 0), "is_top_floor": int(z == n_zones - 1), "has_hvac": 1, "is_conditioned": 1, **_shade(rng, mi % 2 == 0)})
            ezb.append({"model_id": mid, "zone_instance_name": zn, "building_instance_name": "Building"})
            base = {"model_id": mid, "zone_name": zn, "construction_name": "Generic", "is_wall": 0, "is_floor": 0, "is_roof": 0, "is_ceiling": 0,
                    "is_outdoors": 0, "is_surface": 0, "is_zone": 0, "is_ground": 0, "is_adiabatic": 0, "sun_exposure": 0, "wind_exposure": 0,
                    "total_thickness": 0.3, "r_value_layers": 2.0, "u_value_total": 0.45, "solar_transmittance": 0.0, "visible_transmittance": 0.0,
                    "front_emissivity": 0.9, "back_emissivity": 0.9, "solar_absorptance": 0.7, "density": 2000.0, "specific_heat": 900.0, **_shade(rng, False)}
            # floor
            fl = dict(base, instance_name=f"{mid}_z{z}_floor", id=rng.integers(1, 1e6), is_floor=1, area=80.0 + 5 * z)
            if z == 0:
                fl.update(is_ground=1)
                # ground-contact floors are weather-exposed by edge_comp_env_weather.sql
                eew.append({"model_id": mid, "city_key": city, "element_name": fl["instance_name"], "is_ext_wall": 0, "is_ext_roof": 0, "surface_area": fl["area"]})
            else:
                fl.update(is_surface=1)
            E.append(fl)
            # roof / ceiling
            rf = dict(base, instance_name=f"{mid}_z{z}_roof", id=rng.integers(1, 1e6), area=80.0 + 5 * z)
            if z == n_zones - 1:
                rf.update(is_roof=1, is_outdoors=1, sun_exposure=1, wind_exposure=1)
            else:
                rf.update(is_ceiling=1, is_surface=1)
            E.append(rf)
            # two exterior walls with a window each
            for w in range(2):
                wn = f"{mid}_z{z}_wall_{w}"
                E.append(dict(base, instance_name=wn, id=rng.integers(1, 1e6), is_wall=1, is_outdoors=1, sun_exposure=1, wind_exposure=1, area=30.0))
                an = f"{wn}_Glz0"
                A.append({"model_id": mid, "instance_name": an, "id": rng.integers(1, 1e6), "building_surface_name": wn, "construction_name": "Glazing",
                          "is_window": 1, "is_door": 0, "is_glass_door": 0, "is_tubular_daylight": 0, "area": 6.0, "total_thickness": 0.006,
                          "r_value_layers": 0.0, "u_value_total": 2.8, "solar_transmittance": 0.6, "visible_transmittance": 0.7, "front_emissivity": 0.84,
                          "back_emissivity": 0.84, "solar_absorptance": 0.1, "density": 2500.0, "specific_heat": 750.0, **_shade(rng, False)})
                eae.append({"model_id": mid, "aperture_name": an, "element_name": wn})
                eew.append({"model_id": mid, "city_key": city, "element_name": wn, "is_ext_wall": 1, "is_ext_roof": 0, "surface_area": 30.0})
                eaw.append({"model_id": mid, "city_key": city, "aperture_name": an, "is_parent_ext_wall": 1, "is_parent_ext_roof": 0, "aperture_area": 6.0})
            if z == n_zones - 1:
                eew.append({"model_id": mid, "city_key": city, "element_name": f"{mid}_z{z}_roof", "is_ext_wall": 0, "is_ext_roof": 1, "surface_area": 80.0 + 5 * z})
        for e in E:
            if e["model_id"] == mid:
                eez.append({"model_id": mid, "comp_element_name": e["instance_name"], "zone_name": e["zone_name"]})
        # adjacency: floor of zone1 ↔ ceiling of zone0
        eadj.append({"model_id": mid, "element_1_name": f"{mid}_z0_roof", "element_2_name": f"{mid}_z1_floor", "shared_area": 80.0})
    nodes = {"building": pd.DataFrame(B), "comp_zone": pd.DataFrame(Z), "comp_element": pd.DataFrame(E), "comp_aperture": pd.DataFrame(A), "weather_city": pd.DataFrame(W)}
    edges = {"edge_comp_zone_building": pd.DataFrame(ezb), "edge_comp_element_zone": pd.DataFrame(eez), "edge_comp_aperture_element": pd.DataFrame(eae),
             "edge_comp_element_adjacency": pd.DataFrame(eadj), "edge_comp_env_weather": pd.DataFrame(eew), "edge_comp_aperture_env_weather": pd.DataFrame(eaw)}
    for df in list(nodes.values()) + list(edges.values()):
        if "id" in df.columns:
            df["id"] = df["id"].astype(int)
    return c10_spec(), {"model_ids": model_ids, "nodes": nodes, "edges": edges}


def build_graph(spec: ConfigSpec, payload: Dict[str, Any], model_id: str):
    """Build the NetworkX graph of one model with the project's real builder."""
    from beg.build.networkx_graphs import build_nx_graph_for_model

    nodes = {t: df[df["model_id"] == model_id].copy() for t, df in payload["nodes"].items()}
    edges = {t: df[df["model_id"] == model_id].copy() for t, df in payload["edges"].items()}
    built = build_nx_graph_for_model(spec.full_name, model_id, nodes, edges, spec.node_tables, edge_specs={e.table: e.raw for e in spec.edges})
    return built.G
