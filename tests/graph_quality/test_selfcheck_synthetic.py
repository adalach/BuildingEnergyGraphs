"""
Self-tests of the validation library on synthetic data.

Every test plants one defect into an otherwise clean dataset and asserts that
exactly the intended check reports it.  These tests need no database and no
generated data, so they always run — they protect the checks themselves.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from beg.validation.config_checks import check_config_spec
from beg.validation.core import Report, Status
from beg.validation.cross_config_checks import check_graphs_consistent, check_payloads_consistent
from beg.validation.dataframe_checks import run_dataframe_checks
from beg.validation.label_checks import check_labels
from beg.validation.nx_checks import check_graph_collection, run_graph_checks
from beg.validation.spec import config_spec_from_dict

from .synthetic import C10_CONFIG, build_graph, c10_spec, make_synthetic

pytestmark = pytest.mark.graph_quality


def failed(results) -> set:
    return {r.name for r in results if r.status == Status.FAIL}


def warned(results) -> set:
    return {r.name for r in results if r.status == Status.WARN}


def clone(payload):
    return {"model_ids": list(payload["model_ids"]), "nodes": {k: v.copy() for k, v in payload["nodes"].items()}, "edges": {k: v.copy() for k, v in payload["edges"].items()}}


@pytest.fixture(scope="module")
def clean():
    spec, payload = make_synthetic(n_models=4)
    return spec, payload


# --------------------------------------------------------------------------- clean data
def test_clean_dataframes_pass(clean):
    spec, payload = clean
    res = run_dataframe_checks(spec, payload, min_models_for_variance=1000)
    assert failed(res) == set(), Report(res).failures


def test_clean_graphs_pass(clean):
    spec, payload = clean
    graphs = [build_graph(spec, payload, m) for m in payload["model_ids"]]
    res = []
    for G, m in zip(graphs, payload["model_ids"]):
        res += run_graph_checks(G, spec, expected_model_id=m)
    res += check_graph_collection(graphs, spec, min_graphs_for_variance=1000)
    assert failed(res) == set(), Report(res).failures
    # the important positive results are actually produced
    names = {r.name for r in res}
    assert {"nx.node_types_present", "nx.edge_types_declared", "nx.weakly_connected", "nx.child_has_parent", "nx.weather_edges_match_exposure", "nx.collection_attr_schema_stable"} <= names


def test_clean_config_passes():
    res = check_config_spec(c10_spec())
    assert failed(res) == set()


# --------------------------------------------------------------------------- config defects
def test_config_weather_variant_mismatch():
    cfg = copy.deepcopy(C10_CONFIG)
    cfg["prefix"] = "component_hetero_no_weather"
    spec = config_spec_from_dict(cfg, name="component_hetero_no_weather")
    assert "config.weather_variant" in failed(check_config_spec(spec))


def test_config_edge_endpoint_not_declared():
    cfg = copy.deepcopy(C10_CONFIG)
    cfg["edges"][0]["tgt_cat"] = "typo_table"
    assert "config.edge_endpoint_declared" in failed(check_config_spec(config_spec_from_dict(cfg)))


def test_config_isolated_node_table():
    cfg = copy.deepcopy(C10_CONFIG)
    cfg["nodes"].append({"table": "orphan_table"})
    res = check_config_spec(config_spec_from_dict(cfg))
    assert {"config.node_tables_connected", "config.type_graph_connected"} <= failed(res)


def test_config_homo_with_two_node_types():
    cfg = copy.deepcopy(C10_CONFIG)
    cfg["type"] = "homogeneous"
    cfg["prefix"] = "component_homo_external_weather"
    res = check_config_spec(config_spec_from_dict(cfg, name="component_homo_external_weather"))
    assert {"config.homo_single_node_type", "config.homo_self_type_edges"} <= failed(res)


def test_config_sql_transform_missing(tmp_path):
    (tmp_path / "PRIMARY_ORDER.txt").write_text("building.sql\n")
    (tmp_path / "building.sql").write_text("CREATE TABLE building AS SELECT 1;")
    res = check_config_spec(c10_spec(), transforms_dir=tmp_path)
    assert {"config.sql_transform_files_exist", "config.sql_transforms_in_primary_order"} <= failed(res)
    assert "config.edge_table_has_producer" in failed(res)


# --------------------------------------------------------------------------- dataframe defects
def test_df_dangling_edge_and_missing_node(clean):
    spec, payload = clean
    p = clone(payload)
    el = p["nodes"]["comp_element"]
    victim = el.iloc[0]["instance_name"]
    p["nodes"]["comp_element"] = el[el["instance_name"] != victim]
    res = run_dataframe_checks(spec, p, min_models_for_variance=1000)
    assert "df.edge_endpoints_resolve" in failed(res)
    G = build_graph(spec, p, payload["model_ids"][0])
    assert "nx.no_missing_from_nodes" in failed(run_graph_checks(G, spec))


def test_df_missing_node_table(clean):
    spec, payload = clean
    p = clone(payload)
    del p["nodes"]["comp_aperture"]
    assert "df.node_tables_present" in failed(run_dataframe_checks(spec, p, min_models_for_variance=1000))


def test_df_null_feature_value(clean):
    spec, payload = clean
    p = clone(payload)
    p["nodes"]["comp_element"].loc[p["nodes"]["comp_element"].index[:2], "area"] = np.nan
    res = run_dataframe_checks(spec, p, min_models_for_variance=1000)
    assert "df.column_partially_missing" in failed(res)
    G = build_graph(spec, p, payload["model_ids"][0])
    assert "nx.node_attrs_complete" in failed(run_graph_checks(G, spec))


def test_df_all_null_column_is_uncomputable(clean):
    spec, payload = clean
    p = clone(payload)
    p["nodes"]["comp_zone"]["ceiling_height"] = np.nan
    assert "df.column_all_missing" in failed(run_dataframe_checks(spec, p, min_models_for_variance=1000))


def test_df_value_domain_violation(clean):
    spec, payload = clean
    p = clone(payload)
    p["nodes"]["comp_element"].loc[p["nodes"]["comp_element"].index[0], "u_value_total"] = -3.0
    p["nodes"]["comp_aperture"].loc[p["nodes"]["comp_aperture"].index[0], "solar_transmittance"] = 1.7
    res = run_dataframe_checks(spec, p, min_models_for_variance=1000)
    viol = [r for r in res if r.name == "df.value_domain" and r.status == Status.FAIL]
    assert {v.details["column"] for v in viol} >= {"u_value_total", "solar_transmittance"}
    G = build_graph(spec, p, payload["model_ids"][0])
    assert "nx.node_attr_domain" in failed(run_graph_checks(G, spec))


def test_df_one_hot_not_exclusive(clean):
    spec, payload = clean
    p = clone(payload)
    b = p["nodes"]["building"]
    b.loc[b.index[0], ["vintage_pre_1980", "vintage_2007"]] = 1   # two categories active
    b.loc[b.index[1], "vintage"] = "2016"                         # label disagrees with active one-hot
    res = run_dataframe_checks(spec, p, min_models_for_variance=1000)
    assert "df.one_hot_exclusive" in failed(res)
    assert "df.one_hot_matches_label" in failed(res)
    G = build_graph(spec, p, payload["model_ids"][0])
    assert "nx.node_one_hot_exclusive" in failed(run_graph_checks(G, spec))


def test_df_boundary_flags_exclusive(clean):
    spec, payload = clean
    p = clone(payload)
    el = p["nodes"]["comp_element"]
    el.loc[el.index[0], ["is_outdoors", "is_ground"]] = 1
    assert "df.one_hot_exclusive" in failed(run_dataframe_checks(spec, p, min_models_for_variance=1000))


def test_df_duplicate_node_key_within_and_across_tables(clean):
    spec, payload = clean
    p = clone(payload)
    z = p["nodes"]["comp_zone"]
    p["nodes"]["comp_zone"] = pd.concat([z, z.iloc[[0]]], ignore_index=True)
    a = p["nodes"]["comp_aperture"].copy()
    a.loc[a.index[0], "instance_name"] = p["nodes"]["comp_element"].iloc[0]["instance_name"]
    p["nodes"]["comp_aperture"] = a
    res = run_dataframe_checks(spec, p, min_models_for_variance=1000)
    assert {"df.node_key_unique", "df.node_key_unique_across_tables"} <= failed(res)


def test_df_singleton_violation(clean):
    spec, payload = clean
    p = clone(payload)
    b = p["nodes"]["building"]
    extra = b.iloc[[0]].copy()
    extra["instance_name"] = "Building_2"
    p["nodes"]["building"] = pd.concat([b, extra], ignore_index=True)
    res = run_dataframe_checks(spec, p, min_models_for_variance=1000)
    assert "df.singleton_node_table" in failed(res)
    G = build_graph(spec, p, payload["model_ids"][0])
    assert "nx.singleton_node_type" in failed(run_graph_checks(G, spec))


def test_df_child_with_two_parents_and_orphan(clean):
    spec, payload = clean
    p = clone(payload)
    ez = p["nodes"]["comp_zone"]
    e = p["edges"]["edge_comp_element_zone"]
    # element 0 gets a second zone; element 1 loses its zone edge
    dup = e.iloc[[0]].copy()
    dup["zone_name"] = ez.iloc[1]["instance_name"]
    e = pd.concat([e.drop(e.index[1]), dup], ignore_index=True)
    p["edges"]["edge_comp_element_zone"] = e
    res = run_dataframe_checks(spec, p, min_models_for_variance=1000)
    assert {"df.edge_child_single_parent", "df.edge_child_has_parent"} <= failed(res)
    G = build_graph(spec, p, payload["model_ids"][0])
    assert {"nx.child_single_parent", "nx.child_has_parent"} <= failed(run_graph_checks(G, spec))


def test_df_model_missing_from_table(clean):
    spec, payload = clean
    p = clone(payload)
    m = payload["model_ids"][-1]
    p["nodes"]["comp_zone"] = p["nodes"]["comp_zone"][p["nodes"]["comp_zone"]["model_id"] != m]
    assert "df.node_table_model_coverage" in failed(run_dataframe_checks(spec, p, min_models_for_variance=1000))


def test_df_weather_edge_missing_for_exposed_element(clean):
    spec, payload = clean
    p = clone(payload)
    w = p["edges"]["edge_comp_env_weather"]
    p["edges"]["edge_comp_env_weather"] = w.iloc[1:]
    res = run_dataframe_checks(spec, p, min_models_for_variance=1000)
    assert "df.weather_edges_match_exposure" in failed(res)
    G = build_graph(spec, p, payload["model_ids"][0])
    assert "nx.weather_edges_match_exposure" in failed(run_graph_checks(G, spec))


def test_df_weather_edge_to_interior_element(clean):
    spec, payload = clean
    p = clone(payload)
    w = p["edges"]["edge_comp_env_weather"]
    m = payload["model_ids"][0]
    extra = pd.DataFrame([{"model_id": m, "city_key": "testcity", "element_name": f"{m}_z1_floor", "is_ext_wall": 0, "is_ext_roof": 0, "surface_area": 80.0}])  # interior floor
    p["edges"]["edge_comp_env_weather"] = pd.concat([w, extra], ignore_index=True)
    assert "df.weather_edges_match_exposure" in failed(run_dataframe_checks(spec, p, min_models_for_variance=1000))


def test_df_ground_floor_with_sun_exposure(clean):
    spec, payload = clean
    p = clone(payload)
    el = p["nodes"]["comp_element"]
    idx = el.index[el["is_ground"] == 1][0]
    el.loc[idx, "sun_exposure"] = 1
    assert "df.ground_no_exposure" in failed(run_dataframe_checks(spec, p, min_models_for_variance=1000))


def test_df_aperture_larger_than_parent(clean):
    spec, payload = clean
    p = clone(payload)
    a = p["nodes"]["comp_aperture"]
    a.loc[a.index[0], "area"] = 500.0
    assert "df.aperture_within_parent" in failed(run_dataframe_checks(spec, p, min_models_for_variance=1000))


def test_df_floor_flags_inconsistent(clean):
    spec, payload = clean
    p = clone(payload)
    z = p["nodes"]["comp_zone"]
    z.loc[z.index[0], "is_top_floor"] = 1  # ground-floor zone flagged as top floor while a higher zone exists
    assert "df.floor_flags_consistent" in failed(run_dataframe_checks(spec, p, min_models_for_variance=1000))


def test_df_all_zero_shading_dataset(clean):
    spec, payload = clean
    p = clone(payload)
    for t in ("building", "comp_zone"):
        p["nodes"][t]["shade_count"] = 0
        p["nodes"][t]["shade_ray_count"] = 0
    res = run_dataframe_checks(spec, p, min_models_for_variance=1)
    assert "df.shading_not_all_zero" in failed(res)


def test_df_constant_feature_warns(clean):
    spec, payload = clean
    res = run_dataframe_checks(spec, payload, min_models_for_variance=1)
    assert "df.constant_feature" in warned(res)  # synthetic data has many constants by construction


def test_df_weather_city_mismatch(clean):
    spec, payload = clean
    p = clone(payload)
    w = p["nodes"]["weather_city"]
    w.loc[w.index[0], "city_key"] = "othercity"
    p["nodes"]["weather_city"] = w
    res = run_dataframe_checks(spec, p, min_models_for_variance=1000)
    assert "df.weather_city_matches_model" in failed(res)


def test_df_self_loop_and_double_stored_adjacency(clean):
    spec, payload = clean
    p = clone(payload)
    adj = p["edges"]["edge_comp_element_adjacency"]
    m = payload["model_ids"][0]
    rev = adj.iloc[[0]].rename(columns={"element_1_name": "element_2_name", "element_2_name": "element_1_name"})
    loop = pd.DataFrame([{"model_id": m, "element_1_name": f"{m}_z0_floor", "element_2_name": f"{m}_z0_floor", "shared_area": 1.0}])
    p["edges"]["edge_comp_element_adjacency"] = pd.concat([adj, rev, loop], ignore_index=True)
    res = run_dataframe_checks(spec, p, min_models_for_variance=1000)
    assert {"df.edge_no_self_loops", "df.edge_not_double_stored"} <= failed(res)


# --------------------------------------------------------------------------- graph-only defects
def test_nx_disconnected_graph(clean):
    spec, payload = clean
    G = build_graph(spec, payload, payload["model_ids"][0])
    zone_edges = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if d.get("rel") == "belongs_to" and G.nodes[u]["node_type"] in ("comp_zone", "building") and G.nodes[v]["node_type"] in ("comp_zone", "building")]
    G.remove_edges_from(zone_edges)
    res = run_graph_checks(G, spec)
    assert {"nx.weakly_connected", "nx.no_isolated_nodes"} <= failed(res)


def test_nx_unexpected_edge_type_and_reverse_of_one_directional(clean):
    spec, payload = clean
    G = build_graph(spec, payload, payload["model_ids"][0])
    weather = next(n for n, d in G.nodes(data=True) if d["node_type"] == "weather_city")
    zone = next(n for n, d in G.nodes(data=True) if d["node_type"] == "comp_zone")
    element = next(v for u, v, d in G.edges(data=True) if d.get("rel") == "weather_for")
    G.add_edge(weather, zone, rel="weather_for", edge_type="weather_for", is_reverse=0)   # not in config
    G.add_edge(element, weather, rel="weather_for", edge_type="weather_for", is_reverse=0)  # reverse of one-directional
    res = run_graph_checks(G, spec)
    assert {"nx.edge_types_declared", "nx.one_directional_respected"} <= failed(res)


def test_nx_bidirectional_asymmetry(clean):
    spec, payload = clean
    G = build_graph(spec, payload, payload["model_ids"][0])
    u, v, k = next((u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if d.get("rel") == "part_of" and d.get("is_reverse") == 1)
    G.remove_edge(u, v, k)
    assert "nx.bidirectional_symmetric" in failed(run_graph_checks(G, spec))


def test_nx_node_id_not_contiguous(clean):
    spec, payload = clean
    G = build_graph(spec, payload, payload["model_ids"][0])
    n0 = next(iter(G.nodes))
    G.nodes[n0]["node_id"] = 999
    assert "nx.node_id_contiguous" in failed(run_graph_checks(G, spec))


def test_nx_attr_none_and_nan(clean):
    spec, payload = clean
    G = build_graph(spec, payload, payload["model_ids"][0])
    el = [n for n, d in G.nodes(data=True) if d["node_type"] == "comp_element"]
    G.nodes[el[0]]["area"] = float("nan")
    G.nodes[el[1]]["density"] = None
    res = run_graph_checks(G, spec)
    assert {"nx.node_attr_finite", "nx.node_attr_types"} <= failed(res)


def test_nx_collection_schema_unstable(clean):
    spec, payload = clean
    graphs = [build_graph(spec, payload, m) for m in payload["model_ids"]]
    for _, d in graphs[0].nodes(data=True):
        if d["node_type"] == "comp_zone":
            del d["ceiling_height"]
    assert "nx.collection_attr_schema_stable" in failed(check_graph_collection(graphs, spec, min_graphs_for_variance=1000))


def test_nx_graph_level_weather_none_detected(clean):
    spec, payload = clean
    G = build_graph(spec, payload, payload["model_ids"][0])
    G.graph["derived_from"] = "C6_component_homo_no_weather"
    G.graph["graph_feature_cols"] = ["t_mean_c_01", "summer_wind_speed_mean_ms"]
    G.graph["graph_feature_values"] = {"t_mean_c_01": 3.2, "summer_wind_speed_mean_ms": None}
    res = run_graph_checks(G, spec, graph_level_weather=True, graph_weather_dim=3)
    assert {"nx.graph_weather_finite", "nx.graph_weather_dim"} <= failed(res)


def test_nx_model_id_mismatch(clean):
    spec, payload = clean
    G = build_graph(spec, payload, payload["model_ids"][0])
    assert "nx.graph_model_id_matches" in failed(run_graph_checks(G, spec, expected_model_id="somebody_else_0001"))


# --------------------------------------------------------------------------- cross-config
def test_cross_zone_count_disagrees(clean):
    spec, payload = clean
    G1 = build_graph(spec, payload, payload["model_ids"][0])
    G2 = build_graph(spec, payload, payload["model_ids"][0])
    z = next(n for n, d in G2.nodes(data=True) if d["node_type"] == "comp_zone")
    G2.remove_node(z)
    res = check_graphs_consistent({"A": G1, "B": G2})
    assert "cross.zone_count_agrees" in failed(res)


def test_cross_payload_model_sets_and_building_attrs(clean):
    spec, payload = clean
    p2 = clone(payload)
    p2["model_ids"] = p2["model_ids"][:-1]
    p2["nodes"]["building"].loc[p2["nodes"]["building"].index[0], "people_per_floor_area"] = 0.9
    res = check_payloads_consistent({"a": payload, "b": p2}, {"a": spec, "b": spec})
    assert {"cross.model_ids_identical", "cross.building_attrs_agree"} <= failed(res)


# --------------------------------------------------------------------------- labels
def test_labels_checks():
    df = pd.DataFrame({"building_id": ["a_0001", "a_0002", "a_0002", "a_0003", "a_0004"], "eui_kwh_per_m2": [150.0, np.nan, 120.0, -5.0, 200.0]})
    res = check_labels(df, dataset_model_ids=["a_0001", "a_0002", "a_0003", "a_0004", "a_0005", "A_0001"])
    assert {"labels.ids_unique", "labels.finite", "labels.positive", "labels.cover_dataset"} <= failed(res)
    assert "labels.id_normalisation" in warned(res)
    clean_df = pd.DataFrame({"building_id": [f"a_{i:04d}" for i in range(50)], "eui_kwh_per_m2": np.linspace(100, 400, 50)})
    assert failed(check_labels(clean_df, dataset_model_ids=clean_df["building_id"])) == set()
