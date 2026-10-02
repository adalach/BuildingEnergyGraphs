"""
Self-tests of the PyG checks on hand-built ``Data`` / ``HeteroData`` objects
and a miniature ``training_ready`` payload.  Skipped when torch / PyG are absent.
"""

from __future__ import annotations

import math

import pytest

from beg.validation.core import Status
from beg.validation.pyg_checks import (
    _sha256_ids,
    check_graph,
    check_graph_collection,
    check_training_ready_alignment,
    check_training_ready_payload,
    torch_available,
)

from .synthetic import c10_spec

pytestmark = pytest.mark.graph_quality

if not torch_available():  # pragma: no cover
    pytest.skip("torch / torch_geometric not available", allow_module_level=True)

import torch  # noqa: E402
from torch_geometric.data import Data, HeteroData  # noqa: E402


def failed(results) -> set:
    return {r.name for r in results if r.status == Status.FAIL}


def warned(results) -> set:
    return {r.name for r in results if r.status == Status.WARN}


def _bidir(pairs):
    src = [a for a, b in pairs] + [b for a, b in pairs]
    tgt = [b for a, b in pairs] + [a for a, b in pairs]
    return torch.tensor([src, tgt], dtype=torch.long)


def make_hetero(seed: int = 0) -> HeteroData:
    g = torch.Generator().manual_seed(seed)
    hd = HeteroData()
    hd.model_id = f"city_{seed:04d}"
    hd["building"].x = torch.rand((1, 6), generator=g)
    hd["comp_zone"].x = torch.rand((2, 5), generator=g)
    hd["comp_element"].x = torch.rand((6, 12), generator=g)
    hd["comp_aperture"].x = torch.rand((2, 9), generator=g)
    hd["weather_city"].x = torch.rand((1, 20), generator=g)
    # zone -> building (belongs_to) + reverse store
    hd["comp_zone", "belongs_to", "building"].edge_index = torch.tensor([[0, 1], [0, 0]])
    hd["building", "belongs_to", "comp_zone"].edge_index = torch.tensor([[0, 0], [0, 1]])
    # element -> zone
    hd["comp_element", "belongs_to", "comp_zone"].edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 1, 1]])
    hd["comp_zone", "belongs_to", "comp_element"].edge_index = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 1, 2, 3, 4, 5]])
    # aperture -> element
    hd["comp_aperture", "part_of", "comp_element"].edge_index = torch.tensor([[0, 1], [1, 4]])
    hd["comp_element", "part_of", "comp_aperture"].edge_index = torch.tensor([[1, 4], [0, 1]])
    # adjacency (self type, symmetric)
    hd["comp_element", "adjacent_surface", "comp_element"].edge_index = _bidir([(2, 3)])
    hd["comp_element", "adjacent_surface", "comp_element"].edge_attr = torch.ones((2, 1))
    # weather (one-directional)
    hd["weather_city", "weather_for", "comp_element"].edge_index = torch.tensor([[0, 0, 0], [0, 1, 4]])
    hd["weather_city", "weather_for", "comp_element"].edge_attr = torch.ones((3, 4))
    hd["weather_city", "weather_for", "comp_aperture"].edge_index = torch.tensor([[0, 0], [0, 1]])
    hd["weather_city", "weather_for", "comp_aperture"].edge_attr = torch.ones((2, 3))
    hd.y = torch.tensor([0.3])
    return hd


def make_homo(seed: int = 0, n: int = 6) -> Data:
    g = torch.Generator().manual_seed(seed)
    ei = _bidir([(i, i + 1) for i in range(n - 1)])
    d = Data(x=torch.rand((n, 7), generator=g), edge_index=ei, edge_attr=torch.ones((ei.size(1), 2)), y=torch.tensor([0.1]))
    d.model_id = f"city_{seed:04d}"
    d.weather_city = torch.rand((223,), generator=g)
    return d


def test_clean_hetero_passes():
    res = check_graph(make_hetero(), c10_spec(), model_id="city_0000")
    assert failed(res) == set(), [str(r) for r in res if r.status == Status.FAIL]
    assert "pyg.graph_valid" in {r.name for r in res}


def test_hetero_defects():
    spec = c10_spec()
    hd = make_hetero()
    hd["comp_element"].x[0, 0] = float("nan")
    hd["comp_aperture", "part_of", "comp_element"].edge_index = torch.tensor([[0, 1], [1, 99]])  # out of bounds
    del hd["building", "belongs_to", "comp_zone"]  # missing reverse store
    hd["comp_element", "weather_for", "weather_city"].edge_index = torch.tensor([[0], [0]])  # reverse of one-directional
    hd["weather_city", "weather_for", "comp_element"].edge_attr = torch.ones((2, 4))  # wrong row count
    res = check_graph(hd, spec, model_id="somebody_else")
    assert {"pyg.tensor_finite", "pyg.edge_index_bounds", "pyg.reverse_store_present", "pyg.one_directional_respected", "pyg.edge_attr_rows", "pyg.model_id_matches"} <= failed(res)


def test_hetero_unexpected_and_missing_types():
    spec = c10_spec()
    hd = make_hetero()
    hd["ghost"].x = torch.zeros((1, 2))
    del hd["comp_zone"]
    res = check_graph(hd, spec)
    assert "pyg.node_types_match_config" in failed(res)


def test_hetero_asymmetric_self_type_edges():
    hd = make_hetero()
    hd["comp_element", "adjacent_surface", "comp_element"].edge_index = torch.tensor([[2], [3]])
    hd["comp_element", "adjacent_surface", "comp_element"].edge_attr = torch.ones((1, 1))
    assert "pyg.edges_symmetric" in failed(check_graph(hd, c10_spec()))


def test_clean_homo_passes():
    res = check_graph(make_homo(), None, model_id="city_0000", expect_symmetric=True, graph_feat_key="weather_city", graph_feat_dim=223)
    assert failed(res) == set()


def test_homo_defects():
    d = make_homo()
    d.edge_index = torch.tensor([[0, 1, 2, 2], [1, 2, 3, 2]])  # asymmetric + self loop
    d.x[1, 2] = float("inf")
    d.weather_city = torch.zeros((100,))
    res = check_graph(d, None, model_id="city_0000", expect_symmetric=True, graph_feat_key="weather_city", graph_feat_dim=223)
    assert {"pyg.no_self_loops", "pyg.edges_symmetric", "pyg.tensor_finite", "pyg.graph_feature_dim", "pyg.graph_feature_nonzero"} <= failed(res)


def test_collection_dim_instability_and_constants():
    spec = c10_spec()
    graphs = [make_hetero(i) for i in range(25)]
    for g in graphs:
        g["comp_zone"].x[:, 0] = 0.0   # all-zero column
        g["comp_zone"].x[:, 1] = 5.0   # constant column
    res = check_graph_collection(graphs, spec, model_ids=[g.model_id for g in graphs])
    assert failed(res) == set()
    assert {"pyg.feature_all_zero", "pyg.feature_constant"} <= warned(res)
    graphs[3]["comp_element"].x = torch.rand((6, 13))
    assert "pyg.feature_dim_stable" in failed(check_graph_collection(graphs, spec))


def _payload(n: int = 12, kind: str = "hetero"):
    graphs = [make_hetero(i) if kind == "hetero" else make_homo(i) for i in range(n)]
    ids = [g.model_id for g in graphs]
    labels = {m: 100.0 + 10 * i for i, m in enumerate(ids)}
    tr, va, te = list(range(0, 8)), [8, 9], [10, 11]
    tr_vals = [labels[ids[i]] for i in tr]
    mean = sum(tr_vals) / len(tr_vals)
    std = math.sqrt(sum((v - mean) ** 2 for v in tr_vals) / (len(tr_vals) - 1))
    for i, g in enumerate(graphs):
        g.y = torch.tensor([(labels[ids[i]] - mean) / std])
    split = {"train_idx": tr, "val_idx": va, "test_idx": te, "train_ids": [ids[i] for i in tr], "val_ids": [ids[i] for i in va], "test_ids": [ids[i] for i in te], "seed": 0, "method": "random"}
    split.update({f"{s}_ids_sha": _sha256_ids(split[f"{s}_ids"]) for s in ("train", "val", "test")})
    return {
        "schema_version": "beg-training-ready-v4", "config_name": "C10_component_hetero_external_weather", "graph_kind": kind,
        "config_signature_hash": "abc", "dataset_ids": ids, "graphs": graphs, "split": split,
        "label": {"label_col": "eui_kwh_per_m2", "train_mean_raw": mean, "train_std_raw": std, "space": "zscore(train_only_raw)"},
        "labels_raw": {"by_index": [labels[m] for m in ids], "by_id": labels, "space": "raw"},
    }


def test_training_ready_clean():
    p = _payload()
    meta = {"config_name": p["config_name"], "graph_kind": "hetero", "config_signature_hash": "abc", "n_graphs": 12, "split_sizes": {"train": 8, "val": 2, "test": 2}}
    res = check_training_ready_payload(p, c10_spec(), meta=meta)
    assert failed(res) == set(), [str(r) for r in res if r.status == Status.FAIL]


def test_training_ready_defects():
    p = _payload()
    p["split"]["val_idx"] = [7, 8]                     # overlap with train, index 9 uncovered
    p["labels_raw"]["by_index"][0] = -1.0               # non-positive label
    p["label"]["train_mean_raw"] += 50                  # wrong stats
    p["split"]["test_ids"] = ["tampered", "x"]          # sha mismatch + ids≠idx
    res = check_training_ready_payload(p, c10_spec(), meta={"n_graphs": 13})
    assert {"pyg.tr_split_disjoint", "pyg.tr_split_covers", "pyg.tr_labels_positive", "pyg.tr_label_stats", "pyg.tr_split_sha", "pyg.tr_split_ids_match_idx", "pyg.tr_meta_matches", "pyg.tr_graph_y_denorm"} <= failed(res)


def test_training_ready_alignment():
    a, b = _payload(), _payload()
    b["split"]["train_ids"][0], b["split"]["val_ids"][0] = b["split"]["val_ids"][0], b["split"]["train_ids"][0]
    b["labels_raw"]["by_id"][b["dataset_ids"][0]] += 1.0
    res = check_training_ready_alignment({"A": a, "B": b})
    assert {"pyg.tr_split_aligned", "pyg.tr_labels_aligned"} <= failed(res)
    assert failed(check_training_ready_alignment({"A": _payload(), "B": _payload()})) == set()
