"""
PyG checks: per-model graph files (``outputs/pyg_graphs``) and the
``training_ready.pt`` packages (``outputs/pyg_datasets``).  Skipped when
torch / torch_geometric are not importable.
"""

from __future__ import annotations

import pytest

from beg.validation.loaders import (
    iter_pyg_graph_files,
    list_training_ready,
    load_meta_json,
    resolve_spec_for_output_name,
)
from beg.validation.pyg_checks import (
    check_graph,
    check_graph_collection,
    check_training_ready_alignment,
    check_training_ready_payload,
    load_graph_file,
    torch_available,
)

from .conftest import assert_ok

pytestmark = pytest.mark.graph_quality

GRAPH_FEAT_KEY = "weather_city"   # GNN.DEFAULT_GRAPH_FEAT_KEY for derived homo graph-weather configs
GRAPH_FEAT_DIM = 223

_TR_CACHE: dict = {}


def _need_torch():
    if not torch_available():
        pytest.skip("torch / torch_geometric not available")


def test_pyg_graph_files(ctx, pyg_config_dir):
    _need_torch()
    spec, derived = resolve_spec_for_output_name(pyg_config_dir.name, ctx.paths.specs())
    if spec is None:
        pytest.skip(f"no config for {pyg_config_dir.name}")
    graphs, ids, results = [], [], []
    for mid, path in iter_pyg_graph_files(pyg_config_dir, limit=ctx.max_models):
        g = load_graph_file(path)
        graphs.append(g)
        ids.append(mid)
        kw = {"model_id": mid}
        if derived:
            kw.update(graph_feat_key=GRAPH_FEAT_KEY, graph_feat_dim=GRAPH_FEAT_DIM)
        if spec.homogeneous:
            kw["expect_symmetric"] = all(e.bidirectional for e in spec.edges)
        results += check_graph(g, spec, **kw)
    if not graphs:
        pytest.skip("no .pt files")
    results += check_graph_collection(graphs, spec, model_ids=ids)
    assert_ok(results, fail_on_warn=ctx.fail_on_warn, context=f"{pyg_config_dir.name} ({len(graphs)} graphs)")


def _load_tr(path):
    if path not in _TR_CACHE:
        _TR_CACHE[path] = load_graph_file(path)
    return _TR_CACHE[path]


def test_training_ready_package(ctx, training_ready_item):
    _need_torch()
    name, (pt_path, meta_path) = training_ready_item
    spec, _ = resolve_spec_for_output_name(name, ctx.paths.specs())
    payload = _load_tr(pt_path)
    results = check_training_ready_payload(payload, spec, meta=load_meta_json(meta_path), max_graphs_checked=ctx.max_models)
    assert_ok(results, fail_on_warn=ctx.fail_on_warn, context=name)


def test_training_ready_alignment(ctx):
    _need_torch()
    items = {k: v for k, v in list_training_ready(ctx.pyg_datasets).items() if ctx.wanted(k)}
    if len(items) < 2:
        pytest.skip("need ≥2 training_ready packages")
    payloads = {k: _load_tr(pt) for k, (pt, _) in items.items()}
    assert_ok(check_training_ready_alignment(payloads), fail_on_warn=ctx.fail_on_warn, context="training_ready alignment")


def test_training_ready_matches_labels(ctx, training_ready_item):
    """labels_raw inside the package must equal labels.csv."""
    _need_torch()
    name, (pt_path, _) = training_ready_item
    if not ctx.labels_csv.exists():
        pytest.skip("labels.csv missing")
    import pandas as pd

    payload = _load_tr(pt_path)
    lab = pd.read_csv(ctx.labels_csv)
    # Package ids are lower case while labels.csv keeps the model's own casing
    # (see docs/limitations.md), so ids are normalised before joining, as every
    # loader in `beg` does.
    lab.index = lab["building_id"].astype(str).str.strip().str.lower()
    lab = lab["eui_kwh_per_m2"]
    by_id = payload["labels_raw"]["by_id"]

    missing = [m for m in by_id if str(m).lower() not in lab.index]
    assert not missing, f"{name}: {len(missing)} dataset ids not in labels.csv e.g. {missing[:5]}"
    diff = [m for m in by_id if abs(float(by_id[m]) - float(lab[str(m).lower()])) > 1e-6]
    assert not diff, f"{name}: {len(diff)} labels differ from labels.csv e.g. {diff[:5]}"
