"""
NetworkX graph checks on graphs pickled per model, if you have them: ``--nx-pickles``
points at a folder with one sub-folder per configuration (named like
``C5_zone_hetero_external_weather``) holding one ``<model_id>.pkl`` per model.

To validate freshly built graphs, build them with
``beg.build.networkx_graphs.build_nx_graph_for_model`` and call
``beg.validation.nx_checks.run_graph_checks`` directly (see README).
"""

from __future__ import annotations

import pytest

from beg.validation.cross_config_checks import check_graphs_consistent
from beg.validation.loaders import (
    list_nx_pickle_configs,
    load_nx_pickles,
    resolve_spec_for_output_name,
)
from beg.validation.nx_checks import check_graph_collection, run_graph_checks

from .conftest import assert_ok

pytestmark = pytest.mark.graph_quality

_CACHE: dict = {}


def _graphs(ctx, nx_config_dir):
    if nx_config_dir not in _CACHE:
        _CACHE[nx_config_dir] = load_nx_pickles(nx_config_dir, limit=ctx.max_models)
    return _CACHE[nx_config_dir]


def _spec(ctx, nx_config_dir):
    spec, derived = resolve_spec_for_output_name(nx_config_dir.name, ctx.paths.specs())
    if spec is None:
        pytest.skip(f"no config found for output folder {nx_config_dir.name}")
    return spec, derived


def test_each_graph(ctx, nx_config_dir):
    spec, derived = _spec(ctx, nx_config_dir)
    graphs = _graphs(ctx, nx_config_dir)
    if not graphs:
        pytest.skip("no pickles")
    results = []
    for mid, G in graphs.items():
        results += run_graph_checks(G, spec, expected_model_id=mid, graph_level_weather=derived or None)
    assert_ok(results, fail_on_warn=ctx.fail_on_warn, context=f"{nx_config_dir.name} ({len(graphs)} graphs)")


def test_graph_collection(ctx, nx_config_dir):
    spec, _ = _spec(ctx, nx_config_dir)
    graphs = _graphs(ctx, nx_config_dir)
    if not graphs:
        pytest.skip("no pickles")
    assert_ok(check_graph_collection(list(graphs.values()), spec), fail_on_warn=ctx.fail_on_warn, context=nx_config_dir.name)


def test_same_building_across_configs(ctx):
    dirs = [d for d in list_nx_pickle_configs(ctx.nx_pickles) if ctx.wanted(d.name)]
    if len(dirs) < 2:
        pytest.skip("need ≥2 config folders of pickles")
    per_model: dict = {}
    for d in dirs:
        for mid, G in _graphs(ctx, d).items():
            per_model.setdefault(mid, {})[d.name] = G
    results = []
    for mid, gs in per_model.items():
        if len(gs) >= 2:
            results += check_graphs_consistent(gs, model_id=mid)
    if not results:
        pytest.skip("no model appears in ≥2 configs")
    assert_ok(results, fail_on_warn=ctx.fail_on_warn, context="cross-config (nx)")
