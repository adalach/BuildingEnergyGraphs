"""Labels (labels.csv) and cross-configuration consistency of the DataFrames."""

from __future__ import annotations

import pytest

from beg.validation.cross_config_checks import check_payloads_consistent
from beg.validation.label_checks import check_labels, load_labels
from beg.validation.loaders import fetch_payload_from_db, make_engine_from_env

from .conftest import assert_ok

pytestmark = pytest.mark.graph_quality


def _dataset_ids(ctx):
    """Model ids of the generated dataset, from whichever artefact is available."""
    out = ctx.dataframes_out()
    if out is not None:
        ids = set()
        for p in out.get("data", {}).values():
            ids.update(map(str, p.get("model_ids") or []))
        if ids:
            return ids
    from beg.validation.loaders import iter_pyg_graph_files, list_pyg_config_dirs

    for d in list_pyg_config_dirs(ctx.pyg_graphs):
        ids = {mid for mid, _ in iter_pyg_graph_files(d)}
        if ids:
            return ids
    return None


def test_labels_file(ctx):
    if not ctx.labels_csv.exists():
        pytest.skip(f"{ctx.labels_csv} missing")
    labels = load_labels(ctx.labels_csv)
    results = check_labels(labels, dataset_model_ids=_dataset_ids(ctx))
    assert_ok(results, fail_on_warn=ctx.fail_on_warn, context=str(ctx.labels_csv))


def test_configs_describe_same_buildings(ctx):
    payloads = {}
    if ctx.use_db:
        engine = make_engine_from_env()
        if engine is None:
            pytest.skip("BEG_DSN not set")
        # use the same model ids for every config
        first = next(iter(ctx.specs.values()))
        base = fetch_payload_from_db(engine, first, schema=ctx.db_schema, max_models=ctx.max_models)
        payloads[first.name] = base
        for s in ctx.specs.values():
            if s.name != first.name:
                payloads[s.name] = fetch_payload_from_db(engine, s, schema=ctx.db_schema, model_ids=base["model_ids"])
    else:
        out = ctx.dataframes_out()
        if out is None:
            pytest.skip("DataFrames not enabled (--with-dataframes / --db)")
        payloads = {k: v for k, v in out.get("data", {}).items() if k in ctx.specs}
    if len(payloads) < 2:
        pytest.skip("need ≥2 configs")
    assert_ok(check_payloads_consistent(payloads, ctx.specs), fail_on_warn=ctx.fail_on_warn, context="cross-config (dataframes)")
