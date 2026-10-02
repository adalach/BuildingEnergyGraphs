"""
DataFrame-level checks on the graph DataFrames (``out["data"][config]``).

Enabled with ``--dataframes-pkl`` / ``--with-dataframes`` (a pickled
``build_graph_dataframes`` result, if you have one) or ``--db`` (fetches straight
from Postgres, limited by ``--max-models``).
"""

from __future__ import annotations

import pytest

from beg.validation.dataframe_checks import (
    check_edge_integrity,
    check_feature_variance,
    check_missing_values,
    check_model_coverage,
    check_node_uniqueness,
    check_physical_consistency,
    check_table_schema,
    check_tables_present,
    check_value_domains,
    check_weather_tables,
    flatten_payload,
)
from beg.validation.loaders import fetch_payload_from_db, make_engine_from_env

from .conftest import assert_ok

pytestmark = pytest.mark.graph_quality

_CACHE: dict = {}


@pytest.fixture
def payload(ctx, df_payload_item):
    source, spec = df_payload_item
    key = (source, spec.name)
    if key in _CACHE:
        return spec, _CACHE[key]
    if source == "db":
        engine = make_engine_from_env()
        if engine is None:
            pytest.skip("BEG_DSN not set")
        raw = fetch_payload_from_db(engine, spec, schema=ctx.db_schema, max_models=ctx.max_models)
    else:
        out = ctx.dataframes_out()
        if out is None:
            pytest.skip(f"{ctx.dataframes_pkl} not found")
        data = out.get("data", {})
        if spec.name not in data:
            pytest.skip(f"config {spec.name} not in pickle (has {sorted(data)[:10]})")
        raw = data[spec.name]
    flat = flatten_payload(raw)
    _CACHE[key] = flat
    return spec, flat


def _run(ctx, spec, results, name):
    return assert_ok(results, fail_on_warn=ctx.fail_on_warn, context=f"{spec.full_name} / {name}")


def test_tables_present_and_schema(ctx, payload):
    spec, p = payload
    _run(ctx, spec, check_tables_present(spec, p["nodes"], p["edges"]) + check_table_schema(spec, p["nodes"], p["edges"]), "tables")


def test_model_coverage(ctx, payload):
    spec, p = payload
    _run(ctx, spec, check_model_coverage(spec, p["nodes"], p["edges"], p["model_ids"]), "coverage")


def test_node_uniqueness(ctx, payload):
    spec, p = payload
    _run(ctx, spec, check_node_uniqueness(spec, p["nodes"]), "node keys")


def test_edge_integrity(ctx, payload):
    spec, p = payload
    _run(ctx, spec, check_edge_integrity(spec, p["nodes"], p["edges"]), "edges")


def test_missing_values(ctx, payload):
    spec, p = payload
    _run(ctx, spec, check_missing_values(spec, p["nodes"], p["edges"]), "missing values")


def test_value_domains(ctx, payload):
    spec, p = payload
    _run(ctx, spec, check_value_domains(spec, p["nodes"]), "value domains")


def test_feature_variance(ctx, payload):
    spec, p = payload
    _run(ctx, spec, check_feature_variance(spec, p["nodes"], p["edges"]), "variance")


def test_physical_consistency(ctx, payload):
    spec, p = payload
    _run(ctx, spec, check_physical_consistency(spec, p["nodes"], p["edges"]), "physics")


def test_weather_tables(ctx, payload):
    spec, p = payload
    _run(ctx, spec, check_weather_tables(spec, p["nodes"]), "weather")
