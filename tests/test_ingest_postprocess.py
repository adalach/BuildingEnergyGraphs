"""Post-processing must not destroy references between objects.

The three ingestion passes judge a column by how many distinct values it holds.
That is the right question for a feature and the wrong one for a reference: a
column naming another object is an identifier, and how often a given name recurs
is an accident of the model set. A reference such as ``Construction.layer_5``
with two to five distinct names falls in the one-hot window, and must keep its
values and its foreign key instead of becoming indicator columns.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table
from sqlalchemy import text as sa_text

from beg.db import get_engine
from beg.ingest import postprocess as pp

pytestmark = pytest.mark.db

SCHEMA = "beg_test_postprocess"


@pytest.fixture()
def probe_schema():
    """A throwaway schema holding one table shaped like ``ep_construction``."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(sa_text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no database available: {exc}")

    with engine.begin() as conn:
        conn.execute(sa_text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
        conn.execute(sa_text(f"CREATE SCHEMA {SCHEMA}"))

    meta = MetaData(schema=SCHEMA)
    table = Table(
        "ep_construction",
        meta,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("model_id", String),
        Column("instance_name", String),
        Column("outside_layer", String),   # reference, many values
        Column("layer_5", String),         # reference, two values + NULLs
        Column("layer_9", String),         # reference, entirely NULL
        Column("surface_type", String),    # genuine categorical
    )
    meta.create_all(engine)

    rows = [
        {"model_id": "m1", "instance_name": "Wall", "outside_layer": f"Brick {i}",
         "layer_5": None, "layer_9": None, "surface_type": "Wall"}
        for i in range(10)
    ]
    rows += [
        {"model_id": "m1", "instance_name": "Roof A", "outside_layer": "Membrane",
         "layer_5": "Generic Acoustic Tile", "layer_9": None, "surface_type": "Roof"},
        {"model_id": "m1", "instance_name": "Roof B", "outside_layer": "Membrane",
         "layer_5": "Generic Gypsum Board", "layer_9": None, "surface_type": "Roof"},
    ]
    with engine.begin() as conn:
        conn.execute(table.insert(), rows)

    yield engine, {("ep_construction", c) for c in ("outside_layer", "layer_5", "layer_9")}

    with engine.begin() as conn:
        conn.execute(sa_text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))


def _columns(engine) -> set[str]:
    return set(
        pd.read_sql(
            "SELECT column_name FROM information_schema.columns "
            f"WHERE table_schema = '{SCHEMA}' AND table_name = 'ep_construction'",
            engine,
        )["column_name"]
    )


def test_reference_column_is_not_one_hot_encoded(probe_schema):
    engine, references = probe_schema

    pp.one_hot_encode_columns(engine, SCHEMA, reference_columns=references)
    columns = _columns(engine)

    assert "layer_5" in columns, "a material reference was flattened into indicator columns"
    assert not [c for c in columns if c.startswith("layer_5_")]

    # The genuine categorical is still encoded, so the pass has not simply stopped.
    assert "surface_type" not in columns
    assert {"surface_type_wall", "surface_type_roof"} <= columns


def test_reference_column_survives_pruning_unless_empty(probe_schema):
    engine, references = probe_schema

    dropped, _analyzed = pp.prune_invariant_columns(
        engine, SCHEMA, reference_columns=references
    )
    columns = _columns(engine)

    # Two distinct values among many NULLs: which rows have a fifth layer is
    # exactly what the column says, so it stays.
    assert "layer_5" in columns
    # Entirely NULL carries nothing, reference or not.
    assert "layer_9" not in columns
    assert ("ep_construction", "layer_9") in {(r.table, r.column) for r in dropped.itertuples()}


def test_unprotected_reference_column_is_lost(probe_schema):
    """Without the exemption the column is destroyed — the check is not vacuous."""
    engine, _references = probe_schema

    pp.one_hot_encode_columns(engine, SCHEMA)  # no reference_columns passed
    columns = _columns(engine)

    assert "layer_5" not in columns
    assert {"layer_5_generic_acoustic_tile", "layer_5_generic_gypsum_board"} <= columns
