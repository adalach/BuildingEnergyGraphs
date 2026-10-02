"""Converting transform outputs to tables must not destroy dependent views.

A transform typically builds two relations: ``X_cache`` as a materialized view
holding the work, and ``X`` as a plain view selecting from it. In ``tables`` mode
the runner turns both into physical tables.

Converting them one at a time is order-dependent and silently destructive:
``DROP MATERIALIZED VIEW ... CASCADE`` on ``X_cache`` takes ``X`` with it, and
every downstream transform then fails with "relation ... does not exist". The
batch conversion keeps both, in either order.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text as sa_text

from beg.db import get_engine
from beg.db.utils import get_relkind
from beg.pipeline.transform_runner import (
    convert_relation_to_table,
    convert_relations_to_tables,
)

pytestmark = pytest.mark.db

SCHEMA = "beg_test_transform_runner"


@pytest.fixture()
def cache_and_view():
    """A materialized view with a plain view selecting from it, as transforms build."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(sa_text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no database available: {exc}")

    def build():
        with engine.begin() as con:
            con.execute(sa_text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
            con.execute(sa_text(f"CREATE SCHEMA {SCHEMA}"))
            con.execute(sa_text(f"CREATE TABLE {SCHEMA}.source (id int, v text)"))
            con.execute(sa_text(f"INSERT INTO {SCHEMA}.source VALUES (1, 'a'), (2, 'b')"))
            con.execute(
                sa_text(f"CREATE MATERIALIZED VIEW {SCHEMA}.x_cache AS SELECT * FROM {SCHEMA}.source")
            )
            con.execute(sa_text(f"CREATE VIEW {SCHEMA}.x AS SELECT * FROM {SCHEMA}.x_cache"))

    build()
    yield engine, build

    with engine.begin() as con:
        con.execute(sa_text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))


def test_batch_conversion_keeps_the_dependent_view(cache_and_view):
    engine, _build = cache_and_view

    # The cache first: the order that loses the view when converted one at a time.
    convert_relations_to_tables(engine, SCHEMA, ["x_cache", "x"])

    assert get_relkind(engine, SCHEMA, "x_cache") == "r"
    assert get_relkind(engine, SCHEMA, "x") == "r", "the dependent view was dropped by cascade"

    with engine.connect() as con:
        rows = con.execute(sa_text(f"SELECT count(*) FROM {SCHEMA}.x")).scalar()
    assert rows == 2, "the converted table lost its rows"


def test_batch_conversion_is_order_independent(cache_and_view):
    engine, build = cache_and_view

    for order in (["x_cache", "x"], ["x", "x_cache"]):
        build()
        convert_relations_to_tables(engine, SCHEMA, order)
        assert get_relkind(engine, SCHEMA, "x") == "r", f"lost the view with order {order}"
        assert get_relkind(engine, SCHEMA, "x_cache") == "r", f"lost the cache with order {order}"


def test_one_at_a_time_loses_the_view(cache_and_view):
    """Converting the cache alone drops the view, so the batch tests are not vacuous."""
    engine, _build = cache_and_view

    convert_relation_to_table(engine, SCHEMA, "x_cache")

    assert get_relkind(engine, SCHEMA, "x_cache") == "r"
    assert get_relkind(engine, SCHEMA, "x") is None, (
        "expected the cascade to drop the dependent view; if this now survives, "
        "the single-relation helper has changed and this test is stale"
    )
