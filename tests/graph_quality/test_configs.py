"""Static configuration checks — no database, no generated data required."""

from __future__ import annotations

import pytest

from beg.validation.config_checks import check_config_spec, check_configs_dir

from .conftest import assert_ok

pytestmark = pytest.mark.graph_quality


def test_configs_dir_is_consistent(ctx):
    """ids / prefixes unique across the directory, every config loads."""
    results = check_configs_dir(ctx.paths.configs_dir, transforms_dir=ctx.paths.transforms_dir, names=ctx.config_names)
    # keep only directory-level results here; per-config detail is asserted below
    dir_level = [r for r in results if r.config is None]
    assert_ok(dir_level, fail_on_warn=ctx.fail_on_warn, context=str(ctx.paths.configs_dir))


def test_config_static(ctx, spec):
    """naming convention, node/edge wiring, weather variant, SQL transforms & PRIMARY_ORDER."""
    results = check_config_spec(spec, transforms_dir=ctx.paths.transforms_dir)
    assert_ok(results, fail_on_warn=ctx.fail_on_warn, context=spec.full_name)


def test_config_expected_structure(spec):
    """The derived expectations must be non-trivial (guards against a silently empty spec)."""
    assert spec.node_tables, "config declares no node tables"
    assert spec.expected_triples or len(spec.node_tables) == 1, "config declares no relations"
    if spec.homogeneous:
        assert len(set(spec.node_tables)) == 1
    else:
        assert len(set(spec.node_tables)) >= 2
