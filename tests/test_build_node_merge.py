"""Merging node tables must keep the first non-null value per column, in table order."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from beg.build.pyg_homo import _clean_and_merge_node_tables  # noqa: E402

pytestmark = pytest.mark.torch

KEY = "instance_name"


def test_holes_are_filled_from_later_tables_in_order():
    first = pd.DataFrame({KEY: ["a", "b", "c"], "x": [1.0, np.nan, 3.0], "y": [np.nan, 2.0, np.nan]})
    second = pd.DataFrame({KEY: ["b", "c", "d"], "x": [9.0, 30.0, 4.0], "y": [5.0, 6.0, 7.0]})

    merged = _clean_and_merge_node_tables({"t1": first, "t2": second}, ["t1", "t2"], node_key=KEY)
    rows = merged.set_index(KEY)

    assert list(merged[KEY]) == ["a", "b", "c", "d"], "first-appearance order must survive"
    # b and c appear in both: the earlier table wins where it has a value...
    assert rows.loc["b", "x"] == 9.0, "b has no x in t1, so t2 fills it"
    assert rows.loc["c", "x"] == 3.0, "c has x in t1, so t2 must not overwrite it"
    # ...and the later table fills only what is missing.
    assert rows.loc["c", "y"] == 6.0
    assert pd.isna(rows.loc["a", "y"]), "missing everywhere stays missing"
    assert rows.loc["b", "__node_table__"] == "t1", "provenance is the first table seen"


def test_unique_keys_pass_through_unchanged():
    frame = pd.DataFrame({KEY: ["a", "b"], "x": [1.0, 2.0]})
    merged = _clean_and_merge_node_tables({"t": frame}, ["t"], node_key=KEY)

    assert list(merged[KEY]) == ["a", "b"]
    assert list(merged["x"]) == [1.0, 2.0]
    assert "__node_table__" in merged.columns


def test_column_order_is_preserved():
    """The builder reads columns positionally in places, so order is load-bearing."""
    first = pd.DataFrame({KEY: ["a", "a"], "alpha": [1.0, np.nan], "beta": [np.nan, 2.0]})
    merged = _clean_and_merge_node_tables({"t": first}, ["t"], node_key=KEY)

    assert list(merged.columns) == [KEY, "alpha", "beta", "__node_table__"]
    assert merged.loc[0, "alpha"] == 1.0 and merged.loc[0, "beta"] == 2.0
