"""A search that lacks a whole model family on a configuration is reported as incomplete."""

from __future__ import annotations

import pandas as pd

from beg.report.results import coverage_problems

CONFIGS = ["C1_zone_homo_no_weather", "C2_zone_homo_graph_weather"]
HOMO_FAMILIES = ["gcn", "gin_mean", "gin_set2set", "sage_attn"]


def _runs(cells) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "config_id": config.split("_")[0],
                "config_name": config,
                "graph_kind": "homo",
                "model_key": model,
                "run_fingerprint": f"{config}/{model}",
            }
            for config, model in cells
        ]
    )


def test_a_family_without_a_completed_run_is_reported():
    cells = [(c, m) for c in CONFIGS for m in HOMO_FAMILIES if (c, m) != (CONFIGS[1], "gcn")]
    problems = coverage_problems(_runs(cells), expected_variants=1)
    assert any("no completed run: C2_zone_homo_graph_weather/gcn" in p for p in problems), problems


def test_every_family_present_reports_no_empty_cell():
    cells = [(c, m) for c in CONFIGS for m in HOMO_FAMILIES]
    problems = coverage_problems(_runs(cells), expected_variants=1)
    assert not any("no completed run" in p for p in problems), problems
