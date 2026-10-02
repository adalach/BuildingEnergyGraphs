from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from beg.pipeline.sql import ensure_required_source_tables, execute_pipeline_sql


class _Result:
    def __init__(self, *, scalar=None, row=None):
        self._scalar = scalar
        self._row = row

    def scalar_one(self):
        return self._scalar

    def one(self):
        return self._row


class _Connection:
    def __init__(self, results):
        self._results = results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, statement):
        return self._results.pop(0)


class _Engine:
    def __init__(self, *results):
        self.results = list(results)

    def connect(self):
        return _Connection(self.results)


class ShadingPreconditionTests(unittest.TestCase):
    def test_irrelevant_configs_do_not_touch_database(self):
        self.assertFalse(
            ensure_required_source_tables(
                engine=None,
                raw_schema="ep",
                sql_transforms=["building.sql"],
            )
        )

    def test_existing_complete_nonzero_table_is_accepted(self):
        engine = _Engine(
            _Result(
                row=SimpleNamespace(
                    eligible_models=2,
                    row_count=12,
                    covered_models=2,
                    nonzero_rows=7,
                )
            ),
        )
        with patch("beg.pipeline.sql.get_relkind", return_value="r"):
            changed = ensure_required_source_tables(
                engine=engine,
                raw_schema="ep",
                sql_transforms=["surface_shading_metrics.sql"],
                verbose=False,
            )
        self.assertFalse(changed)

    def test_missing_table_runs_producer_and_prints_message(self):
        engine = _Engine(
            _Result(scalar=2),
            _Result(
                row=SimpleNamespace(
                    eligible_models=2,
                    row_count=12,
                    covered_models=2,
                    nonzero_rows=7,
                )
            ),
        )
        output = StringIO()
        with (
            patch(
                "beg.pipeline.sql.get_relkind",
                side_effect=[None, "r"],
            ),
            patch(
                "beg.pipeline.shading.populate_surface_shading_metrics",
                return_value=12,
            ) as producer,
            redirect_stdout(output),
        ):
            changed = ensure_required_source_tables(
                engine=engine,
                raw_schema="ep",
                sql_transforms=["surface_shading_metrics.sql"],
                verbose=True,
            )

        self.assertTrue(changed)
        producer.assert_called_once_with(
            engine,
            schema="ep",
            replace=False,
            resume=False,
        )
        self.assertIn("is missing", output.getvalue())
        self.assertIn("Running beg.pipeline.shading now.", output.getvalue())

    def test_complete_all_zero_table_stops_pipeline(self):
        engine = _Engine(
            _Result(
                row=SimpleNamespace(
                    eligible_models=2,
                    row_count=12,
                    covered_models=2,
                    nonzero_rows=0,
                )
            ),
        )
        with patch("beg.pipeline.sql.get_relkind", return_value="r"):
            with self.assertRaisesRegex(RuntimeError, "every shading metric is zero"):
                ensure_required_source_tables(
                    engine=engine,
                    raw_schema="ep",
                    sql_transforms=["surface_shading_metrics.sql"],
                    verbose=False,
                )

    def test_auto_population_forces_transform_rebuild(self):
        configs = pd.DataFrame(
            [
                {
                    "config_name": "zone_homo_no_weather",
                    "type": "homogeneous",
                    "nodes": [{"table": "zones_zone"}],
                    "edges": [],
                    "sql_transforms": ["surface_shading_metrics.sql"],
                }
            ]
        )
        with (
            patch("beg.pipeline.sql.load_configs_from_dir", return_value=configs),
            patch(
                "beg.pipeline.sql.build_unified_transform_list",
                return_value=["surface_shading_metrics.sql"],
            ),
            patch(
                "beg.pipeline.sql.ensure_required_source_tables",
                return_value=True,
            ),
            patch(
                "beg.pipeline.sql.execute_sql_transforms",
                return_value={"surface_shading_metrics.sql": True},
            ) as execute_transforms,
        ):
            execute_pipeline_sql(
                configs_dir="configs",
                config_names=["zone_homo_no_weather"],
                rebuild_transforms=False,
                engine=object(),
                verbose=False,
            )

        self.assertTrue(execute_transforms.call_args.kwargs["rebuild"])


if __name__ == "__main__":
    unittest.main()
