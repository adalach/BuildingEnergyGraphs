"""
pytest configuration for the graph dataset quality suite.

Run everything that is available on this machine::

    pytest tests/graph_quality -q

Useful options (all optional — a missing artefact skips its tests)::

    --repo-root PATH         project root (default: auto-detected)
    --configs-dir PATH       configs folder (default: <root>/configs)
    --config NAME            restrict to config name(s) (repeatable),
                             e.g. component_hetero_external_weather
    --dataframes-pkl PATH    pickled build_graph_dataframes result, if you have one
                             (default <root>/outputs/graph_dataframes.pkl)
    --with-dataframes        enable DataFrame-level tests from that pickle (implied by
                             --dataframes-pkl)
    --db                     fetch DataFrames from Postgres instead (BEG_DSN)
    --db-schema NAME         schema to read (default ep_transformed)
    --max-models N           limit models fetched from DB / graphs loaded per config (default 200)
    --nx-pickles PATH        folder of NetworkX graphs pickled per model, one sub-folder
                             per configuration, if you have them
    --pyg-graphs PATH        per-model PyG root (default outputs/pyg_graphs)
    --pyg-datasets PATH      training_ready root (default outputs/pyg_datasets)
    --labels PATH            labels csv (default <root>/data/labels.csv)
    --fail-on-warn           treat WARN results as failures
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

HERE = Path(__file__).resolve()
for cand in [HERE.parents[2], *HERE.parents]:
    if (cand / "beg").is_dir():
        if str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
        break

from beg.validation.core import CheckResult, Report, Status  # noqa: E402
from beg.validation.loaders import RepoPaths, resolve_spec_for_output_name  # noqa: E402


def pytest_addoption(parser: pytest.Parser) -> None:
    g = parser.getgroup("graph-quality")
    g.addoption("--repo-root", default=None)
    g.addoption("--configs-dir", default=None)
    g.addoption("--config", action="append", default=None, help="config name(s) to test")
    g.addoption("--dataframes-pkl", default=None)
    g.addoption("--with-dataframes", action="store_true", default=False)
    g.addoption("--db", action="store_true", default=False)
    g.addoption("--db-schema", default="ep_transformed")
    g.addoption("--max-models", type=int, default=200)
    g.addoption("--nx-pickles", default=None)
    g.addoption("--pyg-graphs", default=None)
    g.addoption("--pyg-datasets", default=None)
    g.addoption("--labels", default=None)
    g.addoption("--fail-on-warn", action="store_true", default=False)


# --------------------------------------------------------------------------
# shared state
# --------------------------------------------------------------------------

class Ctx:
    def __init__(self, config: pytest.Config):
        self.paths = RepoPaths(Path(config.getoption("--repo-root")) if config.getoption("--repo-root") else None)
        if config.getoption("--configs-dir"):
            self.paths.configs_dir = Path(config.getoption("--configs-dir"))
        self.config_names: Optional[List[str]] = config.getoption("--config")
        self.max_models: int = config.getoption("--max-models")
        self.fail_on_warn: bool = config.getoption("--fail-on-warn")
        self.with_dataframes: bool = config.getoption("--with-dataframes") or bool(config.getoption("--dataframes-pkl"))
        self.dataframes_pkl = Path(config.getoption("--dataframes-pkl")) if config.getoption("--dataframes-pkl") else self.paths.graph_dataframes_pkl
        self.use_db: bool = config.getoption("--db")
        self.db_schema: str = config.getoption("--db-schema")
        self.nx_pickles = Path(config.getoption("--nx-pickles")) if config.getoption("--nx-pickles") else self.paths.nx_pickles_root
        self.pyg_graphs = Path(config.getoption("--pyg-graphs")) if config.getoption("--pyg-graphs") else self.paths.pyg_graphs_root
        self.pyg_datasets = Path(config.getoption("--pyg-datasets")) if config.getoption("--pyg-datasets") else self.paths.pyg_datasets_root
        self.labels_csv = Path(config.getoption("--labels")) if config.getoption("--labels") else self.paths.labels_csv
        self._specs = None
        self._out = None

    @property
    def specs(self) -> Dict[str, Any]:
        if self._specs is None:
            self._specs = self.paths.specs(self.config_names)
        return self._specs

    def wanted(self, output_name: str) -> bool:
        if not self.config_names:
            return True
        spec, _ = resolve_spec_for_output_name(output_name, self.paths.specs())
        return spec is not None and spec.name in self.config_names

    def dataframes_out(self) -> Optional[Dict[str, Any]]:
        if self._out is None and self.with_dataframes and self.dataframes_pkl.exists():
            from beg.validation.loaders import load_graph_dataframes

            self._out = load_graph_dataframes(self.dataframes_pkl)
        return self._out


_CTX: Optional[Ctx] = None


def pytest_configure(config: pytest.Config) -> None:
    global _CTX
    _CTX = Ctx(config)
    config.addinivalue_line("markers", "graph_quality: dataset quality checks")


@pytest.fixture(scope="session")
def ctx() -> Ctx:
    assert _CTX is not None
    return _CTX


# --------------------------------------------------------------------------
# assertion helper
# --------------------------------------------------------------------------

def assert_ok(results: List[CheckResult], *, fail_on_warn: bool = False, context: str = "") -> Report:
    """Fail the test on any FAIL (or WARN when requested); surface WARNs as pytest warnings."""
    rep = Report(list(results))
    for w in rep.warnings:
        warnings.warn(f"[WARN] {w}", stacklevel=2)
    bad = rep.failures + (rep.warnings if fail_on_warn else [])
    if bad:
        lines = [f"{len(bad)} failing checks{(' for ' + context) if context else ''}:"]
        lines += [f"  - {r}" for r in bad[:60]]
        if len(bad) > 60:
            lines.append(f"  … {len(bad) - 60} more")
        skipped = rep.by_status(Status.SKIP)
        if skipped:
            lines.append(f"({len(skipped)} checks skipped)")
        pytest.fail("\n".join(lines), pytrace=False)
    return rep


# --------------------------------------------------------------------------
# parametrisation
# --------------------------------------------------------------------------

def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    ctx = _CTX
    assert ctx is not None
    if "spec" in metafunc.fixturenames:
        specs = list(ctx.specs.values())
        metafunc.parametrize("spec", specs, ids=[s.full_name for s in specs])
    if "nx_config_dir" in metafunc.fixturenames:
        from beg.validation.loaders import list_nx_pickle_configs

        dirs = [d for d in list_nx_pickle_configs(ctx.nx_pickles) if ctx.wanted(d.name)]
        metafunc.parametrize("nx_config_dir", dirs, ids=[d.name for d in dirs])
    if "pyg_config_dir" in metafunc.fixturenames:
        from beg.validation.loaders import list_pyg_config_dirs

        dirs = [d for d in list_pyg_config_dirs(ctx.pyg_graphs) if ctx.wanted(d.name)]
        metafunc.parametrize("pyg_config_dir", dirs, ids=[d.name for d in dirs])
    if "training_ready_item" in metafunc.fixturenames:
        from beg.validation.loaders import list_training_ready

        items = [(k, v) for k, v in list_training_ready(ctx.pyg_datasets).items() if ctx.wanted(k)]
        metafunc.parametrize("training_ready_item", items, ids=[k for k, _ in items])
    if "df_payload_item" in metafunc.fixturenames:
        items: List[Any] = []
        ids: List[str] = []
        if ctx.use_db:
            for s in ctx.specs.values():
                items.append(("db", s))
                ids.append(s.full_name)
        elif ctx.with_dataframes and ctx.dataframes_pkl.exists():
            # payload names inside the pickle are the config names (without C-prefix)
            for s in ctx.specs.values():
                items.append(("pkl", s))
                ids.append(s.full_name)
        metafunc.parametrize("df_payload_item", items, ids=ids)
