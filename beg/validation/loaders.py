"""
Locate and load the artefacts the checks operate on.

The check functions take plain Python objects; pytest and the command-line entry
point (``scripts/run_graph_quality_checks.py``) use these helpers to find:

* configs                    ``configs/C<n>.json``
* graph DataFrames           a pickled ``build_graph_dataframes`` result, if you have one,
                             or a live DB fetch
* NetworkX graphs            pickled per model, one folder per configuration, if you have them
* PyG per-model graphs       ``outputs/pyg_graphs/{homo,hetero}/<Cn_config>/<model_id>.pt``
* training packages          ``outputs/pyg_datasets/<Cn_config>/training_ready.pt``
                             (+ ``training_ready.meta.json``)
* labels                     ``data/labels.csv``

``outputs/`` and ``data/`` are ``beg.settings.output_dir()`` and ``data_dir()``, so
``BEG_OUTPUT_DIR`` and ``BEG_DATA_DIR`` apply, unless a project root is given explicitly.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from beg.settings import data_dir, output_dir
from beg.validation.spec import ConfigSpec, load_config_specs

FULL_NAME_RE = re.compile(r"^C(\d+)_(?P<name>.+)$")

# Derived homogeneous graph-weather variants are built downstream from base configs.
DERIVED_GRAPH_WEATHER_BASE: Dict[str, str] = {
    "zone_homo_graph_weather": "zone_homo_no_weather",
    "component_homo_graph_weather": "component_homo_no_weather",
}


def find_project_root(start: Optional[Path] = None) -> Path:
    p = Path(start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "beg").is_dir() and (cand / "configs").is_dir():
            return cand
    return Path(__file__).resolve().parents[2]


def strip_c_prefix(full_name: str) -> str:
    m = FULL_NAME_RE.match(str(full_name))
    return m.group("name") if m else str(full_name)


def resolve_spec_for_output_name(full_name: str, specs: Dict[str, ConfigSpec]) -> Tuple[Optional[ConfigSpec], bool]:
    """Return ``(spec, is_graph_weather_derived)`` for an output folder name.

    ``C7_component_homo_graph_weather`` gives the ``component_homo_no_weather`` spec and ``True``.
    """
    name = strip_c_prefix(full_name)
    if name in specs:
        return specs[name], False
    base = DERIVED_GRAPH_WEATHER_BASE.get(name)
    if base and base in specs:
        return specs[base], True
    # experiment configs may use the prefix directly
    for s in specs.values():
        if s.full_name == full_name:
            return s, False
    return None, False


# --------------------------------------------------------------------------
# Graph DataFrames
# --------------------------------------------------------------------------

def load_graph_dataframes(path: Path | str) -> Dict[str, Any]:
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def fetch_payload_from_db(engine: Any, spec: ConfigSpec, *, schema: str = "ep_transformed", max_models: Optional[int] = None, model_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Re-implements the fetch step of ``beg.pipeline.build_graph_dataframes`` for one config
    (no transforms are executed) so checks can run straight against Postgres."""
    import pandas as pd
    from sqlalchemy import text as sql_text

    main = spec.primary_node_table
    with engine.connect() as con:
        if model_ids is None:
            q = f'SELECT DISTINCT model_id FROM "{schema}"."{main}" ORDER BY model_id' + (f" LIMIT {int(max_models)}" if max_models else "")
            model_ids = [r[0] for r in con.execute(sql_text(q)).fetchall()]
    mids = list(model_ids)
    nodes = {}
    for t in spec.node_tables:
        nodes[t] = pd.read_sql(sql_text(f'SELECT * FROM "{schema}"."{t}" WHERE model_id = ANY(:mids)'), engine, params={"mids": mids}) if mids else pd.DataFrame()
    edges = {}
    for t in spec.edge_tables:
        edges[t] = pd.read_sql(sql_text(f'SELECT * FROM "{schema}"."{t}" WHERE model_id = ANY(:mids)'), engine, params={"mids": mids}) if mids else pd.DataFrame()
    return {"model_ids": mids, "nodes": nodes, "edges": edges}


def make_engine_from_env() -> Optional[Any]:
    """Engine for the configured database, or None when BEG_DSN is unset.

    The checks that need a database skip themselves on None, so the artefact
    checks still run.
    """
    from beg.db import get_engine
    from beg.settings import dsn as configured_dsn

    if not configured_dsn(required=False):
        return None
    return get_engine()


# --------------------------------------------------------------------------
# NetworkX pickles
# --------------------------------------------------------------------------

def list_nx_pickle_configs(pickles_root: Path | str) -> List[Path]:
    root = Path(pickles_root)
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def load_nx_pickles(config_dir: Path | str, *, limit: Optional[int] = None) -> Dict[str, Any]:
    """``{model_id: graph}`` for one folder of pickles (raw graphs or ``BuiltGraph`` objects)."""
    out: Dict[str, Any] = {}
    for p in sorted(Path(config_dir).glob("*.pkl"))[: limit or None]:
        with p.open("rb") as fh:
            obj = pickle.load(fh)
        G = getattr(obj, "G", obj)
        out[p.stem] = G
    return out


# --------------------------------------------------------------------------
# PyG
# --------------------------------------------------------------------------

def list_pyg_config_dirs(pyg_graphs_root: Path | str) -> List[Path]:
    root = Path(pyg_graphs_root)
    dirs: List[Path] = []
    for kind in ("homo", "hetero"):
        d = root / kind
        if d.is_dir():
            dirs.extend(sorted(p for p in d.iterdir() if p.is_dir()))
    return dirs


def iter_pyg_graph_files(config_dir: Path | str, *, limit: Optional[int] = None) -> Iterator[Tuple[str, Path]]:
    files = sorted(Path(config_dir).glob("*.pt"))
    for p in files[: limit or None]:
        yield p.stem, p


def list_training_ready(datasets_root: Path | str) -> Dict[str, Tuple[Path, Optional[Path]]]:
    root = Path(datasets_root)
    out: Dict[str, Tuple[Path, Optional[Path]]] = {}
    if not root.is_dir():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        pt = d / "training_ready.pt"
        if pt.exists():
            meta = d / "training_ready.meta.json"
            out[d.name] = (pt, meta if meta.exists() else None)
    return out


def load_meta_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# convenience: standard repository layout
# --------------------------------------------------------------------------

class RepoPaths:
    def __init__(self, root: Optional[Path] = None):
        self.root = find_project_root(root)
        self.configs_dir = self.root / "configs"
        self.transforms_dir = self.root / "beg" / "sql" / "transforms"
        self.outputs = self.root / "outputs" if root is not None else output_dir()
        self.graph_dataframes_pkl = self.outputs / "graph_dataframes.pkl"
        self.nx_pickles_root = self.outputs / "nx_graphs"
        self.pyg_graphs_root = self.outputs / "pyg_graphs"
        self.pyg_datasets_root = self.outputs / "pyg_datasets"
        self.labels_csv = (self.root / "data" if root is not None else data_dir()) / "labels.csv"

    def specs(self, names: Optional[Sequence[str]] = None) -> Dict[str, ConfigSpec]:
        return load_config_specs(self.configs_dir, names)
