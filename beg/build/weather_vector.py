"""Graph-level weather vectors, for the derived configurations C2 and C7.

C2 and C7 differ from C1 and C6 only in carrying the city's weather as a single
vector attached to the graph. They have no configuration file: their graphs are
copies of the base graphs with one extra attribute.

``beg.config.naming`` knows which configuration derives from which, through
``ConfigId.derived_from``.

**The normalisation is fitted on every model being built, before the train/test
split happens in BEG_03_Datasets.** Test-set weather statistics therefore inform
the training features of C2 and C7. The published results were produced this way,
and fitting on the training split would change C2 and C7; see ``docs/limitations.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text as sa_text

from beg.build.export import prune_graph_dir, safe_artifact_stem

#: Attribute the graph-level weather vector is stored under. `beg.models.gnn`
#: reads this name, so changing it breaks every trained run.
WEATHER_ATTR_NAME = "weather_city"

#: Columns of ``ep.weather_city`` that identify rather than describe.
WEATHER_EXCLUDE_COLS = frozenset(
    {"model_id", "instance_name", "city_key", "epw_file_name", "ddy_file_name",
     "country", "state_province", "source", "wmo", "time_zone"}
)


@dataclass
class WeatherFeatures:
    """Normalised per-model weather vectors, and the statistics used."""

    vectors: dict = field(repr=False, default_factory=dict)
    columns: list = field(repr=False, default_factory=list)
    means: pd.Series | None = field(repr=False, default=None)
    stds: pd.Series | None = field(repr=False, default=None)

    def __len__(self) -> int:
        return len(self.vectors)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "models_with_weather": len(self.vectors),
                    "feature_columns": len(self.columns),
                    "normalisation": "z-score over all models built",
                }
            ]
        )

    def describe_columns(self, n: int = 8) -> pd.DataFrame:
        """The first ``n`` feature columns with the statistics used to scale them."""
        if self.means is None:
            return pd.DataFrame(columns=["column", "mean", "std"])
        cols = self.columns[:n]
        return pd.DataFrame(
            {"column": cols, "mean": self.means[cols].values, "std": self.stds[cols].values}
        )


def load_weather_features(
    engine,
    model_ids: Sequence[str],
    schema: str = "ep",
    table: str = "weather_city",
) -> WeatherFeatures:
    """Read ``ep.weather_city``, keep the numeric columns, and z-score them.

    Statistics are computed over ``model_ids``, the models being built, not the
    training split (see the module docstring).
    """
    import torch

    with engine.connect() as con:
        exists = con.execute(
            sa_text("SELECT to_regclass(:name)"), {"name": f"{schema}.{table}"}
        ).scalar()
        if exists is None:
            raise RuntimeError(
                f"No weather source at {schema}.{table}. Run BEG_01_Ingest first."
            )
        frame = pd.read_sql_query(sa_text(f'SELECT * FROM "{schema}"."{table}"'), con)

    if "model_id" not in frame.columns:
        raise RuntimeError(f"{schema}.{table} has no model_id column")

    frame["model_id"] = frame["model_id"].astype(str).str.strip()
    wanted = {str(m) for m in model_ids}
    frame = (
        frame[frame["model_id"].isin(wanted)]
        .sort_values("model_id")
        .drop_duplicates(subset="model_id", keep="first")
    )

    numeric = [
        c
        for c in frame.select_dtypes(include=[np.number]).columns
        if c not in WEATHER_EXCLUDE_COLS
    ]
    if not numeric:
        raise RuntimeError("No numeric weather columns remained after exclusions.")

    means = frame[numeric].mean(skipna=True)
    # Population std (ddof=0); a constant column gets std 1 to avoid dividing by zero.
    stds = frame[numeric].std(skipna=True, ddof=0).replace(0, 1.0)

    normalised = frame.copy()
    normalised[numeric] = (normalised[numeric] - means) / stds

    vectors = {
        str(row["model_id"]): torch.tensor(
            row[numeric].to_numpy(dtype=np.float32), dtype=torch.float32
        )
        for _, row in normalised[["model_id"] + numeric].iterrows()
    }

    return WeatherFeatures(vectors=vectors, columns=numeric, means=means, stds=stds)


def _unwrap_graph(obj: Any) -> Any:
    """The graph itself, unwrapped from a dict, list or tuple container if it has one."""
    if isinstance(obj, dict):
        for key in ("data", "graph", "pyg", "g"):
            if key in obj:
                return obj[key]
    if isinstance(obj, (list, tuple)) and obj:
        return obj[0]
    return obj


def _save_like(original: Any, graph: Any, out_path: Path) -> None:
    """Save ``graph`` in the same container shape ``original`` used."""
    import torch

    if isinstance(original, dict):
        for key in ("data", "graph", "pyg", "g"):
            if key in original:
                updated = dict(original)
                updated[key] = graph
                torch.save(updated, out_path)
                return
    elif isinstance(original, (list, tuple)) and original:
        updated = list(original)
        updated[0] = graph
        torch.save(tuple(updated) if isinstance(original, tuple) else updated, out_path)
        return
    torch.save(graph, out_path)


@dataclass
class WeatherAttachResult:
    rows: list = field(repr=False, default_factory=list)

    @property
    def saved(self) -> int:
        return sum(r["saved"] for r in self.rows)

    def summary(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(
                columns=["base_config", "weather_config", "input_graphs", "saved",
                         "skipped_existing", "missing_weather", "removed_stale"]
            )
        return pd.DataFrame(self.rows).sort_values("base_config").reset_index(drop=True)


def attach_weather_vectors(
    *,
    base_configs: Sequence[str],
    full_name_of: dict,
    weather: WeatherFeatures,
    graphs_root: Path,
    force_rebuild: bool = False,
) -> WeatherAttachResult:
    """Copy each base configuration's graphs, adding the graph-level weather vector.

    ``base_configs`` are the configurations something derives *from* — C1 and C6.
    The destination is the derived configuration's full name, taken from
    ``beg.config.naming``. A model with no weather row is skipped and counted in
    ``missing_weather``; it never gets a zero vector.
    """
    import torch

    from beg.config.naming import CONFIGS, resolve

    homo_root = Path(graphs_root) / "homo"
    result = WeatherAttachResult()

    derived_of = {c.derived_from: c for c in CONFIGS if c.derived_from}

    for base in base_configs:
        base_id = resolve(base).id
        derived = derived_of.get(base_id)
        if derived is None:
            continue

        base_full = str(full_name_of[base])
        in_dir = homo_root / base_full
        out_dir = homo_root / derived.full_name

        available = sorted(in_dir.glob("*.pt"))
        keep = [p.stem for p in available if p.stem in weather.vectors]
        removed = prune_graph_dir(out_dir, keep, wipe_all=force_rebuild)

        saved = skipped = missing = 0
        for path in available:
            out_path = out_dir / path.name
            if out_path.exists() and not force_rebuild:
                skipped += 1
                continue

            vector = weather.vectors.get(path.stem)
            if vector is None:
                missing += 1
                continue

            original = torch.load(path, map_location="cpu", weights_only=False)
            graph = _unwrap_graph(original)
            setattr(graph, WEATHER_ATTR_NAME, vector)
            _save_like(original, graph, out_path)
            saved += 1

        result.rows.append(
            {
                "base_config": base_full,
                "weather_config": derived.full_name,
                "input_graphs": len(available),
                "saved": saved,
                "skipped_existing": skipped,
                "missing_weather": missing,
                "removed_stale": removed,
            }
        )

    return result


__all__ = [
    "WEATHER_ATTR_NAME",
    "WEATHER_EXCLUDE_COLS",
    "WeatherFeatures",
    "WeatherAttachResult",
    "load_weather_features",
    "attach_weather_vectors",
    "safe_artifact_stem",
]
