"""Persistent GNN run inventory and notebook-facing coverage views."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from beg.config.naming import sort_key

RUN_INDEX_COLUMNS = [
    "run_dir",
    "status",
    "config",
    "graph_kind",
    "model_key",
    "variant_idx",
    "run_fingerprint",
    "dataset_fingerprint",
    "implementation_fingerprint",
    "started_utc",
    "finished_utc",
    "epochs_ran",
    "best_val_epoch",
    "best_val_mse_z",
    "test_rmse_raw",
    "test_r2_raw",
    "train_seconds",
    "error_type",
    "error_message",
]


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def discover_run_inventory(runs_dir: Path) -> pd.DataFrame:
    """Read every run folder, including interrupted and failed attempts."""
    rows: list[dict[str, Any]] = []
    if not runs_dir.exists():
        return pd.DataFrame(columns=RUN_INDEX_COLUMNS)

    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        meta_path = run_dir / "run_meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        status = str(meta.get("status", "unknown"))
        if (run_dir / "_DONE.txt").exists():
            status = "done"
        elif status == "running":
            # A new notebook kernel cannot know whether an old process survived.
            # Surface this conservatively and require an explicit retry choice.
            status = "interrupted"

        metrics = meta.get("metrics") or {}
        error = meta.get("error") or {}
        train_hparams = meta.get("train_hparams") or {}
        model_overrides = meta.get("model_overrides") or {}
        row = {
            "run_dir": str(run_dir),
            "status": status,
            "config": str(meta.get("config_name", "")),
            "graph_kind": str(meta.get("graph_kind", "")),
            "model_key": str(meta.get("model_key", "")),
            "variant_idx": _as_int(meta.get("variant_idx")),
            "run_fingerprint": str(meta.get("run_fingerprint", "")),
            "dataset_fingerprint": str(meta.get("dataset_fingerprint", "")),
            "implementation_fingerprint": str(meta.get("implementation_fingerprint", "")),
            "started_utc": meta.get("started_utc"),
            "finished_utc": meta.get("finished_utc"),
            "epochs_ran": _as_int(meta.get("epochs_ran")),
            "best_val_epoch": _as_int(meta.get("best_val_epoch")),
            "best_val_mse_z": _as_float(meta.get("best_val_mse_z")),
            "test_rmse_raw": _as_float(metrics.get("test_rmse_raw")),
            "test_r2_raw": _as_float(metrics.get("test_r2_raw")),
            "train_seconds": _as_float(meta.get("train_seconds")),
            "error_type": error.get("type"),
            "error_message": error.get("message"),
            **{f"model__{key}": value for key, value in model_overrides.items() if key != "pool_node_types"},
            **{f"train__{key}": value for key, value in train_hparams.items()},
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=RUN_INDEX_COLUMNS)
    return pd.DataFrame(rows).sort_values(
        ["config", "model_key", "variant_idx", "started_utc"], na_position="last"
    ).reset_index(drop=True)


def write_run_index(runs_dir: Path) -> pd.DataFrame:
    """Refresh the cheap-to-read global run index after each attempt."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    inventory = discover_run_inventory(runs_dir)
    target = runs_dir / "run_index.csv"
    temp = target.with_suffix(".tmp.csv")
    inventory.to_csv(temp, index=False)
    temp.replace(target)
    return inventory


def training_plan_dataframe(plan: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Make a notebook-displayable view without serialising graph payloads."""
    rows = []
    for item in plan:
        rows.append(
            {
                "action": item.get("action"),
                "existing_status": item.get("existing_status"),
                "config": item.get("config_name"),
                "graph_kind": item.get("graph_kind"),
                "model_key": item.get("model_key"),
                "variant_idx": item.get("variant_idx"),
                "run_fingerprint": str(item.get("run_fingerprint", ""))[:12],
                "model_overrides": item.get("model_overrides"),
                "train_hparams": item.get("train_hparams"),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["action", "existing_status", "config", "graph_kind", "model_key", "variant_idx", "run_fingerprint"]
        )
    return pd.DataFrame(rows).sort_values(["graph_kind", "config", "model_key", "variant_idx"]).reset_index(drop=True)


def build_coverage_tables(
    *,
    inventory: pd.DataFrame,
    plan: Iterable[Mapping[str, Any]],
    configs: Sequence[str],
    model_keys: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Coverage of the plan by the runs on disk: detailed, compact and per configuration.

    A planned run is completed when a finished run carries its fingerprint, or the
    fingerprint it has under the torch/CUDA build that trained it
    (``existing_run_fingerprint``, set by ``beg.train.guard.match_runs_from_other_builds``).
    """
    configs = sorted(configs, key=sort_key)
    status: dict[str, str] = {}
    rmse: dict[str, float] = {}
    for row in inventory.to_dict("records") if not inventory.empty else []:
        fingerprint, state = str(row.get("run_fingerprint", "")), str(row.get("status", ""))
        if status.get(fingerprint) != "done":
            status[fingerprint] = state
        if state == "done":
            rmse[fingerprint] = _as_float(row.get("test_rmse_raw"))

    counts: dict[tuple[str, str], dict[str, Any]] = {
        (config, model_key): {"selected": 0, "done": 0, "failed": 0, "interrupted": 0, "rmse": []}
        for config in configs
        for model_key in model_keys
    }
    for item in plan:
        cell = counts.get((item.get("config_name"), item.get("model_key")))
        if cell is None:
            continue
        cell["selected"] += 1
        fingerprints = [item.get("run_fingerprint"), item.get("existing_run_fingerprint")]
        states = {status.get(str(fp)) for fp in fingerprints if fp}
        if "done" in states:
            cell["done"] += 1
            cell["rmse"] += [rmse[str(fp)] for fp in fingerprints if fp and str(fp) in rmse]
        elif "failed" in states:
            cell["failed"] += 1
        elif states & {"interrupted", "running"}:
            cell["interrupted"] += 1

    coverage = pd.DataFrame(
        [
            {
                "config": config,
                "model_key": model_key,
                "selected_variants": cell["selected"],
                "completed_variants": cell["done"],
                "failed_variants": cell["failed"],
                "interrupted_variants": cell["interrupted"],
                "best_test_rmse_raw": min(cell["rmse"]) if cell["rmse"] else float("nan"),
            }
            for (config, model_key), cell in counts.items()
        ]
    )
    compact = coverage.copy()
    compact["coverage"] = compact.apply(
        lambda row: f"{int(row.completed_variants)}/{int(row.selected_variants)}"
        + (f"  !{int(row.failed_variants)}" if int(row.failed_variants) else "")
        + (f"  ~{int(row.interrupted_variants)}" if int(row.interrupted_variants) else ""),
        axis=1,
    )
    matrix = compact.pivot(index="config", columns="model_key", values="coverage").reindex(index=list(configs), columns=list(model_keys))
    by_config = (
        coverage.groupby("config", as_index=False)[["selected_variants", "completed_variants", "failed_variants", "interrupted_variants"]]
        .sum()
        .sort_values("config", key=lambda s: s.map(sort_key))
        .reset_index(drop=True)
    )
    return coverage, matrix, by_config


def plot_coverage_matrix(coverage: pd.DataFrame, *, title: str = "GNN variant coverage"):
    """Plot completed/selected variants by configuration and model family."""
    import matplotlib.pyplot as plt

    configs = list(dict.fromkeys(coverage["config"].tolist()))
    model_keys = list(dict.fromkeys(coverage["model_key"].tolist()))
    if not configs or not model_keys:
        raise ValueError("Coverage is empty; select at least one config and one model family.")

    lookup = coverage.set_index(["config", "model_key"])
    # Cells with nothing planned (a model family the configuration's graph kind cannot use) stay blank
    values = np.full((len(configs), len(model_keys)), np.nan)
    labels = np.full(values.shape, "", dtype=object)
    for i, config in enumerate(configs):
        for j, model_key in enumerate(model_keys):
            row = lookup.loc[(config, model_key)]
            selected = int(row["selected_variants"])
            if not selected:
                continue
            done = int(row["completed_variants"])
            values[i, j] = done / selected
            suffix = ""
            if int(row["failed_variants"]):
                suffix += f"\n!{int(row['failed_variants'])} failed"
            if int(row["interrupted_variants"]):
                suffix += f"\n~{int(row['interrupted_variants'])} paused"
            labels[i, j] = f"{done}/{selected}{suffix}"

    cmap = plt.get_cmap("YlGn").copy()
    cmap.set_bad("#f2f2f2")
    fig, ax = plt.subplots(figsize=(max(8, 1.55 * len(model_keys)), max(4, 0.6 * len(configs) + 2)))
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(model_keys)), labels=model_keys, rotation=30, ha="right")
    ax.set_yticks(range(len(configs)), labels=configs)
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            colour = "white" if values[i, j] > 0.6 else "black"
            ax.text(j, i, labels[i, j], ha="center", va="center", fontsize=8, color=colour)
    fig.colorbar(image, ax=ax, label="completed share of selected variants")
    fig.tight_layout()
    return fig, ax
