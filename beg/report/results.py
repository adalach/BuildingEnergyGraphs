"""Turning completed runs into the paper's numbers.

The input is `outputs/GNNruns/`, one directory per run. The output is a small set
of tables: which configuration won, by how much, and how stable that answer is
across the search.

Two rules shape everything here.

**Metrics are recomputed.** :func:`beg.train.verified_runs.verify_runs` recomputes
every metric from each run's own ``history.csv`` and ``test_predictions.csv`` and
checks it against what was stored; nothing proceeds if any run disagrees with itself.

**Selection is on validation, reporting is on test.** A run is chosen for a
configuration by its best validation MSE, and only then is its test score read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from beg.config.naming import CONFIG_ORDER, resolve, sort_key
from beg.train.search import MODEL_SEARCH_SPACES

#: Metrics reported for a run. ``_z`` is in the normalised target space the model
#: trained on; ``_raw`` is denormalised back to kWh/m2/yr and is what the paper
#: quotes.
TEST_METRICS = ["test_mse_z", "test_rmse_raw", "test_r2_raw"]

#: The metric a run is selected by. Validation, never test.
SELECTION_METRIC = "best_val_mse_z"


def load_verified_runs(
    runs_dir: Path,
    *,
    experiment_max_epochs: Optional[int] = 100,
    strict: bool = True,
) -> dict[str, Any]:
    """Load every completed run, with its metrics recomputed and audited.

    ``experiment_max_epochs`` keeps only runs trained under that epoch budget;
    runs with another budget (e.g. two-epoch smoke tests) are returned as
    ``excluded``. Pass ``None`` to keep everything.

    Returns a dict with ``runs`` (the experiment), ``excluded`` (completed runs
    outside the budget), ``audit`` (the per-run metric verification) and
    ``invalid`` (directories that could not be read, which includes runs that
    failed before writing predictions).
    """
    from beg.train.verified_runs import verify_runs

    manifest, audit, invalid = verify_runs(Path(runs_dir), strict=strict)

    if manifest.empty:
        raise RuntimeError(f"No completed runs under {runs_dir}.")

    if experiment_max_epochs is None:
        runs, excluded = manifest, manifest.iloc[0:0]
    else:
        budget = pd.to_numeric(manifest["max_epochs"], errors="coerce")
        keep = budget == int(experiment_max_epochs)
        runs, excluded = manifest.loc[keep].copy(), manifest.loc[~keep].copy()

    if runs.empty:
        raise RuntimeError(
            f"No runs at max_epochs={experiment_max_epochs}. Present budgets: "
            f"{sorted(pd.to_numeric(manifest['max_epochs'], errors='coerce').dropna().unique())}"
        )

    runs["config_id"] = runs["config_name"].map(lambda c: resolve(c).id)
    return {"runs": runs.reset_index(drop=True), "excluded": excluded.reset_index(drop=True),
            "audit": audit, "invalid": invalid}


def verification_summary(audit: pd.DataFrame, invalid: pd.DataFrame) -> pd.DataFrame:
    """One row per check, for a notebook to display before trusting anything."""
    match_columns = [c for c in audit.columns if c.startswith("matches_")]
    rows = [
        {
            "check": "runs whose stored metrics match a recomputation from raw files",
            "n": int(audit["all_metrics_match"].astype(bool).sum()) if len(audit) else 0,
            "of": len(audit),
            "ok": bool(audit["all_metrics_match"].astype(bool).all()) if len(audit) else True,
        }
    ]
    for column in match_columns:
        mismatches = int((~audit[column].astype(bool)).sum()) if len(audit) else 0
        rows.append(
            {
                "check": column.replace("matches_", "recomputed "),
                "n": len(audit) - mismatches,
                "of": len(audit),
                "ok": mismatches == 0,
            }
        )
    rows.append(
        {
            "check": "run directories that could not be read",
            "n": len(invalid),
            "of": len(invalid),
            "ok": True,  # reported, not fatal: a failed run legitimately has no predictions
        }
    )
    return pd.DataFrame(rows)


def coverage_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Variants per configuration and model, in configuration order."""
    table = (
        runs.groupby(["config_name", "model_key"])["run_fingerprint"]
        .nunique()
        .unstack(fill_value=0)
    )
    return table.reindex(sorted(table.index, key=sort_key))


def coverage_problems(runs: pd.DataFrame, *, expected_variants: int) -> list[str]:
    """Every way the run set falls short of the declared experiment.

    A partial set can still be tabulated; the notebook shows these problems above
    the tables so that a thin cell is not read as a result.
    """
    problems = []

    missing_configs = [c for c in CONFIG_ORDER if c not in set(runs["config_id"])]
    if missing_configs:
        problems.append(f"configurations with no runs at all: {missing_configs}")

    present = set(zip(runs["config_name"], runs["model_key"]))
    kinds = runs.groupby("config_name")["graph_kind"].first()
    absent = [
        f"{config}/{model}"
        for config in sorted(kinds.index, key=sort_key)
        for model, spec in sorted(MODEL_SEARCH_SPACES.items())
        if spec["graph_kind"] == kinds[config] and (config, model) not in present
    ]
    if absent:
        problems.append(f"{len(absent)} cell(s) with no completed run: {', '.join(absent)}")

    cells = runs.groupby(["config_name", "model_key"])["run_fingerprint"].nunique()
    short = cells[cells < expected_variants]
    if len(short):
        detail = ", ".join(f"{cfg}/{model} {n}/{expected_variants}" for (cfg, model), n in short.items())
        problems.append(f"{len(short)} cell(s) below {expected_variants} variants: {detail}")

    duplicated = int(runs["run_fingerprint"].duplicated().sum())
    if duplicated:
        problems.append(
            f"{duplicated} run(s) share a fingerprint with another, so the same "
            "configuration was trained twice under different directory names"
        )
    return problems


def select_validation_best(runs: pd.DataFrame, *, by: str = "config_name") -> pd.DataFrame:
    """The single best run per group, chosen on validation loss.

    Ties break on ``run_dir``, so the choice is reproducible. ``by`` is
    ``config_name`` for the headline table and ``["config_name", "model_key"]`` for
    the per-model breakdown.
    """
    group = [by] if isinstance(by, str) else list(by)
    frame = runs.copy()
    for column in [SELECTION_METRIC, "best_val_epoch", *TEST_METRICS]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    best = (
        frame.dropna(subset=[SELECTION_METRIC])
        .sort_values([*group, SELECTION_METRIC, "run_dir"], kind="mergesort")
        .groupby(group, as_index=False)
        .first()
    )
    return best.sort_values("config_name", key=lambda s: s.map(sort_key)).reset_index(drop=True)


def summarise_validation_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread of validation MSE per configuration and model.

    This characterises the hyperparameter sweep without using held-out metrics.
    """
    frame = runs.copy()
    frame[SELECTION_METRIC] = pd.to_numeric(frame[SELECTION_METRIC], errors="coerce")

    summary = (
        frame.dropna(subset=[SELECTION_METRIC])
        .groupby(["config_id", "config_name", "model_family", "model_key"], as_index=False)
        .agg(
            n_runs=("run_fingerprint", "nunique"),
            validation_mse_z_mean=(SELECTION_METRIC, "mean"),
            validation_mse_z_std=(SELECTION_METRIC, "std"),
            validation_mse_z_min=(SELECTION_METRIC, "min"),
        )
    )
    return (
        summary.sort_values(
            ["config_name", "model_family", "model_key"],
            key=lambda s: s.map(sort_key) if s.name == "config_name" else s,
            kind="mergesort",
        ).reset_index(drop=True)
    )


def summarise_best_runs(best: pd.DataFrame) -> pd.DataFrame:
    """The headline table: one selected run per configuration."""
    columns = [
        "config_id", "config_name", "graph_kind", "model_family", "model_key",
        "variant_idx", SELECTION_METRIC, "best_val_epoch", "epochs_ran", *TEST_METRICS,
        "train_seconds", "run_dir",
    ]
    table = best[[c for c in columns if c in best.columns]].copy()
    return table.sort_values("config_name", key=lambda s: s.map(sort_key)).reset_index(drop=True)


def configuration_ranking(runs: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    """How the configurations order, by selected and typical validation MSE.

    The ranking by the median of the whole search shows whether the selected-run
    ordering holds beyond one hyperparameter variant.
    """
    frame = runs.copy()
    frame[SELECTION_METRIC] = pd.to_numeric(frame[SELECTION_METRIC], errors="coerce")

    median = (
        frame.groupby("config_name")[SELECTION_METRIC]
        .agg(median_validation_mse="median", worst_validation_mse="max", n_runs="size")
        .reset_index()
    )
    table = (
        best[["config_id", "config_name", "model_key", SELECTION_METRIC]]
        .rename(columns={SELECTION_METRIC: "selected_validation_mse", "model_key": "selected_model"})
        .merge(median, on="config_name", how="inner")
    )
    table["rank_by_selected_validation"] = table["selected_validation_mse"].rank(ascending=True).astype(int)
    table["rank_by_median_validation"] = table["median_validation_mse"].rank(ascending=True).astype(int)
    table["rank_shift"] = table["rank_by_median_validation"] - table["rank_by_selected_validation"]
    return table.sort_values("rank_by_selected_validation").reset_index(drop=True)


def rank_agreement(ranking: pd.DataFrame) -> float:
    """Spearman correlation between selected and median validation rankings.

    The columns are already ranks, so their Pearson correlation is Spearman's
    coefficient.
    """
    return float(ranking["rank_by_selected_validation"].corr(ranking["rank_by_median_validation"]))


def weather_effect(best: pd.DataFrame) -> pd.DataFrame:
    """What each weather variant buys, within a granularity and graph kind.

    The comparison the configurations were designed for: C1→C2 and C6→C7 add a
    graph-level vector, C3→C4 and C8→C9 widen the building node, C3→C5 and
    C8→C10 add a weather node. Each pair differs in exactly one thing.
    """
    by_id = best.set_index("config_id")
    pairs = [
        ("zone", "homo", "C1", "C2", "graph-level vector"),
        ("component", "homo", "C6", "C7", "graph-level vector"),
        ("zone", "hetero", "C3", "C4", "building-node features"),
        ("component", "hetero", "C8", "C9", "building-node features"),
        ("zone", "hetero", "C3", "C5", "weather node"),
        ("component", "hetero", "C8", "C10", "weather node"),
    ]
    rows = []
    for level, kind, base, variant, how in pairs:
        if base not in by_id.index or variant not in by_id.index:
            continue
        rows.append(
            {
                "level": level,
                "graph_kind": kind,
                "baseline": base,
                "with_weather": variant,
                "weather_enters_as": how,
                "r2_baseline": float(by_id.loc[base, "test_r2_raw"]),
                "r2_with_weather": float(by_id.loc[variant, "test_r2_raw"]),
                "delta_r2": float(by_id.loc[variant, "test_r2_raw"] - by_id.loc[base, "test_r2_raw"]),
            }
        )
    return pd.DataFrame(rows)


def training_cost(runs: pd.DataFrame) -> pd.DataFrame:
    """What the search cost, per configuration.

    ``train_seconds`` is only comparable across runs from one machine; the
    ``hardware`` block in each ``run_meta.json`` records which.
    """
    frame = runs.copy()
    frame["train_seconds"] = pd.to_numeric(frame["train_seconds"], errors="coerce")
    frame["epochs_ran"] = pd.to_numeric(frame["epochs_ran"], errors="coerce")

    summary = (
        frame.groupby(["config_id", "config_name", "graph_kind"], as_index=False)
        .agg(
            n_runs=("run_fingerprint", "nunique"),
            hours_total=("train_seconds", lambda s: s.sum() / 3600.0),
            minutes_mean=("train_seconds", lambda s: s.mean() / 60.0),
            epochs_mean=("epochs_ran", "mean"),
        )
    )
    summary["seconds_per_epoch"] = (
        summary["minutes_mean"] * 60.0 / summary["epochs_mean"].replace(0, np.nan)
    )
    return summary.sort_values("config_name", key=lambda s: s.map(sort_key)).reset_index(drop=True)


def load_history(runs_dir: Path, run_dir_name: str) -> pd.DataFrame:
    """One run's per-epoch history."""
    return pd.read_csv(Path(runs_dir) / str(run_dir_name) / "history.csv")


def load_predictions(runs_dir: Path, run_dir_name: str) -> pd.DataFrame:
    """One run's per-model test predictions, denormalised where available."""
    return pd.read_csv(Path(runs_dir) / str(run_dir_name) / "test_predictions.csv")


# ---------------------------------------------------------------------------
# Stability, ranking and building order
# ---------------------------------------------------------------------------

#: What counts as a stable run, defined once and used by every all-run summary.
STABILITY_RULE = {
    "best_val_mse_z": ("<", 10.0),
}


def stable_runs(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split runs into stable and unstable under :data:`STABILITY_RULE`.

    A run that trained but produced numerically meaningless predictions is not a
    point in the search; averaged in, one overflow would dominate a configuration's
    mean. The unstable runs are returned so the notebook can report how many there
    were.
    """
    keep = pd.Series(True, index=runs.index)
    for column, (op, threshold) in STABILITY_RULE.items():
        values = pd.to_numeric(runs[column], errors="coerce")
        keep &= (values < threshold) if op == "<" else (values > threshold)
        keep &= values.notna()
    return runs.loc[keep].copy(), runs.loc[~keep].copy()


#: Pairs of buildings closer than this in simulated EUI are also scored
#: separately: ordering two buildings a few kWh apart correctly is close to a
#: coin toss for any model, so the all-pairs figure understates what matters.
MINIMUM_EUI_GAP = 25.0


def _pairwise_concordance(y_true: np.ndarray, y_pred: np.ndarray, minimum_gap: float) -> tuple[int, float]:
    """Share of building pairs whose predicted order matches the simulated order.

    Pairs tied in the simulated target have no order to recover and are left
    out, as are pairs closer than ``minimum_gap`` when a gap is given.
    """
    left, right = np.triu_indices(len(y_true), 1)
    true_diff = y_true[left] - y_true[right]
    pred_diff = y_pred[left] - y_pred[right]
    retained = true_diff != 0 if minimum_gap == 0 else np.abs(true_diff) >= minimum_gap
    count = int(retained.sum())
    if count == 0:
        return 0, float("nan")
    return count, float((np.sign(true_diff[retained]) == np.sign(pred_diff[retained])).mean())


def building_order_fidelity(
    best: pd.DataFrame,
    runs_dir: Path,
    *,
    configs: Optional[list[str]] = None,
    minimum_eui_gap: float = MINIMUM_EUI_GAP,
) -> pd.DataFrame:
    """Whether a selected model ranks buildings the way the simulation does.

    A surrogate used to shortlist design options is judged by ordering, not by
    error: it is useful if it puts the better building first. Reported as
    Spearman's rho, Kendall's tau, and the share of building pairs ordered
    correctly -- over every pair, and over pairs at least ``minimum_eui_gap``
    kWh/m2/yr apart.
    """
    from scipy.stats import kendalltau, spearmanr

    frame = best if configs is None else best[best["config_id"].isin(configs)]
    rows = []
    for _, row in frame.iterrows():
        predictions = load_predictions(runs_dir, row["run_dir"])
        y_true = predictions["y_true_raw"].to_numpy(dtype=float)
        y_pred = predictions["y_pred_raw"].to_numpy(dtype=float)
        all_pairs, all_share = _pairwise_concordance(y_true, y_pred, 0.0)
        far_pairs, far_share = _pairwise_concordance(y_true, y_pred, minimum_eui_gap)
        rows.append(
            {
                "config_id": row["config_id"],
                "model_key": row["model_key"],
                "n_test_buildings": len(predictions),
                "spearman_rho": float(spearmanr(y_true, y_pred).statistic),
                "kendall_tau": float(kendalltau(y_true, y_pred).statistic),
                "pairs": all_pairs,
                "pairs_ordered_correctly": all_share,
                "min_gap_kwh_per_m2": minimum_eui_gap,
                "pairs_apart": far_pairs,
                "pairs_apart_ordered_correctly": far_share,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model size
# ---------------------------------------------------------------------------


def parameter_counts(runs: pd.DataFrame, runs_dir: Path, graphs_root: Path) -> pd.DataFrame:
    """Trainable parameters of every run's model, rebuilt from its metadata.

    Each model is reconstructed from the architecture recorded in its
    ``run_meta.json`` and one graph of its configuration, then run forward once
    so lazily sized layers take their final shape. The count depends on input
    width, which is fixed per configuration, so one graph each is enough.

    Takes the experiment's verified runs, since the run directory also holds trial
    runs that are not part of the search.
    """
    import json
    import warnings

    import torch
    from torch_geometric.loader import DataLoader

    from beg.models.gnn import create_model

    runs_dir, graphs_root = Path(runs_dir), Path(graphs_root)
    manifests = pd.concat(
        [pd.read_parquet(graphs_root / f"graphs_manifest_{kind}.parquet") for kind in ("homo", "hetero")],
        ignore_index=True,
    )
    first = manifests.sort_values(["config_full_name", "model_id"]).groupby("config_full_name").first()
    # Manifest paths are relative to graphs_root.
    graph_path = {str(cfg): graphs_root / str(path) for cfg, path in first["path"].items()}

    graphs: dict[str, Any] = {}
    rows = []
    for run in runs.itertuples(index=False):
        meta = json.loads((runs_dir / str(run.run_dir) / "run_meta.json").read_text(encoding="utf-8"))
        config = str(meta["config_name"])
        if config not in graphs:
            graphs[config] = torch.load(graph_path[config], map_location="cpu", weights_only=False)

        loader_kwargs: dict[str, Any] = {"batch_size": 1, "shuffle": False}
        if meta["graph_kind"] == "hetero":
            loader_kwargs["exclude_keys"] = ["feature_names"]
        # C5 and C10 carry a weather node that only sends messages, and PyG warns
        # that it is never updated. The warning names a source file by absolute
        # path, so it is silenced to keep it out of the stored notebook output.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="There exist node types")
            model = create_model(str(meta["model_key"]), graphs[config], **dict(meta.get("model_overrides") or {}))
            model.eval()
            with torch.no_grad():
                model(next(iter(DataLoader([graphs[config]], **loader_kwargs))))

        rows.append(
            {
                "run_dir": str(run.run_dir),
                "config_id": resolve(config).id,
                "config_name": config,
                "graph_kind": meta["graph_kind"],
                "model_family": meta["model_family"],
                "model_key": meta["model_key"],
                "variant_idx": meta["variant_idx"],
                "hidden": (meta.get("model_overrides") or {}).get("hidden"),
                "num_layers": (meta.get("model_overrides") or {}).get("num_layers"),
                "trainable_parameters": int(sum(q.numel() for q in model.parameters() if q.requires_grad)),
                "total_parameters": int(sum(q.numel() for q in model.parameters())),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["config_name", "model_key", "variant_idx"], key=lambda c: c.map(sort_key) if c.name == "config_name" else c)
        .reset_index(drop=True)
    )


def parameter_count_ranges(counts: pd.DataFrame) -> pd.DataFrame:
    """Parameter ranges per model family: the appendix table."""
    return (
        counts.groupby(["model_family", "model_key"], as_index=False)
        .agg(
            runs=("run_dir", "count"),
            configurations=("config_name", "nunique"),
            trainable_min=("trainable_parameters", "min"),
            trainable_median=("trainable_parameters", "median"),
            trainable_max=("trainable_parameters", "max"),
        )
        .sort_values(["model_family", "model_key"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Computational cost
# ---------------------------------------------------------------------------

#: Modification times have two-second resolution once a run directory has passed
#: through a ZIP archive, which is how runs trained on another machine come back.
TIMESTAMP_RESOLUTION_S = 2.0


def load_simulation_timing(path: Path) -> pd.DataFrame:
    """EnergyPlus wall-clock time and SQL output size, one row per model."""
    frame = pd.read_csv(path)
    required = {"model_id", "simulation_time_s", "sql_size_bytes"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{path} is missing columns: {sorted(missing)}")
    frame["sql_size_mib"] = pd.to_numeric(frame["sql_size_bytes"], errors="coerce") / 1024.0**2
    frame["simulation_time_s"] = pd.to_numeric(frame["simulation_time_s"], errors="coerce")
    return frame


def model_package_sizes(runs: pd.DataFrame, runs_dir: Path) -> pd.DataFrame:
    """The on-disk size of every run's deployable model package."""
    runs_dir = Path(runs_dir)
    rows = []
    for run in runs.itertuples(index=False):
        package = runs_dir / str(run.run_dir) / "model_package.pt"
        if not package.exists():
            raise FileNotFoundError(f"Missing model package for {run.run_dir}")
        rows.append(
            {
                "run_dir": str(run.run_dir),
                "config_name": run.config_name,
                "graph_kind": run.graph_kind,
                "package_mib": package.stat().st_size / 1024.0**2,
            }
        )
    return pd.DataFrame(rows)


def prediction_time_upper_bound(best: pd.DataFrame, runs_dir: Path) -> pd.DataFrame:
    """An upper bound on prediction time per building, for each selected run.

    The trainer does not time inference, so the only record is the gap between
    the last checkpoint and the predictions file written after it. The timestamp
    resolution, :data:`TIMESTAMP_RESOLUTION_S`, is added to every gap and the
    result divided by the number of test buildings. It bounds the cost from
    above; it does not measure it. The published figure uses the same method.
    """
    runs_dir = Path(runs_dir)
    rows = []
    for _, row in best.iterrows():
        directory = runs_dir / str(row["run_dir"])
        predictions = directory / "test_predictions.csv"
        checkpoint = directory / "last_checkpoint.pt"
        recorded = max(0.0, predictions.stat().st_mtime - checkpoint.stat().st_mtime)
        n_test = len(pd.read_csv(predictions, usecols=["model_id"]))
        rows.append(
            {
                "config_id": row["config_id"],
                "model_key": row["model_key"],
                "recorded_gap_s": recorded,
                "n_test_buildings": n_test,
                "ms_per_building_upper_bound": (recorded + TIMESTAMP_RESOLUTION_S) * 1000.0 / n_test,
            }
        )
    return pd.DataFrame(rows)


def computational_cost_summary(
    runs: pd.DataFrame,
    simulation: pd.DataFrame,
    packages: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    n_models: int,
) -> pd.DataFrame:
    """Simulating the dataset against training and deploying its surrogates.

    ``n_models`` is the dataset size the prediction time is extrapolated to.
    """
    train_s = pd.to_numeric(runs["train_seconds"], errors="coerce")
    sim_s = simulation["simulation_time_s"].dropna()
    sql_bytes = pd.to_numeric(simulation["sql_size_bytes"], errors="coerce").dropna()
    package_bytes = packages["package_mib"] * 1024.0**2
    mean_ms = float(prediction["ms_per_building_upper_bound"].mean())

    rows = [
        ("EnergyPlus simulations timed", f"{len(sim_s):,} of {len(simulation):,}"),
        ("EnergyPlus time, total", f"{sim_s.sum() / 3600:,.1f} h"),
        ("EnergyPlus time, median per model", f"{sim_s.median():.1f} s"),
        ("EnergyPlus SQL output, total", f"{sql_bytes.sum() / 1024**4:.2f} TiB"),
        ("GNN search, total training time", f"{train_s.sum() / 3600:,.1f} h over {len(runs):,} runs"),
        ("GNN training as share of simulation time", f"{100 * train_s.sum() / sim_s.sum():.2f} %"),
        ("model packages, total", f"{package_bytes.sum() / 1024**3:.2f} GiB"),
        ("model packages as share of SQL output", f"{100 * package_bytes.sum() / sql_bytes.sum():.3f} %"),
        ("prediction, upper bound per building", f"{mean_ms:.3f} ms (mean over selected runs)"),
        (f"prediction, upper bound for all {n_models:,} models", f"{mean_ms * n_models / 1000:.1f} s"),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value"])
