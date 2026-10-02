"""Read completed runs and recompute their metrics from each run's own files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from beg.settings import output_dir

DEFAULT_RUNS_DIR = output_dir() / "GNNruns"

METADATA_FIELDS = {
    "status": "status",
    "config_name": "config_name",
    "graph_kind": "graph_kind",
    "model_family": "model_family",
    "model_name": "model_name",
    "model_key": "model_key",
    "variant_idx": "variant_idx",
    "train_seconds": "train_seconds",
    "run_fingerprint": "run_fingerprint",
    "dataset_fingerprint": "dataset_fingerprint",
    "implementation_fingerprint": "implementation_fingerprint",
    "started_utc": "started_utc",
    "finished_utc": "finished_utc",
    "error_type": "error.type",
    "error_message": "error.message",
    "batch_size": "train_hparams.batch_size",
    "grad_clip_norm": "train_hparams.grad_clip_norm",
    "learning_rate": "train_hparams.learning_rate",
    "weight_decay": "train_hparams.weight_decay",
    "max_epochs": "train_hparams.max_epochs",
    "patience": "train_hparams.patience",
    "train_seed": "train_hparams.train_seed",
    "hidden": "model_overrides.hidden",
    "num_layers": "model_overrides.num_layers",
    "dropout": "model_overrides.dropout",
    "pooling": "model_overrides.pooling",
    "use_layernorm": "model_overrides.use_layernorm",
    "use_residual": "model_overrides.use_residual",
    "set2set_steps": "model_overrides.set2set_steps",
}

def _get_path(data: dict[str, Any], path: str, default: Any = np.nan) -> Any:
    current: Any = data
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _numeric_column(frame: pd.DataFrame, column: str, path: Path) -> np.ndarray:
    if column not in frame.columns:
        raise KeyError(f"{path} is missing required column {column!r}")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _history_metrics(history_path: Path) -> dict[str, float]:
    history = pd.read_csv(history_path)
    epochs = _numeric_column(history, "epoch", history_path)
    train = _numeric_column(history, "train_mse_z", history_path)
    validation = _numeric_column(history, "val_mse_z", history_path)
    valid = np.isfinite(epochs) & np.isfinite(train) & np.isfinite(validation)
    if not valid.any():
        raise ValueError(f"{history_path} has no finite epoch/train/validation rows")

    epochs = epochs[valid]
    train = train[valid]
    validation = validation[valid]
    best_index = int(np.argmin(validation))
    return {
        "best_val_epoch": int(epochs[best_index]),
        "best_val_mse_z": float(validation[best_index]),
        "epochs_ran": int(epochs[-1]),
        "final_train_mse_z": float(train[-1]),
        "final_val_mse_z": float(validation[-1]),
        "final_val_minus_train_mse_z": float(validation[-1] - train[-1]),
    }


def _prediction_metrics(predictions_path: Path) -> dict[str, float]:
    predictions = pd.read_csv(predictions_path)
    true_z = _numeric_column(predictions, "y_true_z", predictions_path)
    pred_z = _numeric_column(predictions, "y_pred_z", predictions_path)
    true_raw = _numeric_column(predictions, "y_true_raw", predictions_path)
    pred_raw = _numeric_column(predictions, "y_pred_raw", predictions_path)

    z_mask = np.isfinite(true_z) & np.isfinite(pred_z)
    raw_mask = np.isfinite(true_raw) & np.isfinite(pred_raw)
    if not z_mask.any() or not raw_mask.any():
        raise ValueError(f"{predictions_path} has no finite prediction pairs")

    z_error = pred_z[z_mask] - true_z[z_mask]
    raw_error = pred_raw[raw_mask] - true_raw[raw_mask]
    raw_truth = true_raw[raw_mask]
    raw_total = float(np.sum((raw_truth - np.mean(raw_truth)) ** 2))
    if raw_total <= 0.0:
        raise ValueError(f"{predictions_path} has zero raw target variance")

    return {
        "test_mae_z": float(np.mean(np.abs(z_error))),
        "test_mse_z": float(np.mean(z_error**2)),
        "test_rmse_z": float(np.sqrt(np.mean(z_error**2))),
        "test_mse_raw": float(np.mean(raw_error**2)),
        "test_rmse_raw": float(np.sqrt(np.mean(raw_error**2))),
        "test_r2_raw": float(1.0 - np.sum(raw_error**2) / raw_total),
    }


def _matches(stored: Any, recomputed: float) -> bool:
    try:
        return bool(np.isclose(float(stored), recomputed, rtol=1e-5, atol=1e-8))
    except (TypeError, ValueError):
        return False


def verify_runs(
    runs_dir: Path = DEFAULT_RUNS_DIR,
    *,
    strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Completed runs, with metrics recomputed from ``history.csv`` and ``test_predictions.csv``.

    Returns ``(runs, audit, invalid)``: one row per completed run, the stored
    against the recomputed metrics, and the directories that could not be used
    (including runs recorded as failed). With ``strict``, a run whose stored
    metrics disagree with the recomputation, or an unreadable run, raises.
    """

    records: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, str]] = []

    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir() and not path.name.startswith(".")):
        meta_path = run_dir / "run_meta.json"
        done_path = run_dir / "_DONE.txt"
        history_path = run_dir / "history.csv"
        predictions_path = run_dir / "test_predictions.csv"
        # A run recorded as failed has no predictions; report it as not_done.
        status = None
        if meta_path.exists():
            try:
                status = json.loads(meta_path.read_text(encoding="utf-8")).get("status")
            except (OSError, json.JSONDecodeError):
                status = None
        if status is not None and status != "done":
            invalid_rows.append(
                {"run_dir": run_dir.name, "problem": "not_done", "details": str(status)}
            )
            continue

        required = (meta_path, done_path, history_path, predictions_path)
        if not all(path.exists() for path in required):
            invalid_rows.append(
                {
                    "run_dir": run_dir.name,
                    "problem": "missing_required",
                    "details": ", ".join(path.name for path in required if not path.exists()),
                }
            )
            continue

        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if metadata.get("status") != "done":
                invalid_rows.append(
                    {"run_dir": run_dir.name, "problem": "unexpected_status", "details": str(metadata.get("status"))}
                )
                continue
            history = _history_metrics(history_path)
            predictions = _prediction_metrics(predictions_path)
        except Exception as error:
            invalid_rows.append(
                {"run_dir": run_dir.name, "problem": type(error).__name__, "details": str(error)}
            )
            continue

        record = {"run_dir": run_dir.name}
        record.update({name: _get_path(metadata, path) for name, path in METADATA_FIELDS.items()})
        record.update(history)
        record.update(predictions)
        records.append(record)

        stored_metrics = {
            "best_val_mse_z": metadata.get("best_val_mse_z"),
            "best_val_epoch": metadata.get("best_val_epoch"),
            "epochs_ran": metadata.get("epochs_ran"),
            "test_mae_z": _get_path(metadata, "metrics.test_mae_z"),
            "test_mse_z": _get_path(metadata, "metrics.test_mse_z"),
            "test_rmse_raw": _get_path(metadata, "metrics.test_rmse_raw"),
            "test_r2_raw": _get_path(metadata, "metrics.test_r2_raw"),
        }
        audit = {"run_dir": run_dir.name}
        audit.update({f"stored_{name}": value for name, value in stored_metrics.items()})
        audit.update({f"recomputed_{name}": record[name] for name in stored_metrics})
        audit.update({f"matches_{name}": _matches(value, record[name]) for name, value in stored_metrics.items()})
        audit["all_metrics_match"] = bool(all(audit[f"matches_{name}"] for name in stored_metrics))
        audit_rows.append(audit)

    manifest = pd.DataFrame(records)
    audit_df = pd.DataFrame(audit_rows)
    invalid_df = pd.DataFrame(invalid_rows)
    if manifest.empty:
        raise RuntimeError(f"No valid completed runs found under {runs_dir}")

    numeric_columns = [
        "variant_idx", "train_seconds", "best_val_epoch", "best_val_mse_z", "epochs_ran",
        "final_train_mse_z", "final_val_mse_z", "final_val_minus_train_mse_z",
        "test_mae_z", "test_mse_z", "test_rmse_z", "test_mse_raw", "test_rmse_raw", "test_r2_raw",
        "batch_size", "learning_rate", "weight_decay", "max_epochs", "patience",
        "hidden", "num_layers", "dropout", "set2set_steps",
    ]
    for column in numeric_columns:
        if column in manifest.columns:
            manifest[column] = pd.to_numeric(manifest[column], errors="coerce")

    # Runs recorded as failed are reported, not counted as problems.
    suspicious = invalid_df if invalid_df.empty else invalid_df.loc[invalid_df["problem"] != "not_done"]
    mismatched = 0 if audit_df.empty else int((~audit_df["all_metrics_match"]).sum())
    if strict and (not suspicious.empty or mismatched):
        raise RuntimeError(
            f"Raw-folder verification failed under {runs_dir}: "
            f"{mismatched} run(s) whose stored metrics disagree with a recomputation, "
            f"{len(suspicious)} unreadable or incomplete run(s). "
            f"Runs recorded as failed are excluded from this count and reported separately."
        )

    return manifest.sort_values("run_dir", kind="mergesort").reset_index(drop=True), audit_df, invalid_df
