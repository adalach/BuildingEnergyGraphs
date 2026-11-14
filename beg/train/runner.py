"""Reusable, resumable GNN training runner, driven by ``BEG_04_Train``."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from beg.models.gnn import create_model, get_model_spec, hetero_default_pool_node_types
from beg.train.dashboard import discover_run_inventory, write_run_index


def torch_load_trusted(path: Path):
    """Load local training packages which contain PyG graph objects."""
    return torch.load(path, map_location="cpu", weights_only=False)


def load_training_ready_packages(datasets_dir: Path) -> dict[str, dict[str, Any]]:
    """Discover training packages without loading their multi-gigabyte graph tensors.

    The returned descriptors contain metadata and a package path.  The runner
    materialises exactly one package at a time immediately before its run.
    """
    if not datasets_dir.exists():
        raise FileNotFoundError(f"Missing datasets directory: {datasets_dir}")
    payloads = {}
    for config_dir in sorted(path for path in datasets_dir.iterdir() if path.is_dir()):
        package = config_dir / "training_ready.pt"
        metadata_path = config_dir / "training_ready.meta.json"
        if package.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            payloads[config_dir.name] = {"package_path": package, "metadata": metadata}
    if not payloads:
        raise FileNotFoundError(f"No training_ready.pt packages found under {datasets_dir}")
    return payloads


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _sha256_hex(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=_jsonable).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


#: The modules whose contents change what a run produces: the models, the
#: hyperparameters and search space, and the training loop itself. Reporting
#: modules are deliberately absent — redrawing a dashboard must not invalidate
#: a completed run.
IMPLEMENTATION_SOURCES = ("models/gnn.py", "train/search.py", "train/runner.py")


def implementation_fingerprint() -> str:
    """Hash the local model and training code that affects a run's result.

    Half of what makes a run content-addressed: combined with the dataset
    fingerprint, it means a plan skips work already done and never presents
    results from two different revisions of the code as comparable.

    Missing a source is an error rather than something to skip. This function
    previously named the predecessor's files — ``GNNmodels.py``,
    ``advei_training.py`` — and filtered on ``path.exists()``. After the refactor
    renamed them none matched, so it silently hashed an empty source set and
    every revision of the models fingerprinted identically.
    """
    package_root = Path(__file__).resolve().parent.parent
    source_paths = [package_root / relative for relative in IMPLEMENTATION_SOURCES]

    missing = [str(path) for path in source_paths if not path.exists()]
    if missing:
        raise RuntimeError(
            "Cannot fingerprint the implementation: these sources are missing, so "
            f"runs could not be told apart across revisions: {missing}. "
            "Update IMPLEMENTATION_SOURCES if a module was renamed."
        )

    return _sha256_hex(
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "sources": {
                relative: _source_hash(path)
                for relative, path in zip(IMPLEMENTATION_SOURCES, source_paths, strict=True)
            },
        }
    )


def dataset_fingerprint(metadata: Mapping[str, Any]) -> str:
    """Fingerprint the stable, cheap-to-read training-ready metadata contract."""
    split = metadata["split"]
    return _sha256_hex(
        {
            "config_name": metadata["config_name"],
            "graph_kind": metadata["graph_kind"],
            "schema_version": metadata.get("schema_version"),
            "config_signature_hash": metadata.get("config_signature_hash"),
            "n_graphs": metadata.get("n_graphs"),
            "split_sizes": metadata.get("split_sizes"),
            "split": {
                "train_ids_sha": split.get("train_ids_sha"),
                "val_ids_sha": split.get("val_ids_sha"),
                "test_ids_sha": split.get("test_ids_sha"),
            },
        }
    )


def run_fingerprint(
    *,
    dataset_fp: str,
    implementation_fp: str,
    model_key: str,
    model_overrides: Mapping[str, Any],
    train_hparams: Mapping[str, Any],
    deterministic: bool,
) -> str:
    return _sha256_hex(
        {
            "dataset_fingerprint": dataset_fp,
            "implementation_fingerprint": implementation_fp,
            "model_key": model_key,
            "model_overrides": dict(model_overrides),
            "train_hparams": dict(train_hparams),
            "deterministic": bool(deterministic),
        }
    )


def set_all_seeds(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            pass


def _make_loaders(payload: Mapping[str, Any], batch_size: int, graph_kind: str):
    graphs = payload["graphs"]
    split = payload["split"]
    train_graphs = [graphs[index] for index in split["train_idx"]]
    val_graphs = [graphs[index] for index in split["val_idx"]]
    test_graphs = [graphs[index] for index in split["test_idx"]]
    loader_kwargs: dict[str, Any] = {"batch_size": int(batch_size), "num_workers": 0}
    if graph_kind == "hetero":
        loader_kwargs["exclude_keys"] = ["feature_names"]
    return (
        DataLoader(train_graphs, shuffle=True, **loader_kwargs),
        DataLoader(val_graphs, shuffle=False, **loader_kwargs),
        DataLoader(test_graphs, shuffle=False, **loader_kwargs),
        test_graphs,
    )


@torch.no_grad()
def _eval_epoch(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    predictions, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        predictions.append(model(batch).view(-1).detach().cpu())
        targets.append(batch.y.view(-1).detach().cpu())
    prediction = torch.cat(predictions, dim=0).numpy()
    target = torch.cat(targets, dim=0).numpy()
    mse = float(np.mean((prediction - target) ** 2))
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(prediction - target)))}


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: str,
    grad_clip_norm: Optional[float],
) -> float:
    model.train()
    total_loss, total_count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch).view(-1)
        target = batch.y.view(-1)
        loss = F.mse_loss(prediction, target)
        loss.backward()
        if grad_clip_norm is not None and math.isfinite(float(grad_clip_norm)) and float(grad_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        optimizer.step()
        total_loss += float(loss.item()) * int(target.numel())
        total_count += int(target.numel())
    return total_loss / max(total_count, 1)


def _extract_model_ids(batch: Any, fallback_count: int) -> list[str]:
    model_ids = getattr(batch, "model_id", None)
    if model_ids is None:
        return [f"graph_{index}" for index in range(fallback_count)]
    if isinstance(model_ids, str):
        return [model_ids]
    if isinstance(model_ids, (list, tuple)):
        return [str(value[0]) if isinstance(value, (list, tuple)) and len(value) == 1 else str(value) for value in model_ids]
    return [str(model_ids)]


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    return float("nan") if total <= 0 else 1.0 - residual / total


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_jsonable), encoding="utf-8")


def build_training_plan(
    *,
    payloads: Mapping[str, Mapping[str, Any]],
    jobs: Sequence[Mapping[str, Any]],
    runs_dir: Path,
    target_configs: Optional[Sequence[str]] = None,
    deterministic: bool = False,
    retry_incomplete: bool = False,
) -> list[dict[str, Any]]:
    """Cross selected model variants with compatible graph configurations."""
    existing = discover_run_inventory(runs_dir)
    existing_status = (
        existing.drop_duplicates("run_fingerprint", keep="last").set_index("run_fingerprint")["status"].to_dict()
        if not existing.empty
        else {}
    )
    selected_configs = sorted(payloads) if target_configs is None else [name for name in target_configs if name in payloads]
    code_fingerprint = implementation_fingerprint()
    plan: list[dict[str, Any]] = []

    for config_name in selected_configs:
        descriptor = payloads[config_name]
        metadata = descriptor["metadata"]
        graph_kind = str(metadata["graph_kind"])
        data_fingerprint = dataset_fingerprint(metadata)
        for job in jobs:
            if str(job["graph_kind"]) != graph_kind:
                continue
            model_overrides = dict(job["model_overrides"])
            if graph_kind == "hetero":
                # Resolve the concrete node types only after this configuration
                # is loaded for training.  This keeps dashboard/plan creation
                # lightweight even when all packages together exceed RAM.
                model_overrides.setdefault("pool_node_types", "__auto_from_graph__")
            hparams = dict(job["train_hparams"])
            fingerprint = run_fingerprint(
                dataset_fp=data_fingerprint,
                implementation_fp=code_fingerprint,
                model_key=str(job["model_key"]),
                model_overrides=model_overrides,
                train_hparams=hparams,
                deterministic=deterministic,
            )
            previous_status = existing_status.get(fingerprint)
            if previous_status == "done":
                action = "skip_done"
            elif previous_status and not retry_incomplete:
                action = "skip_incomplete"
            elif previous_status:
                action = "resume"
            else:
                action = "run"
            folder_name = f"{int(metadata['n_graphs']):05d}_{graph_kind}_{config_name}_{job['model_key']}_v{int(job['variant_idx']):03d}_{fingerprint[:12]}"
            plan.append(
                {
                    "action": action,
                    "existing_status": previous_status,
                    "config_name": config_name,
                    "graph_kind": graph_kind,
                    "model_key": str(job["model_key"]),
                    "variant_idx": int(job["variant_idx"]),
                    "model_overrides": model_overrides,
                    "train_hparams": hparams,
                    "dataset_fingerprint": data_fingerprint,
                    "implementation_fingerprint": code_fingerprint,
                    "run_fingerprint": fingerprint,
                    "out_dir": runs_dir / folder_name,
                    "package_path": Path(descriptor["package_path"]),
                    "metadata": metadata,
                }
            )
    return plan


def _load_checkpoint(path: Path) -> Optional[dict[str, Any]]:
    return torch_load_trusted(path) if path.exists() else None


def _train_one_run(item: Mapping[str, Any], *, device: str, deterministic: bool, resume_incomplete: bool) -> dict[str, Any]:
    payload = torch_load_trusted(Path(item["package_path"]))
    out_dir = Path(item["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path, history_path = out_dir / "run_meta.json", out_dir / "history.csv"
    checkpoint_path = out_dir / "last_checkpoint.pt"
    done_path = out_dir / "_DONE.txt"
    if done_path.exists():
        done_path.unlink()

    train_hparams = dict(item["train_hparams"])
    set_all_seeds(int(train_hparams["train_seed"]), deterministic=deterministic)
    train_loader, val_loader, test_loader, test_graphs = _make_loaders(payload, int(train_hparams["batch_size"]), str(item["graph_kind"]))
    sample_graph = payload["graphs"][payload["split"]["train_idx"][0]]
    spec = get_model_spec(str(item["model_key"]))
    model_overrides = dict(item["model_overrides"])
    if model_overrides.get("pool_node_types") == "__auto_from_graph__":
        model_overrides["pool_node_types"] = [
            node_type for node_type in hetero_default_pool_node_types(sample_graph.edge_types) if node_type != "weather_city"
        ]
    model = create_model(str(item["model_key"]), sample_graph, **model_overrides).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_hparams["learning_rate"]), weight_decay=float(train_hparams["weight_decay"]))

    old_meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    meta = {
        "status": "running",
        "started_utc": old_meta.get("started_utc", _now_utc_iso()),
        "resumed_utc": _now_utc_iso() if item["action"] == "resume" else None,
        "resume_count": int(old_meta.get("resume_count", 0)) + int(item["action"] == "resume"),
        "config_name": item["config_name"],
        "graph_kind": item["graph_kind"],
        "model_key": item["model_key"],
        "model_name": spec.name,
        "model_family": spec.family,
        "variant_idx": item["variant_idx"],
        "model_overrides": model_overrides,
        "train_hparams": train_hparams,
        # What it ran on. `train_seconds` is meaningless without this once runs
        # come from more than one machine, which they do as soon as training
        # moves to a GPU box while the data is prepared elsewhere. The
        # implementation fingerprint already keeps CPU and CUDA results from
        # being treated as interchangeable; this says which is which.
        "hardware": {
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(0) if str(device).startswith("cuda") else platform.processor() or platform.machine()
            ),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
        },
        "dataset_fingerprint": item["dataset_fingerprint"],
        "implementation_fingerprint": item["implementation_fingerprint"],
        "run_fingerprint": item["run_fingerprint"],
        "dataset": {
            "label_col": payload["label"]["label_col"],
            "label_mean_raw_train": float(payload["label"]["train_mean_raw"]),
            "label_std_raw_train": float(payload["label"]["train_std_raw"]),
            "config_signature_hash": payload.get("config_signature_hash"),
            "split": {key: payload["split"].get(key) for key in ("train_ids_sha", "val_ids_sha", "test_ids_sha")},
        },
    }
    _write_json(meta_path, meta)

    history: list[dict[str, Any]] = []
    best_val_mse, best_epoch, best_state, start_epoch = float("inf"), -1, None, 1
    checkpoint = _load_checkpoint(checkpoint_path) if resume_incomplete else None
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = list(checkpoint["history"])
        best_val_mse = float(checkpoint["best_val_mse_z"])
        best_epoch = int(checkpoint["best_val_epoch"])
        best_state = checkpoint["best_state_dict"]
        start_epoch = int(checkpoint["epoch"]) + 1

    started = time.time()
    # A resumed run has already been paid for once. Timing only the resumed part
    # would report a run interrupted at epoch 80 and finished in 20 more as
    # costing 20 epochs, and `train_seconds` is what the paper's computational
    # cost figure reads.
    seconds_before_resume = float(old_meta.get("train_seconds") or 0.0)
    max_epochs, patience = int(train_hparams["max_epochs"]), int(train_hparams["patience"])
    # Notebooks are committed with their outputs and every tqdm refresh is stored
    # as its own stream chunk, so a bar that repaints per epoch turns one long run
    # into hundreds of near-identical lines of cursor-movement escapes. Stepping
    # by iterations bounds it at ~10 writes per run however long the run takes,
    # while still moving often enough to watch. Same reasoning as beg.ingest.loader.
    with tqdm(
        total=max_epochs,
        initial=min(start_epoch - 1, max_epochs),
        desc=f"{item['config_name']} · {item['model_key']} v{item['variant_idx']}",
        leave=False,
        unit="epoch",
        file=sys.stdout,
        miniters=max(1, int(max_epochs) // 10),
        mininterval=0,
    ) as epoch_bar:
        for epoch in range(start_epoch, max_epochs + 1):
            epoch_started = time.time()
            train_mse = _train_one_epoch(
                model, train_loader, optimizer, device=device, grad_clip_norm=train_hparams.get("grad_clip_norm")
            )
            validation = _eval_epoch(model, val_loader, device)
            history.append(
                {
                    "epoch": epoch,
                    "train_mse_z": train_mse,
                    "val_mse_z": validation["mse"],
                    "val_rmse_z": validation["rmse"],
                    "val_mae_z": validation["mae"],
                    # Recorded rather than divided out of train_seconds: early
                    # stopping makes the epoch count vary between runs, and a
                    # resumed run's total spans two sittings.
                    "epoch_seconds": float(time.time() - epoch_started),
                }
            )
            if validation["mse"] < best_val_mse - 1e-12:
                best_val_mse, best_epoch = float(validation["mse"]), epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            pd.DataFrame(history).to_csv(history_path, index=False)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                    "best_val_mse_z": best_val_mse,
                    "best_val_epoch": best_epoch,
                    "best_state_dict": best_state,
                },
                checkpoint_path,
            )
            epoch_bar.update(1)
            epoch_bar.set_postfix(val_mse=f"{validation['mse']:.4f}", best=f"{best_val_mse:.4f}", patience=f"{epoch - best_epoch}/{patience}")
            if epoch - best_epoch >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training produced no valid validation checkpoint.")
    model.load_state_dict(best_state)
    test_z = _eval_epoch(model, test_loader, device)
    prediction_kwargs: dict[str, Any] = {"batch_size": max(16, int(train_hparams["batch_size"]) * 4), "shuffle": False}
    if item["graph_kind"] == "hetero":
        prediction_kwargs["exclude_keys"] = ["feature_names"]
    prediction_loader = DataLoader(test_graphs, **prediction_kwargs)
    prediction_rows = []
    model.eval()
    with torch.no_grad():
        for batch in prediction_loader:
            batch = batch.to(device)
            predictions = model(batch).view(-1).detach().cpu().numpy()
            targets = batch.y.view(-1).detach().cpu().numpy()
            model_ids = _extract_model_ids(batch, int(getattr(batch, "num_graphs", len(predictions)) or len(predictions)))
            prediction_rows.extend(
                {"model_id": str(model_ids[index]), "y_true_z": float(targets[index]), "y_pred_z": float(predictions[index])}
                for index in range(min(len(predictions), len(targets), len(model_ids)))
            )
    predictions_df = pd.DataFrame(prediction_rows).sort_values("model_id").reset_index(drop=True)
    mean_raw, std_raw = float(payload["label"]["train_mean_raw"]), float(payload["label"]["train_std_raw"])
    predictions_df["y_true_raw"] = predictions_df["y_true_z"] * std_raw + mean_raw
    predictions_df["y_pred_raw"] = predictions_df["y_pred_z"] * std_raw + mean_raw
    predictions_df["err_raw"] = predictions_df["y_pred_raw"] - predictions_df["y_true_raw"]
    y_true, y_pred = predictions_df["y_true_raw"].to_numpy(float), predictions_df["y_pred_raw"].to_numpy(float)
    mse_raw = float(np.mean((y_pred - y_true) ** 2))
    metrics = {
        "test_mse_z": float(test_z["mse"]),
        "test_rmse_z": float(test_z["rmse"]),
        "test_mae_z": float(test_z["mae"]),
        "test_mse_raw": mse_raw,
        "test_rmse_raw": float(np.sqrt(mse_raw)),
        "test_r2_raw": float(_r2_score(y_true, y_pred)),
    }
    torch.save(model.state_dict(), out_dir / "model_state_dict.pt")
    torch.save(
        {"model_key": item["model_key"], "model_overrides": model_overrides, "train_hparams": train_hparams, "best_val_epoch": best_epoch, "best_val_mse_z": best_val_mse, "state_dict": model.state_dict()},
        out_dir / "model_package.pt",
    )
    predictions_df.to_csv(out_dir / "test_predictions.csv", index=False)
    meta.update(
        {
            "status": "done",
            "finished_utc": _now_utc_iso(),
            "train_seconds": seconds_before_resume + float(time.time() - started),
            "epochs_ran": len(history),
            "best_val_epoch": best_epoch,
            "best_val_mse_z": best_val_mse,
            "metrics": metrics,
        }
    )
    _write_json(meta_path, meta)
    done_path.write_text(f"DONE {meta['finished_utc']}", encoding="utf-8")
    return {"status": "done", "run_dir": str(out_dir), "config": item["config_name"], "model_key": item["model_key"], "variant_idx": item["variant_idx"], **metrics}


def run_training_plan(
    plan: Iterable[Mapping[str, Any]],
    *,
    runs_dir: Path,
    deterministic: bool = False,
    device: Optional[str] = None,
    continue_on_error: bool = True,
) -> pd.DataFrame:
    """Run a pre-reviewed plan with total-run and per-epoch tqdm progress bars."""
    items = list(plan)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    executable = [item for item in items if item["action"] in {"run", "resume"}]
    results: list[dict[str, Any]] = []
    with tqdm(total=len(executable), desc="Selected GNN runs", unit="run", file=sys.stdout, mininterval=0) as run_bar:
        for item in items:
            if item["action"] not in {"run", "resume"}:
                results.append({"status": item["action"], "config": item["config_name"], "model_key": item["model_key"], "variant_idx": item["variant_idx"]})
                continue
            run_bar.set_postfix(config=item["config_name"], model=item["model_key"], variant=item["variant_idx"])
            try:
                result = _train_one_run(item, device=device, deterministic=deterministic, resume_incomplete=item["action"] == "resume")
            except Exception as error:
                out_dir = Path(item["out_dir"])
                meta_path = out_dir / "run_meta.json"
                meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
                meta.update({"status": "failed", "finished_utc": _now_utc_iso(), "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}})
                _write_json(meta_path, meta)
                result = {"status": "failed", "run_dir": str(out_dir), "config": item["config_name"], "model_key": item["model_key"], "variant_idx": item["variant_idx"], "error": str(error)}
                if not continue_on_error:
                    write_run_index(runs_dir)
                    raise
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            results.append(result)
            write_run_index(runs_dir)
            run_bar.update(1)
    return pd.DataFrame(results)
