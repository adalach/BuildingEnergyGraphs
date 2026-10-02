"""What a run cost is a number the paper prints, so it has to survive a restart.

The computational cost figure in ``BEG_05_Results`` reads ``train_seconds``
from every run, so a resumed run must report the time of every sitting, not only
the last.

These tests train a tiny model so the whole plan runs in seconds.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

pytestmark = pytest.mark.torch


MODEL_IDS = ["a_0001", "b_0002", "c_0003", "d_0004", "e_0005", "f_0006"]
SPLIT = {
    "dataset_ids": MODEL_IDS,
    "idx_train": [0, 1, 2, 3],
    "idx_val": [4],
    "idx_test": [5],
    "train_ids": MODEL_IDS[:4],
    "val_ids": [MODEL_IDS[4]],
    "test_ids": [MODEL_IDS[5]],
    "seed": 1234,
    "method": "random_model_split",
    "group_column": None,
    "n_groups": 0,
    "n_bins": 0,
    "score": 0.0,
    "objective_breakdown": {},
}


@pytest.fixture()
def trained_run(tmp_path):
    """One finished run of a tiny homogeneous model, and the paths around it."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data

    from beg.datasets.tools import (
        export_all_configs,
        load_graph_manifests,
        train_label_normalisation,
    )

    cfg_full = "C1_zone_homo_no_weather"
    cfg_dir = tmp_path / "graphs" / "homo" / cfg_full
    cfg_dir.mkdir(parents=True)

    rows = []
    for model_id in MODEL_IDS:
        graph = Data(
            x=torch.randn(4, 3),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
            edge_attr=torch.zeros(3, 0),
        )
        torch.save(graph, cfg_dir / f"{model_id}.pt")
        rows.append(
            {
                "config_full_name": cfg_full, "config_id": "C1", "model_id": model_id,
                "graph_type": "homo", "path": f"homo/{cfg_full}/{model_id}.pt", "bytes": 1,
            }
        )

    graphs_root = tmp_path / "graphs"
    frame = pd.DataFrame(rows)
    frame.to_parquet(graphs_root / "graphs_manifest_homo.parquet", index=False)
    frame.iloc[0:0].to_parquet(graphs_root / "graphs_manifest_hetero.parquet", index=False)

    labels = {mid: float(index + 1) for index, mid in enumerate(MODEL_IDS)}
    label_meta = train_label_normalisation(SPLIT, labels, label_col="eui")
    datasets_dir = tmp_path / "pyg_datasets"
    export_all_configs(
        graphs_root=graphs_root, output_dir=datasets_dir,
        manifests=load_graph_manifests(graphs_root), configs=["C1"],
        dataset_ids=MODEL_IDS, split=SPLIT, label_map_raw=labels,
        label_meta=label_meta, progress=False,
    )

    runs_dir = tmp_path / "GNNruns"
    return datasets_dir, runs_dir, cfg_full


def _plan(datasets_dir, runs_dir, cfg_full, *, retry=False, epochs=3):
    from beg.train.runner import build_training_plan, load_training_ready_packages
    from beg.train.search import expand_model_jobs

    jobs = expand_model_jobs(
        model_keys=["gcn"],
        max_variants_per_model=1,
        train_overrides={"max_epochs": epochs, "patience": epochs, "batch_size": 2},
    )
    return build_training_plan(
        payloads=load_training_ready_packages(datasets_dir), jobs=jobs, runs_dir=runs_dir,
        target_configs=[cfg_full], retry_incomplete=retry,
    )


def _run_dir(runs_dir):
    return next(path for path in runs_dir.iterdir() if path.is_dir())


def test_a_finished_run_records_its_cost(trained_run):
    from beg.train.runner import run_training_plan

    datasets_dir, runs_dir, cfg_full = trained_run
    run_training_plan(_plan(datasets_dir, runs_dir, cfg_full), runs_dir=runs_dir, continue_on_error=False)

    meta = json.loads((_run_dir(runs_dir) / "run_meta.json").read_text())

    assert meta["status"] == "done"
    assert meta["train_seconds"] > 0
    assert meta["epochs_ran"] == 3
    assert meta["hardware"]["device"] in {"cpu", "cuda"}


def test_history_records_each_epoch_separately(trained_run):
    """Per-epoch cost, not inferred by dividing the total.

    Early stopping makes the epoch count differ between runs, and a resumed run's
    total spans two sittings, so the division would be wrong in both directions.
    """
    from beg.train.runner import run_training_plan

    datasets_dir, runs_dir, cfg_full = trained_run
    run_training_plan(_plan(datasets_dir, runs_dir, cfg_full), runs_dir=runs_dir, continue_on_error=False)

    history = pd.read_csv(_run_dir(runs_dir) / "history.csv")

    assert "epoch_seconds" in history.columns
    assert len(history) == 3
    assert (history["epoch_seconds"] > 0).all()


def test_a_resumed_run_reports_what_it_really_cost(trained_run):
    """A resumed run's ``train_seconds`` includes the earlier sitting."""
    from beg.train.runner import run_training_plan

    datasets_dir, runs_dir, cfg_full = trained_run
    run_training_plan(_plan(datasets_dir, runs_dir, cfg_full), runs_dir=runs_dir, continue_on_error=False)

    run_dir = _run_dir(runs_dir)
    meta_path = run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text())
    first_sitting = float(meta["train_seconds"])

    # Put it back to how an interrupted run looks on disk.
    meta["status"] = "running"
    meta_path.write_text(json.dumps(meta))
    (run_dir / "_DONE.txt").unlink()

    plan = _plan(datasets_dir, runs_dir, cfg_full, retry=True)
    assert [item["action"] for item in plan] == ["resume"], (
        "the same hyperparameters must fingerprint to the same run, or this is not a resume"
    )
    run_training_plan(plan, runs_dir=runs_dir, continue_on_error=False)

    resumed = json.loads(meta_path.read_text())

    assert resumed["resume_count"] == 1
    assert resumed["train_seconds"] >= first_sitting, (
        "a resumed run reported less than it had already spent; the earlier "
        "sitting was dropped rather than carried forward"
    )
    assert len([path for path in runs_dir.iterdir() if path.is_dir()]) == 1
