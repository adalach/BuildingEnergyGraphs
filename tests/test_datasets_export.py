"""The packaging stage: what it must not leak, and what it must not rebuild.

The label normalisation is fitted on the training split alone; a leak of the test
distribution into the target would flatter all ten configurations equally and go
unnoticed in the comparison between them. A rerun reuses a package unless what
it was built from has changed.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from beg.datasets.tools import (
    config_signature_for,
    export_all_configs,
    load_graph_manifests,
    train_label_normalisation,
)

pytestmark = pytest.mark.torch


MODEL_IDS = ["lodz_0011", "metz_0014", "genoa_0020", "turin_0011"]
SPLIT = {
    "dataset_ids": MODEL_IDS,
    "idx_train": [0, 1],
    "idx_val": [2],
    "idx_test": [3],
    "train_ids": ["lodz_0011", "metz_0014"],
    "val_ids": ["genoa_0020"],
    "test_ids": ["turin_0011"],
    "seed": 1234,
    "method": "random_model_split",
    "group_column": None,
    "n_groups": 0,
    "n_bins": 0,
    "score": 0.0,
    "objective_breakdown": {},
}


@pytest.fixture()
def graph_set(tmp_path):
    """A C1 export directory with one graph per model, plus its manifest."""
    torch = pytest.importorskip("torch")
    from torch_geometric.data import Data

    cfg_full = "C1_zone_homo_no_weather"
    cfg_dir = tmp_path / "homo" / cfg_full
    cfg_dir.mkdir(parents=True)

    rows = []
    for i, model_id in enumerate(MODEL_IDS):
        graph = Data(
            x=torch.zeros(3 + i, 4),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            edge_attr=torch.zeros(2, 0),
        )
        # Files keep the database's own spelling; ids downstream are lower case.
        name = model_id.title()
        torch.save(graph, cfg_dir / f"{name}.pt")
        rows.append(
            {
                "config_full_name": cfg_full,
                "config_id": "C1",
                "model_id": name,
                "graph_type": "homo",
                "path": f"homo/{cfg_full}/{name}.pt",
                "bytes": 1,
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_parquet(tmp_path / "graphs_manifest_homo.parquet", index=False)
    frame.iloc[0:0].to_parquet(tmp_path / "graphs_manifest_hetero.parquet", index=False)
    return tmp_path, cfg_full


def test_manifests_fold_model_ids_to_lower_case(graph_set):
    """Files are named as the database spells a model; label ids are lower case."""
    graphs_root, _cfg = graph_set

    manifests = load_graph_manifests(graphs_root)

    assert set(manifests["homo"]["model_id"]) == set(MODEL_IDS)
    # The original spelling survives where it is needed: the path on disk.
    assert manifests["homo"]["path"].str.contains("Lodz_0011").any()


def test_label_normalisation_uses_the_training_split_only():
    labels = {"lodz_0011": 10.0, "metz_0014": 20.0, "genoa_0020": 900.0, "turin_0011": 1000.0}

    meta = train_label_normalisation(SPLIT, labels, label_col="eui")

    assert meta["train_mean_raw"] == pytest.approx(15.0)
    assert meta["train_std_raw"] == pytest.approx(5.0)
    assert meta["space"] == "zscore(train_only_raw)"


def test_label_normalisation_survives_a_constant_training_split():
    """A zero standard deviation would divide every target by zero."""
    labels = dict.fromkeys(MODEL_IDS, 42.0)

    meta = train_label_normalisation(SPLIT, labels, label_col="eui")

    assert meta["train_std_raw"] == 1.0


def test_export_writes_then_reuses(graph_set):
    graphs_root, cfg_full = graph_set
    manifests = load_graph_manifests(graphs_root)
    labels = {mid: float(i + 1) for i, mid in enumerate(MODEL_IDS)}
    label_meta = train_label_normalisation(SPLIT, labels, label_col="eui")
    out = graphs_root / "datasets"

    common = dict(
        graphs_root=graphs_root, output_dir=out, manifests=manifests, configs=["C1"],
        dataset_ids=MODEL_IDS, split=SPLIT, label_map_raw=labels, label_meta=label_meta,
        progress=False,
    )

    first = export_all_configs(**common)
    assert first[cfg_full]["reused"] is False
    assert (out / cfg_full / "training_ready.pt").exists()
    assert (out / cfg_full / "graph_sizes.csv").exists()

    second = export_all_configs(**common)
    assert second[cfg_full]["reused"] is True
    # The diagnostics still get their sizes back, without loading the graphs.
    assert list(second[cfg_full]["size_detail"]["num_nodes"]) == [3, 4, 5, 6]

    forced = export_all_configs(**common, force_rebuild=True)
    assert forced[cfg_full]["reused"] is False


def test_a_changed_split_invalidates_the_package(graph_set):
    """Reuse must not survive a change that alters what BEG_04 would train on."""
    graphs_root, cfg_full = graph_set
    manifests = load_graph_manifests(graphs_root)
    labels = {mid: float(i + 1) for i, mid in enumerate(MODEL_IDS)}
    label_meta = train_label_normalisation(SPLIT, labels, label_col="eui")
    out = graphs_root / "datasets"

    common = dict(
        graphs_root=graphs_root, output_dir=out, manifests=manifests, configs=["C1"],
        dataset_ids=MODEL_IDS, split=SPLIT, label_map_raw=labels, label_meta=label_meta,
        progress=False,
    )
    export_all_configs(**common)

    moved = dict(SPLIT)
    moved["train_ids"] = ["lodz_0011", "genoa_0020"]
    moved["val_ids"] = ["metz_0014"]

    again = export_all_configs(**{**common, "split": moved})

    assert again[cfg_full]["reused"] is False, "a different split reused a stale package"


def test_missing_graph_is_refused_not_silently_dropped(graph_set):
    """A short configuration must fail, not quietly shrink the dataset."""
    graphs_root, _cfg = graph_set
    manifests = load_graph_manifests(graphs_root)
    labels = {mid: 1.0 for mid in [*MODEL_IDS, "absent_0001"]}
    label_meta = train_label_normalisation(SPLIT, labels, label_col="eui")

    with pytest.raises(RuntimeError, match="no graph for"):
        export_all_configs(
            graphs_root=graphs_root, output_dir=graphs_root / "datasets", manifests=manifests,
            configs=["C1"], dataset_ids=[*MODEL_IDS, "absent_0001"], split=SPLIT,
            label_map_raw=labels, label_meta=label_meta, progress=False,
        )


def test_signature_is_stable_and_split_sensitive(graph_set):
    """BEG_04 fingerprints this, so it must change when, and only when, it should."""
    torch = pytest.importorskip("torch")
    graphs_root, cfg_full = graph_set
    graph = torch.load(
        graphs_root / "homo" / cfg_full / "Lodz_0011.pt", map_location="cpu", weights_only=False
    )
    label_meta = {"label_col": "eui", "train_mean_raw": 1.0, "train_std_raw": 2.0}
    args = dict(cfg_name=cfg_full, sample_graph=graph, label_meta=label_meta, n_graphs=4)

    _, first = config_signature_for(split=SPLIT, **args)
    _, again = config_signature_for(split=SPLIT, **args)
    assert first == again

    reseeded = {**SPLIT, "seed": 99}
    _, other = config_signature_for(split=reseeded, **args)
    assert other != first


def test_meta_json_carries_what_the_trainer_fingerprints(graph_set):
    from beg.train.runner import dataset_fingerprint

    graphs_root, cfg_full = graph_set
    manifests = load_graph_manifests(graphs_root)
    labels = {mid: float(i + 1) for i, mid in enumerate(MODEL_IDS)}
    label_meta = train_label_normalisation(SPLIT, labels, label_col="eui")
    out = graphs_root / "datasets"

    export_all_configs(
        graphs_root=graphs_root, output_dir=out, manifests=manifests, configs=["C1"],
        dataset_ids=MODEL_IDS, split=SPLIT, label_map_raw=labels, label_meta=label_meta,
        progress=False,
    )

    meta = json.loads((out / cfg_full / "training_ready.meta.json").read_text())
    assert dataset_fingerprint(meta)
    assert meta["split_sizes"] == {"train": 2, "val": 1, "test": 1}
