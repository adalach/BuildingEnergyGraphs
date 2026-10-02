"""Rerunning a finished BEG_02_Graphs must not re-enter the database.

:func:`reuse_existing_export` decides from the export directory whether every
graph of a configuration is on disk. Complete means complete: anything less falls
through to the full build, which fetches the frames from the database.
"""

from __future__ import annotations

import pytest

from beg.build.export import reuse_existing_export
from beg.config.naming import resolve

pytestmark = pytest.mark.torch


@pytest.fixture()
def homo_graphs(tmp_path):
    """A C1 export directory, and a helper to populate it with named models."""
    torch = pytest.importorskip("torch")
    from torch_geometric.data import Data

    config = resolve("C1")
    cfg_dir = tmp_path / "homo" / config.full_name
    cfg_dir.mkdir(parents=True)

    def write(model_ids, edge_feature_width=0):
        for model_id in model_ids:
            graph = Data(
                x=torch.zeros(3, 4),
                edge_index=torch.tensor([[0, 1], [1, 2]]),
                edge_attr=torch.zeros(2, edge_feature_width),
            )
            torch.save(graph, cfg_dir / f"{model_id}.pt")

    return tmp_path, config, write


def test_complete_directory_is_reused(homo_graphs):
    root, config, write = homo_graphs
    write(["Lodz_0011", "Metz_0014"])

    done = reuse_existing_export(root, config, {"lodz_0011", "metz_0014"})

    assert done is not None
    assert done["kind"] == "homo"
    assert done["model_ids"] == {"Lodz_0011", "Metz_0014"}
    assert done["row"]["saved"] == 0
    assert done["row"]["skipped_existing"] == 2
    assert done["row"]["config_full_name"] == config.full_name


def test_partial_directory_falls_through(homo_graphs):
    """An interrupted run must finish, not be mistaken for a complete one."""
    root, config, write = homo_graphs
    write(["Lodz_0011"])

    assert reuse_existing_export(root, config, {"lodz_0011", "metz_0014"}) is None


def test_missing_directory_falls_through(homo_graphs):
    root, config, _write = homo_graphs

    assert reuse_existing_export(root, config, {"lodz_0011"}) is None


def test_extra_graphs_on_disk_still_count_as_complete(homo_graphs):
    """Graphs for unlabelled models do not make an export incomplete.

    ``keep_model_ids`` narrows an export to models that have a target, so a
    directory left over from a wider run legitimately holds more than is wanted.
    """
    root, config, write = homo_graphs
    write(["Lodz_0011", "Metz_0014", "Genoa_0011"])

    done = reuse_existing_export(root, config, {"lodz_0011"})

    assert done is not None
    assert done["model_ids"] == {"Lodz_0011"}
    assert done["row"]["model_ids"] == 1


def test_model_ids_match_case_insensitively(homo_graphs):
    """Files keep the model's own casing; labels are lowercased on the way in."""
    root, config, write = homo_graphs
    write(["Lodz_0011"])

    done = reuse_existing_export(root, config, {"LODZ_0011"})

    assert done is not None
    assert done["model_ids"] == {"Lodz_0011"}


def test_edge_feature_width_is_read_back_from_a_graph(homo_graphs):
    """The summary reports the real width, which file names cannot carry."""
    root, config, write = homo_graphs
    write(["Lodz_0011"], edge_feature_width=5)

    done = reuse_existing_export(root, config, {"lodz_0011"})

    assert done["row"]["edge_feature_dim"] == 5
