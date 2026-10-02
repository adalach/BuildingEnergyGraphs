"""The published results must remain the output of the code in this repository.

Every one of the 816 runs behind the paper records an implementation
fingerprint: a hash of the torch and CUDA builds and of the three source files
that decide what a run produces. Recomputed from the code here, under the
recorded builds, it matches -- which is what lets the repository claim its own
code produced the numbers.

Editing any of the three files, even a comment, changes the fingerprint: the
repository then describes code that did not produce its results, and
``BEG_04_Train`` treats all 816 runs as stale and retrains them. These tests fail
as soon as that happens.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import beg

PROVENANCE = Path(beg.__file__).resolve().parent.parent / "paper_assets" / "data" / "published_runs_provenance.json"


@pytest.fixture(scope="module")
def provenance() -> dict:
    return json.loads(PROVENANCE.read_text(encoding="utf-8"))


def test_the_training_sources_are_the_ones_that_produced_the_runs(provenance):
    """Each training source hashes as recorded; a failure names the file that changed."""
    package = Path(beg.__file__).resolve().parent
    changed = [
        relative
        for relative, recorded in provenance["sources_sha256"].items()
        if hashlib.sha256((package / relative).read_bytes()).hexdigest() != recorded
    ]
    assert not changed, (
        f"beg/{', beg/'.join(changed)} changed since the published runs were trained. "
        "The repository no longer contains the code behind its results, and BEG_04 "
        "would retrain all of them. Revert the change, or retrain and update "
        "paper_assets/data/published_runs_provenance.json."
    )


@pytest.mark.torch
def test_the_hashed_sources_are_the_ones_the_fingerprint_covers(provenance):
    pytest.importorskip("torch")
    from beg.train.runner import IMPLEMENTATION_SOURCES

    assert set(provenance["sources_sha256"]) == set(IMPLEMENTATION_SOURCES)


@pytest.mark.torch
def test_the_code_reproduces_the_recorded_fingerprint(provenance):
    """End to end: the fingerprint the runs carry, recomputed from this code."""
    pytest.importorskip("torch")
    from beg.train.runner import IMPLEMENTATION_SOURCES, _sha256_hex, _source_hash

    package = Path(beg.__file__).resolve().parent
    recomputed = _sha256_hex(
        {
            "torch": provenance["torch"],
            "cuda": provenance["cuda"],
            "sources": {rel: _source_hash(package / rel) for rel in IMPLEMENTATION_SOURCES},
        }
    )
    assert recomputed == provenance["implementation_fingerprint"]


def test_the_published_runs_carry_that_fingerprint(provenance):
    import pandas as pd

    manifest = pd.read_csv(PROVENANCE.parent / "published_runs.csv")
    assert len(manifest) == provenance["runs"]
    assert set(manifest["implementation_fingerprint"]) == {provenance["implementation_fingerprint"]}
