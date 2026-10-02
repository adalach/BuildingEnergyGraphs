"""Runs are content-addressed, and this is the half that can rot silently.

``run_fingerprint`` combines a dataset fingerprint with an implementation
fingerprint so a plan skips completed work and never presents results from two
revisions of the code as comparable. The dataset half is checked by the package
metadata; the implementation half hashes source files *by path*, so a renamed or
moved source must fail here instead of silently dropping out of the hash.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from beg.train.runner import IMPLEMENTATION_SOURCES, implementation_fingerprint  # noqa: E402

pytestmark = pytest.mark.torch


def test_every_named_source_exists():
    """A renamed module must fail loudly, not drop out of the hash."""
    from pathlib import Path

    import beg

    package_root = Path(beg.__file__).resolve().parent
    missing = [rel for rel in IMPLEMENTATION_SOURCES if not (package_root / rel).exists()]

    assert not missing, (
        f"implementation_fingerprint names sources that do not exist: {missing}. "
        "Runs could not be told apart across revisions."
    )


def test_the_sources_are_the_ones_that_decide_a_result():
    """Models, hyperparameters and the training loop — and not the reporting."""
    assert set(IMPLEMENTATION_SOURCES) == {
        "models/gnn.py",
        "train/search.py",
        "train/runner.py",
    }, (
        "The hashed set changed. Adding a module invalidates every completed run; "
        "removing one lets a change to it pass unnoticed. Reporting modules belong "
        "outside, so redrawing a dashboard does not invalidate a run."
    )


def test_a_missing_source_raises(monkeypatch):
    monkeypatch.setattr(
        "beg.train.runner.IMPLEMENTATION_SOURCES", ("models/gnn.py", "train/not_a_module.py")
    )

    with pytest.raises(RuntimeError, match="missing"):
        implementation_fingerprint()


def test_editing_a_model_changes_the_fingerprint(tmp_path):
    """The property the whole mechanism rests on."""
    from pathlib import Path

    import beg

    source = Path(beg.__file__).resolve().parent / "models" / "gnn.py"
    before = implementation_fingerprint()
    original = source.read_text(encoding="utf-8")
    try:
        source.write_text(original + "\n# fingerprint probe\n", encoding="utf-8")
        after = implementation_fingerprint()
    finally:
        source.write_text(original, encoding="utf-8")

    assert after != before, "a change to the model code left the fingerprint unchanged"
    assert implementation_fingerprint() == before, "restoring the file did not restore the hash"


def test_fingerprint_is_stable_between_calls():
    assert implementation_fingerprint() == implementation_fingerprint()
