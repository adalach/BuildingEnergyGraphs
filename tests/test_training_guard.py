"""BEG_04_Train trains only when confirmed, and not again what another build trained."""

import json
from pathlib import Path

import pytest

from beg.train.guard import (
    DONE_OTHER_BUILD,
    TrainingNotConfirmed,
    require_training_confirmed,
    runs_from_other_builds,
)

HERE = ("2.14.0", None)


def _run(runs_dir, name, torch, cuda, **meta):
    (runs_dir / name).mkdir(parents=True)
    (runs_dir / name / "run_meta.json").write_text(
        json.dumps({"status": "done", "hardware": {"torch": torch, "cuda": cuda}, **meta})
    )
    (runs_dir / name / "_DONE.txt").write_text("done\n")


def test_unconfirmed_plan_that_trains_stops():
    plan = [{"action": "run"}, {"action": "skip_done"}, {"action": "resume"}]
    with pytest.raises(TrainingNotConfirmed) as err:
        require_training_confirmed(plan, confirmed=False, build=HERE, device="cpu")
    assert "train 2 runs on CPU" in str(err.value)
    assert "CONFIRM_TRAINING = True" in str(err.value)


def test_confirmed_or_nothing_to_train_passes():
    assert require_training_confirmed([{"action": "run"}], confirmed=True, build=HERE, device="cpu") == 1
    done = [{"action": "skip_done"}, {"action": DONE_OTHER_BUILD}, {"action": "skip_incomplete"}]
    assert require_training_confirmed(done, confirmed=False, build=HERE, device="cpu") == 0


def test_runs_from_this_build_are_not_counted_as_foreign(tmp_path):
    _run(tmp_path, "local", "2.14.0", None)
    _run(tmp_path, "remote", "2.6.0+cu124", "12.4")
    assert runs_from_other_builds(tmp_path, HERE) == {"torch 2.6.0+cu124, CUDA 12.4": 1}


def test_a_run_from_another_build_counts_as_done_only_when_everything_else_matches(tmp_path):
    pytest.importorskip("torch")
    import beg
    from beg.train.guard import match_runs_from_other_builds
    from beg.train.runner import IMPLEMENTATION_SOURCES, _sha256_hex, _source_hash, run_fingerprint

    package = Path(beg.__file__).resolve().parent
    sources = {rel: _source_hash(package / rel) for rel in IMPLEMENTATION_SOURCES}
    remote_code = _sha256_hex({"torch": "2.6.0+cu124", "cuda": "12.4", "sources": sources})
    fields = dict(dataset_fp="d" * 64, model_key="gcn", model_overrides={"hidden": 64},
                  train_hparams={"max_epochs": 100})
    remote_run = run_fingerprint(implementation_fp=remote_code, deterministic=False, **fields)
    _run(tmp_path, "remote", "2.6.0+cu124", "12.4",
         run_fingerprint=remote_run, implementation_fingerprint=remote_code)

    def item(**changes):
        base = {"action": "run", "dataset_fingerprint": fields["dataset_fp"], "model_key": "gcn",
                "model_overrides": {"hidden": 64}, "train_hparams": {"max_epochs": 100},
                "implementation_fingerprint": "local-code", "run_fingerprint": "local-run"}
        return {**base, **changes}

    same, other_hparams = match_runs_from_other_builds(
        [item(), item(train_hparams={"max_epochs": 2})], runs_dir=tmp_path, deterministic=False
    )
    assert same["action"] == DONE_OTHER_BUILD
    assert same["existing_run_fingerprint"] == remote_run
    assert same["trained_with"] == "torch 2.6.0+cu124, CUDA 12.4"
    assert other_hparams["action"] == "run"

    (deterministic,) = match_runs_from_other_builds([item()], runs_dir=tmp_path, deterministic=True)
    assert deterministic["action"] == "run"
