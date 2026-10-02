"""Decide what a training plan trains on this machine, and stop unless confirmed.

A run fingerprint includes the torch version and CUDA build, so a run trained on
another machine never matches a plan built here by fingerprint alone.
:func:`match_runs_from_other_builds` recomputes each planned fingerprint under
the builds that trained the existing runs and marks a planned run as done when
one of them matches: same dataset, same model and training code, same model and
training hyperparameters. :func:`require_training_confirmed` then stops the
execute cell of ``BEG_04_Train`` whenever anything is left to train and
``CONFIRM_TRAINING`` is not set.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

TRAINING_ACTIONS = frozenset({"run", "resume"})

#: Plan action for a run already trained under another torch/CUDA build.
DONE_OTHER_BUILD = "skip_done_other_build"

#: Plan actions that mean the run exists and is complete.
DONE_ACTIONS = frozenset({"skip_done", DONE_OTHER_BUILD})


class TrainingNotConfirmed(RuntimeError):
    """The plan would train, and ``CONFIRM_TRAINING`` is not set."""


def _build(torch_version: str | None, cuda: str | None) -> str:
    return f"torch {torch_version or '?'}, CUDA {cuda or 'none'}"


def _run_metas(runs_dir: Path):
    for meta_path in sorted(Path(runs_dir).glob("*/run_meta.json")):
        try:
            yield meta_path.parent, json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue


def runs_from_other_builds(runs_dir: Path, build: tuple[str, str | None]) -> Counter:
    """Existing runs trained under a torch/CUDA build other than ``build``, counted per build."""
    local = _build(*build)
    counts: Counter = Counter()
    for _run_dir, meta in _run_metas(runs_dir):
        hardware = meta.get("hardware") or {}
        trained = _build(hardware.get("torch"), hardware.get("cuda"))
        if trained != local:
            counts[trained] += 1
    return counts


def match_runs_from_other_builds(
    plan: Iterable[Mapping[str, Any]],
    *,
    runs_dir: Path,
    deterministic: bool,
) -> list[dict[str, Any]]:
    """The plan, with runs that another torch/CUDA build already trained marked as done.

    For every build recorded in a finished run's ``hardware`` block, the
    implementation fingerprint is recomputed from this repository's training
    sources under that build. Where it equals the fingerprint the run recorded,
    the code is unchanged, and each planned run to train is fingerprinted again
    under that build. A match turns the item's action into ``skip_done_other_build``
    and records the existing run in ``existing_run_fingerprint``, ``out_dir`` and
    ``trained_with``.
    """
    from beg.train.runner import IMPLEMENTATION_SOURCES, _sha256_hex, _source_hash, run_fingerprint

    package_root = Path(__file__).resolve().parent.parent
    sources = {rel: _source_hash(package_root / rel) for rel in IMPLEMENTATION_SOURCES}

    done: dict[str, Path] = {}
    builds: dict[tuple[str | None, str | None], set[str]] = {}
    for run_dir, meta in _run_metas(runs_dir):
        if meta.get("status") != "done" or not (run_dir / "_DONE.txt").exists():
            continue
        done[str(meta.get("run_fingerprint", ""))] = run_dir
        hardware = meta.get("hardware") or {}
        key = (hardware.get("torch"), hardware.get("cuda"))
        builds.setdefault(key, set()).add(str(meta.get("implementation_fingerprint", "")))

    unchanged = {}
    for (torch_version, cuda), recorded in builds.items():
        fingerprint = _sha256_hex({"torch": torch_version, "cuda": cuda, "sources": sources})
        if fingerprint in recorded:
            unchanged[(torch_version, cuda)] = fingerprint

    matched = []
    for item in plan:
        item = dict(item)
        if item.get("action") in TRAINING_ACTIONS:
            for (torch_version, cuda), implementation_fp in unchanged.items():
                if implementation_fp == item.get("implementation_fingerprint"):
                    continue
                fingerprint = run_fingerprint(
                    dataset_fp=item["dataset_fingerprint"],
                    implementation_fp=implementation_fp,
                    model_key=item["model_key"],
                    model_overrides=item["model_overrides"],
                    train_hparams=item["train_hparams"],
                    deterministic=deterministic,
                )
                if fingerprint in done:
                    item.update(
                        action=DONE_OTHER_BUILD,
                        existing_status="done",
                        existing_run_fingerprint=fingerprint,
                        out_dir=done[fingerprint],
                        trained_with=_build(torch_version, cuda),
                    )
                    break
        matched.append(item)
    return matched


def require_training_confirmed(
    plan: Iterable[Mapping[str, Any]],
    *,
    confirmed: bool,
    build: tuple[str, str | None],
    device: str,
) -> int:
    """Raise :class:`TrainingNotConfirmed` if ``plan`` would train and ``confirmed`` is False.

    Returns the number of runs the plan trains. A plan with nothing to train
    passes whatever ``confirmed`` is.
    """
    to_train = sum(1 for item in plan if item.get("action") in TRAINING_ACTIONS)
    if to_train == 0 or confirmed:
        return to_train
    raise TrainingNotConfirmed(
        f"This plan would train {to_train} runs on {device.upper()} ({_build(*build)}).\n"
        "To train on this machine, set CONFIRM_TRAINING = True in the first cell and run this "
        "cell again."
    )
