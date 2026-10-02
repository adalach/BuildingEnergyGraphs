"""Checks that hold without a database, a GPU or any downloaded data.

These are the tests that keep the repository publishable: every module imports,
no credential or machine path is committed, in files or in stored notebook
output, and the paths the pipeline uses resolve relative to the repository.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def test_repo_root_is_this_repository():
    from beg.settings import repo_root

    assert repo_root() == REPO_ROOT
    assert (repo_root() / "pyproject.toml").exists()


def test_paths_are_repository_relative(monkeypatch):
    from beg import settings

    monkeypatch.delenv("BEG_DATA_DIR", raising=False)
    monkeypatch.delenv("BEG_OUTPUT_DIR", raising=False)
    assert settings.data_dir() == REPO_ROOT / "data"
    assert settings.output_dir() == REPO_ROOT / "outputs"

    monkeypatch.setenv("BEG_OUTPUT_DIR", "outputs/subset")
    assert settings.output_dir() == REPO_ROOT / "outputs" / "subset"


def test_missing_dsn_raises_with_instructions(monkeypatch):
    from beg import settings

    monkeypatch.setattr(settings, "_load_env", lambda: None)
    monkeypatch.delenv("BEG_DSN", raising=False)

    assert settings.dsn(required=False) is None
    with pytest.raises(RuntimeError, match="BEG_DSN is not set"):
        settings.dsn()


def test_redacted_dsn_hides_the_password():
    from beg.settings import redacted_dsn

    fixture = "postgresql+psycopg2://beg:hunter2@127.0.0.1:5432/beg"  # allow-secret: test fixture
    out = redacted_dsn(fixture)
    assert "hunter2" not in out
    assert out == "postgresql+psycopg2://beg:***@127.0.0.1:5432/beg"


#: Packages behind an optional extra; their absence is a setup choice, not a break.
OPTIONAL_DEPS = ("torch", "matplotlib", "plotly", "honeybee", "ladybug")


def _module_names() -> list[str]:
    """Every module under ``beg/``, named from the filesystem.

    Nothing is imported to list them, so one failing module cannot hide the
    others.
    """
    import beg

    root = Path(beg.__path__[0])
    names = ["beg"]
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).with_suffix("")
        parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
        if parts:
            names.append("beg." + ".".join(parts))
    return names


def test_every_module_imports():
    """A rename or a moved helper must not leave a dangling import behind.

    Any exception counts: a module missing its own imports raises ``NameError``,
    not ``ImportError``.
    """
    failures = []
    for name in _module_names():
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - reporting every break is the point
            if isinstance(exc, ImportError) and any(d in str(exc) for d in OPTIONAL_DEPS):
                continue
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failures, "modules failed to import:\n" + "\n".join(failures)


def test_no_secrets_or_machine_paths_committed():
    from check_no_secrets import REPO_ROOT as SCRIPT_ROOT
    from check_no_secrets import all_files, check

    assert SCRIPT_ROOT == REPO_ROOT
    problems = check(all_files())
    assert not problems, "publication blockers found:\n" + "\n".join(problems)


@pytest.mark.parametrize("path", sorted((REPO_ROOT / "notebooks").glob("*.ipynb")), ids=lambda p: p.name)
def test_notebook_is_valid_and_outputs_are_publishable(path: Path):
    """Stored outputs carry nothing machine-specific.

    Notebooks are published with their outputs so they can be read without being
    run; ``check_no_secrets.notebook_problems`` checks what those outputs contain.
    """
    from check_no_secrets import notebook_problems

    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb.get("nbformat", 0) >= 4

    problems = notebook_problems(path, path.name)
    assert not problems, "notebook output is not publishable:\n" + "\n".join(problems)


@pytest.mark.parametrize("path", sorted((REPO_ROOT / "configs").rglob("*.json")), ids=lambda p: p.name)
def test_config_is_valid_json(path: Path):
    json.loads(path.read_text(encoding="utf-8"))


def test_every_package_file_is_tracked_by_git():
    """A published package has to contain its own source.

    Every `.py` file under `beg/` must be tracked by git; an unanchored `.gitignore`
    pattern such as `build/` also matches `beg/build/`. `test_every_module_imports`
    cannot see an untracked file, since it walks the working tree.
    """
    import subprocess

    repo_check = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if repo_check.returncode != 0:
        pytest.skip("not a git working tree")

    tracked = set(
        subprocess.run(
            ["git", "ls-files", "beg"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.split()
    )
    on_disk = {
        p.relative_to(REPO_ROOT).as_posix()
        for p in (REPO_ROOT / "beg").rglob("*.py")
        if "__pycache__" not in p.parts
    }

    missing = sorted(on_disk - tracked)
    assert not missing, (
        "package files exist but git does not track them; they would be absent "
        "from a clone:\n  " + "\n  ".join(missing)
    )


def test_keyword_arguments_match_the_functions_they_call():
    """Catch a caller left behind when a parameter was removed.

    A call like ``f(gone=1)`` imports fine and raises only when the line runs,
    which in this repository can be an hour into a pipeline. This walks every
    call in ``beg`` whose target is a function defined in ``beg`` and checks the
    keywords against the real signature.
    """
    import ast
    import importlib
    import inspect

    problems = []
    for path in sorted((REPO_ROOT / "beg").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("beg"):
                for alias in node.names:
                    imported[alias.asname or alias.name] = f"{node.module}.{alias.name}"

        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            target = imported.get(node.func.id)
            if target is None:
                continue
            module_name, _, attribute = target.rpartition(".")
            try:
                function = getattr(importlib.import_module(module_name), attribute)
                signature = inspect.signature(function)
            except Exception:  # noqa: BLE001 - optional extras, classes, non-callables
                continue
            if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
                continue
            for keyword in node.keywords:
                if keyword.arg and keyword.arg not in signature.parameters:
                    problems.append(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno}: "
                        f"{attribute}() has no parameter {keyword.arg!r}"
                    )

    assert not problems, "calls disagree with their signatures:\n" + "\n".join(problems)
