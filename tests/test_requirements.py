"""The install instructions match what the code actually imports.

A package imported by a notebook but missing from its declared needs fails for
a user half-way through a run, with an error that names a module rather than
the pip command that fixes it. These tests keep ``beg.requirements`` (which the
notebooks and ``beg doctor`` report from) in step with the notebooks' imports
and with ``pyproject.toml``.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

from beg.requirements import NOTEBOOK_NEEDS, REQUIREMENTS

REPO_ROOT = Path(__file__).resolve().parent.parent
STDLIB = set(sys.stdlib_module_names)


def _imports(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module
            yield from (f"{node.module}.{a.name}" for a in node.names)


def _module_file(name: str) -> Path | None:
    base = REPO_ROOT.joinpath(*name.split("."))
    return next((p for p in (base.with_suffix(".py"), base / "__init__.py") if p.exists()), None)


def _third_party_closure(names) -> set[str]:
    """Every third-party top-level module reachable from ``names`` through beg.

    Static and generous: an import inside a function counts, so a
    lazily imported package still has to be declared.
    """
    seen, found, todo = set(), set(), list(names)
    while todo:
        name = todo.pop()
        top = name.split(".")[0]
        if top in STDLIB:
            continue
        if top != "beg":
            found.add(top)
            continue
        parts = name.split(".")
        todo += [".".join(parts[:i]) for i in range(1, len(parts))]
        path = _module_file(name)
        if path is None or path in seen:
            continue
        seen.add(path)
        todo += list(_imports(ast.parse(path.read_text(encoding="utf-8"))))
    return found


@pytest.mark.parametrize("notebook", sorted(NOTEBOOK_NEEDS))
def test_notebook_needs_cover_what_it_imports(notebook):
    cells = json.loads((REPO_ROOT / "notebooks" / f"{notebook}.ipynb").read_text())["cells"]
    code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    code = "\n".join(line for line in code.splitlines() if not line.lstrip().startswith(("%", "!")))

    imported = _third_party_closure(_imports(ast.parse(code)))
    undeclared = imported - set(NOTEBOOK_NEEDS[notebook])
    assert not undeclared, (
        f"{notebook} imports {sorted(undeclared)}, which beg.requirements.NOTEBOOK_NEEDS "
        "does not list, so its environment check would pass and the notebook then fail"
    )


def _normalise(distribution: str) -> str:
    return re.sub(r"[-_.]+", "-", distribution).lower()


def test_requirements_are_declared_in_pyproject_under_their_extra():
    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    sections = {"core": project["dependencies"], **project["optional-dependencies"]}
    declared = {
        extra: {_normalise(re.split(r"[<>=!~\[; ]", spec, maxsplit=1)[0]) for spec in specs}
        for extra, specs in sections.items()
    }
    wrong = [
        f"{r.distribution} (said to be in {r.extra})"
        for r in REQUIREMENTS
        if _normalise(r.distribution) not in declared.get(r.extra, set())
    ]
    assert not wrong, f"not declared in pyproject.toml where beg.requirements says: {wrong}"


def test_every_third_party_import_in_the_package_is_known():
    imported = _third_party_closure(
        f"beg.{'.'.join(p.relative_to(REPO_ROOT / 'beg').with_suffix('').parts)}"
        for p in (REPO_ROOT / "beg").rglob("*.py")
    )
    known = {r.module for r in REQUIREMENTS}
    assert imported - known == set()
