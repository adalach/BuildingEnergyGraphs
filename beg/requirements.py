"""What each notebook needs installed, and a check that says so in plain words.

The package is split into extras (see ``pyproject.toml``) so that, for example,
the training notebook can run on a machine without the Honeybee stack. Each
notebook calls :func:`check_environment` in its first cell, and ``beg doctor``
prints the same report for all five.

Only the standard library is imported here, so the check works in exactly the
environments it is meant to diagnose.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class Requirement:
    module: str          #: import name
    distribution: str    #: name on PyPI, as written in pyproject.toml
    extra: str           #: pyproject extra that provides it ("core" = always installed)
    purpose: str


REQUIREMENTS: tuple[Requirement, ...] = (
    Requirement("pandas", "pandas", "core", "every stage"),
    Requirement("numpy", "numpy", "core", "every stage"),
    Requirement("sqlalchemy", "SQLAlchemy", "core", "the database"),
    Requirement("psycopg2", "psycopg2-binary", "core", "the PostgreSQL driver"),
    Requirement("networkx", "networkx", "core", "graph building"),
    Requirement("pyarrow", "pyarrow", "core", "the DataFrame cache"),
    Requirement("dotenv", "python-dotenv", "core", "reading .env"),
    Requirement("tqdm", "tqdm", "core", "progress bars"),
    Requirement("tabulate", "tabulate", "core", "printed tables"),
    Requirement("scipy", "scipy", "core", "split balance and statistics"),
    Requirement("shapely", "shapely", "core", "the surface shading transform"),
    Requirement("torch", "torch", "train", "PyG graphs, datasets and training"),
    Requirement("torch_geometric", "torch-geometric", "train", "PyG graphs, datasets and training"),
    Requirement("matplotlib", "matplotlib", "viz", "figures"),
    Requirement("plotly", "plotly", "viz", "3D building previews in the graph explorer"),
    Requirement("honeybee", "honeybee-core", "viz", "3D building previews in the graph explorer"),
    Requirement("ladybug", "ladybug-core", "viz", "3D building previews in the graph explorer"),
    Requirement("ladybug_geometry", "ladybug-geometry", "viz", "3D building previews in the graph explorer"),
    Requirement("IPython", "ipython", "notebooks", "running the notebooks"),
    Requirement("ipykernel", "ipykernel", "notebooks", "running the notebooks"),
)

def _modules(*groups: str) -> tuple[str, ...]:
    """Import names of whole extras (``"core"``) and of single modules (``"matplotlib"``)."""
    extras = {r.extra for r in REQUIREMENTS}
    out = []
    for g in groups:
        out += [r.module for r in REQUIREMENTS if r.extra == g] if g in extras else [g]
    return tuple(out)


#: What each notebook imports, directly or through ``beg``, listed per module:
#: BEG_04 needs matplotlib for its dashboard but not the rest of the ``viz`` extra.
#: tests/test_requirements.py checks this against the notebooks' actual imports.
NOTEBOOK_NEEDS: dict[str, tuple[str, ...]] = {
    "BEG_01_Ingest": _modules("core", "matplotlib", "notebooks"),
    "BEG_02_Graphs": _modules("core", "train", "viz", "notebooks"),
    "BEG_03_Datasets": _modules("core", "train", "matplotlib", "notebooks"),
    "BEG_04_Train": _modules("core", "train", "matplotlib", "notebooks"),
    "BEG_05_Results": _modules("core", "train", "matplotlib", "notebooks"),
}


def install_command(notebook: str, modules=None) -> str:
    """The pip command that adds ``modules`` (default: everything) for ``notebook``.

    An extra the notebook needs in full is named as an extra; from one it needs
    only part of, just those packages are named (``matplotlib`` for BEG_04_Train).
    """
    needs = set(NOTEBOOK_NEEDS[notebook])
    wanted = needs if modules is None else set(modules) & needs
    extras, packages = set(), []
    for extra in sorted({r.extra for r in REQUIREMENTS if r.module in wanted}):
        members = [r for r in REQUIREMENTS if r.extra == extra]
        if extra == "core" or all(r.module in needs for r in members):
            extras.add(extra)
        else:
            packages += [r.distribution for r in members if r.module in wanted]
    words = ["pip", "install"]
    if extras:
        named = sorted(extras - {"core"})
        words += ["-e", f'".[{",".join(named)}]"' if named else '"."']
    return " ".join(words + packages)


def missing(modules=None) -> list[Requirement]:
    """Requirements among ``modules`` (all of them by default) not importable here."""
    wanted = set(modules) if modules is not None else {r.module for r in REQUIREMENTS}
    return [r for r in REQUIREMENTS
            if r.module in wanted and importlib.util.find_spec(r.module) is None]


def check_environment(notebook: str) -> bool:
    """Print whether this environment can run ``notebook``, and how to fix it if not.

    Returns ``True`` when everything is there. Does not raise: the first cells of
    a notebook often work without an optional extra, and the report says which
    steps will not.
    """
    gaps = missing(NOTEBOOK_NEEDS[notebook])
    if not gaps:
        print(f"environment ok for {notebook}")
        return True
    print(f"{notebook} needs packages this environment does not have:")
    for r in gaps:
        print(f"  - {r.distribution:<18} ({r.extra}) for {r.purpose}")
    print("Install them with the virtual environment active, then restart the kernel:")
    print(f"  {install_command(notebook, [r.module for r in gaps])}")
    return False
