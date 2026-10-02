#!/usr/bin/env python3
"""Pack everything needed to run BEG_04_Train on another machine into one zip.

Training is the only stage that needs no database: ``beg.train`` and
``beg.models`` read ``outputs/pyg_datasets`` and nothing else, so it can run on
whichever machine has the GPU.

The archive holds the package, the training notebook, the ten training-ready
packages and the instructions. ``.env`` is never packed, and the contents are
checked for credentials and paths from this machine before the archive is
written.

Usage::

    python scripts/pack_training_bundle.py
    python scripts/pack_training_bundle.py --out somewhere/else.zip
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

from beg.settings import display_path, output_dir, repo_root

#: Files and directories the training stage needs, relative to the repository.
#: ``README.md`` and ``LICENSE`` are here because ``pyproject.toml`` names them:
#: without them ``pip install -e .`` fails on the far machine.
INCLUDE = [
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "beg",
    "configs",
    "notebooks/BEG_04_Train.ipynb",
    "docs/training-on-another-machine.md",
]

#: Never packed, whatever else matches.
EXCLUDE_NAMES = {".env", ".env.local", "__pycache__", ".DS_Store", ".git", ".ipynb_checkpoints"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".log"}

#: A credential, or a path belonging to whoever built the archive.
LEAKS = re.compile(
    r"(postgres(ql)?\+?\w*://[^\s\"']+|/Users/[A-Za-z0-9._-]+/|/home/[A-Za-z0-9._-]+/|[A-Z]:\\\\Users\\\\[A-Za-z0-9._-]+)"
)

#: Scanned for leaks. Binary payloads are checked separately and more cheaply.
TEXT_SUFFIXES = {".py", ".toml", ".md", ".json", ".txt", ".cfg", ".ini", ".csv", ".ipynb", ".sql"}


def iter_files(root: Path, relative: str):
    """Every file under one INCLUDE entry, with the excludes applied."""
    source = root / relative
    if not source.exists():
        return
    candidates = [source] if source.is_file() else sorted(source.rglob("*"))
    for path in candidates:
        if not path.is_file():
            continue
        if any(part in EXCLUDE_NAMES for part in path.parts):
            continue
        if path.suffix in EXCLUDE_SUFFIXES or path.name in EXCLUDE_NAMES:
            continue
        yield path


def scan_for_leaks(paths: list[Path]) -> list[str]:
    """Report credentials or build-machine paths in anything shippable."""
    problems = []
    for path in paths:
        if path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in LEAKS.finditer(text):
            line = text[: match.start()].count("\n") + 1
            problems.append(f"{path}:{line}: {match.group(0)[:60]}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="archive path")
    parser.add_argument("--readme", type=Path, default=None, help="Windows README to include")
    parser.add_argument("--no-datasets", action="store_true", help="code only, for a quick check")
    args = parser.parse_args()

    root = repo_root()
    out_path = args.out or (output_dir() / "beg-train-windows.zip")
    readme = args.readme or (root / "docs" / "README-WINDOWS.md")
    prefix = out_path.stem

    if not readme.exists():
        print(f"Missing Windows README: {readme}", file=sys.stderr)
        return 1

    members: list[tuple[Path, str]] = [(readme, f"{prefix}/README-WINDOWS.md")]
    for relative in INCLUDE:
        for path in iter_files(root, relative):
            members.append((path, f"{prefix}/{path.relative_to(root).as_posix()}"))

    datasets_root = output_dir() / "pyg_datasets"
    if not args.no_datasets:
        if not datasets_root.exists():
            print(f"Missing training packages: {datasets_root}", file=sys.stderr)
            return 1
        configs = sorted(p for p in datasets_root.iterdir() if p.is_dir())
        if not configs:
            print(f"No configurations under {datasets_root}", file=sys.stderr)
            return 1
        for path in iter_files(datasets_root, "."):
            members.append((path, f"{prefix}/outputs/pyg_datasets/{path.relative_to(datasets_root).as_posix()}"))
        print(f"training packages : {len(configs)} configurations")

    problems = scan_for_leaks([path for path, _ in members])
    if problems:
        print("Refusing to pack: credentials or build-machine paths found.", file=sys.stderr)
        for problem in problems[:20]:
            print(f"  {problem}", file=sys.stderr)
        return 1

    raw_bytes = sum(path.stat().st_size for path, _ in members)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for index, (path, arcname) in enumerate(members, start=1):
            archive.write(path, arcname)
            if index % 500 == 0 or index == len(members):
                print(f"  packed {index}/{len(members)} files", end="\r", flush=True)
    print()

    packed = out_path.stat().st_size
    print(f"files             : {len(members)}")
    print(f"uncompressed      : {raw_bytes / 1e9:.2f} GB")
    print(f"archive           : {packed / 1e9:.2f} GB  ({display_path(out_path)})")
    print("checked for credentials and build-machine paths: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
