#!/usr/bin/env python3
"""Refuse to let credentials, machine paths or bulky binaries into the repository.

Runs as a pre-commit hook and in CI. It checks every git-tracked file for:

1. a connection string carrying a password;
2. an absolute path inside somebody's home directory;
3. megabytes of base64 image data embedded in a notebook;
4. a ``.env`` file staged for commit.

**Notebook outputs are allowed, and are scanned.** They ship so the notebooks can
be read without running them, and every cell's outputs are checked for home paths
and connection strings.

Anything a notebook prints must therefore go through
``beg.settings.display_path()`` for paths and ``beg.settings.redacted_dsn()`` for
connection strings. Watch out for objects whose *plain-text* representation
differs from their HTML one -- ``IPython.display.FileLink`` renders a relative
link but reprs as an absolute path, and Jupyter stores both.

Usage
-----
    python scripts/check_no_secrets.py            # check tracked files
    python scripts/check_no_secrets.py --all      # check every file in the tree
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: A DSN with a non-empty, non-placeholder password.
DSN_WITH_PASSWORD = re.compile(
    r"(?:postgres(?:ql)?(?:\+\w+)?|mysql|mongodb)://[^:\s/]+:"
    r"(?!change-me\b)(?!\$\{)(?!\*)[^@\s\"']+@"
)

#: Absolute paths into a user home directory.
HOME_PATH = re.compile(r"(?:/Users/|/home/|C:\\\\Users\\\\)[A-Za-z0-9._-]+/")

#: Files that are allowed to contain example connection strings.
ALLOWLIST = {
    ".env.example",
    "docker-compose.yml",
    "scripts/check_no_secrets.py",
    "docs/installation.md",
    "docs/database.md",
}

#: A line carrying this marker is exempt. Use it only for strings that are not
#: credentials at all - a test fixture, or a throwaway CI service password - and
#: always state the reason after the marker.
PRAGMA = "allow-secret"

TEXT_SUFFIXES = {
    ".py", ".ipynb", ".sql", ".md", ".txt", ".yml", ".yaml",
    ".json", ".cfg", ".toml", ".sh", ".cff",
}


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [REPO_ROOT / name for name in out.split("\0") if name]


def ignored_files() -> set[Path]:
    """Everything git is configured to ignore, resolved to absolute paths."""
    out = subprocess.run(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return {REPO_ROOT / name for name in out.split("\0") if name}


def all_files() -> list[Path]:
    """Every file that could still end up in the repository.

    Used by ``--all``: untracked files are included, git-ignored files are not,
    since they cannot reach a clone.
    """
    skip = {".git", ".venv", "venv", "__pycache__", "data", "outputs", "node_modules", "tmp"}
    ignored = ignored_files()
    return [
        p
        for p in REPO_ROOT.rglob("*")
        if p.is_file()
        and not skip & set(p.relative_to(REPO_ROOT).parts)
        and p not in ignored
    ]


#: Embedded base64 payloads (images, mostly) allowed per notebook, in bytes.
MAX_EMBEDDED_BYTES = 1_500_000

#: Output MIME types whose payload is base64 rather than text.
BINARY_MIME_PREFIXES = ("image/", "application/pdf")


def _output_texts(cell: dict):
    """Yield every piece of *text* an output carries, with a label for reporting."""
    for i, out in enumerate(cell.get("outputs") or []):
        for key in ("text", "traceback"):
            value = out.get(key)
            if value:
                yield f"output {i} {key}", value if isinstance(value, str) else "".join(value)
        for mime, value in (out.get("data") or {}).items():
            if mime.startswith(BINARY_MIME_PREFIXES):
                continue
            yield (
                f"output {i} {mime}",
                value if isinstance(value, str) else "".join(value),
            )


def _embedded_bytes(cell: dict) -> int:
    """Approximate decoded size of base64 payloads in a cell's outputs."""
    total = 0
    for out in cell.get("outputs") or []:
        for mime, value in (out.get("data") or {}).items():
            if not mime.startswith(BINARY_MIME_PREFIXES):
                continue
            payload = value if isinstance(value, str) else "".join(value)
            total += len(payload) * 3 // 4
    return total


def notebook_problems(path: Path, rel: str) -> list[str]:
    """Scan a notebook's source *and* its stored outputs, cell by cell.

    Problems are reported by cell number, which is what a reader of the notebook
    sees.
    """
    problems: list[str] = []
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"{rel}: not readable as a notebook ({exc})"]

    embedded = 0
    for index, cell in enumerate(nb.get("cells", [])):
        source = cell.get("source") or []
        regions = [("source", source if isinstance(source, str) else "".join(source))]
        regions.extend(_output_texts(cell))
        embedded += _embedded_bytes(cell)

        for label, text in regions:
            for line in text.splitlines():
                if PRAGMA in line:
                    continue
                if DSN_WITH_PASSWORD.search(line):
                    problems.append(
                        f"{rel}: cell {index} {label}: connection string with a password "
                        "(print beg.settings.redacted_dsn() instead)"
                    )
                if HOME_PATH.search(line):
                    problems.append(
                        f"{rel}: cell {index} {label}: absolute path inside a home directory "
                        "(print beg.settings.display_path(...) instead)"
                    )

    if embedded > MAX_EMBEDDED_BYTES:
        problems.append(
            f"{rel}: {embedded / 1e6:.1f} MB of embedded image data in outputs "
            f"(limit {MAX_EMBEDDED_BYTES / 1e6:.1f} MB); lower the figure DPI or "
            "clear that cell's output"
        )
    return problems


def is_tracked(path: Path) -> bool:
    """True when git has this file under version control."""
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return result.returncode == 0


def check(paths: list[Path]) -> list[str]:
    problems: list[str] = []
    for path in paths:
        rel = path.relative_to(REPO_ROOT).as_posix()

        # A local .env is the correct place for the password. It is only a problem
        # if git has it; --all walks the working tree, where it legitimately exists.
        if rel == ".env" or rel.endswith("/.env"):
            if is_tracked(path):
                problems.append(f"{rel}: .env must never be committed")
            continue

        if path.suffix not in TEXT_SUFFIXES:
            continue

        # Notebooks get a dedicated pass that also scans their stored outputs and
        # reports cell numbers.
        if path.suffix == ".ipynb":
            if rel not in ALLOWLIST:
                problems.extend(notebook_problems(path, rel))
            continue

        if rel in ALLOWLIST:
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        lines = text.splitlines()
        for lineno, line in enumerate(lines, start=1):
            previous = lines[lineno - 2] if lineno > 1 else ""
            if PRAGMA in line or PRAGMA in previous:
                continue
            if DSN_WITH_PASSWORD.search(line):
                problems.append(f"{rel}:{lineno}: connection string with a password")
            if HOME_PATH.search(line):
                problems.append(f"{rel}:{lineno}: absolute path inside a home directory")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="check every file in the working tree, not only tracked files",
    )
    args = parser.parse_args(argv)

    paths = all_files() if args.all else tracked_files()
    problems = check(paths)

    if problems:
        print(f"{len(problems)} problem(s) found:\n", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            "\nNothing here belongs in a public repository. "
            "Use BEG_DSN from .env for connections, beg.settings for paths.",
            file=sys.stderr,
        )
        return 1

    print(f"checked {len(paths)} files: no credentials, home paths or oversized notebook output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
