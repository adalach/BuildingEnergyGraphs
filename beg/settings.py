"""Repository-relative paths and the single database-connection setting.

Paths are derived from the repository root unless an environment variable
overrides them, and the database is addressed only through the ``BEG_DSN``
environment variable, which is read from a local ``.env`` file when one is present.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

try:  # optional: without python-dotenv, .env is not read
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

DSN_ENV_VAR = "BEG_DSN"

_DSN_MISSING = f"""\
{DSN_ENV_VAR} is not set, so beg does not know which database to use.

  1. cp .env.example .env
  2. edit .env and set {DSN_ENV_VAR}
  3. start a server (docker compose up -d) and run: beg db init

See docs/installation.md for the four supported PostgreSQL setups.
"""


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Absolute path of the repository root.

    Resolved from this file's location, so it does not depend on the working
    directory.
    """
    return Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def _load_env() -> None:
    if load_dotenv is not None:
        load_dotenv(repo_root() / ".env", override=False)


def _dir_from_env(var: str, default: str) -> Path:
    """``var`` from the environment or ``.env``; a relative path is taken from the repository root."""
    _load_env()
    raw = os.environ.get(var)
    path = Path(raw).expanduser() if raw else Path(default)
    return (repo_root() / path).resolve()


def data_dir() -> Path:
    """Raw inputs (epJSON models, weather files, labels). Override: ``BEG_DATA_DIR``."""
    return _dir_from_env("BEG_DATA_DIR", "data")


def output_dir() -> Path:
    """Generated artefacts (graphs, datasets, runs). Override: ``BEG_OUTPUT_DIR``."""
    return _dir_from_env("BEG_OUTPUT_DIR", "outputs")


def display_path(path: Path | str) -> str:
    """A path safe to print, and to store in a published notebook's output.

    Relative to the repository root where possible, otherwise the bare filename
    (``BEG_DATA_DIR`` and ``BEG_OUTPUT_DIR`` may point outside the repository).
    Never an absolute path.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(repo_root()).as_posix()
    except ValueError:
        return resolved.name


def dsn(required: bool = True) -> str | None:
    """Return the configured database DSN.

    Parameters
    ----------
    required:
        When True (default) a missing ``BEG_DSN`` raises ``RuntimeError`` with
        setup instructions. When False the function returns ``None`` instead.
    """
    _load_env()
    value = os.environ.get(DSN_ENV_VAR, "").strip()
    if value:
        return value
    if required:
        raise RuntimeError(_DSN_MISSING)
    return None


def redacted_dsn(value: str | None = None) -> str:
    """The DSN with its password replaced, safe to print or log."""
    import re

    raw = value if value is not None else (dsn(required=False) or "<unset>")
    return re.sub(r"://([^:/@]+):[^@]*@", r"://\1:***@", raw)
