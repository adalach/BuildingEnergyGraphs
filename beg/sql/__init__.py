"""SQL transforms: the definition of every node and edge table.

A configuration is a declared selection from the transforms in ``transforms/``.
Execution order is the intersection of the configuration's ``sql_transforms``
list with ``PRIMARY_ORDER.txt``, a hand-maintained flat list: a new transform
must be placed after everything it reads from, and nothing checks this.
"""

from pathlib import Path

SQL_DIR = Path(__file__).resolve().parent
TRANSFORMS_DIR = SQL_DIR / "transforms"
PRIMARY_ORDER_FILE = TRANSFORMS_DIR / "PRIMARY_ORDER.txt"

__all__ = [
    "SQL_DIR",
    "TRANSFORMS_DIR",
    "PRIMARY_ORDER_FILE",
    "read_primary_order",
    "transform_path",
]


def read_primary_order() -> list[str]:
    """Transform file names in execution order, comments and blanks removed."""
    lines = PRIMARY_ORDER_FILE.read_text(encoding="utf-8").splitlines()
    return [s for line in lines if (s := line.strip()) and not s.startswith("#")]


def transform_path(name: str) -> Path:
    """Absolute path of a transform, with or without the ``.sql`` suffix."""
    return TRANSFORMS_DIR / (name if name.endswith(".sql") else f"{name}.sql")
