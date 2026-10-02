"""
Core result containers shared by all validation modules.

A check returns a list of :class:`CheckResult`.  Each result has a *status*
(PASS / WARN / FAIL / SKIP), a short human-readable *message* and an optional
*details* dict with machine-readable evidence (counts, example ids, ...).

FAIL  — the dataset is wrong and must not be used for training/comparison.
WARN  — suspicious but possibly legitimate (e.g. a constant feature column).
SKIP  — the check could not run (missing optional input such as torch).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence


class Status(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


# Alias for callers that think in terms of "severity".
Severity = Status


@dataclass
class CheckResult:
    name: str
    status: Status
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    config: Optional[str] = None
    model_id: Optional[str] = None
    scope: str = ""  # e.g. "config", "dataframe", "nx", "pyg", "labels", "cross"

    # ------------------------------------------------------------------ helpers
    @classmethod
    def ok(cls, name: str, message: str = "", **kw: Any) -> "CheckResult":
        return cls(name=name, status=Status.PASS, message=message, **kw)

    @classmethod
    def warn(cls, name: str, message: str, **kw: Any) -> "CheckResult":
        return cls(name=name, status=Status.WARN, message=message, **kw)

    @classmethod
    def fail(cls, name: str, message: str, **kw: Any) -> "CheckResult":
        return cls(name=name, status=Status.FAIL, message=message, **kw)

    @classmethod
    def skip(cls, name: str, message: str, **kw: Any) -> "CheckResult":
        return cls(name=name, status=Status.SKIP, message=message, **kw)

    @property
    def failed(self) -> bool:
        return self.status == Status.FAIL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "config": self.config,
            "model_id": self.model_id,
            "scope": self.scope,
            "details": _jsonable(self.details),
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        where = " ".join(x for x in [self.config or "", self.model_id or ""] if x)
        where = f" [{where}]" if where else ""
        return f"{self.status.value:4s} {self.name}{where}: {self.message}"


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of numpy / pandas scalars and sets for JSON dumps."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return str(obj)
        return obj
    if hasattr(obj, "item"):
        try:
            return _jsonable(obj.item())
        except Exception:  # pragma: no cover
            return str(obj)
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


@dataclass
class Report:
    """A collection of CheckResults with summary / rendering helpers."""

    results: List[CheckResult] = field(default_factory=list)

    def add(self, results: Iterable[CheckResult] | CheckResult) -> "Report":
        if isinstance(results, CheckResult):
            self.results.append(results)
        else:
            self.results.extend(results)
        return self

    # ------------------------------------------------------------ filtering
    def by_status(self, status: Status) -> List[CheckResult]:
        return [r for r in self.results if r.status == status]

    @property
    def failures(self) -> List[CheckResult]:
        return self.by_status(Status.FAIL)

    @property
    def warnings(self) -> List[CheckResult]:
        return self.by_status(Status.WARN)

    @property
    def ok(self) -> bool:
        return not self.failures

    def counts(self) -> Dict[str, int]:
        c = Counter(r.status.value for r in self.results)
        return {s.value: c.get(s.value, 0) for s in Status}

    # ------------------------------------------------------------ rendering
    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        payload = {"summary": self.counts(), "results": [r.to_dict() for r in self.results]}
        txt = json.dumps(payload, indent=indent, default=str)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(txt)
        return txt

    def to_markdown(self, *, max_rows_per_status: int = 200, include_pass: bool = False) -> str:
        c = self.counts()
        lines = [
            "# Graph dataset quality report",
            "",
            "| FAIL | WARN | PASS | SKIP |",
            "|---:|---:|---:|---:|",
            f"| {c['FAIL']} | {c['WARN']} | {c['PASS']} | {c['SKIP']} |",
            "",
        ]
        order = [Status.FAIL, Status.WARN, Status.SKIP] + ([Status.PASS] if include_pass else [])
        for status in order:
            rows = self.by_status(status)
            if not rows:
                continue
            lines.append(f"## {status.value} ({len(rows)})")
            lines.append("")
            lines.append("| check | config | model | message |")
            lines.append("|---|---|---|---|")
            for r in rows[:max_rows_per_status]:
                msg = r.message.replace("|", "\\|").replace("\n", " ")
                lines.append(f"| `{r.name}` | {r.config or ''} | {r.model_id or ''} | {msg} |")
            if len(rows) > max_rows_per_status:
                lines.append(f"| … | | | {len(rows) - max_rows_per_status} more |")
            lines.append("")
        # Aggregate by check name for FAIL/WARN so repeated per-model issues are readable
        agg = Counter((r.name, r.status.value) for r in self.results if r.status in (Status.FAIL, Status.WARN))
        if agg:
            lines.append("## Issues grouped by check")
            lines.append("")
            lines.append("| check | status | occurrences |")
            lines.append("|---|---|---:|")
            for (name, status), n in sorted(agg.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"| `{name}` | {status} | {n} |")
            lines.append("")
        return "\n".join(lines)

    def print_summary(self, *, show: Sequence[Status] = (Status.FAIL, Status.WARN)) -> None:  # pragma: no cover
        print(f"Results: {self.counts()}")
        for st in show:
            for r in self.by_status(st):
                print(" ", r)


# ---------------------------------------------------------------------------
# Small helpers used by many check modules
# ---------------------------------------------------------------------------

def examples(values: Iterable[Any], n: int = 5) -> List[Any]:
    """Return up to *n* example values (deterministically ordered when possible)."""
    vals = list(values)
    try:
        vals = sorted(set(vals), key=lambda v: str(v))
    except Exception:  # pragma: no cover
        pass
    return vals[:n]


def pct(part: float, whole: float) -> float:
    return 100.0 * float(part) / float(whole) if whole else 0.0
