"""
Checks on the target labels (``data/labels.csv``: ``building_id,eui_kwh_per_m2``)
and their coverage of the generated dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from beg.validation.core import CheckResult, examples, pct

SCOPE = "labels"


def load_labels(path: Path | str, *, id_col: str = "building_id", label_col: str = "eui_kwh_per_m2") -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in (id_col, label_col) if c not in df.columns]
    if missing:
        raise ValueError(f"labels file lacks columns {missing}; has {list(df.columns)}")
    return df


def check_labels(
    labels: pd.DataFrame,
    *,
    id_col: str = "building_id",
    label_col: str = "eui_kwh_per_m2",
    dataset_model_ids: Optional[Iterable[str]] = None,
    plausible_range: tuple = (5.0, 3000.0),
    outlier_sigma: float = 5.0,
) -> List[CheckResult]:
    kw = dict(scope=SCOPE)
    r: List[CheckResult] = []
    n = len(labels)
    if n == 0:
        return [CheckResult.fail("labels.nonempty", "labels table is empty", **kw)]
    ids = labels[id_col].astype(str)
    dup = ids[ids.duplicated(keep=False)]
    if len(dup):
        r.append(CheckResult.fail("labels.ids_unique", f"{dup.nunique()} building_ids appear more than once e.g. {examples(dup)}", **kw))
    else:
        r.append(CheckResult.ok("labels.ids_unique", f"{n} unique building ids", **kw))
    y = pd.to_numeric(labels[label_col], errors="coerce")
    nan = int(y.isna().sum())
    if nan:
        r.append(CheckResult.fail("labels.finite", f"{nan} labels are missing/non-numeric e.g. {examples(ids[y.isna()])}", **kw))
    inf = int(np.isinf(y.fillna(0)).sum())
    if inf:
        r.append(CheckResult.fail("labels.finite", f"{inf} infinite labels", **kw))
    nonpos = int((y <= 0).sum())
    if nonpos:
        r.append(CheckResult.fail("labels.positive", f"{nonpos} labels ≤ 0 (EUI must be positive; simulation failed?) e.g. {examples(ids[y <= 0])}", **kw))
    lo, hi = plausible_range
    out_range = (y < lo) | (y > hi)
    k = int(out_range.sum())
    if k:
        r.append(CheckResult.warn("labels.plausible_range", f"{k} labels outside [{lo}, {hi}] kWh/m²a e.g. {examples(list(zip(ids[out_range], y[out_range].round(1))))}", **kw))
    valid = y.dropna()
    if len(valid) > 10:
        mu, sd = float(valid.mean()), float(valid.std())
        z = ((valid - mu) / sd).abs() if sd > 0 else pd.Series(0.0, index=valid.index)
        k = int((z > outlier_sigma).sum())
        if k:
            r.append(CheckResult.warn("labels.outliers", f"{k} labels beyond {outlier_sigma}σ (mean={mu:.1f}, std={sd:.1f}) e.g. {examples(list(zip(ids[z > outlier_sigma], valid[z > outlier_sigma].round(1))))}", **kw))
        r.append(CheckResult.ok("labels.distribution", f"n={len(valid)} mean={mu:.1f} std={sd:.1f} min={valid.min():.1f} p50={valid.median():.1f} max={valid.max():.1f}", details={"mean": mu, "std": sd, "min": float(valid.min()), "max": float(valid.max())}, **kw))
        if sd == 0:
            r.append(CheckResult.fail("labels.variance", "all labels identical", **kw))
        # suspicious repeated exact values (copy/paste or default)
        vc = valid.round(6).value_counts()
        rep = vc[vc > max(3, 0.005 * len(valid))]
        if len(rep):
            r.append(CheckResult.warn("labels.repeated_values", f"label values repeated unusually often: {dict(rep.head(5))}", **kw))
    if dataset_model_ids is not None:
        ds = {str(m) for m in dataset_model_ids}
        lab = set(ids)
        norm = lambda s: str(s).strip().lower()  # noqa: E731
        lab_norm = {norm(label) for label in lab}
        missing_exact = sorted(ds - lab)
        # ids that join only after case/whitespace normalisation (the pipeline lower-cases ids)
        norm_only = [m for m in missing_exact if norm(m) in lab_norm]
        missing = [m for m in missing_exact if norm(m) not in lab_norm]
        if missing:
            r.append(CheckResult.fail("labels.cover_dataset", f"{len(missing)}/{len(ds)} dataset models have no label (even after case normalisation) e.g. {examples(missing)}", details={"missing": missing[:100]}, **kw))
        else:
            r.append(CheckResult.ok("labels.cover_dataset", f"all {len(ds)} dataset models have a label", **kw))
        if norm_only:
            r.append(CheckResult.warn("labels.id_normalisation", f"{len(norm_only)} dataset ids match a label only after case/whitespace normalisation (e.g. {examples(norm_only, 3)}) — every join must normalise consistently", **kw))
        unused = sorted(label for label in lab if label not in ds and norm(label) not in {norm(m) for m in ds})
        if unused:
            r.append(CheckResult.warn("labels.unused", f"{len(unused)} labelled buildings have no graph in the dataset ({pct(len(unused), len(lab)):.1f}%) e.g. {examples(unused)}", **kw))
        dup_norm = ids.map(norm)
        dn = dup_norm[dup_norm.duplicated(keep=False)]
        if len(dn) and not len(dup):
            r.append(CheckResult.fail("labels.ids_unique_normalised", f"{dn.nunique()} building_ids collide after case normalisation e.g. {examples(dn)}", **kw))
    return r
