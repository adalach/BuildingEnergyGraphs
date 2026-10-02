"""Parsing ``Energy+.schema.epJSON`` and pre-scanning the model set.

Two things have to be known before any table can be created: what EnergyPlus
says each object type looks like, and which object types the model set actually
uses. Tables are built for the intersection, so an object the schema describes
but no model contains never reaches the database.

The schema also carries the *reference labels* that drive foreign keys: an object
advertises labels under its ``name.reference`` key, and a field declared
``object_list`` names the labels it points at.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from beg.ingest.schema_map import return_as_list


@dataclass
class EnergyPlusSchema:
    """The parsed schema: field metadata per object, plus the label maps."""

    path: Path
    version: str | None
    tables: dict[str, pd.DataFrame] = field(repr=False, default_factory=dict)
    label_to_objects: dict[str, set[str]] = field(repr=False, default_factory=dict)
    object_to_labels: dict[str, set[str]] = field(repr=False, default_factory=dict)

    def __len__(self) -> int:
        return len(self.tables)

    def summary(self) -> pd.DataFrame:
        """One-row frame describing what was parsed."""
        return pd.DataFrame(
            [
                {
                    "schema_file": self.path.name,
                    "energyplus_version": self.version or "unknown",
                    "object_types": len(self.tables),
                    "reference_labels": len(self.label_to_objects),
                    "objects_advertising_labels": len(self.object_to_labels),
                }
            ]
        )

    def fields_of(self, obj_name: str) -> pd.DataFrame:
        """Field metadata for one object type, as parsed."""
        return self.tables[obj_name]


def schema_version(schema_data: dict) -> str | None:
    """The EnergyPlus version a schema describes, read from ``Version``'s default."""
    try:
        version_obj = schema_data["properties"]["Version"]["patternProperties"]
        first = next(iter(version_obj.values()))
        return str(first["properties"]["version_identifier"]["default"])
    except (KeyError, StopIteration, TypeError):
        return None


def load_energyplus_schema(path: Path) -> EnergyPlusSchema:
    """Parse the EnergyPlus schema into per-object field tables and label maps.

    Each object's fields become one row per field in a DataFrame, with the union
    of all field-metadata keys as columns, so downstream code can read ``type``,
    ``default``, ``enum``, ``minimum``, ``object_list`` and friends uniformly.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"EnergyPlus schema not found at {path}. See docs/data.md for where it comes from."
        )

    schema_data = json.loads(path.read_text(encoding="utf-8"))
    parsed = EnergyPlusSchema(path=path, version=schema_version(schema_data))

    for obj_name, obj_schema in schema_data.get("properties", {}).items():
        # EnergyPlus places object field definitions under patternProperties
        patt = obj_schema.get("patternProperties", {})
        obj_def = next(iter(patt.values()), None)
        if not obj_def or "properties" not in obj_def:
            # Skip an object without field definitions
            continue

        props = obj_def["properties"]
        req_set = set(obj_def.get("required", []))

        # Union all top-level keys across fields for a consistent/wide DataFrame
        all_keys = set().union(*(p.keys() for p in props.values())) if props else set()

        rows = []
        for fname, fdef in props.items():
            row = {"field": fname, "required": fname in req_set}
            for k in all_keys:
                row[k] = fdef.get(k, None)
            rows.append(row)

        parsed.tables[obj_name] = pd.DataFrame(rows)

        # Advertised reference labels drive foreign key inference later
        name_block = obj_schema.get("name") or obj_def.get("name") or {}
        refs = return_as_list(name_block.get("reference"))
        if refs:
            parsed.object_to_labels.setdefault(obj_name, set()).update(refs)
            for lbl in refs:
                parsed.label_to_objects.setdefault(lbl, set()).add(obj_name)

    return parsed


def scan_used_objects(epjson_files) -> tuple[set[str], Counter]:
    """Pre-scan the model set: which object types occur, and how many instances.

    Returns ``(used_objects, object_counts)``. The table set must be complete
    before the first row is written, so every file is read here and again during
    ingestion.
    """
    used_objects: set[str] = set()
    object_counts: Counter = Counter()

    for fp in epjson_files:
        data = json.loads(Path(fp).read_text(encoding="utf-8"))
        for obj_name, instances in data.items():
            used_objects.add(obj_name)
            if isinstance(instances, (dict, list)):
                object_counts[obj_name] += len(instances)
            else:
                # Rare: a scalar or null payload counts as one instance
                object_counts[obj_name] += 1

    return used_objects, object_counts


def usage_frame(object_counts: Counter, schema: EnergyPlusSchema | None = None) -> pd.DataFrame:
    """Object types by instance count, flagged with whether the schema describes them."""
    frame = pd.DataFrame(
        {"object": list(object_counts.keys()), "instances": list(object_counts.values())}
    )
    if schema is not None:
        frame["in_schema"] = frame["object"].isin(schema.tables)
    return frame.sort_values("instances", ascending=False).reset_index(drop=True)


def model_files(epjson_dir: Path) -> list[Path]:
    """Every epJSON file in a directory, matching either extension spelling."""
    epjson_dir = Path(epjson_dir)
    return sorted({*epjson_dir.glob("*.epjson"), *epjson_dir.glob("*.epJSON")})
