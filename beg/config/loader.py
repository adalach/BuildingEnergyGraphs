"""Loading graph configurations, and resolving ``extends``.

A configuration declares which SQL transforms to run and which of their outputs
become node and edge tables. The heterogeneous configurations of each level
(C3-C5, C8-C10) extend a base file in ``configs/base/`` and state only their
difference from it; C1 and C6 are self-contained.

Resolving ``extends`` is the only transformation applied; an invalid
configuration raises.

Variant syntax::

    {
      "extends": "base/zone_hetero.json",
      "id": "C5",
      "prefix": "zone_hetero_external_weather",
      "weather": "separate node",
      "description": "...",
      "sql_transforms": {"add": ["edge_zones_env_weather.sql"]},
      "edges": {"add": [ ... ], "remove": [ ... ]}
    }

For each of ``sql_transforms``, ``nodes`` and ``edges`` the variant may give
either an ``{"add": [...], "remove": [...]}`` object, applied to the inherited
list in that order, or a plain list, which replaces it outright.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from beg.config.validator import ConfigValidationError, validate_config

#: Keys whose values are lists that a variant may extend or replace.
LIST_KEYS = ("sql_transforms", "nodes", "edges")

#: How deep an ``extends`` chain may go before we call it a cycle.
MAX_EXTENDS_DEPTH = 8


def _merge_list(inherited: list[Any], directive: Any, key: str, source: Path) -> list[Any]:
    if isinstance(directive, list):
        return list(directive)
    if not isinstance(directive, dict):
        raise ConfigValidationError(
            f"{source}: '{key}' must be a list or an object with 'add'/'remove', "
            f"got {type(directive).__name__}"
        )
    unknown = set(directive) - {"add", "remove"}
    if unknown:
        raise ConfigValidationError(
            f"{source}: '{key}' accepts only 'add' and 'remove', got {sorted(unknown)}"
        )

    removals = directive.get("remove", [])
    result = [item for item in inherited if item not in removals]
    if len(result) + len(removals) != len(inherited):
        never_matched = [r for r in removals if r not in inherited]
        if never_matched:
            raise ConfigValidationError(
                f"{source}: '{key}' removes entries the base does not contain: {never_matched}"
            )
    return result + list(directive.get("add", []))


def resolve_config(path: Path, _depth: int = 0) -> dict[str, Any]:
    """Read one configuration file and apply its ``extends`` chain."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if _depth > MAX_EXTENDS_DEPTH:
        raise ConfigValidationError(f"{path}: 'extends' chain deeper than {MAX_EXTENDS_DEPTH}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    parent = raw.pop("extends", None)
    if parent is None:
        return raw

    base = resolve_config((path.parent / parent).resolve(), _depth + 1)
    merged = dict(base)
    for key, value in raw.items():
        if key in LIST_KEYS:
            merged[key] = _merge_list(base.get(key, []), value, key, path)
        else:
            merged[key] = value
    return merged


def load_config_json(config_path: Path) -> dict[str, Any]:
    """Load, resolve and validate one configuration file."""
    config_path = Path(config_path)
    cfg = resolve_config(config_path)
    validate_config(cfg, config_path.stem)
    return cfg


def config_path_for(configs_dir: Path, name: str) -> Path:
    """Locate a configuration file by id (``C5``) or slug (``zone_hetero_external_weather``)."""
    configs_dir = Path(configs_dir)
    candidates = [f"{name}.json"]

    try:
        from beg.config.naming import resolve as resolve_name

        config = resolve_name(name)
        candidates.insert(0, f"{config.id}.json")
    except (KeyError, ImportError):
        pass

    for candidate in candidates:
        path = configs_dir / candidate
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No configuration for {name!r} in {configs_dir}; tried {', '.join(candidates)}"
    )


def discover_config_paths(
    configs_dir: Path, names: list[str] | tuple[str, ...] | None = None
) -> dict[str, Path]:
    """Map configuration id to its file, for the configurations that have one.

    Names are resolved through :mod:`beg.config.naming`.

    Normally eight entries, not ten: **C2 and C7 have no file**. They are derived
    at materialisation time by attaching a graph-level weather vector to C1 and C6.
    """
    from beg.config.naming import CONFIG_ORDER, resolve

    configs_dir = Path(configs_dir)
    wanted = list(names) if names else list(CONFIG_ORDER)

    found: dict[str, Path] = {}
    for name in wanted:
        try:
            config_id = resolve(name).id
        except KeyError:
            config_id = name
        try:
            found[config_id] = config_path_for(configs_dir, name)
        except FileNotFoundError:
            continue  # derived configurations, and any not present in this directory
    return found


def load_configs_from_dir(configs_dir: Path, config_names: list[str]) -> pd.DataFrame:
    """Load several configurations into the ``configs_df`` the pipeline expects.

    Returns one row per configuration with the columns ``config_name``,
    ``config_path``, ``description``, ``homogeneous``, ``type``,
    ``sql_transforms``, ``nodes``, ``edges``, ``prefix`` and ``config_full_name``.

    Every configuration is attempted before raising, so all errors are reported
    together.
    """
    configs_dir = Path(configs_dir)
    if not configs_dir.exists():
        raise FileNotFoundError(f"Configs directory not found: {configs_dir}")

    rows: list[dict[str, Any]] = []
    errors: list[tuple[str, str]] = []

    for idx, config_name in enumerate(config_names, start=1):
        try:
            config_file = config_path_for(configs_dir, config_name)
            cfg = load_config_json(config_file)
            homogeneous = cfg["type"] == "homogeneous"
            rows.append(
                {
                    "config_name": cfg.get("prefix", config_name),
                    "config_path": config_file,
                    "description": cfg.get("description", ""),
                    "homogeneous": homogeneous,
                    "type": cfg["type"],
                    "sql_transforms": cfg["sql_transforms"],
                    "nodes": cfg["nodes"],
                    "edges": cfg["edges"],
                    "prefix": cfg["prefix"],
                    "config_full_name": infer_config_full_name(
                        config_name=cfg.get("prefix", config_name),
                        index=idx,
                        config_id=cfg.get("id"),
                    ),
                }
            )
        except (FileNotFoundError, json.JSONDecodeError, ConfigValidationError) as exc:
            errors.append((config_name, str(exc)))

    if errors:
        message = "Failed to load configuration(s):\n\n"
        for config_name, error in errors:
            message += f"  {config_name}:\n     {error}\n\n"
        raise ConfigValidationError(message)

    return pd.DataFrame(rows)


def infer_config_full_name(
    config_name: str,
    index: int,
    config_id: str | None = None,
) -> str:
    """Directory-ordering name, e.g. ``C5_zone_hetero_external_weather``.

    Falls back to the load-order index when the configuration carries no ``id``.
    """
    raw_prefix = str(config_id).strip() if config_id is not None else ""
    prefix = raw_prefix or f"C{index}"
    return f"{prefix}_{config_name}"
