"""
Configuration validation.

Strict validation: an invalid configuration raises, and nothing is rewritten.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Required top-level keys
REQUIRED_KEYS = ['prefix', 'type', 'nodes', 'edges', 'sql_transforms']

# Valid configuration types
VALID_TYPES = ['homogeneous', 'heterogeneous']


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""
    pass


def validate_config(cfg: Dict[str, Any], config_name: str = "config") -> None:
    """
    Strict validation of configuration structure; ``cfg`` is not modified.

    Args:
        cfg: Configuration dictionary
        config_name: Name for error messages

    Raises:
        ConfigValidationError: If configuration is invalid
    """
    # Check required keys
    validate_required_keys(cfg, config_name)

    # Validate type field
    validate_type_field(cfg, config_name)

    # Validate sql_transforms
    validate_sql_transforms(cfg.get('sql_transforms', []), config_name)

    # Validate nodes structure
    validate_nodes_structure(cfg.get('nodes', []), config_name)

    # Validate edges structure
    validate_edges_structure(cfg.get('edges', []), config_name)

    # Validate type consistency
    if cfg['type'] == 'homogeneous':
        validate_homogeneous_consistency(cfg, config_name)
    else:
        validate_heterogeneous_consistency(cfg, config_name)


def validate_required_keys(cfg: Dict[str, Any], config_name: str) -> None:
    """
    Check all required keys exist.

    Args:
        cfg: Configuration dictionary
        config_name: Name for error messages

    Raises:
        ConfigValidationError: If any required key is missing
    """
    missing = [key for key in REQUIRED_KEYS if key not in cfg]
    if missing:
        raise ConfigValidationError(
            f"Config '{config_name}' is missing required keys: {missing}\n"
            f"Required keys: {REQUIRED_KEYS}"
        )


def validate_type_field(cfg: Dict[str, Any], config_name: str) -> None:
    """
    Validate 'type' field is valid.

    Args:
        cfg: Configuration dictionary
        config_name: Name for error messages

    Raises:
        ConfigValidationError: If type is invalid
    """
    config_type = cfg.get('type')

    if config_type not in VALID_TYPES:
        raise ConfigValidationError(
            f"Config '{config_name}' has invalid type: '{config_type}'\n"
            f"Valid types: {VALID_TYPES}"
        )


def validate_homogeneous_consistency(cfg: Dict[str, Any], config_name: str) -> None:
    """
    Validate homogeneous configuration consistency.

    Homogeneous graphs must have:
    - Exactly 1 node type
    - Exactly 1 edge type (or edges that collapse to 1 type)

    Args:
        cfg: Configuration dictionary
        config_name: Name for error messages

    Raises:
        ConfigValidationError: If inconsistent with homogeneous type
    """
    nodes = cfg.get('nodes', [])
    edges = cfg.get('edges', [])

    # Check node count
    if len(nodes) != 1:
        raise ConfigValidationError(
            f"Config '{config_name}' is type='homogeneous' but has {len(nodes)} node types.\n"
            f"Homogeneous graphs must have exactly 1 node type.\n"
            f"Either:\n"
            f"  1. Change type to 'heterogeneous', OR\n"
            f"  2. Remove extra node types to have only 1"
        )

    # Several edge definitions are fine if all connect the one node type: warn only
    if len(edges) > 1:
        import warnings
        warnings.warn(
            f"Config '{config_name}' is homogeneous but has {len(edges)} edge definitions. "
            "This is OK if they all connect the same node type.",
            UserWarning,
            stacklevel=2,
        )


def validate_heterogeneous_consistency(cfg: Dict[str, Any], config_name: str) -> None:
    """
    Validate heterogeneous configuration consistency.

    Heterogeneous graphs must have:
    - 2+ node types OR 2+ edge types

    Args:
        cfg: Configuration dictionary
        config_name: Name for error messages

    Raises:
        ConfigValidationError: If inconsistent with heterogeneous type
    """
    nodes = cfg.get('nodes', [])
    edges = cfg.get('edges', [])

    # Must have multiple types somewhere
    if len(nodes) == 1 and len(edges) <= 1:
        raise ConfigValidationError(
            f"Config '{config_name}' is type='heterogeneous' but has only {len(nodes)} node type(s) "
            f"and {len(edges)} edge type(s).\n"
            f"Heterogeneous graphs must have 2+ node types OR 2+ edge types.\n"
            f"Either:\n"
            f"  1. Change type to 'homogeneous', OR\n"
            f"  2. Add more node/edge types"
        )


def validate_nodes_structure(nodes: List[Any], config_name: str) -> None:
    """
    Validate nodes list structure.

    Each node can be:
    - String: "zones_zone"
    - Dict: {"table": "zones_zone", "rules": {...}}

    Args:
        nodes: Nodes list from config
        config_name: Name for error messages

    Raises:
        ConfigValidationError: If structure is invalid
    """
    if not isinstance(nodes, list):
        raise ConfigValidationError(
            f"Config '{config_name}' has invalid 'nodes' field: must be a list, got {type(nodes).__name__}"
        )

    if len(nodes) == 0:
        raise ConfigValidationError(
            f"Config '{config_name}' has empty 'nodes' list. Must have at least 1 node type."
        )

    for i, node in enumerate(nodes):
        if isinstance(node, str):
            # Simple string format is OK
            continue
        elif isinstance(node, dict):
            # Dict format must have 'table' key
            if 'table' not in node:
                raise ConfigValidationError(
                    f"Config '{config_name}' node #{i} is a dict but missing 'table' key.\n"
                    f"Node dict format: {{'table': 'table_name', 'rules': {{...}}}}"
                )
        else:
            raise ConfigValidationError(
                f"Config '{config_name}' node #{i} has invalid type: {type(node).__name__}\n"
                f"Must be string (table name) or dict with 'table' key"
            )


def validate_edges_structure(edges: List[Any], config_name: str) -> None:
    """
    Validate edges list structure.

    Each edge must be a dict with required keys.

    Args:
        edges: Edges list from config
        config_name: Name for error messages

    Raises:
        ConfigValidationError: If structure is invalid
    """
    if not isinstance(edges, list):
        raise ConfigValidationError(
            f"Config '{config_name}' has invalid 'edges' field: must be a list, got {type(edges).__name__}"
        )

    # Empty edges list is OK (though unusual)
    if len(edges) == 0:
        import warnings
        warnings.warn(
            f"Config '{config_name}' has empty 'edges' list. Graph will have no edges.",
            UserWarning,
            stacklevel=2,
        )
        return

    for i, edge in enumerate(edges):
        if not isinstance(edge, dict):
            raise ConfigValidationError(
                f"Config '{config_name}' edge #{i} must be a dict, got {type(edge).__name__}"
            )

        # Every edge needs edge_src; a plain foreign-key edge uses edge_src "fk"
        if 'edge_src' not in edge:
            raise ConfigValidationError(
                f"Config '{config_name}' edge #{i} missing 'edge_src' key.\n"
                f"Required keys: edge_src, src_cat, tgt_cat"
            )


def validate_sql_transforms(sql_transforms: List[str], config_name: str) -> None:
    """
    Validate sql_transforms list.

    Args:
        sql_transforms: SQL transforms list from config
        config_name: Name for error messages

    Raises:
        ConfigValidationError: If structure is invalid
    """
    if not isinstance(sql_transforms, list):
        raise ConfigValidationError(
            f"Config '{config_name}' has invalid 'sql_transforms' field: "
            f"must be a list, got {type(sql_transforms).__name__}"
        )

    for i, transform in enumerate(sql_transforms):
        if not isinstance(transform, str):
            raise ConfigValidationError(
                f"Config '{config_name}' sql_transforms[{i}] must be a string (SQL file name), "
                f"got {type(transform).__name__}: {transform}"
            )

        # Check it ends with .sql
        if not transform.endswith('.sql'):
            raise ConfigValidationError(
                f"Config '{config_name}' sql_transforms[{i}] should end with '.sql': {transform}"
            )
