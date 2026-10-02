"""Configuration identity, loading and validation.

``naming`` holds the C1-C10 definitions and is the only place any part of this
project decides what a configuration is called. ``loader`` reads the JSON
declarations in ``configs/`` and resolves ``extends`` into a full configuration.
"""

from beg.config.naming import (
    CONFIG_ORDER,
    CONFIGS,
    DECLARED,
    DERIVED,
    HETERO_CONFIGS,
    HOMO_CONFIGS,
    ConfigId,
    resolve,
    sort_key,
)

__all__ = [
    "ConfigId",
    "CONFIGS",
    "CONFIG_ORDER",
    "DECLARED",
    "DERIVED",
    "HOMO_CONFIGS",
    "HETERO_CONFIGS",
    "resolve",
    "sort_key",
]
