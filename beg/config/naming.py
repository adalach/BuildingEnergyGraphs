"""The single source of truth for configuration identity.

Every configuration name and the canonical C1-C10 order live here; notebooks,
scripts and figure generators import them.

A configuration has three axes:

``level``            ``zone`` | ``component``
``graph_type``       ``hetero`` | ``homo``
``weather_variant``  ``no_weather`` | ``building_weather`` | ``external_weather``
                     | ``graph_weather``

Eight configurations are declared in ``configs/``. Two more, C2 and C7, have no
config file: they are *derived* at materialisation time by attaching a graph-level
weather vector to the homogeneous base configuration, C1 and C6 respectively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Level = Literal["zone", "component"]
GraphType = Literal["hetero", "homo"]
WeatherVariant = Literal["no_weather", "building_weather", "external_weather", "graph_weather"]


@dataclass(frozen=True)
class ConfigId:
    """One graph configuration in the benchmark."""

    id: str
    level: Level
    graph_type: GraphType
    weather_variant: WeatherVariant
    description: str
    #: Base configuration this one is derived from, or None when declared in configs/.
    derived_from: str | None = None

    @property
    def slug(self) -> str:
        """``zone_homo_no_weather`` - the three axes, as used in file names."""
        return f"{self.level}_{self.graph_type}_{self.weather_variant}"

    @property
    def full_name(self) -> str:
        """``C1_zone_homo_no_weather`` - the identifier used for directories and manifests."""
        return f"{self.id}_{self.slug}"

    @property
    def config_filename(self) -> str:
        """The file in ``configs/`` that declares this configuration."""
        if self.derived_from is not None:
            raise ValueError(
                f"{self.id} is derived from {self.derived_from} and has no config file"
            )
        return f"{self.id}.json"

    @property
    def is_derived(self) -> bool:
        return self.derived_from is not None

    @property
    def has_weather(self) -> bool:
        return self.weather_variant != "no_weather"


CONFIGS: tuple[ConfigId, ...] = (
    ConfigId("C1", "zone", "homo", "no_weather", "Zone graph, homogeneous, no weather"),
    ConfigId(
        "C2",
        "zone",
        "homo",
        "graph_weather",
        "Zone graph, homogeneous, weather as a graph-level feature vector",
        derived_from="C1",
    ),
    ConfigId("C3", "zone", "hetero", "no_weather", "Zone and building nodes, no weather"),
    ConfigId(
        "C4", "zone", "hetero", "building_weather", "Zone and building nodes, weather in the building node"
    ),
    ConfigId(
        "C5", "zone", "hetero", "external_weather", "Zone and building nodes, weather as a separate node"
    ),
    ConfigId("C6", "component", "homo", "no_weather", "Component graph, homogeneous, no weather"),
    ConfigId(
        "C7",
        "component",
        "homo",
        "graph_weather",
        "Component graph, homogeneous, weather as a graph-level feature vector",
        derived_from="C6",
    ),
    ConfigId(
        "C8", "component", "hetero", "no_weather", "Zone, element, aperture and building nodes, no weather"
    ),
    ConfigId(
        "C9",
        "component",
        "hetero",
        "building_weather",
        "Zone, element, aperture and building nodes, weather in the building node",
    ),
    ConfigId(
        "C10",
        "component",
        "hetero",
        "external_weather",
        "Zone, element, aperture and building nodes, weather as a separate node",
    ),
)

#: Canonical order for tables and figures: zone family first, then component.
CONFIG_ORDER: tuple[str, ...] = tuple(c.id for c in CONFIGS)

BY_ID: dict[str, ConfigId] = {c.id: c for c in CONFIGS}
BY_FULL_NAME: dict[str, ConfigId] = {c.full_name: c for c in CONFIGS}
BY_SLUG: dict[str, ConfigId] = {c.slug: c for c in CONFIGS}

#: Configurations declared by a file in configs/.
DECLARED: tuple[ConfigId, ...] = tuple(c for c in CONFIGS if not c.is_derived)

#: Configurations built by attaching graph-level weather to a base configuration.
DERIVED: dict[str, str] = {c.id: c.derived_from for c in CONFIGS if c.derived_from}

HOMO_CONFIGS: tuple[str, ...] = tuple(c.id for c in CONFIGS if c.graph_type == "homo")
HETERO_CONFIGS: tuple[str, ...] = tuple(c.id for c in CONFIGS if c.graph_type == "hetero")


def resolve(name: str) -> ConfigId:
    """Look a configuration up by id (``C5``), full name or slug, case-insensitively."""
    key = name.strip()
    for table in (BY_ID, BY_FULL_NAME, BY_SLUG):
        if key in table:
            return table[key]

    lowered = key.lower()
    for table in (BY_FULL_NAME, BY_SLUG):
        for candidate, config in table.items():
            if candidate.lower() == lowered:
                return config

    raise KeyError(
        f"unknown configuration {name!r}; expected one of {', '.join(CONFIG_ORDER)} "
        f"or a full name such as {CONFIGS[0].full_name}"
    )


def sort_key(name: str) -> int:
    """Sort configurations C1..C10 numerically rather than lexicographically."""
    return CONFIG_ORDER.index(resolve(name).id)

