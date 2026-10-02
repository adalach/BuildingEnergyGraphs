"""The paper's configuration-topology figure: one panel per distinct topology.

Weather enters a graph in three ways and only one of them changes the shape:
``external_weather`` adds a weather node, while ``graph_weather`` attaches a
vector to the graph and ``building_weather`` widens the building node's features.
Neither of the latter two is visible in a drawing, so C1/C2, C3/C4, C6/C7 and
C8/C9 each share a panel and the ten configurations make six.

The panels are read from the exported ``.pt`` files, which carry the topology and
the type of every node and edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from beg.build.graph_utils import pyg_to_nx
from beg.config.naming import CONFIGS, resolve

#: Fill colour per node type. The homogeneous configurations have one node type
#: by construction, named here for the level they describe.
NODE_COLOURS: dict[str, str] = {
    "building": "#A2AD00",
    "zones_zone": "#F9BA00",
    "comp_zone": "#D64C13",
    "comp_element": "#F9BA00",
    "comp_aperture": "#98C6EA",
    "weather_city": "#0065BD",
    "zone": "#F9BA00",
    "component": "#50369A",
}

#: Ring colour, where the figure draws one. A zone carrying its components'
#: features is drawn as an orange disc with a red ring, to separate it from the
#: plain zone of a component-level graph.
NODE_RINGS: dict[str, str] = {
    "zones_zone": "#D64C13",
    "zone": "#D64C13",
}

#: Legend wording per node type.
NODE_LABELS: dict[str, str] = {
    "building": "Building",
    "zones_zone": "Zone with component features",
    "comp_zone": "Zone",
    "comp_element": "Component: Wall / Floor / Roof / Ceiling",
    "comp_aperture": "Aperture",
    "weather_city": "Weather",
    "zone": "Zone with component features",
    "component": "Unified component",
}

#: Colour and legend wording per edge role.
EDGE_STYLES: dict[str, tuple[str, str]] = {
    "unified": ("#0065BD", "Unified edge"),
    "adjacency": ("#8C8C8C", "Adjacency"),
    "weather": ("#E37222", "Weather → Exposed object"),
    "building_zone": ("#FF4EA7", "Building ⇆ Zone membership"),
    "zone_component": ("#609818", "Zone ⇆ Component membership"),
    "component_aperture": ("#F9BA00", "Component ⇆ Aperture membership"),
}

#: Typography, in points at the figure's native size. The figure is authored
#: large and scaled to the page, so these are chosen to land near the panel
#: titles after scaling, as the published legend does.
TITLE_FONT = 18
COUNT_FONT = 15
LEGEND_FONT = 17
LEGEND_MARKER = 19
LEGEND_LINE = 5.0

#: How far from the centre each node type sits. The published panels are drawn
#: as concentric shells, which is what makes a component graph readable: the
#: building at the centre, its zones around it, their components around those,
#: and the apertures on the rim.
NODE_SHELL_ORDER: dict[str, int] = {
    "weather_city": 0,
    "building": 0,
    "comp_zone": 1,
    "zones_zone": 1,
    "zone": 1,
    "comp_element": 2,
    "component": 2,
    "comp_aperture": 3,
}


def edge_role(rel: str, src_type: str, tgt_type: str) -> str:
    """Which legend entry an edge belongs to, from its relation and endpoints.

    ``belongs_to`` carries two different meanings depending on what it connects —
    a zone to its building, or a component to its zone — so the endpoints decide,
    not the relation name alone.
    """
    rel = str(rel)
    if rel == "unified":
        return "unified"
    if "adjacent" in rel:
        return "adjacency"
    if "weather" in rel or "weather_city" in (src_type, tgt_type):
        return "weather"
    if "building" in (src_type, tgt_type):
        return "building_zone"
    if "aperture" in src_type or "aperture" in tgt_type:
        return "component_aperture"
    return "zone_component"


@dataclass
class TopologyPanel:
    """One panel: the configurations sharing a topology, and a graph showing it."""

    title: str
    config_ids: list[str]
    representative: str
    graph: Any

    @property
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.graph.number_of_edges()


def distinct_topologies() -> list[tuple[str, list[str]]]:
    """Group the ten configurations by the shape they produce.

    Returns ``(representative_id, [ids sharing it])`` in configuration order. Two
    configurations share a topology when they agree on level, graph type, and
    whether weather enters as its own node.
    """
    groups: dict[tuple, list[str]] = {}
    for config in CONFIGS:
        key = (
            config.level,
            config.graph_type,
            config.weather_variant == "external_weather",
        )
        groups.setdefault(key, []).append(config.id)

    ordered = sorted(groups.values(), key=lambda ids: int(ids[0].lstrip("C")))
    return [(ids[0], ids) for ids in ordered]


def load_topology_panels(
    graphs_root: Path,
    model_id: str,
    *,
    topologies: list[tuple[str, list[str]]] | None = None,
) -> list[TopologyPanel]:
    """Read one exported graph per distinct topology, for a single building."""
    import torch

    graphs_root = Path(graphs_root)
    panels = []

    for representative, ids in topologies or distinct_topologies():
        config = resolve(representative)
        kind = "homo" if config.graph_type == "homo" else "hetero"
        path = graphs_root / kind / config.full_name / f"{model_id}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"No exported graph for {config.full_name} / {model_id}: run the "
                "export cells above, or choose a FIGURE_MODEL that this run built."
            )

        graph = torch.load(path, map_location="cpu", weights_only=False)
        panels.append(
            TopologyPanel(
                title=" / ".join(ids),
                config_ids=list(ids),
                representative=representative,
                graph=pyg_to_nx(graph, node_type=config.level),
            )
        )

    return panels


def _node_size_for(n_nodes: int, scale: float) -> float:
    """Big discs on the small graphs, smaller on the dense ones, as in the paper."""
    return scale * (1100.0 if n_nodes <= 24 else 170.0)


def _edge_width_for(n_nodes: int, scale: float) -> float:
    return scale * (2.2 if n_nodes <= 24 else 1.3)


def _concentric(graph, shells: list[list], *, radii=(0.0, 0.40, 0.72, 1.0)):
    """Place each shell on its own circle, ordered so children follow parents.

    Each ring is sorted by the angle of its parent in the ring inside it, which
    gathers it into the fans the published figure shows: one wedge per zone, its
    components on the next ring out, their apertures beyond those.
    """
    import numpy as np

    pos: dict = {}
    ring_of = {node: ring for ring, nodes in enumerate(shells) for node in nodes}
    for ring, nodes in enumerate(shells):
        radius = radii[min(ring, len(radii) - 1)]

        if ring == 0:
            if len(nodes) == 1:
                pos[nodes[0]] = np.array([0.0, 0.0])
            else:
                for i, node in enumerate(nodes):
                    angle = 2 * np.pi * i / len(nodes) + np.pi / 2
                    pos[node] = 0.12 * np.array([np.cos(angle), np.sin(angle)])
            continue

        # Zones all hang off the one building at the centre, so their parent
        # angle says nothing. Take their order from the zone grid instead: the
        # ring then walks around the building the way the zones touch, and
        # interzone edges run along the ring rather than across it.
        zone_ring = [n for n in nodes if str(graph.nodes[n].get("node_type", "")) in ZONE_TYPES]
        grid_angle = {}
        if len(zone_ring) > 1:
            grid = _zone_grid(graph, zone_ring)
            grid_angle = {n: float(np.arctan2(xy[1], xy[0])) for n, xy in grid.items()}

        def parent_angle(node, grid_angle=grid_angle, ring=ring):
            if node in grid_angle:
                return grid_angle[node]
            # Parents are the neighbours on the ring directly inside, and only
            # those. The weather node reaches every component from the centre;
            # counting it would pull every component toward one angle and
            # scramble the wedges it is meant to preserve.
            neighbours = set(graph.predecessors(node)) | set(graph.successors(node))
            inner = [n for n in neighbours if ring_of.get(n) == ring - 1 and n in pos]
            angles = [np.arctan2(*pos[n][::-1]) for n in inner
                      if float(np.hypot(*pos[n])) > 1e-9]
            if not angles:
                return 0.0
            # Circular mean: a plain mean of -179 and +179 degrees is 0, the wrong side.
            return float(np.angle(np.mean(np.exp(1j * np.array(angles)))))

        # Sort by parent angle, space evenly, then rotate the whole ring so the
        # sequence lines up with its parents and each wedge of children sits
        # beside its parent.
        ordered = sorted(nodes, key=parent_angle)
        n = max(len(ordered), 1)
        even = 2 * np.pi * np.arange(len(ordered)) / n
        targets = np.array([parent_angle(node) for node in ordered])
        offset = float(np.angle(np.mean(np.exp(1j * (targets - even)))))
        for node, angle in zip(ordered, even + offset, strict=True):
            pos[node] = radius * np.array([np.cos(angle), np.sin(angle)])
    return pos


#: Node types that are zones, at either level.
ZONE_TYPES = frozenset({"zone", "zones_zone", "comp_zone"})

#: Node types that sit inside a zone: once a panel has these, it is drawn as rings.
CONTAINED_TYPES = frozenset({"comp_element", "component", "comp_aperture"})


def _zone_grid(graph, zones: list) -> dict:
    """Zones laid out the way they touch, with no crossing interzone edges.

    Each connected group of zones (zones joined by shared walls, floors or
    ceilings) is laid out on its own and the groups stand side by side:

    * a **ring** of contacts - the common case, and what a building of two floors
      with two zones each produces - becomes a regular polygon, so a 2x2 building
      is a square. The lowest-index zone takes the bottom-left corner and the walk
      goes up first, which with the ``zone<i>_L<floor>`` naming puts zone 0 in the
      left column and the ground floor in the bottom row;
    * a **chain** becomes a vertical column;
    * anything else uses a planar layout when one exists, so nothing crosses, and
      Kamada-Kawai otherwise.

    With the building node at the centre, spokes to the corners of a convex
    polygon never cross its sides, so C3/C4 stay crossing-free as well.
    """
    import networkx as nx
    import numpy as np

    contacts = nx.Graph()
    contacts.add_nodes_from(zones)
    contacts.add_edges_from(
        (u, v) for u, v in graph.to_undirected().edges()
        if u in contacts and v in contacts and u != v
    )

    def index_of(node):
        return graph.nodes[node].get("index", 0)

    def ring_order(sub):
        start = min(sub.nodes, key=index_of)
        order, previous, current = [start], None, min(sub.neighbors(start), key=index_of)
        previous = start
        while current != start:
            order.append(current)
            step = [n for n in sub.neighbors(current) if n != previous][0]
            previous, current = current, step
        return order

    def group_layout(sub) -> dict:
        n = sub.number_of_nodes()
        if n == 1:
            return {next(iter(sub.nodes)): np.zeros(2)}
        degrees = [d for _, d in sub.degree()]
        if n >= 3 and all(d == 2 for d in degrees):
            order = ring_order(sub)
            # Start at the bottom-left corner and go clockwise, i.e. up first.
            angles = np.deg2rad(225.0) - 2 * np.pi * np.arange(n) / n
            return {node: np.array([np.cos(a), np.sin(a)]) for node, a in zip(order, angles)}
        if nx.is_tree(sub) and max(degrees) <= 2:
            ends = [node for node, d in sub.degree() if d == 1]
            order = list(nx.dfs_preorder_nodes(sub, source=min(ends, key=index_of)))
            return {node: np.array([0.0, float(row)]) for row, node in enumerate(order)}
        is_planar, _ = nx.check_planarity(sub)
        if is_planar:
            return nx.planar_layout(sub)
        try:
            return nx.kamada_kawai_layout(sub)
        except ImportError:  # kamada_kawai needs scipy
            return nx.spring_layout(sub, seed=7)

    groups = sorted(nx.connected_components(contacts), key=lambda c: min(index_of(n) for n in c))
    pos: dict = {}
    x_cursor = 0.0
    for group in groups:
        local = group_layout(contacts.subgraph(group))
        xy = np.array(list(local.values()), dtype=float)
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        span = max(float((hi - lo).max()), 1.0)
        for node, v in local.items():
            pos[node] = np.array([x_cursor, 0.0]) + (np.asarray(v, dtype=float) - lo) / span
        x_cursor += (hi[0] - lo[0]) / span + 1.0

    # Centre on the origin and fit the largest dimension to [-1, 1], keeping the
    # building's aspect ratio.
    if pos:
        xy = np.array(list(pos.values()))
        centre = (xy.max(axis=0) + xy.min(axis=0)) / 2
        span = max(float((xy.max(axis=0) - xy.min(axis=0)).max()), 1e-9)
        pos = {n: (v - centre) / span * 2.0 for n, v in pos.items()}
    return pos


def panel_layout(graph, *, seed: int = 7):
    """Positions for one panel, matching the published figure.

    * **Zone panels** (C1-C5) - zones on the building's own grid, so a 2x2
      building draws as a square with nothing crossing. The building node sits at
      the centre of the grid and the weather node below it, which keeps every
      membership edge inside its own quadrant.
    * **Component panels** (C8-C10) - concentric rings, innermost first: building
      and weather, then zones, then components, then apertures, so containment
      stays visible.
    * **The unified component graph** (C6/C7) - a spring layout, since all its
      nodes share one type.
    """
    import networkx as nx
    import numpy as np

    types = {str(data.get("node_type", "")) for _, data in graph.nodes(data=True)}

    if types & CONTAINED_TYPES and len(types) > 1:
        shells: dict[int, list] = {}
        for node, data in graph.nodes(data=True):
            shells.setdefault(NODE_SHELL_ORDER.get(str(data.get("node_type", "")), 2), []).append(node)
        return _concentric(graph, [shells[k] for k in sorted(shells)])

    zones = [n for n, d in graph.nodes(data=True) if str(d.get("node_type", "")) in ZONE_TYPES]
    if zones:
        pos = _zone_grid(graph, zones)
        others = [n for n in graph.nodes if n not in pos]
        buildings = [n for n in others if graph.nodes[n].get("node_type") == "building"]
        weather = [n for n in others if graph.nodes[n].get("node_type") == "weather_city"]
        # The building takes the centre of the grid; weather sits a step below it.
        # With zones on the corners, each membership edge runs into its own
        # quadrant and cannot cross an interzone edge on the grid's sides.
        # With weather present the two share the centre line, far enough apart
        # that the markers do not touch at the panel's short height.
        lift = 0.32 if weather else 0.0
        for node in buildings:
            pos[node] = np.array([0.0, lift])
        for i, node in enumerate(weather):
            pos[node] = np.array([0.0, -lift - 0.3 * i])
        for node in others:
            pos.setdefault(node, np.array([0.0, 0.0]))
        return pos

    return nx.spring_layout(graph, seed=seed, k=1.5 / max(graph.number_of_nodes(), 1) ** 0.4)


#: Legend order: what every panel shares, then the zone level, then the
#: component level. Matches the order a reader meets them, left to right.
LEGEND_ORDER = (
    "Building",
    "Weather",
    "Zone with component features",
    "Unified component",
    "Zone",
    "Component: Wall / Floor / Roof / Ceiling",
    "Aperture",
)


def _ordered_node_handles(handles):
    rank = {label: i for i, label in enumerate(LEGEND_ORDER)}
    return sorted(handles, key=lambda h: rank.get(h.get_label(), len(rank)))


def _row_major(handles, *, ncol: int):
    """Reorder so a legend reads across rows.

    Matplotlib fills legend columns top to bottom, which strands the last entry
    alone in the right-hand column. Interleaving by row makes the columns fill
    evenly and the entries read left to right.
    """
    nrow = -(-len(handles) // ncol)
    grid = [handles[r * ncol:(r + 1) * ncol] for r in range(nrow)]
    return [row[c] for c in range(ncol) for row in grid if c < len(row)]


def _panel_frame(fig, ax, *, radius: float = 0.018, header: float = 0.085) -> None:
    """The rounded outline the published panels sit in.

    Unfilled and behind everything: a figure-level patch is drawn after the axes
    when their z-orders tie, so a white fill here would cover the graph it is
    supposed to frame. ``header`` lifts the top edge over the title and the node
    and edge counts, which belong inside the panel.
    """
    from matplotlib.patches import FancyBboxPatch

    box = ax.get_position()
    pad = 0.008
    fig.patches.append(FancyBboxPatch(
        (box.x0 - pad, box.y0 - pad),
        box.width + 2 * pad, box.height + pad + header,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        transform=fig.transFigure, facecolor="none",
        edgecolor="#D9DADB", linewidth=1.1, zorder=-1, clip_on=False,
    ))


def plot_topologies(
    panels: list[TopologyPanel],
    *,
    ncols: int = 3,
    panel_size: tuple[float, float] = (5.6, 4.6),
    node_scale: float = 1.0,
    edge_scale: float = 1.0,
    layout_seed: int = 7,
    title: str | None = None,
):
    """Draw the panels in a grid, with a shared node and edge legend.

    ``node_scale`` and ``edge_scale`` multiply the built-in sizes, which are
    already set to the weight the paper figure uses; raise them for a poster,
    lower them for a dense configuration.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D

    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
    )
    axes = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    seen_node_types: list[str] = []
    seen_edge_roles: list[str] = []

    # More axes than panels when the grid does not divide evenly; the spares
    # are switched off below.
    for ax, panel in zip(axes, panels, strict=False):
        G = panel.graph
        n_nodes = panel.n_nodes
        pos = panel_layout(G, seed=layout_seed)

        node_size = _node_size_for(n_nodes, node_scale)
        width = _edge_width_for(n_nodes, edge_scale)

        by_role: dict[str, list] = {}
        for u, v, data in G.edges(data=True):
            role = edge_role(
                data.get("rel", "unified"),
                str(data.get("src_type", "")),
                str(data.get("tgt_type", "")),
            )
            by_role.setdefault(role, []).append((u, v))

        for role, edge_list in by_role.items():
            colour = EDGE_STYLES[role][0]
            nx.draw_networkx_edges(
                G, pos, ax=ax, edgelist=edge_list, edge_color=colour,
                width=width, arrows=True, arrowsize=8 if n_nodes <= 24 else 5,
                arrowstyle="-|>", connectionstyle="arc3,rad=0.09",
                node_size=node_size, alpha=0.9,
            )
            if role not in seen_edge_roles:
                seen_edge_roles.append(role)

        by_type: dict[str, list] = {}
        for node, data in G.nodes(data=True):
            by_type.setdefault(str(data.get("node_type", "node")), []).append(node)

        for ntype, node_list in by_type.items():
            nx.draw_networkx_nodes(
                G, pos, ax=ax, nodelist=node_list,
                node_color=NODE_COLOURS.get(ntype, "#999999"),
                edgecolors=NODE_RINGS.get(ntype, "#FFFFFF"),
                linewidths=2.0 if ntype in NODE_RINGS else 0.6,
                node_size=node_size,
            )
            if ntype not in seen_node_types:
                seen_node_types.append(ntype)

        ax.set_title(panel.title, fontsize=TITLE_FONT, fontweight="bold", pad=30)
        ax.text(0.5, 1.015, f"N={panel.n_nodes}, E={panel.n_edges}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=COUNT_FONT)
        # The aspect stays free, so the content fills the landscape panel: the
        # rings are ellipses and the zone grid is wide, as in the published panels.
        ax.margins(0.10)
        ax.set_axis_off()
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes[len(panels):]:
        ax.set_axis_off()

    drawn_axes = list(axes[:len(panels)])

    # A homogeneous zone graph and a heterogeneous one use different type names
    # for what the legend calls the same thing, so entries collapse by wording.
    node_handles, node_seen = [], set()
    for t in seen_node_types:
        label = NODE_LABELS.get(t, t)
        if label in node_seen:
            continue
        node_seen.add(label)
        node_handles.append(
            Line2D(
                [], [], marker="o", linestyle="none", markersize=LEGEND_MARKER,
                markerfacecolor=NODE_COLOURS.get(t, "#999999"),
                markeredgecolor=NODE_RINGS.get(t, "#FFFFFF"),
                markeredgewidth=3.0 if t in NODE_RINGS else 0.6,
                label=label,
            )
        )

    edge_handles = [
        Line2D([], [], color=EDGE_STYLES[r][0], linewidth=LEGEND_LINE, label=EDGE_STYLES[r][1])
        for r in seen_edge_roles
    ]

    # Two wide, short legends across the full width, nodes above edges, at the
    # size of the panel titles.
    node_handles = _row_major(_ordered_node_handles(node_handles), ncol=4)
    edge_handles = _row_major(edge_handles, ncol=3)
    common = dict(frameon=False, fontsize=LEGEND_FONT, handlelength=2.6,
                  handletextpad=0.6, columnspacing=2.2, borderaxespad=0.2)
    fig.legend(handles=node_handles, loc="lower center",
               ncol=min(4, len(node_handles)), bbox_to_anchor=(0.5, 0.075), **common)
    fig.legend(handles=edge_handles, loc="lower center",
               ncol=min(3, len(edge_handles)), bbox_to_anchor=(0.5, 0.0), **common)
    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold")

    fig.tight_layout(rect=(0, 0.21, 1, 0.96 if title else 0.99), w_pad=4.0, h_pad=5.0)

    # Frames last: they are figure-level patches, so they need the positions
    # tight_layout settled on.
    for ax in drawn_axes:
        _panel_frame(fig, ax)
    return fig
