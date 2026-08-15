"""The one figure: the network laid out by tier with SPOF nodes highlighted."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

from .model import SupplyNetwork  # noqa: E402
from .spof import rank_spofs  # noqa: E402

PART_COLOUR = "#4c6ef5"
SITE_COLOUR = "#adb5bd"
SUPPLIER_COLOUR = "#868e96"
SPOF_COLOUR = "#e03131"


def _layered_layout(net: SupplyNetwork, g: nx.DiGraph) -> dict:
    """Columns by role and tier: finished goods left, raw material suppliers right."""
    layers: dict[str, list[str]] = {}
    for node, data in g.nodes(data=True):
        if data["kind"] == "part":
            key = f"1part{data['tier']}"
        elif data["kind"] == "site":
            key = "2site"
        else:
            key = f"3supplier{data['tier']}"
        layers.setdefault(key, []).append(node)
    # Each layer is stretched over the same vertical range, otherwise the site column
    # (one node per plant) dominates the canvas and the BOM structure disappears.
    pos = {}
    for x, key in enumerate(sorted(layers)):
        nodes = sorted(layers[key])
        n = len(nodes)
        for i, node in enumerate(nodes):
            spread = 2.0 * min(1.0, n / 8.0)
            y = 0.0 if n == 1 else (i / (n - 1) - 0.5) * spread
            pos[node] = (float(x), y)
    return pos


def spof_figure(net: SupplyNetwork, path: str | Path, top_n: int = 8) -> Path:
    """Draw the network and mark the highest revenue-at-risk single points of failure."""
    g = net.to_graph()
    spofs = rank_spofs(net, top_n=top_n)
    flagged = {s.node_id for s in spofs}
    pos = _layered_layout(net, g)

    colours, sizes = [], []
    for node, data in g.nodes(data=True):
        if node in flagged:
            colours.append(SPOF_COLOUR)
            sizes.append(180)
        elif data["kind"] == "part":
            colours.append(PART_COLOUR)
            sizes.append(60)
        elif data["kind"] == "site":
            colours.append(SITE_COLOUR)
            sizes.append(35)
        else:
            colours.append(SUPPLIER_COLOUR)
            sizes.append(45)

    fig, ax = plt.subplots(figsize=(13, 7))
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#dee2e6", width=0.6, arrows=False)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colours, node_size=sizes, linewidths=0)
    labels = {f"part:{p.part_id}": p.part_id for p in net.finished_goods()}
    labels.update({s.node_id: s.label for s in spofs[:5]})
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=7.5)

    worst = spofs[0] if spofs else None
    subtitle = (
        f"top single point of failure {worst.label} ({worst.countries}): "
        f"{worst.revenue_share:.0%} of annual revenue at risk"
        if worst
        else "no single point of failure found"
    )
    ax.set_title(f"Multi-tier supply network, single points of failure highlighted\n{subtitle}", fontsize=11)
    ax.set_xlabel("finished good -> subassembly -> component -> material -> site -> supplier", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=PART_COLOUR, label="part"),
        plt.Line2D([], [], marker="o", linestyle="", color=SITE_COLOUR, label="site"),
        plt.Line2D([], [], marker="o", linestyle="", color=SUPPLIER_COLOUR, label="supplier"),
        plt.Line2D([], [], marker="o", linestyle="", color=SPOF_COLOUR, label=f"top {top_n} SPOF"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
