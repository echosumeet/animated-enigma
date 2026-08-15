"""Single-point-of-failure detection, ranked by revenue at risk.

Two structural signals are combined: articulation points of the undirected projection
(nodes whose removal disconnects the network) and sole-sourced parts. Neither is a
ranking on its own -- degree and betweenness rank the busiest node, not the one that
stops the line. Every candidate is scored by re-running the availability kernel with
that node removed, so the ordering is in dollars.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import networkx as nx

from .flow import revenue_at_risk, site_flow
from .model import SupplyNetwork


@dataclass
class Spof:
    node_id: str
    kind: str  # "site" | "supplier"
    label: str
    revenue_at_risk: float
    revenue_share: float
    articulation: bool
    sole_source_parts: int
    mean_recovery_days: float
    countries: str

    def as_dict(self) -> dict:
        return asdict(self)


def articulation_points(net: SupplyNetwork) -> set[str]:
    """Cut vertices of the undirected projection of the typed graph."""
    g = net.to_graph().to_undirected()
    return set(nx.articulation_points(g))


def sole_source_parts(net: SupplyNetwork) -> dict[str, str]:
    """part_id -> site_id for every part with a single effective source."""
    out = {}
    for p in net.parts:
        srcs = net.sources(p.part_id)
        if not srcs:
            continue
        top = max(srcs, key=lambda e: e.share)
        if len(srcs) == 1 or top.share >= 0.99:
            out[p.part_id] = top.site_id
    return out


def rank_spofs(net: SupplyNetwork, flex: float = 1.0, top_n: int = 15) -> list[Spof]:
    """Rank sites and suppliers by the revenue that stops if they go down."""
    total_revenue = sum(p.annual_revenue for p in net.finished_goods())
    arts = articulation_points(net)
    sole = sole_source_parts(net)
    sole_by_site: dict[str, int] = {}
    for site_id in sole.values():
        sole_by_site[site_id] = sole_by_site.get(site_id, 0) + 1

    flows: dict[str, float] = {}
    for fg in net.finished_goods():
        for site_id, v in site_flow(net, fg.part_id).items():
            flows[site_id] = flows.get(site_id, 0.0) + v

    rows: list[Spof] = []
    for s in net.sites:
        rar = revenue_at_risk(net, [s.site_id], flex=flex)
        if rar <= 0 and flows.get(s.site_id, 0.0) <= 0:
            continue
        rows.append(
            Spof(
                node_id=f"site:{s.site_id}",
                kind="site",
                label=s.site_id,
                revenue_at_risk=rar,
                revenue_share=rar / total_revenue if total_revenue else 0.0,
                articulation=f"site:{s.site_id}" in arts,
                sole_source_parts=sole_by_site.get(s.site_id, 0),
                mean_recovery_days=s.mean_recovery_days,
                countries=s.country,
            )
        )
    for sup in net.suppliers:
        site_ids = net.supplier_sites(sup.supplier_id)
        if len(site_ids) < 2:
            continue  # identical to its single site; do not double count
        rar = revenue_at_risk(net, site_ids, flex=flex)
        if rar <= 0:
            continue
        rows.append(
            Spof(
                node_id=f"supplier:{sup.supplier_id}",
                kind="supplier",
                label=sup.supplier_id,
                revenue_at_risk=rar,
                revenue_share=rar / total_revenue if total_revenue else 0.0,
                articulation=f"supplier:{sup.supplier_id}" in arts,
                sole_source_parts=sum(sole_by_site.get(sid, 0) for sid in site_ids),
                mean_recovery_days=max(net.site(sid).mean_recovery_days for sid in site_ids),
                countries=",".join(sorted({net.site(sid).country for sid in site_ids})),
            )
        )
    rows.sort(key=lambda r: (-r.revenue_at_risk, r.label))
    return rows[:top_n]


def degree_ranking(net: SupplyNetwork, top_n: int = 15) -> list[str]:
    """Baseline ranking by degree centrality, for comparison against `rank_spofs`."""
    g = net.to_graph().to_undirected()
    nodes = [n for n in g.nodes if n.startswith(("site:", "supplier:"))]
    deg = sorted(nodes, key=lambda n: (-g.degree(n), n))
    return deg[:top_n]
