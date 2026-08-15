"""N-tier expansion, value flow, and the availability kernel.

The kernel is deliberately one function. SPOF ranking, Monte Carlo simulation and
mitigation scoring all answer the same question -- "what fraction of the finished good
can still be built with this set of sites down" -- so they all call `output_fraction`.
Keeping one implementation is what stops the three views from disagreeing in a review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .model import SupplyNetwork


@dataclass(frozen=True)
class ExpandedPart:
    part_id: str
    depth: int
    qty_per_fg: float
    annual_units: float
    annual_spend: float
    sole_source: bool


def topo_order(net: SupplyNetwork, root: str) -> list[str]:
    """Parts reachable from `root`, parents before children (BOM is a DAG)."""
    seen: dict[str, int] = {}

    def visit(pid: str, depth: int) -> None:
        if seen.get(pid, -1) >= depth:
            return
        seen[pid] = depth
        for e in net.children(pid):
            visit(e.child, depth + 1)

    visit(root, 0)
    return sorted(seen, key=lambda p: (seen[p], p))


def depths(net: SupplyNetwork, root: str) -> dict[str, int]:
    """Longest-path depth of each reachable part below the finished good."""
    out: dict[str, int] = {}

    def visit(pid: str, depth: int) -> None:
        if out.get(pid, -1) >= depth:
            return
        out[pid] = depth
        for e in net.children(pid):
            visit(e.child, depth + 1)

    visit(root, 0)
    return out


def expand(net: SupplyNetwork, root: str) -> list[ExpandedPart]:
    """Explode a finished good down to raw materials, accumulating quantity per unit.

    A part reached by several paths gets the summed quantity-per, which is the point:
    multi-path parts are where tier-1 diversity quietly evaporates.
    """
    d = depths(net, root)
    order = sorted(d, key=lambda p: (d[p], p))
    qty = {pid: 0.0 for pid in order}
    qty[root] = 1.0
    for pid in order:
        for e in net.children(pid):
            qty[e.child] += qty[pid] * e.qty_per
    fg_units = net.part(root).annual_units
    rows = []
    for pid in order:
        srcs = net.sources(pid)
        units = qty[pid] * fg_units
        rows.append(
            ExpandedPart(
                part_id=pid,
                depth=d[pid],
                qty_per_fg=qty[pid],
                annual_units=units,
                annual_spend=units * net.part(pid).unit_cost,
                sole_source=len(srcs) == 1 or (bool(srcs) and max(s.share for s in srcs) >= 0.99),
            )
        )
    return rows


def site_flow(net: SupplyNetwork, root: str) -> dict[str, float]:
    """Annual spend flowing through each site for one finished good."""
    flow: dict[str, float] = {}
    for row in expand(net, root):
        for e in net.sources(row.part_id):
            flow[e.site_id] = flow.get(e.site_id, 0.0) + row.annual_spend * e.share
    return flow


def supplier_flow(net: SupplyNetwork, root: str) -> dict[str, float]:
    """Annual spend flowing through each supplier, summed over its sites."""
    out: dict[str, float] = {}
    for site_id, value in site_flow(net, root).items():
        sup = net.site_supplier(site_id)
        out[sup] = out.get(sup, 0.0) + value
    return out


def output_fraction(
    net: SupplyNetwork,
    root: str,
    down_sites: Iterable[str] = (),
    flex: float = 1.0,
    exempt_parts: Iterable[str] = (),
    order: list[str] | None = None,
) -> float:
    """Buildable fraction of `root` with `down_sites` unavailable.

    Part availability is the surviving allocation share, scaled by `flex` (the amount
    surviving sites can over-produce) and capped at 1. Availability propagates up the
    BOM as a minimum, not a product: you cannot substitute a surplus of one component
    for a shortage of another. `exempt_parts` are covered by buffer stock this period.
    """
    down = set(down_sites)
    exempt = set(exempt_parts)
    order = topo_order(net, root) if order is None else order
    out: dict[str, float] = {}
    for pid in reversed(order):
        srcs = net.sources(pid)
        if not srcs or pid in exempt:
            avail = 1.0
        else:
            surviving = sum(e.share for e in srcs if e.site_id not in down)
            avail = min(1.0, surviving * flex)
        worst = min((out[e.child] for e in net.children(pid)), default=1.0)
        out[pid] = min(avail, worst)
    return out[root]


def revenue_at_risk(
    net: SupplyNetwork,
    down_sites: Iterable[str],
    flex: float = 1.0,
    buffers: Mapping[str, float] | None = None,
) -> float:
    """Annual revenue that cannot be produced with `down_sites` unavailable."""
    covered = [p for p, days in (buffers or {}).items() if days > 0]
    total = 0.0
    for fg in net.finished_goods():
        frac = output_fraction(net, fg.part_id, down_sites, flex, covered)
        total += fg.annual_revenue * (1.0 - frac)
    return total
