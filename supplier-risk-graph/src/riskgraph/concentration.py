"""Concentration measures and hidden shared sub-tier supplier detection.

HHI is computed on spend shares, by tier, by supplier and by country. The headline
capability is `hidden_dependencies`: a supplier deep in the network that several
*different* tier-1 suppliers independently depend on. Tier-1 HHI reads healthy, the
sourcing review passes, and the whole product still stops when one plant floods.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from .flow import depths, expand, output_fraction, revenue_at_risk
from .model import SupplyNetwork


def hhi(shares: dict[str, float]) -> float:
    """Herfindahl-Hirschman index on the 0-1 scale (1.0 = single source)."""
    total = sum(shares.values())
    if total <= 0:
        return 0.0
    return sum((v / total) ** 2 for v in shares.values())


def effective_sources(shares: dict[str, float]) -> float:
    """1/HHI -- the number of equally sized suppliers the position behaves like."""
    h = hhi(shares)
    return 1.0 / h if h > 0 else 0.0


def _spend_by(net: SupplyNetwork, root: str, key: str) -> dict[int | str, dict[str, float]]:
    """Spend split by supplier, grouped by tier / country / 'all'."""
    out: dict[int | str, dict[str, float]] = {}
    for row in expand(net, root):
        for e in net.sources(row.part_id):
            site = net.site(e.site_id)
            sup = site.supplier_id
            value = row.annual_spend * e.share
            if key == "tier":
                group: int | str = net.supplier(sup).tier
                member = sup
            elif key == "country":
                group, member = "all", site.country
            else:
                group, member = "all", sup
            bucket = out.setdefault(group, {})
            bucket[member] = bucket.get(member, 0.0) + value
    return out


def hhi_by_tier(net: SupplyNetwork, root: str) -> dict[int, float]:
    return {t: hhi(v) for t, v in sorted(_spend_by(net, root, "tier").items())}


def geographic_hhi(net: SupplyNetwork, root: str) -> tuple[float, dict[str, float]]:
    spend = _spend_by(net, root, "country").get("all", {})
    total = sum(spend.values()) or 1.0
    return hhi(spend), {k: v / total for k, v in sorted(spend.items(), key=lambda x: -x[1])}


def supplier_hhi(net: SupplyNetwork, root: str) -> float:
    return hhi(_spend_by(net, root, "supplier").get("all", {}))


@dataclass
class HiddenDependency:
    supplier_id: str
    depth: int
    tier1_suppliers: int
    tier1_share_covered: float
    parts: int
    revenue_at_risk: float  # across every finished good the supplier feeds
    revenue_share: float  # share of THIS root's output lost if the supplier goes down
    countries: str

    def as_dict(self) -> dict:
        return asdict(self)


def _tier1_ancestors(net: SupplyNetwork, root: str) -> dict[str, set[str]]:
    """For each part, the set of depth-1 parts (tier-1 subassemblies) above it."""
    d = depths(net, root)
    order = sorted(d, key=lambda p: (d[p], p))
    anc: dict[str, set[str]] = {p: set() for p in order}
    for pid in order:
        for e in net.children(pid):
            inherited = {pid} if d[pid] == 1 else anc[pid]
            anc[e.child] |= inherited
    return anc


def hidden_dependencies(
    net: SupplyNetwork,
    root: str,
    min_depth: int = 2,
    min_branches: int = 2,
) -> list[HiddenDependency]:
    """Sub-tier suppliers that several independent tier-1 branches converge on.

    A supplier qualifies when it sits at or below `min_depth` and is reached through
    at least `min_branches` distinct tier-1 subassemblies. Ranked by the revenue that
    stops if it goes down, which is the number a sourcing director will act on.
    """
    anc = _tier1_ancestors(net, root)
    d = depths(net, root)
    total_t1 = sum(1 for p, dd in d.items() if dd == 1)

    reach: dict[str, set[str]] = {}
    parts_of: dict[str, set[str]] = {}
    min_seen: dict[str, int] = {}
    for row in expand(net, root):
        if row.depth < min_depth:
            continue
        for e in net.sources(row.part_id):
            sup = net.site_supplier(e.site_id)
            reach.setdefault(sup, set()).update(anc.get(row.part_id, set()))
            parts_of.setdefault(sup, set()).add(row.part_id)
            min_seen[sup] = min(min_seen.get(sup, 99), row.depth)

    out: list[HiddenDependency] = []
    for sup, branches in reach.items():
        if len(branches) < min_branches:
            continue
        site_ids = net.supplier_sites(sup)
        frac = output_fraction(net, root, site_ids)
        rar = revenue_at_risk(net, site_ids)
        out.append(
            HiddenDependency(
                supplier_id=sup,
                depth=min_seen[sup],
                tier1_suppliers=len(branches),
                tier1_share_covered=len(branches) / total_t1 if total_t1 else 0.0,
                parts=len(parts_of[sup]),
                revenue_at_risk=rar,
                revenue_share=1.0 - frac,
                countries=",".join(sorted({net.site(s).country for s in site_ids})),
            )
        )
    # Ties on revenue share are common once several nodes can each stop the line.
    # Break them by depth first: the deeper node is the one nobody is looking at.
    out.sort(key=lambda r: (-r.revenue_share, -r.depth, -r.tier1_suppliers, r.supplier_id))
    return out
