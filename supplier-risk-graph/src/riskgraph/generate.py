"""Synthetic multi-tier network generator.

The generating process mirrors what an OEM actually sees: a shallow, well-diversified
tier 1, thinning diversity as you go down, and a small number of deep nodes that many
independent-looking branches converge on. The planted diamond is explicit -- a shared
sub-tier material sole-sourced from one site -- because a detector nobody can point a
known-answer instance at is not a detector.
"""

from __future__ import annotations

import numpy as np

from .model import BomEdge, Part, Site, Supplier, SupplyEdge, SupplyNetwork

COUNTRIES = [
    ("DE", "EMEA", 0.030, 28.0),
    ("US", "AMER", 0.035, 24.0),
    ("MX", "AMER", 0.055, 34.0),
    ("CN", "APAC", 0.070, 45.0),
    ("TW", "APAC", 0.085, 62.0),
    ("VN", "APAC", 0.060, 40.0),
    ("JP", "APAC", 0.045, 33.0),
    ("IN", "APAC", 0.065, 38.0),
]


def _shares(rng: np.random.Generator, n: int) -> list[float]:
    if n == 1:
        return [1.0]
    raw = rng.dirichlet(np.full(n, 2.2))
    raw = np.maximum(raw, 0.05)
    raw = raw / raw.sum()
    return [round(float(x), 4) for x in raw]


def _normalise(shares: list[float]) -> list[float]:
    total = sum(shares)
    fixed = [round(s / total, 6) for s in shares]
    fixed[0] = round(fixed[0] + 1.0 - sum(fixed), 6)
    return fixed


def generate_network(
    seed: int = 7,
    n_products: int = 2,
    n_tier1: int = 8,
    n_tier2: int = 18,
    n_tier3: int = 12,
    diamond_branches: int = 4,
) -> SupplyNetwork:
    """Build a validated three-tier network containing one planted diamond dependency."""
    rng = np.random.default_rng(seed)
    parts: list[Part] = []
    bom: list[BomEdge] = []
    suppliers: list[Supplier] = []
    sites: list[Site] = []
    supply: list[SupplyEdge] = []

    t1 = [f"SA-{i:02d}" for i in range(1, n_tier1 + 1)]
    t2 = [f"CMP-{i:02d}" for i in range(1, n_tier2 + 1)]
    t3 = [f"MAT-{i:02d}" for i in range(1, n_tier3 + 1)]
    critical = "MAT-CRIT"

    for k in range(n_products):
        pid = f"FG-{k + 1}"
        units = float(rng.integers(40_000, 120_000))
        price = float(rng.integers(900, 2400))
        parts.append(
            Part(
                part_id=pid,
                name=f"Finished good {k + 1}",
                tier=0,
                unit_cost=price * 0.62,
                annual_units=units,
                annual_revenue=units * price,
                finished=True,
            )
        )
        for sa in rng.choice(t1, size=min(5, n_tier1), replace=False):
            bom.append(BomEdge(parent=pid, child=str(sa), qty_per=float(rng.integers(1, 3))))

    for pid in t1:
        parts.append(Part(part_id=pid, name=f"Subassembly {pid}", tier=1, unit_cost=float(rng.uniform(60, 260))))
    for pid in t2:
        parts.append(Part(part_id=pid, name=f"Component {pid}", tier=2, unit_cost=float(rng.uniform(8, 70))))
    for pid in t3:
        parts.append(Part(part_id=pid, name=f"Material {pid}", tier=3, unit_cost=float(rng.uniform(0.5, 9))))
    parts.append(Part(part_id=critical, name="Specialty substrate", tier=3, unit_cost=3.4))

    t2_parent: dict[str, str] = {}
    for pid in t1:
        for cmp_id in rng.choice(t2, size=3, replace=False):
            if not any(e.parent == pid and e.child == cmp_id for e in bom):
                bom.append(BomEdge(parent=pid, child=str(cmp_id), qty_per=float(rng.integers(1, 5))))
                t2_parent.setdefault(str(cmp_id), pid)
    for pid in t2:
        for mat in rng.choice(t3, size=2, replace=False):
            if not any(e.parent == pid and e.child == mat for e in bom):
                bom.append(BomEdge(parent=pid, child=str(mat), qty_per=float(rng.uniform(0.2, 4.0))))

    # -- plant the diamond: one deep material under several distinct tier-1 branches
    used_branches: set[str] = set()
    planted: list[str] = []
    for cmp_id in t2:
        branch = t2_parent.get(cmp_id)
        if branch is None or branch in used_branches:
            continue
        used_branches.add(branch)
        planted.append(cmp_id)
        bom.append(BomEdge(parent=cmp_id, child=critical, qty_per=float(rng.uniform(1.0, 3.0))))
        if len(planted) >= diamond_branches:
            break

    # -- suppliers and sites, one supplier tier per part tier
    tier_parts = {1: t1, 2: t2, 3: t3 + [critical]}
    counter = 0
    for tier, plist in tier_parts.items():
        n_sup = max(3, int(len(plist) * (0.9 if tier == 1 else 0.55)))
        pool: list[str] = []
        for _ in range(n_sup):
            counter += 1
            sup_id = f"SUP-{counter:03d}"
            suppliers.append(
                Supplier(
                    supplier_id=sup_id,
                    name=f"Supplier {counter:03d}",
                    tier=tier,
                    financial_health=float(np.clip(rng.normal(0.7, 0.14), 0.05, 0.99)),
                )
            )
            for j in range(int(rng.integers(1, 3))):
                cc, region, base_rate, base_rec = COUNTRIES[int(rng.integers(0, len(COUNTRIES)))]
                site_id = f"{sup_id}-S{j + 1}"
                sites.append(
                    Site(
                        site_id=site_id,
                        supplier_id=sup_id,
                        country=cc,
                        region=region,
                        weekly_capacity=float(rng.integers(4_000, 30_000)),
                        disruption_rate=float(np.clip(base_rate * rng.uniform(0.7, 1.4), 0.005, 0.5)),
                        mean_recovery_days=float(base_rec * (1.0 + 0.35 * (tier - 1)) * rng.uniform(0.8, 1.3)),
                    )
                )
                pool.append(site_id)

        # deeper tiers are thinner: fewer qualified sources per part
        p_multi = {1: 0.97, 2: 0.92, 3: 0.78}[tier]
        for pid in plist:
            if pid == critical:
                continue
            n_src = 1 + int(rng.random() < p_multi) + int(rng.random() < p_multi * 0.45)
            chosen = rng.choice(pool, size=min(n_src, len(pool)), replace=False)
            for site_id, share in zip(chosen, _normalise(_shares(rng, len(chosen)))):
                supply.append(
                    SupplyEdge(
                        part_id=pid,
                        site_id=str(site_id),
                        share=float(share),
                        lead_time_days=float(rng.integers(14, 120)),
                    )
                )
        if tier == 3:
            sole = pool[int(rng.integers(0, len(pool)))]
            supply.append(SupplyEdge(part_id=critical, site_id=sole, share=1.0, lead_time_days=98.0))

    return SupplyNetwork(parts=parts, suppliers=suppliers, sites=sites, bom=bom, supply=supply)
