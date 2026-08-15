"""Mitigation scoring: dual sourcing, buffer stock, pre-qualification.

Every action is scored the same way -- rebuild the network (or the buffer plan) with
the action applied, re-run the simulation under common random numbers, and divide the
reduction in expected annual loss by the annualised cost of the action. Common random
numbers matter: without them the Monte Carlo noise is larger than the effect you are
trying to rank.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from .flow import expand
from .model import Site, Supplier, SupplyEdge, SupplyNetwork
from .simulate import SimResult, simulate

HOLDING_RATE = 0.22  # annual carrying cost as a fraction of inventory value
QUALIFICATION_COST = 250_000.0  # one-off, amortised over the assumed 3-year horizon
PREQUAL_COST = 90_000.0  # annual cost of keeping an alternate audited and tooled
DUAL_SOURCE_PREMIUM = 0.04  # unit-cost premium on volume moved to the second source
AMORTISATION_YEARS = 3.0


@dataclass
class Mitigation:
    action: str
    target: str
    annual_cost: float
    expected_loss_after: float
    p95_loss_after: float
    risk_reduction: float
    p95_reduction: float
    reduction_per_dollar: float

    def as_dict(self) -> dict:
        return asdict(self)


def annual_spend(net: SupplyNetwork) -> dict[str, float]:
    """Annual spend per part, summed across every finished good it feeds."""
    out: dict[str, float] = {}
    for fg in net.finished_goods():
        for row in expand(net, fg.part_id):
            out[row.part_id] = out.get(row.part_id, 0.0) + row.annual_spend
    return out


def with_dual_source(net: SupplyNetwork, part_id: str, second_share: float = 0.35) -> SupplyNetwork:
    """Qualify an independent second source for `part_id` in a different country."""
    data = net.model_dump()
    srcs = sorted(net.sources(part_id), key=lambda e: -e.share)
    if not srcs:
        raise ValueError(f"{part_id} has no sources to dual source")
    incumbent = net.site(srcs[0].site_id)
    alt_sup = f"ALT-{part_id}"
    alt_site = f"{alt_sup}-S1"
    country = "MX" if incumbent.country != "MX" else "PL"
    data["suppliers"].append(
        Supplier(
            supplier_id=alt_sup,
            name=f"Alternate source for {part_id}",
            tier=net.supplier(incumbent.supplier_id).tier,
            financial_health=0.7,
        ).model_dump()
    )
    data["sites"].append(
        Site(
            site_id=alt_site,
            supplier_id=alt_sup,
            country=country,
            region="AMER",
            weekly_capacity=incumbent.weekly_capacity,
            disruption_rate=incumbent.disruption_rate,
            mean_recovery_days=incumbent.mean_recovery_days,
        ).model_dump()
    )
    kept = [e for e in data["supply"] if e["part_id"] != part_id]
    scale = 1.0 - second_share
    rescaled = [
        {**e, "share": round(e["share"] * scale, 6)} for e in data["supply"] if e["part_id"] == part_id
    ]
    residual = 1.0 - sum(e["share"] for e in rescaled)
    rescaled.append(
        SupplyEdge(part_id=part_id, site_id=alt_site, share=round(residual, 6), lead_time_days=45.0).model_dump()
    )
    data["supply"] = kept + rescaled
    return SupplyNetwork.model_validate(data)


def with_prequalification(net: SupplyNetwork, part_id: str, recovery_factor: float = 0.55) -> SupplyNetwork:
    """Pre-qualified alternates do not remove the outage, they shorten it."""
    data = net.model_dump()
    affected = {e.site_id for e in net.sources(part_id)}
    for s in data["sites"]:
        if s["site_id"] in affected:
            s["mean_recovery_days"] = max(1.0, s["mean_recovery_days"] * recovery_factor)
    return SupplyNetwork.model_validate(data)


def score_actions(
    net: SupplyNetwork,
    targets: list[str],
    trials: int = 800,
    seed: int = 11,
    buffer_weeks: float = 4.0,
) -> tuple[SimResult, list[Mitigation]]:
    """Score dual sourcing, buffer stock and pre-qualification for each target part.

    The baseline is deliberately re-run here at the same trial count and seed as the
    action runs. Comparing an action against a baseline sampled with a different budget
    mixes Monte Carlo error into the ranking and can flip the sign of a small effect.
    """
    base = simulate(net, trials=trials, seed=seed)
    spend = annual_spend(net)
    rows: list[Mitigation] = []

    def add(action: str, target: str, cost: float, res: SimResult) -> None:
        reduction = base.expected_loss - res.expected_loss
        rows.append(
            Mitigation(
                action=action,
                target=target,
                annual_cost=cost,
                expected_loss_after=res.expected_loss,
                p95_loss_after=res.p95_loss,
                risk_reduction=reduction,
                p95_reduction=base.p95_loss - res.p95_loss,
                reduction_per_dollar=reduction / cost if cost > 0 else 0.0,
            )
        )

    for part_id in targets:
        part_spend = spend.get(part_id, 0.0)

        res = simulate(with_dual_source(net, part_id), trials=trials, seed=seed)
        cost = QUALIFICATION_COST / AMORTISATION_YEARS + DUAL_SOURCE_PREMIUM * 0.35 * part_spend
        add("dual_source", part_id, cost, res)

        days = buffer_weeks * 7.0
        res = simulate(net, trials=trials, seed=seed, buffers={part_id: days})
        add("buffer_stock", part_id, HOLDING_RATE * part_spend * buffer_weeks / 52.0, res)

        res = simulate(with_prequalification(net, part_id), trials=trials, seed=seed)
        add("prequalify", part_id, PREQUAL_COST, res)

    rows.sort(key=lambda r: -r.reduction_per_dollar)
    return base, rows
