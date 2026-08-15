"""Uncapacitated and capacitated facility location.

Kept as a separate, deliberately small model because it is the right tool for
two jobs: sanity-checking the big network model against a formulation whose
optimum can be verified by enumeration, and demonstrating - with measured
numbers rather than assertion - why the *disaggregated* linking constraint is
worth its extra rows.

The classical result (Balinski 1965; Krarup & Pruzan 1983; see also Cornuejols,
Nemhauser & Wolsey 1990) is that the strong formulation

    x_ij <= y_i    for every facility i and customer j

has an LP relaxation that is frequently integral, while the aggregated form

    sum_j x_ij <= |J| * y_i

has an LP bound that collapses towards "open a fraction of every facility".
Same feasible integer set, same optimum, wildly different search tree. The
benchmark in this repository measures the gap on a generated instance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .instances import Instance
from .modeling import ANY, Model, Solution, quicksum

__all__ = [
    "FacilityLocationResult",
    "uncapacitated_facility_location",
    "capacitated_facility_location",
    "dc_location_subproblem",
    "brute_force_uflp",
]


@dataclass
class FacilityLocationResult:
    status: str
    objective: float | None
    open_facilities: list[str] = field(default_factory=list)
    assignment: dict[str, dict[str, float]] = field(default_factory=dict)
    runtime: float = 0.0
    model_stats: dict[str, int] = field(default_factory=dict)
    lp_bound: float | None = None
    formulation: str = "strong"

    @property
    def integrality_gap(self) -> float | None:
        """``(MILP - LP) / MILP`` for the formulation that produced it."""
        if self.lp_bound is None or not self.objective:
            return None
        return (self.objective - self.lp_bound) / self.objective


def _build(
    fixed_costs: Mapping[str, float],
    unit_cost: Mapping[tuple[str, str], float],
    demand: Mapping[str, float],
    *,
    capacity: Mapping[str, float] | None,
    single_source: bool,
    formulation: str,
    p_facilities: int | None,
    min_volume: Mapping[str, float] | None,
) -> tuple[Model, Any, Any]:
    if formulation not in ("strong", "aggregated"):
        raise ValueError("formulation must be 'strong' or 'aggregated'")
    facilities = list(fixed_costs)
    customers = list(demand)
    pairs = [(f, c) for (f, c) in unit_cost if f in fixed_costs and c in demand]
    if not pairs:
        raise ValueError("unit_cost contains no (facility, customer) pair present in the data")

    m = Model("facility-location", sense="min")
    y = m.add_vars(facilities, name="open", vtype="binary", obj=dict(fixed_costs))
    x = m.add_vars(
        pairs,
        name="serve",
        lb=0.0,
        ub=1.0,
        vtype="binary" if single_source else "continuous",
        obj={(f, c): unit_cost[(f, c)] * demand[c] for (f, c) in pairs},
    )

    for c in customers:
        keys = x.select(ANY, c)
        if not keys:
            raise ValueError(f"customer {c!r} has no feasible facility")
        m.add(x.sum_over(keys) == 1.0, name=f"cover[{c}]", tag="cover")

    if formulation == "strong":
        for f, c in pairs:
            m.add(x[(f, c)] <= y[f], name=f"link[{f},{c}]", tag="link")
    else:
        for f in facilities:
            keys = x.select(f, ANY)
            if keys:
                m.add(x.sum_over(keys) <= len(keys) * y[f], name=f"link_agg[{f}]", tag="link")

    if capacity is not None:
        for f in facilities:
            keys = x.select(f, ANY)
            if not keys:
                continue
            load = quicksum(demand[c] * x[(f, c)] for (_, c) in keys)
            m.add(load - capacity[f] * y[f] <= 0.0, name=f"capacity[{f}]", tag="capacity")
            if min_volume and min_volume.get(f, 0.0) > 0:
                m.add(
                    load - min_volume[f] * y[f] >= 0.0,
                    name=f"min_volume[{f}]",
                    tag="min_volume",
                )

    if p_facilities is not None:
        m.add(y.sum() == float(p_facilities), name="p_facilities", tag="count")

    return m, y, x


def _extract(m: Model, y: Any, x: Any, raw: Solution, formulation: str) -> FacilityLocationResult:
    res = FacilityLocationResult(
        status=raw.status,
        objective=raw.objective,
        runtime=raw.runtime,
        model_stats=m.stats(),
        formulation=formulation,
    )
    if raw.x is None:
        return res
    res.open_facilities = sorted(k for k, v in raw.values(y, integral=True).items() if v > 0.5)
    assign: dict[str, dict[str, float]] = {}
    for (f, c), v in raw.values(x, nonzero=True, tol=1e-7).items():
        assign.setdefault(c, {})[f] = v
    res.assignment = assign
    return res


def uncapacitated_facility_location(
    fixed_costs: Mapping[str, float],
    unit_cost: Mapping[tuple[str, str], float],
    demand: Mapping[str, float] | None = None,
    *,
    formulation: str = "strong",
    p_facilities: int | None = None,
    with_lp_bound: bool = False,
    **solve_kwargs: Any,
) -> FacilityLocationResult:
    """Solve the UFLP. ``demand`` defaults to one unit per customer."""
    customers = sorted({c for (_, c) in unit_cost})
    dem = {c: 1.0 for c in customers} if demand is None else dict(demand)
    m, y, x = _build(
        fixed_costs,
        unit_cost,
        dem,
        capacity=None,
        single_source=False,
        formulation=formulation,
        p_facilities=p_facilities,
        min_volume=None,
    )
    raw = m.solve(**solve_kwargs)
    res = _extract(m, y, x, raw, formulation)
    if with_lp_bound:
        lp = m.solve(relax=True, **solve_kwargs)
        res.lp_bound = lp.objective
    return res


def capacitated_facility_location(
    fixed_costs: Mapping[str, float],
    unit_cost: Mapping[tuple[str, str], float],
    demand: Mapping[str, float],
    capacity: Mapping[str, float],
    *,
    single_source: bool = False,
    min_volume: Mapping[str, float] | None = None,
    formulation: str = "strong",
    p_facilities: int | None = None,
    with_lp_bound: bool = False,
    **solve_kwargs: Any,
) -> FacilityLocationResult:
    """Solve the CFLP, optionally with single sourcing and minimum volumes.

    Single sourcing makes ``x`` binary and turns a problem whose LP relaxation
    is usually near-integral into a genuinely hard one - it is a *service*
    constraint (one bill of lading, one carrier relationship, one point of
    contact per customer) that the cost model does not reward, so it always
    costs money. The point of having it as a switch is to price it.
    """
    m, y, x = _build(
        fixed_costs,
        unit_cost,
        demand,
        capacity=capacity,
        single_source=single_source,
        formulation=formulation,
        p_facilities=p_facilities,
        min_volume=min_volume,
    )
    raw = m.solve(**solve_kwargs)
    res = _extract(m, y, x, raw, formulation)
    if with_lp_bound:
        lp = m.solve(relax=True, **solve_kwargs)
        res.lp_bound = lp.objective
    return res


def dc_location_subproblem(
    instance: Instance, *, include_handling: bool = True
) -> tuple[dict[str, float], dict[tuple[str, str], float], dict[str, float], dict[str, float]]:
    """Collapse the instance to a DC-to-zone facility location problem.

    Upstream echelons are dropped and the outbound lane cost (plus handling) is
    used as the per-unit assignment cost. This is the classic "which DCs do we
    keep" question in the form most planning teams actually pose it, and it is
    the sub-model used for the formulation benchmark.
    """
    fixed = {d.id: d.fixed_cost for d in instance.dcs}
    capacity = {d.id: d.capacity for d in instance.dcs}
    handling = {d.id: (d.handling_cost if include_handling else 0.0) for d in instance.dcs}
    demand = {z.id: instance.zone_demand(z.id) for z in instance.zones}
    weight_share = {
        z.id: (
            sum(
                instance.demand.get((z.id, c.id), 0.0) * c.kg_per_unit
                for c in instance.commodities
            )
            / max(demand[z.id], 1e-9)
        )
        for z in instance.zones
    }
    unit_cost: dict[tuple[str, str], float] = {}
    for ln in instance.lanes_by_echelon("dc", "zone"):
        cost = ln.cost_per_kg * weight_share[ln.dest] + handling[ln.origin]
        key = (ln.origin, ln.dest)
        unit_cost[key] = min(unit_cost.get(key, float("inf")), cost)
    return fixed, unit_cost, demand, capacity


def brute_force_uflp(
    fixed_costs: Mapping[str, float],
    unit_cost: Mapping[tuple[str, str], float],
    demand: Mapping[str, float] | None = None,
) -> tuple[float, list[str]]:
    """Exhaustive search over open sets. Only for small instances, in tests.

    A MILP that agrees with brute force on every subset of an 8-facility
    instance is a MILP whose formulation is right; asserting that it returns
    yesterday's number is not.
    """
    from itertools import combinations

    facilities = sorted(fixed_costs)
    customers = sorted({c for (_, c) in unit_cost})
    dem = {c: 1.0 for c in customers} if demand is None else dict(demand)
    best = float("inf")
    best_set: list[str] = []
    for r in range(1, len(facilities) + 1):
        for combo in combinations(facilities, r):
            total = sum(fixed_costs[f] for f in combo)
            feasible = True
            for c in customers:
                costs = [unit_cost[(f, c)] for f in combo if (f, c) in unit_cost]
                if not costs:
                    feasible = False
                    break
                total += min(costs) * dem[c]
            if feasible and total < best:
                best = total
                best_set = list(combo)
    return best, best_set
