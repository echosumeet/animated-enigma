"""Cost-versus-service efficient frontier.

A service target is a business decision, not an engineering one, and the only
useful way to have that conversation is with a curve rather than a number.  The
frontier here sweeps a service target, sizes the buffer for it, and reports the
inventory investment required - so the question stops being "should we be at 95
or 98?" and becomes "the next two points of fill rate cost this much working
capital; is that worth it?"

Two things this makes visible that a single target never does:

* The curve is convex and steepens hard above ~98%.  On the example item in
  ``benchmarks/`` the last two points of fill rate cost more than the first
  fifteen.  That is the shape of the argument for differentiated service by
  segment rather than one corporate number.
* The fill-rate frontier sits materially below the CSL frontier at the same
  numeric target.  Same items, same variability, same physical service - the gap
  is purely which definition the planning parameter is read against.

Reference: Silver, Pyke & Thomas (2016), Ch. 11 (exchange curves).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .distributions import LeadTimeDemand
from .safety_stock import (
    fill_rate_of_sQ,
    ss_from_cycle_service_level,
    ss_from_fill_rate,
)

__all__ = ["FrontierPoint", "service_frontier", "exchange_curve", "marginal_cost_of_service"]


@dataclass(frozen=True)
class FrontierPoint:
    target: float
    basis: str
    safety_stock: float
    reorder_point: float
    cycle_stock: float
    achieved_csl: float
    achieved_fill: float
    holding_cost: float
    expected_backorder_cost: float

    @property
    def total_cost(self) -> float:
        return float(self.holding_cost + self.expected_backorder_cost)


def service_frontier(
    ltd: LeadTimeDemand,
    Q: float,
    targets: Sequence[float],
    unit_cost: float,
    holding_rate: float,
    basis: str = "fill",
    shortage_cost_per_unit: float = 0.0,
    demand_per_period: float = 1.0,
) -> list[FrontierPoint]:
    """Sweep a service target and return the inventory investment for each point.

    ``basis`` is ``"fill"`` or ``"csl"``.  Holding cost is charged on cycle stock
    plus safety stock at ``unit_cost * holding_rate`` per period.  If a shortage
    cost is supplied, the expected backorder cost per period is added, which turns
    the frontier into a genuine total-cost curve with an interior minimum.
    """
    if basis not in {"fill", "csl"}:
        raise ValueError("basis must be 'fill' or 'csl'")
    h = unit_cost * holding_rate
    cycles_per_period = demand_per_period / Q
    points: list[FrontierPoint] = []
    for target in targets:
        if basis == "fill":
            res = ss_from_fill_rate(ltd, target, Q)
        else:
            res = ss_from_cycle_service_level(ltd, target, Q)
        shortage_per_cycle = ltd.loss(res.reorder_point) - ltd.loss(res.reorder_point + Q)
        points.append(
            FrontierPoint(
                target=float(target),
                basis=basis,
                safety_stock=res.safety_stock,
                reorder_point=res.reorder_point,
                cycle_stock=float(Q / 2.0),
                achieved_csl=res.cycle_service_level,
                achieved_fill=fill_rate_of_sQ(ltd, res.reorder_point, Q),
                holding_cost=float(h * (Q / 2.0 + max(res.safety_stock, -Q / 2.0))),
                expected_backorder_cost=float(
                    shortage_cost_per_unit * shortage_per_cycle * cycles_per_period
                ),
            )
        )
    return points


def marginal_cost_of_service(points: Sequence[FrontierPoint]) -> list[dict[str, float]]:
    """Incremental holding cost per additional point of achieved fill rate.

    The number that ends the "why can't we just be at 99.5?" conversation.
    """
    rows: list[dict[str, float]] = []
    ordered = sorted(points, key=lambda p: p.achieved_fill)
    for a, b in zip(ordered[:-1], ordered[1:]):
        d_service = b.achieved_fill - a.achieved_fill
        if d_service <= 1e-12:
            continue
        rows.append(
            {
                "from_fill": a.achieved_fill,
                "to_fill": b.achieved_fill,
                "delta_holding_cost": float(b.holding_cost - a.holding_cost),
                "cost_per_service_point": float((b.holding_cost - a.holding_cost) / (100.0 * d_service)),
            }
        )
    return rows


def exchange_curve(
    ltd: LeadTimeDemand,
    Q: float,
    unit_cost: float,
    holding_rate: float,
    n_points: int = 30,
    lo: float = 0.80,
    hi: float = 0.995,
) -> dict[str, list[FrontierPoint]]:
    """Both frontiers on a common target grid, ready to plot."""
    targets = list(np.linspace(lo, hi, n_points))
    return {
        "fill": service_frontier(ltd, Q, targets, unit_cost, holding_rate, basis="fill"),
        "csl": service_frontier(ltd, Q, targets, unit_cost, holding_rate, basis="csl"),
    }
