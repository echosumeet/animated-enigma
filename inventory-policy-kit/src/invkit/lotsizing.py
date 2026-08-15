"""Lot sizing: EOQ, all-units quantity discounts, Wagner-Whitin DP, Silver-Meal.

EOQ is the most-taught and least-trusted formula in the field, mostly because the
ordering cost ``K`` is unknowable to two significant figures.  The saving grace is
that the EOQ cost curve is extremely flat: being 40% off the optimal quantity
costs about 6% of the relevant cost.  :func:`eoq_cost_sensitivity` quantifies
that, because it is the argument that lets you stop debating ``K`` and start
shipping.

Wagner-Whitin is the exact dynamic program for deterministic time-varying demand
and is genuinely cheap - ``O(T^2)`` and often ``O(T)`` with the planning-horizon
theorem.  Silver-Meal is the heuristic MRP systems actually run.  Both are here
so the optimality gap can be measured rather than assumed; on the instances in
``benchmarks/`` Silver-Meal is usually within a couple of percent, and knowing
that is worth more than another argument about which one to configure.

References
----------
Harris, F.W. (1913) 'How many parts to make at once', *Factory* 10(2).
Wagner, H.M. and Whitin, T.M. (1958) 'Dynamic version of the economic lot size
model', *Management Science* 5(1), 89-96.
Silver, E.A. and Meal, H.C. (1973) 'A heuristic for selecting lot size quantities',
*Production and Inventory Management* 14(2), 64-74.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "eoq",
    "eoq_total_cost",
    "eoq_cost_sensitivity",
    "PriceBreak",
    "DiscountResult",
    "eoq_all_units_discount",
    "LotSizingPlan",
    "wagner_whitin",
    "silver_meal",
    "least_unit_cost",
    "lot_for_lot",
    "compare_lot_sizing",
    "seasonal_demand_series",
]


def eoq(demand_rate: float, order_cost: float, holding_cost: float) -> float:
    """Classical EOQ ``sqrt(2 K D / h)``, in the same time unit as ``demand_rate``."""
    if demand_rate <= 0 or order_cost <= 0 or holding_cost <= 0:
        raise ValueError("demand_rate, order_cost and holding_cost must be positive")
    return math.sqrt(2.0 * order_cost * demand_rate / holding_cost)


def eoq_total_cost(
    Q: float, demand_rate: float, order_cost: float, holding_cost: float, unit_cost: float = 0.0
) -> float:
    """Relevant cost per time unit: ordering + holding (+ purchase if priced)."""
    if Q <= 0:
        raise ValueError("Q must be positive")
    return float(
        order_cost * demand_rate / Q + holding_cost * Q / 2.0 + unit_cost * demand_rate
    )


def eoq_cost_sensitivity(ratio: float) -> float:
    """Relative cost penalty of ordering ``ratio * Q*`` instead of ``Q*``.

    ``C(rQ*)/C(Q*) = 0.5 * (r + 1/r)``.  At ``r = 1.5`` the penalty is 8.3%; at
    ``r = 2`` it is 25%.  The flatness is why rounding EOQ to a pallet quantity is
    nearly free and why arguing about the third digit of the ordering cost is not
    a good use of anyone's week.
    """
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    return float(0.5 * (ratio + 1.0 / ratio) - 1.0)


@dataclass(frozen=True)
class PriceBreak:
    """All-units price break: unit price ``price`` applies from ``min_qty`` up."""

    min_qty: float
    price: float


@dataclass(frozen=True)
class DiscountResult:
    quantity: float
    unit_price: float
    total_cost: float
    breaks_evaluated: int


def eoq_all_units_discount(
    demand_rate: float,
    order_cost: float,
    holding_rate: float,
    breaks: Sequence[PriceBreak],
) -> DiscountResult:
    """All-units quantity discount EOQ.

    ``holding_rate`` is a *rate* (fraction of unit value per period), so the
    holding cost per unit is ``holding_rate * price`` and therefore depends on the
    price tier - which is what makes the problem non-convex and requires
    evaluating every tier rather than differentiating once.

    Algorithm: for each tier, compute the EOQ at that tier's holding cost.  If it
    falls inside the tier's quantity range it is a candidate; if it falls below
    the tier minimum, the tier minimum is the candidate (cost is increasing to the
    right of EOQ, so the cheapest feasible point is the left edge); if it falls
    above the tier's upper bound the tier is dominated.  Take the cheapest
    candidate on *total* cost including purchase.
    """
    if not breaks:
        raise ValueError("at least one price break is required")
    if holding_rate <= 0:
        raise ValueError("holding_rate must be positive")
    tiers = sorted(breaks, key=lambda b: b.min_qty)
    uppers = [tiers[i + 1].min_qty for i in range(len(tiers) - 1)] + [float("inf")]

    best: tuple[float, float, float] | None = None
    evaluated = 0
    for tier, upper in zip(tiers, uppers):
        h = holding_rate * tier.price
        q_star = eoq(demand_rate, order_cost, h)
        if q_star < tier.min_qty:
            q = tier.min_qty
        elif q_star >= upper:
            continue  # EOQ lies in a cheaper tier's range; that tier handles it
        else:
            q = q_star
        if q <= 0:
            continue
        evaluated += 1
        cost = eoq_total_cost(q, demand_rate, order_cost, h, tier.price)
        if best is None or cost < best[2]:
            best = (q, tier.price, cost)
    if best is None:  # pragma: no cover - unreachable for well-formed break tables
        raise ValueError("no feasible quantity found")
    return DiscountResult(best[0], best[1], best[2], evaluated)


@dataclass
class LotSizingPlan:
    """A production/purchase plan over a finite horizon."""

    orders: list[float]
    setup_cost: float
    holding_cost: float
    method: str
    ending_inventory: list[float] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return float(self.setup_cost + self.holding_cost)

    @property
    def n_setups(self) -> int:
        return int(sum(1 for q in self.orders if q > 1e-9))

    def gap_vs(self, optimal: "LotSizingPlan") -> float:
        """Relative cost gap against an optimal plan (Wagner-Whitin)."""
        if optimal.total_cost <= 0:
            return 0.0
        return float(self.total_cost / optimal.total_cost - 1.0)


def _evaluate_plan(
    demand: Sequence[float], orders: Sequence[float], setup: float, holding: float, method: str
) -> LotSizingPlan:
    """Cost a plan by simulating the inventory balance, end-of-period holding."""
    inv = 0.0
    hold_cost = 0.0
    setups = 0
    ending: list[float] = []
    for d, q in zip(demand, orders):
        if q > 1e-9:
            setups += 1
        inv += q - d
        if inv < -1e-6:
            raise ValueError("plan does not cover demand")
        hold_cost += holding * max(inv, 0.0)
        ending.append(inv)
    return LotSizingPlan(
        orders=list(orders),
        setup_cost=setups * setup,
        holding_cost=hold_cost,
        method=method,
        ending_inventory=ending,
    )


def wagner_whitin(
    demand: Sequence[float], setup_cost: float, holding_cost: float
) -> LotSizingPlan:
    """Exact dynamic program for the deterministic single-item lot-sizing problem.

    Uses the zero-inventory-ordering property: an optimal plan only orders when
    inventory hits zero, so any order covers an integral block of consecutive
    periods.  ``f(t)`` = min cost to cover periods ``t..T``; the recursion is over
    the block endpoint, giving ``O(T^2)``.

    Holding is charged on end-of-period inventory, so an order that arrives and is
    consumed in the same period incurs no holding cost.
    """
    d = [float(x) for x in demand]
    T = len(d)
    if T == 0:
        return LotSizingPlan([], 0.0, 0.0, "wagner-whitin")
    if any(x < 0 for x in d):
        raise ValueError("demand must be non-negative")
    if setup_cost < 0 or holding_cost < 0:
        raise ValueError("costs must be non-negative")

    INF = float("inf")
    f = [INF] * (T + 1)
    nxt = [T] * (T + 1)
    f[T] = 0.0
    for t in range(T - 1, -1, -1):
        # order in period t covering periods t .. j-1
        carry = 0.0
        for j in range(t + 1, T + 1):
            # holding for the block: units for period k are held (k - t) periods
            cost = setup_cost + carry + f[j]
            if cost < f[t] - 1e-12:
                f[t] = cost
                nxt[t] = j
            if j < T:
                # extend the block to cover period j: those units wait (j - t) periods
                carry += holding_cost * (j - t) * d[j]
    orders = [0.0] * T
    t = 0
    while t < T:
        j = nxt[t]
        orders[t] = sum(d[t:j])
        t = j
    return _evaluate_plan(d, orders, setup_cost, holding_cost, "wagner-whitin")


def silver_meal(
    demand: Sequence[float], setup_cost: float, holding_cost: float
) -> LotSizingPlan:
    """Silver-Meal: extend the current order block while cost *per period* falls.

    The classic failure mode is a demand series with a near-zero period followed
    by a spike: average cost per period keeps falling through the cheap period, so
    the heuristic over-extends and then pays a large holding charge on the spike.
    That is exactly where the measured gap to Wagner-Whitin opens up.
    """
    d = [float(x) for x in demand]
    T = len(d)
    orders = [0.0] * T
    t = 0
    while t < T:
        best_len = 1
        best_rate = setup_cost / 1.0
        carry = 0.0
        for length in range(2, T - t + 1):
            carry += holding_cost * (length - 1) * d[t + length - 1]
            rate = (setup_cost + carry) / length
            if rate < best_rate - 1e-12:
                best_rate = rate
                best_len = length
            else:
                break
        orders[t] = sum(d[t : t + best_len])
        t += best_len
    return _evaluate_plan(d, orders, setup_cost, holding_cost, "silver-meal")


def least_unit_cost(
    demand: Sequence[float], setup_cost: float, holding_cost: float
) -> LotSizingPlan:
    """Least-unit-cost: extend the block while cost *per unit* falls.

    Included as a contrast to Silver-Meal - it biases toward longer blocks because
    a large future period can dilute the per-unit cost even when it is expensive
    to carry.
    """
    d = [float(x) for x in demand]
    T = len(d)
    orders = [0.0] * T
    t = 0
    while t < T:
        best_len = 1
        units = d[t]
        best_rate = setup_cost / units if units > 0 else float("inf")
        carry = 0.0
        for length in range(2, T - t + 1):
            carry += holding_cost * (length - 1) * d[t + length - 1]
            units += d[t + length - 1]
            rate = (setup_cost + carry) / units if units > 0 else float("inf")
            if rate < best_rate - 1e-12:
                best_rate = rate
                best_len = length
            else:
                break
        orders[t] = sum(d[t : t + best_len])
        t += best_len
    return _evaluate_plan(d, orders, setup_cost, holding_cost, "least-unit-cost")


def lot_for_lot(
    demand: Sequence[float], setup_cost: float, holding_cost: float
) -> LotSizingPlan:
    """Order exactly this period's demand every period. The zero-holding baseline."""
    return _evaluate_plan(demand, list(demand), setup_cost, holding_cost, "lot-for-lot")


def compare_lot_sizing(
    demand: Sequence[float], setup_cost: float, holding_cost: float
) -> dict[str, LotSizingPlan]:
    """Run every method on one instance so the gaps are directly comparable."""
    return {
        p.method: p
        for p in (
            wagner_whitin(demand, setup_cost, holding_cost),
            silver_meal(demand, setup_cost, holding_cost),
            least_unit_cost(demand, setup_cost, holding_cost),
            lot_for_lot(demand, setup_cost, holding_cost),
        )
    }


def seasonal_demand_series(
    n_periods: int,
    base: float,
    amplitude: float,
    noise_cv: float = 0.0,
    seed: int = 7,
) -> list[float]:
    """Deterministic seasonal demand series used by the examples and benchmarks."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_periods)
    series = base + amplitude * np.sin(2.0 * np.pi * t / 12.0)
    if noise_cv > 0:
        series = series * rng.normal(1.0, noise_cv, size=n_periods)
    return [float(max(0.0, round(x, 2))) for x in series]
