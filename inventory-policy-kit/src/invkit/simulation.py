"""Monte Carlo evaluation of replenishment policies.

The point of this module is falsification.  Analytic safety-stock formulas rest
on assumptions - continuous review, no undershoot, independent demand, backorders
rather than lost sales - and every one of those assumptions is violated somewhere
in a real network.  The only defensible way to ship a formula is to simulate the
policy it produces and check that the service level it promised is the service
level it delivers.  The test suite does exactly that, and it is the strongest
thing in this repository.

Three measurement details that are easy to get wrong and change the answer:

*Cycle service level* is measured per replenishment cycle, not per period.  A
cycle is the window from placing an order to receiving it, and it counts as a
failure if demand goes unmet at any point inside it.  That is the event the
formula ``P(D_L <= s)`` actually describes.  Measuring "fraction of periods with
stock" instead inflates the number and makes a broken reorder point look fine.

*Ready rate* is the per-period version - the fraction of periods ending with
non-negative net inventory.  It is the right comparator for a periodic
order-up-to policy, where every period is its own protection interval.

*Undershoot*.  A continuous-review policy simulated on a period grid crosses the
reorder point mid-period and only reacts at the boundary, so the effective
reorder point sits below ``s`` and realised service comes in under target.  The
``subperiods`` argument splits each period into ``m`` slices to shrink that gap.
This is not a trick to make the formulas look good - it is precisely the
difference between a system polling stock continuously and one running a nightly
batch, and that difference is worth real service points on fast movers.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .policies import Policy

__all__ = [
    "SimulationResult",
    "DemandProcess",
    "simulate_policy",
    "simulate_policy_replications",
]


@dataclass(frozen=True)
class DemandProcess:
    """iid per-period demand with known moments and an exact time-rescaling.

    ``rescaled(f)`` returns the process for a slice of length ``f`` of one period.
    For the gamma family this is exact rather than approximate: scaling the mean
    by ``f`` and the standard deviation by ``sqrt(f)`` leaves the scale parameter
    ``theta = sigma^2 / mu`` unchanged and scales the shape by ``f``, so ``1/f``
    slices sum back to exactly the original per-period distribution.  That is the
    property that makes gamma the right choice for validating continuous-review
    formulas on a discrete grid.
    """

    mean: float
    sd: float
    family: str = "gamma"
    clip_at_zero: bool = True

    def __post_init__(self) -> None:
        if self.mean <= 0 or self.sd <= 0:
            raise ValueError("mean and sd must be positive")
        if self.family not in {"gamma", "normal"}:
            raise ValueError("family must be 'gamma' or 'normal'")

    def rescaled(self, fraction: float) -> "DemandProcess":
        if fraction <= 0:
            raise ValueError("fraction must be positive")
        return DemandProcess(
            mean=self.mean * fraction,
            sd=self.sd * math.sqrt(fraction),
            family=self.family,
            clip_at_zero=self.clip_at_zero,
        )

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.family == "gamma":
            theta = self.sd * self.sd / self.mean
            k = self.mean / theta
            return rng.gamma(shape=k, scale=theta, size=n)
        d = rng.normal(self.mean, self.sd, size=n)
        return np.maximum(d, 0.0) if self.clip_at_zero else d


@dataclass
class SimulationResult:
    """Everything measured in one run."""

    periods: int
    total_demand: float
    demand_filled: float
    n_cycles: int
    n_cycles_with_stockout: int
    avg_on_hand: float
    avg_backorder: float
    avg_inventory_position: float
    n_orders: int
    order_quantity_mean: float
    holding_cost: float = 0.0
    ordering_cost: float = 0.0
    shortage_cost: float = 0.0
    period_stockouts: int = 0
    detail: dict = field(default_factory=dict)

    @property
    def fill_rate(self) -> float:
        return float(self.demand_filled / self.total_demand) if self.total_demand else 1.0

    @property
    def cycle_service_level(self) -> float:
        if self.n_cycles == 0:
            return 1.0
        return float(1.0 - self.n_cycles_with_stockout / self.n_cycles)

    @property
    def ready_rate(self) -> float:
        if self.periods == 0:
            return 1.0
        return float(1.0 - self.period_stockouts / self.periods)

    @property
    def total_cost(self) -> float:
        return float(self.holding_cost + self.ordering_cost + self.shortage_cost)

    def summary(self) -> dict[str, float]:
        return {
            "fill_rate": self.fill_rate,
            "cycle_service_level": self.cycle_service_level,
            "ready_rate": self.ready_rate,
            "avg_on_hand": self.avg_on_hand,
            "avg_backorder": self.avg_backorder,
            "n_orders": float(self.n_orders),
            "total_cost": self.total_cost,
        }


def simulate_policy(
    policy: Policy,
    demand: DemandProcess,
    lead_time_pmf: Mapping[int, float],
    n_periods: int = 20_000,
    warmup: int = 500,
    *,
    subperiods: int = 1,
    lost_sales: bool = False,
    holding_cost: float = 0.0,
    order_cost: float = 0.0,
    shortage_cost: float = 0.0,
    seed: int = 12345,
    initial_position: float | None = None,
) -> SimulationResult:
    """Simulate one long run of a replenishment policy.

    Order arrivals live in a ``heapq`` keyed on the arrival slice, so order
    crossing - a later order arriving before an earlier one - is handled
    correctly.  Crossing is rare with a tight lead time and common with a bimodal
    one (ocean versus expedited air), and a simulator that assumes FIFO arrivals
    overstates service exactly in the case you built the simulator to study.

    Costs are charged on end-of-slice on-hand (holding), per order placed
    (ordering) and per unit-period of backorder (shortage).
    """
    if subperiods < 1:
        raise ValueError("subperiods must be >= 1")
    rng = np.random.default_rng(seed)

    lt_values = np.array(sorted(lead_time_pmf), dtype=int)
    lt_probs = np.array([lead_time_pmf[int(v)] for v in lt_values], dtype=float)
    lt_probs = lt_probs / lt_probs.sum()

    slice_demand = demand.rescaled(1.0 / subperiods)
    total_slices = n_periods * subperiods
    warm_slices = warmup * subperiods
    demand_stream = slice_demand.sample(total_slices, rng)

    if initial_position is None:
        initial_position = getattr(policy, "S", None)
        if initial_position is None:
            initial_position = float(getattr(policy, "s", 0.0)) + float(getattr(policy, "Q", 0.0))
    on_hand = float(initial_position)
    backorder = 0.0
    on_order = 0.0

    arrivals: list[tuple[int, int, float]] = []  # (arrival slice, order id, qty)
    open_orders: dict[int, bool] = {}  # order id -> stockout seen during its cycle
    order_id = 0

    total_demand = 0.0
    filled = 0.0
    sum_on_hand = 0.0
    sum_backorder = 0.0
    sum_position = 0.0
    n_orders = 0
    order_qty_sum = 0.0
    period_stockouts = 0
    n_cycles = 0
    n_cycles_stockout = 0
    hold_c = 0.0
    order_c = 0.0
    short_c = 0.0

    # Continuous-review policies react at every slice; periodic ones only on their
    # review boundary.  Sequence within a slice is receive -> order -> demand, which
    # is the convention that makes the (s, Q) exposure window exactly L periods and
    # the (R, S) protection interval exactly R + L.
    review_every = 1 if policy.continuous else max(1, policy.review_period) * subperiods
    period_stockout_flag = False

    for t in range(total_slices):
        counting = t >= warm_slices

        # 1. receive arrivals due at this slice
        while arrivals and arrivals[0][0] <= t:
            _, oid, qty = heapq.heappop(arrivals)
            on_hand += qty
            on_order -= qty
            had_stockout = open_orders.pop(oid, False)
            if counting:
                n_cycles += 1
                if had_stockout:
                    n_cycles_stockout += 1
            if backorder > 0.0:
                served = min(backorder, on_hand)
                backorder -= served
                on_hand -= served

        # 2. review and order
        if t % review_every == 0:
            position = on_hand - backorder + on_order
            q = policy.order_quantity(position, t // subperiods)
            if q > 1e-9:
                lt_periods = int(rng.choice(lt_values, p=lt_probs))
                arrival = t + max(1, lt_periods * subperiods)
                order_id += 1
                heapq.heappush(arrivals, (arrival, order_id, q))
                open_orders[order_id] = False
                on_order += q
                if counting:
                    n_orders += 1
                    order_qty_sum += q
                    order_c += order_cost

        # 3. demand
        d = float(demand_stream[t])
        on_hand_before = on_hand
        backorder_before = backorder
        served = min(on_hand, d)
        on_hand -= served
        unmet = d - served
        if not lost_sales:
            backorder += unmet
        if counting:
            total_demand += d
            filled += served
        if unmet > 1e-12:
            period_stockout_flag = True
            for oid in open_orders:
                open_orders[oid] = True

        # 4. accounting
        #
        # Inventory is averaged over the slice as the midpoint of its pre- and
        # post-demand values, not sampled at the end.  Depletion within a slice is
        # continuous, so the end-of-slice snapshot understates average on-hand by
        # half a slice of demand - which is exactly the mu/2 discrepancy that
        # makes a simulated holding cost disagree with the textbook Q/2 + SS.
        if counting:
            avg_oh = 0.5 * (on_hand_before + on_hand)
            avg_bo = 0.5 * (backorder_before + backorder)
            sum_on_hand += avg_oh
            sum_backorder += avg_bo
            sum_position += avg_oh - avg_bo + on_order
            hold_c += holding_cost * avg_oh / subperiods
            short_c += shortage_cost * avg_bo / subperiods
        if (t + 1) % subperiods == 0:
            if counting and (period_stockout_flag or backorder > 1e-9):
                period_stockouts += 1
            period_stockout_flag = False

    counted = total_slices - warm_slices
    return SimulationResult(
        periods=int(counted // subperiods),
        total_demand=total_demand,
        demand_filled=filled,
        n_cycles=n_cycles,
        n_cycles_with_stockout=n_cycles_stockout,
        avg_on_hand=sum_on_hand / counted,
        avg_backorder=sum_backorder / counted,
        avg_inventory_position=sum_position / counted,
        n_orders=n_orders,
        order_quantity_mean=order_qty_sum / n_orders if n_orders else 0.0,
        holding_cost=hold_c,
        ordering_cost=order_c,
        shortage_cost=short_c,
        period_stockouts=period_stockouts,
        detail={"subperiods": subperiods, "lost_sales": lost_sales},
    )


def simulate_policy_replications(
    policy: Policy,
    demand: DemandProcess,
    lead_time_pmf: Mapping[int, float],
    n_replications: int = 8,
    n_periods: int = 5_000,
    seed: int = 2026,
    **kwargs,
) -> dict[str, tuple[float, float]]:
    """Independent replications, returning ``(mean, half-width)`` at 95%.

    Reporting a confidence half-width rather than a point estimate is the
    difference between "the simulation validates the formula" and "the simulation
    is consistent with the formula at the resolution we can actually see".
    """
    metrics: dict[str, list[float]] = {}
    for r in range(n_replications):
        res = simulate_policy(
            policy, demand, lead_time_pmf, n_periods=n_periods, seed=seed + 977 * r, **kwargs
        )
        for k, v in res.summary().items():
            metrics.setdefault(k, []).append(v)
    out: dict[str, tuple[float, float]] = {}
    for k, vals in metrics.items():
        arr = np.asarray(vals, dtype=float)
        mean = float(arr.mean())
        hw = float(1.96 * arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0
        out[k] = (mean, hw)
    return out
