"""Replenishment policies: (s,Q), (s,S), (R,S), (R,s,S).

Each policy is a small immutable object with an ``order_quantity`` method that
maps an inventory position to an order.  That signature is deliberately the same
for all four so the Monte Carlo evaluator in :mod:`invkit.simulation` does not
need to know which policy it is running - which is what makes the
analytic-vs-simulated validation in the test suite meaningful rather than
circular.

Choosing between them, in practice:

* ``(s, Q)`` - continuous review, fixed order quantity.  Right when there is a
  real fixed cost per order or a pack/pallet constraint.
* ``(s, S)`` - continuous review, order up to ``S``.  Better when demand arrives
  in large lumps, because ``(s, Q)`` leaves the position below ``s`` after a
  single big order (the "undershoot" problem) and then orders again immediately.
* ``(R, S)`` - periodic review, order up to ``S`` every ``R`` periods.  The
  workhorse when replenishment is calendar-driven (a weekly truck).  Protection
  interval is ``R + L``, not ``L``; getting that wrong is a common and expensive
  error.
* ``(R, s, S)`` - periodic review with a reorder trigger.  Suppresses the
  uneconomically small orders that pure ``(R, S)`` generates on slow movers.

References
----------
Silver, Pyke & Thomas (2016), Ch. 7-8.
Hadley, G. and Whitin, T.M. (1963) *Analysis of Inventory Systems*, Prentice-Hall
- the ``(s, Q)`` fill-rate expression used here.
Ehrhardt, R. and Mosier, C. (1984) 'A revision of the power approximation for
computing (s, S) policies', *Management Science* 30(5), 618-622.
"""

from __future__ import annotations

from dataclasses import dataclass

from .distributions import LeadTimeDemand
from .leadtime import LeadTimeSpec, ltd_gamma, ltd_normal, ltd_stochastic_exact
from .safety_stock import (
    SafetyStockResult,
    cycle_service_of_sQ,
    fill_rate_of_sQ,
    ss_from_cycle_service_level,
    ss_from_fill_rate,
)

__all__ = [
    "Policy",
    "SQPolicy",
    "SSPolicy",
    "RSPolicy",
    "RsSPolicy",
    "build_sQ",
    "build_RS",
    "build_RsS",
    "protection_interval_ltd",
    "expected_cycle_length",
    "average_inventory_sQ",
]


class Policy:
    """Base class: map an inventory position to an order quantity.

    ``continuous`` marks a policy that reacts the instant the position crosses a
    trigger.  The simulator honours it by reviewing at every sub-period slice
    rather than at period boundaries, which is what makes the ``subperiods``
    argument a genuine convergence knob for continuous-review theory instead of a
    cosmetic one.
    """

    review_period: int = 1
    continuous: bool = False

    def order_quantity(self, inventory_position: float, period: int) -> float:
        raise NotImplementedError  # pragma: no cover - interface

    def is_review_period(self, period: int) -> bool:
        return period % self.review_period == 0


@dataclass(frozen=True)
class SQPolicy(Policy):
    """Continuous review: when position drops to ``s`` or below, order ``Q``.

    Multiple lots are ordered if a single lot would not lift the position back
    above ``s`` - otherwise a lumpy demand stream leaves the policy permanently
    behind.
    """

    s: float
    Q: float
    review_period: int = 1
    continuous: bool = True

    def order_quantity(self, inventory_position: float, period: int) -> float:
        if inventory_position > self.s:
            return 0.0
        n_lots = 1
        while inventory_position + n_lots * self.Q <= self.s:
            n_lots += 1
        return float(n_lots * self.Q)


@dataclass(frozen=True)
class SSPolicy(Policy):
    """Continuous review: when position drops to ``s`` or below, order up to ``S``."""

    s: float
    S: float
    review_period: int = 1
    continuous: bool = True

    def __post_init__(self) -> None:
        if self.S <= self.s:
            raise ValueError("S must exceed s")

    def order_quantity(self, inventory_position: float, period: int) -> float:
        if inventory_position > self.s:
            return 0.0
        return float(self.S - inventory_position)


@dataclass(frozen=True)
class RSPolicy(Policy):
    """Periodic review every ``R`` periods: order up to ``S``."""

    R: int
    S: float

    def __post_init__(self) -> None:
        if self.R < 1:
            raise ValueError("R must be at least 1 period")

    @property
    def review_period(self) -> int:  # type: ignore[override]
        return self.R

    def order_quantity(self, inventory_position: float, period: int) -> float:
        return float(max(0.0, self.S - inventory_position))


@dataclass(frozen=True)
class RsSPolicy(Policy):
    """Periodic review every ``R`` periods: if position <= ``s``, order up to ``S``."""

    R: int
    s: float
    S: float

    def __post_init__(self) -> None:
        if self.R < 1:
            raise ValueError("R must be at least 1 period")
        if self.S <= self.s:
            raise ValueError("S must exceed s")

    @property
    def review_period(self) -> int:  # type: ignore[override]
        return self.R

    def order_quantity(self, inventory_position: float, period: int) -> float:
        if inventory_position > self.s:
            return 0.0
        return float(self.S - inventory_position)


def protection_interval_ltd(
    spec: LeadTimeSpec,
    review_period: int = 0,
    family: str = "gamma",
    exact_stochastic: bool = True,
) -> LeadTimeDemand:
    """Demand distribution over the protection interval ``L + R``.

    For continuous review pass ``review_period=0``.  For periodic review pass the
    review period: the order placed at a review does not arrive until ``L`` later
    and must then cover demand until the *next* order arrives, ``R`` after that.
    """
    shifted = spec.with_offset(review_period) if review_period else spec
    if exact_stochastic and len(shifted.lead_time_pmf) > 1:
        return ltd_stochastic_exact(shifted, family=family)
    return ltd_gamma(shifted) if family == "gamma" else ltd_normal(shifted)


def build_sQ(
    spec: LeadTimeSpec,
    Q: float,
    *,
    target_csl: float | None = None,
    target_fill: float | None = None,
    family: str = "gamma",
) -> tuple[SQPolicy, SafetyStockResult]:
    """Build an ``(s, Q)`` policy from either a CSL target or a fill-rate target."""
    if (target_csl is None) == (target_fill is None):
        raise ValueError("specify exactly one of target_csl or target_fill")
    ltd = protection_interval_ltd(spec, review_period=0, family=family)
    if target_csl is not None:
        res = ss_from_cycle_service_level(ltd, target_csl, Q)
    else:
        res = ss_from_fill_rate(ltd, float(target_fill), Q)
    return SQPolicy(s=res.reorder_point, Q=Q), res


def build_RS(
    spec: LeadTimeSpec,
    R: int,
    *,
    target_csl: float | None = None,
    target_fill: float | None = None,
    family: str = "gamma",
) -> tuple[RSPolicy, SafetyStockResult]:
    """Build an ``(R, S)`` policy over the ``R + L`` protection interval.

    For the fill-rate variant the "order quantity" that the type-2 expression
    needs is the expected demand per review cycle, ``R * mu_d`` - that is what one
    replenishment covers.
    """
    ltd = protection_interval_ltd(spec, review_period=R, family=family)
    if (target_csl is None) == (target_fill is None):
        raise ValueError("specify exactly one of target_csl or target_fill")
    cycle_demand = R * spec.demand_mean
    if target_csl is not None:
        res = ss_from_cycle_service_level(ltd, target_csl, cycle_demand)
    else:
        res = ss_from_fill_rate(ltd, float(target_fill), cycle_demand)
    return RSPolicy(R=R, S=res.reorder_point), res


def build_RsS(
    spec: LeadTimeSpec,
    R: int,
    Q: float,
    *,
    target_csl: float | None = None,
    target_fill: float | None = None,
    family: str = "gamma",
) -> tuple[RsSPolicy, SafetyStockResult]:
    """Build an ``(R, s, S)`` policy with ``S = s + Q``.

    ``s`` comes from the service target over the ``R + L`` protection interval and
    ``Q`` is normally the EOQ.  This is the pragmatic construction, not the
    cost-optimal one; Ehrhardt & Mosier's power approximation targets a
    backorder-cost objective instead, which requires a defensible shortage cost.
    Most organisations cannot produce one, and a service target is the honest
    substitute.
    """
    ltd = protection_interval_ltd(spec, review_period=R, family=family)
    if (target_csl is None) == (target_fill is None):
        raise ValueError("specify exactly one of target_csl or target_fill")
    if target_csl is not None:
        res = ss_from_cycle_service_level(ltd, target_csl, Q)
    else:
        res = ss_from_fill_rate(ltd, float(target_fill), Q)
    s = res.reorder_point
    return RsSPolicy(R=R, s=s, S=s + Q), res


def expected_cycle_length(Q: float, demand_per_period: float) -> float:
    """Periods between replenishments under ``(s, Q)``: ``Q / mu_d``."""
    if demand_per_period <= 0:
        raise ValueError("demand rate must be positive")
    return float(Q / demand_per_period)


def average_inventory_sQ(ltd: LeadTimeDemand, s: float, Q: float) -> dict[str, float]:
    """Analytic average inventory decomposition for an ``(s, Q)`` policy.

    ``cycle_stock = Q / 2``, ``safety_stock = s - E[D_L]`` and the expected
    backorder level is the average of the loss function over the inventory
    position, which is uniform on ``[s, s + Q]``.  Reporting on-hand *net* of the
    backorder correction matters on slow movers, where the naive
    ``Q/2 + SS`` overstates on-hand by a visible margin.
    """
    cycle = Q / 2.0
    ss = s - ltd.mean
    # Average backorder level, evaluated by trapezoid over the uniform position.
    grid = [s + Q * i / 200.0 for i in range(201)]
    vals = [ltd.loss(x) for x in grid]
    backorders = sum(
        0.5 * (vals[i] + vals[i + 1]) * (grid[i + 1] - grid[i]) for i in range(len(grid) - 1)
    ) / Q
    return {
        "cycle_stock": float(cycle),
        "safety_stock": float(ss),
        "expected_backorders": float(backorders),
        "expected_on_hand": float(cycle + ss + backorders),
        "cycle_service_level": cycle_service_of_sQ(ltd, s),
        "fill_rate": fill_rate_of_sQ(ltd, s, Q),
    }
