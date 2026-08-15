"""Safety stock from cycle service level, from fill rate, and from empirical errors.

This is the headline module.  Three ways to set the same buffer, and they do not
agree:

**Cycle service level (type 1, "CSL")** is the probability that a replenishment
cycle passes without any stockout.  ``SS = z * sigma_L`` with ``z = Phi^-1(CSL)``.
It is the formula every planning system ships with, and it is almost never the
number the business cares about.  Note what it does *not* contain: the order
quantity.  Two items with identical demand variability but order quantities of
one week and one quarter get the same safety stock, even though the second one
only exposes itself to a stockout a quarter as often.

**Fill rate (type 2)** is the fraction of demand served from stock.  For an
``(s, Q)`` policy the exact expression is

    ``fill = 1 - [ E(D_L - s)^+ - E(D_L - s - Q)^+ ] / Q``

so the buffer required to hit a fill-rate target *falls* as ``Q`` rises.  Solving
it requires inverting the loss function, which has no closed form.  That
inconvenience is the entire reason practitioners quietly substitute CSL and then
wonder why measured fill rate comes in at 99% against a 95% target while
inventory sits 30% above plan.

**Empirical quantile** drops the normality assumption.  When forecast errors are
right-skewed - which they are for anything promotion-driven or intermittent - the
normal fit understates the upper tail and the buffer is too small exactly where
it matters.

References
----------
Silver, Pyke & Thomas (2016), Ch. 7 (B1/B2/B3 shortage measures).
Chopra, S. and Meindl, P. (2016) *Supply Chain Management*, 6th ed., Ch. 12.
Johnson, M.E., Lee, H.L., Davis, T. and Hall, R. (1995) 'Expressions for item
fill rates in periodic inventory systems', *Naval Research Logistics* 42, 57-80.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .distributions import (
    LeadTimeDemand,
    NormalLTD,
    inverse_standard_normal_loss,
)

__all__ = [
    "SafetyStockResult",
    "ss_from_cycle_service_level",
    "ss_from_fill_rate",
    "ss_from_empirical_quantile",
    "fill_rate_of_sQ",
    "cycle_service_of_sQ",
    "achieved_fill_rate_for_csl_target",
    "compare_service_definitions",
]


@dataclass(frozen=True)
class SafetyStockResult:
    """Safety stock plus the reorder point and the service it actually buys."""

    safety_stock: float
    reorder_point: float
    cycle_service_level: float
    fill_rate: float
    basis: str

    def __str__(self) -> str:  # pragma: no cover - display helper
        return (
            f"{self.basis}: SS={self.safety_stock:,.1f} s={self.reorder_point:,.1f} "
            f"CSL={self.cycle_service_level:.4f} fill={self.fill_rate:.4f}"
        )


def cycle_service_of_sQ(ltd: LeadTimeDemand, s: float) -> float:
    """P(no stockout in a replenishment cycle) = P(D_L <= s)."""
    return float(ltd.cdf(s))


def fill_rate_of_sQ(ltd: LeadTimeDemand, s: float, Q: float, exact: bool = True) -> float:
    """Fraction of demand filled from on-hand stock under an ``(s, Q)`` policy.

    The exact form subtracts the loss at ``s + Q``, which accounts for the fact
    that an outstanding order may itself be insufficient.  The approximation
    ``1 - E(D_L - s)^+ / Q`` drops that term; it is accurate when ``Q`` is large
    relative to ``sigma_L`` and conservative (understates fill) otherwise.
    """
    if Q <= 0:
        raise ValueError("Q must be positive")
    shortage = ltd.loss(s)
    if exact:
        shortage -= ltd.loss(s + Q)
    return float(1.0 - shortage / Q)


def ss_from_cycle_service_level(
    ltd: LeadTimeDemand, target_csl: float, Q: float | None = None
) -> SafetyStockResult:
    """Set the reorder point so that ``P(D_L <= s) = target_csl``.

    ``Q`` is optional and used only to report the fill rate the resulting policy
    achieves - which is the number worth looking at.
    """
    if not 0.0 < target_csl < 1.0:
        raise ValueError("target_csl must be in (0, 1)")
    s = ltd.ppf(target_csl)
    ss = s - ltd.mean
    fill = fill_rate_of_sQ(ltd, s, Q) if Q else float("nan")
    return SafetyStockResult(ss, s, target_csl, fill, f"CSL target {target_csl:.3f}")


def ss_from_fill_rate(
    ltd: LeadTimeDemand,
    target_fill: float,
    Q: float,
    exact: bool = True,
    tol: float = 1e-10,
) -> SafetyStockResult:
    """Set the reorder point so that the achieved fill rate equals ``target_fill``.

    For a normal ``ltd`` and the first-order expression this reduces to inverting
    the standard normal loss: ``G(z) = Q * (1 - target) / sigma_L``.  In the
    general case (gamma, empirical, mixture, or the exact two-term form) the
    solve is a bisection on ``s``, which is monotone because the loss function is
    strictly decreasing.

    The result is allowed to be *negative*.  A negative safety stock is not a bug:
    with a large order quantity and a modest fill-rate target, holding stock below
    mean lead-time demand still serves 95% of units, because a stockout that
    occurs once per very long cycle affects a small share of volume.  Systems that
    floor safety stock at zero are leaving that saving on the table.
    """
    if not 0.0 < target_fill < 1.0:
        raise ValueError("target_fill must be in (0, 1)")
    if Q <= 0:
        raise ValueError("Q must be positive")

    if isinstance(ltd, NormalLTD) and not exact:
        g_target = Q * (1.0 - target_fill) / ltd.sigma
        z = inverse_standard_normal_loss(g_target)
        s = ltd.mu + z * ltd.sigma
    else:
        lo = ltd.mean - 12.0 * ltd.sd - Q
        hi = ltd.mean + 12.0 * ltd.sd
        # fill_rate is increasing in s; expand lo until the target is bracketed.
        guard = 0
        while fill_rate_of_sQ(ltd, lo, Q, exact) > target_fill and guard < 200:
            lo -= max(Q, ltd.sd)
            guard += 1
        for _ in range(300):
            mid = 0.5 * (lo + hi)
            if fill_rate_of_sQ(ltd, mid, Q, exact) < target_fill:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol * max(1.0, abs(hi)):
                break
        s = 0.5 * (lo + hi)

    ss = s - ltd.mean
    return SafetyStockResult(
        ss,
        s,
        cycle_service_of_sQ(ltd, s),
        fill_rate_of_sQ(ltd, s, Q, exact),
        f"fill-rate target {target_fill:.3f}",
    )


def ss_from_empirical_quantile(
    errors: np.ndarray,
    target_csl: float,
    lead_time_periods: int,
    n_draws: int = 200_000,
    seed: int = 20260815,
) -> tuple[float, float]:
    """Safety stock from the empirical quantile of aggregated forecast errors.

    Returns ``(empirical_ss, normal_ss)`` so the gap is immediately visible.  The
    per-period error sample is aggregated over the lead time by iid bootstrap;
    the normal comparator uses the same aggregated moments, so the only
    difference between the two numbers is shape.

    Aggregating errors by bootstrap assumes serial independence.  That is the
    honest weak point: real forecast errors are autocorrelated during a demand
    regime shift, and both numbers here will be too small in that case.
    """
    errors = np.asarray(errors, dtype=float)
    rng = np.random.default_rng(seed)
    agg = rng.choice(errors, size=(n_draws, lead_time_periods), replace=True).sum(axis=1)
    empirical_ss = float(np.quantile(agg, target_csl) - agg.mean())
    normal_ss = float(stats.norm.ppf(target_csl) * agg.std(ddof=1))
    return empirical_ss, normal_ss


def achieved_fill_rate_for_csl_target(
    ltd: LeadTimeDemand, target_csl: float, Q: float
) -> float:
    """Fill rate you actually get when you size safety stock off a CSL target."""
    s = ltd.ppf(target_csl)
    return fill_rate_of_sQ(ltd, s, Q)


def compare_service_definitions(
    ltd: LeadTimeDemand, target: float, Q: float
) -> dict[str, float]:
    """Same numeric target read as CSL vs as fill rate: the headline comparison.

    Returns safety stock under each interpretation, the ratio, and the service
    each one delivers on the *other* metric.
    """
    csl_res = ss_from_cycle_service_level(ltd, target, Q)
    fill_res = ss_from_fill_rate(ltd, target, Q)
    ss_csl = csl_res.safety_stock
    ss_fill = fill_res.safety_stock
    return {
        "target": float(target),
        "Q": float(Q),
        "Q_over_sigma": float(Q / ltd.sd),
        "ss_csl_basis": ss_csl,
        "ss_fill_basis": ss_fill,
        "ss_delta": ss_csl - ss_fill,
        "ss_ratio": float(ss_csl / ss_fill) if abs(ss_fill) > 1e-9 else float("inf"),
        "fill_at_csl_basis": csl_res.fill_rate,
        "csl_at_fill_basis": fill_res.cycle_service_level,
    }
