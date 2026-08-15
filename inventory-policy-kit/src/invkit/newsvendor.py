"""Single-period newsvendor with normal and empirical demand.

The newsvendor is the cleanest statement of the only question that matters in
inventory: what is the marginal cost of one unit too many versus one unit too
few?  The optimal order quantity is the critical fractile
``CF = Cu / (Cu + Co)`` of the demand distribution - no optimisation, just a
quantile.

Where it goes wrong in practice is never the formula.  It is that ``Cu``, the
underage cost, gets set to lost gross margin when the real number includes
substitution, the customer who does not come back, and the expediting freight
that gets paid to avoid the stockout in the first place.  Underestimating ``Cu``
by 2x moves the critical fractile from 0.95 to 0.83, which on a moderately
variable item is roughly a third of the buffer.  :func:`critical_fractile_
sensitivity` makes that elasticity explicit so the cost debate happens before the
quantile is computed rather than after the season ends.

The empirical variant matters for the same reason it matters for safety stock: at
a 0.95 fractile you are asking a question about the far right tail, and a normal
fitted to a right-skewed sample answers a different question.

References
----------
Arrow, K.J., Harris, T. and Marschak, J. (1951) 'Optimal inventory policy',
*Econometrica* 19(3), 250-272.
Porteus, E.L. (2002) *Foundations of Stochastic Inventory Theory*, Stanford UP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distributions import EmpiricalLTD, LeadTimeDemand, NormalLTD

__all__ = [
    "NewsvendorResult",
    "critical_fractile",
    "newsvendor_normal",
    "newsvendor_empirical",
    "newsvendor_from_distribution",
    "expected_newsvendor_cost",
    "critical_fractile_sensitivity",
]


@dataclass(frozen=True)
class NewsvendorResult:
    order_quantity: float
    critical_fractile: float
    expected_cost: float
    expected_leftover: float
    expected_shortage: float
    basis: str


def critical_fractile(underage_cost: float, overage_cost: float) -> float:
    """``Cu / (Cu + Co)`` - the service level the economics implies."""
    if underage_cost <= 0 or overage_cost <= 0:
        raise ValueError("costs must be positive")
    return float(underage_cost / (underage_cost + overage_cost))


def expected_newsvendor_cost(
    dist: LeadTimeDemand, Q: float, underage_cost: float, overage_cost: float
) -> tuple[float, float, float]:
    """Expected cost, leftover and shortage at order quantity ``Q``.

    ``E[shortage] = loss(Q)`` and ``E[leftover] = Q - E[D] + loss(Q)`` from the
    identity ``(Q - D)^+ = Q - D + (D - Q)^+``.
    """
    shortage = dist.loss(Q)
    leftover = Q - dist.mean + shortage
    cost = overage_cost * leftover + underage_cost * shortage
    return float(cost), float(leftover), float(shortage)


def newsvendor_from_distribution(
    dist: LeadTimeDemand, underage_cost: float, overage_cost: float, basis: str = "distribution"
) -> NewsvendorResult:
    cf = critical_fractile(underage_cost, overage_cost)
    q = dist.ppf(cf)
    cost, leftover, shortage = expected_newsvendor_cost(dist, q, underage_cost, overage_cost)
    return NewsvendorResult(float(q), cf, cost, leftover, shortage, basis)


def newsvendor_normal(
    mean: float, sd: float, underage_cost: float, overage_cost: float
) -> NewsvendorResult:
    """Normal-demand newsvendor: ``Q* = mu + z_CF * sigma``."""
    return newsvendor_from_distribution(
        NormalLTD(mean, sd), underage_cost, overage_cost, basis="normal"
    )


def newsvendor_empirical(
    sample: np.ndarray, underage_cost: float, overage_cost: float
) -> NewsvendorResult:
    """Empirical-demand newsvendor: the sample quantile at the critical fractile.

    This is also the sample-average-approximation solution to the newsvendor
    program, so it inherits its consistency: as the sample grows it converges to
    the true optimum without ever committing to a shape.
    """
    return newsvendor_from_distribution(
        EmpiricalLTD(sample), underage_cost, overage_cost, basis="empirical"
    )


def critical_fractile_sensitivity(
    dist: LeadTimeDemand, underage_cost: float, overage_cost: float, cu_multipliers=(0.5, 1.0, 2.0)
) -> list[dict[str, float]]:
    """How much the order quantity moves when the underage cost is misjudged.

    Returns one row per multiplier with the implied fractile, the order quantity,
    and the *realised* expected cost of that quantity evaluated at the true costs -
    which is the number that shows whether the mis-estimate actually hurt.
    """
    rows: list[dict[str, float]] = []
    true_cost, _, _ = expected_newsvendor_cost(
        dist, dist.ppf(critical_fractile(underage_cost, overage_cost)), underage_cost, overage_cost
    )
    for m in cu_multipliers:
        cf = critical_fractile(underage_cost * m, overage_cost)
        q = dist.ppf(cf)
        cost, leftover, shortage = expected_newsvendor_cost(dist, q, underage_cost, overage_cost)
        rows.append(
            {
                "cu_multiplier": float(m),
                "assumed_fractile": float(cf),
                "order_quantity": float(q),
                "true_expected_cost": float(cost),
                "cost_penalty_pct": float(100.0 * (cost / true_cost - 1.0)),
                "expected_leftover": float(leftover),
                "expected_shortage": float(shortage),
            }
        )
    return rows
