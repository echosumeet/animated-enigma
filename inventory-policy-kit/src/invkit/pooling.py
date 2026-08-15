"""Risk pooling: the square-root law, and what breaks it.

Consolidating ``n`` independent stocking locations into one reduces safety stock
by a factor of ``sqrt(n)``.  That is the square-root law, and it is the single
most quoted result in network design.  It is also routinely misapplied, in three
specific ways this module makes measurable:

1. **Correlation.** The law assumes independence.  With pairwise correlation
   ``rho`` the pooled standard deviation is ``sigma * sqrt(n + n(n-1)rho)``, so
   the benefit degrades toward zero as ``rho -> 1``.  Regional demand for the same
   SKU is usually positively correlated - a national promotion moves every region
   at once - and the realised saving comes in well below the pitch.
2. **Unequal locations.** Pooling ``n`` locations of very different size gives far
   less than ``sqrt(n)``, because the variance is already concentrated in the
   large one.
3. **Lead time.** Centralising usually lengthens the outbound lead time to the
   customer.  Safety stock scales with ``sqrt(L)``, so a move from 2 days to 5
   days multiplies the requirement by 1.58 and can wipe out the pooling benefit
   entirely.  :func:`pooling_with_lead_time_penalty` computes the break-even.

Reference: Eppen, G.D. (1979) 'Effects of centralization on expected costs in a
multi-location newsboy problem', *Management Science* 25(5), 498-501.
Chopra & Meindl (2016), Ch. 12.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats

__all__ = [
    "PoolingResult",
    "pooled_sd",
    "square_root_law",
    "pooling_with_lead_time_penalty",
    "simulate_pooling",
]


@dataclass(frozen=True)
class PoolingResult:
    n_locations: int
    correlation: float
    decentralised_ss: float
    centralised_ss: float

    @property
    def reduction_pct(self) -> float:
        if self.decentralised_ss <= 0:
            return 0.0
        return float(100.0 * (1.0 - self.centralised_ss / self.decentralised_ss))

    @property
    def effective_sqrt_n(self) -> float:
        """The ``sqrt(n)`` you actually get, given correlation."""
        if self.centralised_ss <= 0:
            return float("inf")
        return float(self.decentralised_ss / self.centralised_ss)


def pooled_sd(sds: Sequence[float], correlation: float = 0.0) -> float:
    """Standard deviation of the sum, under equicorrelation ``rho``."""
    s = np.asarray(sds, dtype=float)
    if not -1.0 <= correlation <= 1.0:
        raise ValueError("correlation must be in [-1, 1]")
    var = float((s ** 2).sum() + correlation * (s.sum() ** 2 - (s ** 2).sum()))
    if var < 0:
        raise ValueError("correlation implies a non-positive-definite covariance matrix")
    return math.sqrt(var)


def square_root_law(
    n_locations: int,
    sd_per_location: float,
    service_level: float = 0.95,
    lead_time: float = 1.0,
    correlation: float = 0.0,
) -> PoolingResult:
    """Decentralised vs centralised safety stock for ``n`` identical locations."""
    if n_locations < 1:
        raise ValueError("n_locations must be >= 1")
    z = float(stats.norm.ppf(service_level))
    root_l = math.sqrt(lead_time)
    decentralised = n_locations * z * sd_per_location * root_l
    centralised = z * pooled_sd([sd_per_location] * n_locations, correlation) * root_l
    return PoolingResult(n_locations, correlation, float(decentralised), float(centralised))


def pooling_with_lead_time_penalty(
    n_locations: int,
    sd_per_location: float,
    lead_time_local: float,
    lead_time_central: float,
    service_level: float = 0.95,
    correlation: float = 0.0,
) -> dict[str, float]:
    """Pooling benefit net of the longer lead time centralisation usually implies.

    Returns the two safety stocks, the net reduction, and the break-even
    centralised lead time at which the pooling benefit is exactly consumed.
    """
    z = float(stats.norm.ppf(service_level))
    decentralised = n_locations * z * sd_per_location * math.sqrt(lead_time_local)
    sd_pool = pooled_sd([sd_per_location] * n_locations, correlation)
    centralised = z * sd_pool * math.sqrt(lead_time_central)
    ratio = (n_locations * sd_per_location) / sd_pool
    breakeven_lt = lead_time_local * ratio ** 2
    return {
        "decentralised_ss": float(decentralised),
        "centralised_ss": float(centralised),
        "net_reduction_pct": float(100.0 * (1.0 - centralised / decentralised)),
        "breakeven_central_lead_time": float(breakeven_lt),
        "pooled_sd": float(sd_pool),
    }


def simulate_pooling(
    n_locations: int,
    mean_per_location: float,
    sd_per_location: float,
    correlation: float = 0.0,
    service_level: float = 0.95,
    n_draws: int = 200_000,
    seed: int = 4242,
) -> dict[str, float]:
    """Empirical check of the square-root law on correlated normal demand.

    Builds an equicorrelated multivariate normal via a single common factor,
    draws demand, and compares the sum of per-location quantile buffers against
    the buffer on the pooled total.  The measured ratio should track
    ``n / sqrt(n + n(n-1)rho)``; where it does not, the discrepancy is Monte Carlo
    error and the returned ``analytic_ratio`` says so.
    """
    rng = np.random.default_rng(seed)
    if not 0.0 <= correlation < 1.0:
        raise ValueError("simulate_pooling supports correlation in [0, 1)")
    common = rng.standard_normal(n_draws)
    idio = rng.standard_normal((n_draws, n_locations))
    a = math.sqrt(correlation)
    b = math.sqrt(1.0 - correlation)
    z = a * common[:, None] + b * idio
    demand = mean_per_location + sd_per_location * z

    local_buffer = 0.0
    for j in range(n_locations):
        local_buffer += float(np.quantile(demand[:, j], service_level)) - mean_per_location
    total = demand.sum(axis=1)
    pooled_buffer = float(np.quantile(total, service_level)) - n_locations * mean_per_location

    analytic_ratio = n_locations / math.sqrt(
        n_locations + n_locations * (n_locations - 1) * correlation
    )
    return {
        "simulated_decentralised_ss": local_buffer,
        "simulated_centralised_ss": pooled_buffer,
        "simulated_ratio": float(local_buffer / pooled_buffer),
        "analytic_ratio": float(analytic_ratio),
        "reduction_pct": float(100.0 * (1.0 - pooled_buffer / local_buffer)),
    }
