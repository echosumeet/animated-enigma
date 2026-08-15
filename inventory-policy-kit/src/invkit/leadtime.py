"""Building lead-time demand distributions, including stochastic lead time.

The single biggest reason reorder points are wrong in practice is that they are
computed from demand variability alone.  Lead time is treated as a planning
parameter - a number in an item master - when it is actually a random variable
with a long right tail (customs, port congestion, a supplier that batches
production).  Ignoring ``sigma_L`` does not shift the reorder point a little;
when the lead-time coefficient of variation approaches the demand CV, the
lead-time term *dominates* the variance.

Two constructions are provided:

1. :func:`ltd_moments` - the classical variance convolution
   ``Var(D_L) = E[L] * Var(d) + E[d]^2 * Var(L)``.  Cheap, standard, and what
   most planning systems should at least be doing.
2. :func:`ltd_stochastic_exact` - the exact mixture over a discrete lead-time
   pmf.  Use it when the lead-time distribution is skewed or bimodal, where
   moment matching to a normal misses the tail that actually causes the stockout.

Reference: Silver, Pyke & Thomas (2016), Sec. 6.7 (variability of lead time);
Eppen, G. and Martin, R. (1988) 'Determining safety stock in the presence of
stochastic lead time and demand', *Management Science* 34(11), 1380-1390.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .distributions import GammaLTD, LeadTimeDemand, MixtureLTD, NormalLTD

__all__ = [
    "LeadTimeSpec",
    "ltd_moments",
    "ltd_normal",
    "ltd_gamma",
    "ltd_stochastic_exact",
    "lead_time_variance_share",
    "undershoot_moments",
    "ltd_with_undershoot",
    "sample_lead_times",
]


@dataclass(frozen=True)
class LeadTimeSpec:
    """Per-period demand moments plus a lead-time distribution.

    ``lead_time_pmf`` maps an integer number of periods to a probability.  A
    deterministic lead time is just ``{L: 1.0}``.
    """

    demand_mean: float
    demand_sd: float
    lead_time_pmf: Mapping[int, float]

    def __post_init__(self) -> None:
        if self.demand_mean <= 0 or self.demand_sd <= 0:
            raise ValueError("demand mean and sd must be positive")
        total = sum(self.lead_time_pmf.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError("lead_time_pmf must sum to 1")
        if any(l < 0 for l in self.lead_time_pmf):
            raise ValueError("lead times must be non-negative")

    @classmethod
    def deterministic(cls, demand_mean: float, demand_sd: float, lead_time: int) -> "LeadTimeSpec":
        return cls(demand_mean, demand_sd, {int(lead_time): 1.0})

    @property
    def lead_time_mean(self) -> float:
        return float(sum(l * p for l, p in self.lead_time_pmf.items()))

    @property
    def lead_time_var(self) -> float:
        m = self.lead_time_mean
        return float(sum(p * (l - m) ** 2 for l, p in self.lead_time_pmf.items()))

    def with_offset(self, extra_periods: float) -> "LeadTimeSpec":
        """Shift every lead time by ``extra_periods`` (e.g. a review period).

        Only meaningful for integer offsets, which is the only case periodic
        review produces.
        """
        offset = int(extra_periods)
        if offset != extra_periods:
            raise ValueError("offset must be an integer number of periods")
        return LeadTimeSpec(
            self.demand_mean,
            self.demand_sd,
            {l + offset: p for l, p in self.lead_time_pmf.items()},
        )


def ltd_moments(spec: LeadTimeSpec) -> tuple[float, float]:
    """Mean and standard deviation of lead-time demand by variance convolution.

    ``E[D_L] = E[L] E[d]`` and ``Var(D_L) = E[L] Var(d) + E[d]^2 Var(L)``, which
    follows from the law of total variance applied to a random sum of iid terms.
    """
    el = spec.lead_time_mean
    vl = spec.lead_time_var
    mean = el * spec.demand_mean
    var = el * spec.demand_sd ** 2 + spec.demand_mean ** 2 * vl
    return float(mean), float(np.sqrt(var))


def lead_time_variance_share(spec: LeadTimeSpec) -> float:
    """Fraction of lead-time demand variance attributable to lead-time variability.

    A diagnostic worth putting in front of a planner: if this is 0.6, then more
    than half the safety stock exists to cover supplier unreliability, and the
    correct fix is a supplier conversation, not a bigger buffer.
    """
    el = spec.lead_time_mean
    demand_term = el * spec.demand_sd ** 2
    lt_term = spec.demand_mean ** 2 * spec.lead_time_var
    total = demand_term + lt_term
    return float(lt_term / total) if total > 0 else 0.0


def ltd_normal(spec: LeadTimeSpec) -> NormalLTD:
    """Normal approximation to lead-time demand using the convolved moments."""
    mean, sd = ltd_moments(spec)
    return NormalLTD(mu=mean, sigma=sd)


def ltd_gamma(spec: LeadTimeSpec) -> GammaLTD:
    """Gamma approximation to lead-time demand using the convolved moments.

    Preferred over the normal whenever the coefficient of variation of lead-time
    demand exceeds roughly 0.3, below which a normal will start putting
    non-trivial mass on negative demand.
    """
    mean, sd = ltd_moments(spec)
    return GammaLTD.from_moments(mean, sd)


def ltd_stochastic_exact(spec: LeadTimeSpec, family: str = "gamma") -> LeadTimeDemand:
    """Exact lead-time demand as a mixture over the lead-time pmf.

    For each possible lead time ``l``, demand over ``l`` periods is built from the
    per-period moments (exactly, in the gamma case, since the gamma is closed
    under convolution at fixed scale).  The mixture of those is the true
    lead-time demand distribution.

    A zero-length lead time contributes a degenerate mass at zero demand, which
    the mixture handles by giving it a tiny-variance component; that is a
    numerical convenience, not a modelling claim.
    """
    if family not in {"gamma", "normal"}:
        raise ValueError("family must be 'gamma' or 'normal'")
    components: list[LeadTimeDemand] = []
    weights: list[float] = []
    for l, p in sorted(spec.lead_time_pmf.items()):
        if p <= 0:
            continue
        mean = max(l, 1e-9) * spec.demand_mean
        sd = max(np.sqrt(max(l, 1e-9)) * spec.demand_sd, 1e-9)
        if family == "gamma":
            components.append(GammaLTD.from_moments(mean, sd))
        else:
            components.append(NormalLTD(mu=mean, sigma=sd))
        weights.append(float(p))
    if len(components) == 1:
        return components[0]
    return MixtureLTD(components, weights)


def undershoot_moments(
    txn_mean: float, txn_sd: float, family: str = "gamma"
) -> tuple[float, float]:
    """Mean and variance of the *undershoot* below a continuous-review trigger.

    A continuous-review ``(s, Q)`` policy does not order when the position equals
    ``s``; it orders when a transaction takes the position *through* ``s``, so the
    position at the moment of ordering is ``s - U`` for some ``U > 0``.  Standard
    reorder-point formulas ignore ``U`` entirely, which is why measured cycle
    service on lumpy items comes in below target even when the arithmetic is
    right.

    By renewal theory ``U`` has the equilibrium (stationary-excess) distribution
    of the transaction size, giving

        ``E[U]   = E[X^2] / (2 E[X])``
        ``E[U^2] = E[X^3] / (3 E[X])``

    The key consequence: ``E[U] = mu/2 + sigma^2/(2 mu)`` for a transaction of
    mean ``mu``.  The second term does **not** vanish as transactions get smaller
    at fixed aggregate demand - it is set by the coefficient of variation of the
    transaction size.  Undershoot is therefore a property of *how customers order*,
    not of how often the system polls stock, and no amount of moving from nightly
    batch to real-time inventory visibility makes it go away.

    The gamma closed form is used because it is exact and because transaction
    sizes are non-negative and right-skewed.  Reference: Zipkin (2000), Sec. 6.6;
    Silver, Pyke & Thomas (2016), Sec. 7.7.
    """
    if family != "gamma":
        raise ValueError("only the gamma transaction family is implemented")
    if txn_mean <= 0 or txn_sd <= 0:
        raise ValueError("transaction moments must be positive")
    theta = txn_sd ** 2 / txn_mean
    k = txn_mean / theta
    m1 = k * theta
    m2 = k * (k + 1.0) * theta ** 2
    m3 = k * (k + 1.0) * (k + 2.0) * theta ** 3
    eu = m2 / (2.0 * m1)
    eu2 = m3 / (3.0 * m1)
    return float(eu), float(max(eu2 - eu * eu, 0.0))


def ltd_with_undershoot(
    spec: LeadTimeSpec,
    txn_mean: float,
    txn_sd: float,
    family: str = "gamma",
) -> LeadTimeDemand:
    """Lead-time demand *plus* undershoot, as one distribution.

    The event a continuous-review reorder point actually has to cover is
    ``D_L + U > s``, not ``D_L > s``.  Convolving the two by moments and
    moment-matching a gamma is accurate enough that simulated cycle service
    lands on target; ignoring ``U`` leaves several points of service on the table
    on any item with a lumpy order profile.
    """
    mean, sd = ltd_moments(spec)
    eu, vu = undershoot_moments(txn_mean, txn_sd, family=family)
    total_mean = mean + eu
    total_sd = float(np.sqrt(sd ** 2 + vu))
    if family == "gamma":
        return GammaLTD.from_moments(total_mean, total_sd)
    return NormalLTD(mu=total_mean, sigma=total_sd)


def sample_lead_times(
    pmf: Mapping[int, float], size: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw integer lead times from a pmf, for simulation."""
    values: Sequence[int] = sorted(pmf)
    probs = np.array([pmf[v] for v in values], dtype=float)
    probs = probs / probs.sum()
    return rng.choice(np.asarray(values, dtype=int), size=size, p=probs)
