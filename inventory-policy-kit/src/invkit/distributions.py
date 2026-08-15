"""Demand distributions with exact first-order loss functions.

Every safety-stock and service-level formula in this package is written against a
small distribution interface rather than against ``scipy.stats.norm``.  That is a
deliberate choice: the single most common modelling error in replenishment systems
is assuming lead-time demand is normal when it is not.  Forecast errors on
intermittent or promotion-driven items are right-skewed and fat-tailed, and a
normal fit silently understates the tail that safety stock is supposed to cover.

Three concrete distributions are provided:

``NormalLTD``
    The textbook default.  Closed-form loss function via the standard normal loss
    ``G(z) = phi(z) - z * (1 - Phi(z))``.

``GammaLTD``
    Non-negative, right-skewed, and *exactly* closed under convolution when the
    scale is shared: ``Gamma(k, theta)`` summed over ``L`` periods is
    ``Gamma(L*k, theta)``.  That makes it the right distribution for validating
    analytic formulas against simulation - there is no truncation-at-zero error to
    explain away, and the lead-time aggregation is exact rather than a
    moment-matched approximation.

``EmpiricalLTD``
    Ordered forecast-error sample.  Quantiles and loss are computed directly from
    the empirical measure, so no shape is assumed at all.

References
----------
Silver, E.A., Pyke, D.F. and Thomas, D.J. (2016) *Inventory and Production
Management in Supply Chains*, 4th ed., CRC Press - Ch. 6-7 for the loss function
and its role in fill-rate calculations.
Zipkin, P.H. (2000) *Foundations of Inventory Management*, McGraw-Hill - Ch. 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats

__all__ = [
    "LeadTimeDemand",
    "NormalLTD",
    "GammaLTD",
    "EmpiricalLTD",
    "MixtureLTD",
    "standard_normal_loss",
    "standard_normal_loss2",
    "inverse_standard_normal_loss",
]

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def standard_normal_loss(z: float | np.ndarray) -> np.ndarray:
    """First-order standard normal loss ``G(z) = E[(Z - z)^+]``.

    ``G(z) = phi(z) - z * (1 - Phi(z))``.  Strictly decreasing on the whole line,
    with ``G(z) -> -z`` as ``z -> -inf`` and ``G(z) -> 0`` as ``z -> +inf``.
    """
    z = np.asarray(z, dtype=float)
    return stats.norm.pdf(z) - z * stats.norm.sf(z)


def standard_normal_loss2(z: float | np.ndarray) -> np.ndarray:
    """Second-order standard normal loss ``G2(z) = integral_z^inf G(u) du``.

    ``G2(z) = 0.5 * [(z^2 + 1) * (1 - Phi(z)) - z * phi(z)]``.  Needed for the
    exact average backorder *level* of a continuous-review ``(s, Q)`` policy,
    where the inventory position is uniform on ``[s, s + Q]``.
    """
    z = np.asarray(z, dtype=float)
    return 0.5 * ((z * z + 1.0) * stats.norm.sf(z) - z * stats.norm.pdf(z))


def inverse_standard_normal_loss(target: float, lo: float = -40.0, hi: float = 40.0) -> float:
    """Solve ``G(z) = target`` for ``z``.

    This inversion is the whole game for fill-rate (type-2) service.  ``G`` has no
    closed-form inverse, but it is smooth and strictly decreasing, so bisection is
    both robust and fast enough that nobody should be using the tabulated
    approximations that still circulate in planning spreadsheets.
    """
    if target <= 0.0:
        raise ValueError("standard normal loss is strictly positive; target must be > 0")
    if target >= standard_normal_loss(lo):
        raise ValueError(f"target {target} is outside the representable range")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(standard_normal_loss(mid)) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


class LeadTimeDemand:
    """Interface for a lead-time demand distribution.

    Implementations must provide ``mean``, ``var``, ``cdf``, ``ppf`` and ``loss``.
    ``loss(x) = E[(D - x)^+]`` is the expected units short if stock ``x`` is on
    hand when the lead-time demand ``D`` arrives; it is the quantity that fill
    rate, expected backorders and shortage cost are all built from.
    """

    @property
    def mean(self) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def var(self) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def sd(self) -> float:
        return math.sqrt(self.var)

    def cdf(self, x: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def ppf(self, p: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def loss(self, x: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def _ppf_bisect(self, p: float, lo: float, hi: float) -> float:
        """Generic quantile by bisection, for mixtures with no closed form."""
        if not 0.0 < p < 1.0:
            raise ValueError("p must be in (0, 1)")
        while self.cdf(hi) < p:
            hi = hi * 2.0 + 1.0
        while self.cdf(lo) > p:
            lo = lo * 2.0 - 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.cdf(mid) < p:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-9 * max(1.0, abs(hi)):
                break
        return 0.5 * (lo + hi)


@dataclass(frozen=True)
class NormalLTD(LeadTimeDemand):
    """Normal lead-time demand with mean ``mu`` and standard deviation ``sigma``."""

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    @property
    def mean(self) -> float:
        return float(self.mu)

    @property
    def var(self) -> float:
        return float(self.sigma ** 2)

    def cdf(self, x: float) -> float:
        return float(stats.norm.cdf(x, loc=self.mu, scale=self.sigma))

    def ppf(self, p: float) -> float:
        return float(stats.norm.ppf(p, loc=self.mu, scale=self.sigma))

    def loss(self, x: float) -> float:
        z = (x - self.mu) / self.sigma
        return float(self.sigma * standard_normal_loss(z))

    def loss2(self, x: float) -> float:
        """Second-order loss ``E[(D - x)^{+2}] / 2``, in demand-squared units."""
        z = (x - self.mu) / self.sigma
        return float(self.sigma ** 2 * standard_normal_loss2(z))


@dataclass(frozen=True)
class GammaLTD(LeadTimeDemand):
    """Gamma lead-time demand, parameterised by shape ``k`` and scale ``theta``.

    Constructed most often through :meth:`from_moments`.  The reason this class
    exists is convolution: if per-period demand is ``Gamma(k, theta)`` then demand
    over ``L`` periods is exactly ``Gamma(L*k, theta)``.  No moment matching, no
    normal approximation, no negative demand.
    """

    k: float
    theta: float

    def __post_init__(self) -> None:
        if self.k <= 0 or self.theta <= 0:
            raise ValueError("gamma shape and scale must be positive")

    @classmethod
    def from_moments(cls, mean: float, sd: float) -> "GammaLTD":
        if mean <= 0 or sd <= 0:
            raise ValueError("mean and sd must be positive")
        theta = sd * sd / mean
        return cls(k=mean / theta, theta=theta)

    def convolve(self, n: float) -> "GammaLTD":
        """Distribution of the sum of ``n`` iid copies (exact for integer ``n``)."""
        if n <= 0:
            raise ValueError("n must be positive")
        return GammaLTD(k=self.k * n, theta=self.theta)

    @property
    def mean(self) -> float:
        return float(self.k * self.theta)

    @property
    def var(self) -> float:
        return float(self.k * self.theta ** 2)

    def cdf(self, x: float) -> float:
        return float(stats.gamma.cdf(x, a=self.k, scale=self.theta))

    def ppf(self, p: float) -> float:
        return float(stats.gamma.ppf(p, a=self.k, scale=self.theta))

    def loss(self, x: float) -> float:
        """``E[(D - x)^+] = k*theta*Sf(x; k+1, theta) - x*Sf(x; k, theta)``."""
        if x <= 0:
            return float(self.mean - x)
        sf_k = stats.gamma.sf(x, a=self.k, scale=self.theta)
        sf_k1 = stats.gamma.sf(x, a=self.k + 1.0, scale=self.theta)
        return float(self.k * self.theta * sf_k1 - x * sf_k)

    def rvs(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.gamma(shape=self.k, scale=self.theta, size=size)


class EmpiricalLTD(LeadTimeDemand):
    """Lead-time demand described by a sample, with no distributional assumption.

    Quantiles use the standard linear-interpolation definition; the loss function
    is the exact empirical mean of ``(d_i - x)^+``.  Use this when you have a
    history of forecast errors and no reason to believe they are symmetric.
    """

    def __init__(self, sample: Sequence[float] | np.ndarray):
        data = np.asarray(sample, dtype=float)
        if data.size < 2:
            raise ValueError("empirical distribution needs at least 2 observations")
        self._data = np.sort(data)

    @property
    def sample(self) -> np.ndarray:
        return self._data

    @property
    def mean(self) -> float:
        return float(self._data.mean())

    @property
    def var(self) -> float:
        return float(self._data.var(ddof=1))

    def cdf(self, x: float) -> float:
        return float(np.searchsorted(self._data, x, side="right") / self._data.size)

    def ppf(self, p: float) -> float:
        return float(np.quantile(self._data, p, method="linear"))

    def loss(self, x: float) -> float:
        return float(np.maximum(self._data - x, 0.0).mean())

    @classmethod
    def from_period_sample(
        cls,
        per_period: Sequence[float] | np.ndarray,
        periods: int,
        n_draws: int = 200_000,
        seed: int = 20260815,
    ) -> "EmpiricalLTD":
        """Aggregate a per-period sample to ``periods`` by iid bootstrap resampling.

        There is no analytic convolution for an arbitrary empirical measure.
        Resampling with a fixed seed keeps the result deterministic and
        reproducible, which matters more here than the last decimal of accuracy.
        """
        data = np.asarray(per_period, dtype=float)
        rng = np.random.default_rng(seed)
        draws = rng.choice(data, size=(n_draws, periods), replace=True).sum(axis=1)
        return cls(draws)


class MixtureLTD(LeadTimeDemand):
    """Finite mixture, used to make lead time itself a random variable.

    When lead time is discrete with pmf ``P(L = l)``, the lead-time demand
    distribution is the mixture of the demand-over-``l``-periods distributions.
    That is exact, unlike the usual variance-convolution shortcut, and the
    difference is not small when the lead-time distribution is bimodal - which it
    is for anything that alternates between ocean and air freight.
    """

    def __init__(self, components: Sequence[LeadTimeDemand], weights: Sequence[float]):
        w = np.asarray(weights, dtype=float)
        if len(components) != w.size:
            raise ValueError("components and weights must be the same length")
        if np.any(w < 0) or not math.isclose(float(w.sum()), 1.0, rel_tol=1e-9):
            raise ValueError("weights must be non-negative and sum to 1")
        self._components = list(components)
        self._weights = w

    @property
    def components(self) -> list[LeadTimeDemand]:
        return list(self._components)

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def mean(self) -> float:
        return float(sum(w * c.mean for w, c in zip(self._weights, self._components)))

    @property
    def var(self) -> float:
        m = self.mean
        second = sum(w * (c.var + c.mean ** 2) for w, c in zip(self._weights, self._components))
        return float(second - m * m)

    def cdf(self, x: float) -> float:
        return float(sum(w * c.cdf(x) for w, c in zip(self._weights, self._components)))

    def loss(self, x: float) -> float:
        return float(sum(w * c.loss(x) for w, c in zip(self._weights, self._components)))

    def ppf(self, p: float) -> float:
        lo = min(c.mean - 6.0 * c.sd for c in self._components)
        hi = max(c.mean + 6.0 * c.sd for c in self._components)
        return self._ppf_bisect(p, lo, hi)
