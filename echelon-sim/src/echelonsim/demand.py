"""Customer demand processes.

Three families, chosen because they separate mechanisms that get conflated:

* :class:`IIDNormal` -- stationary, memoryless. Any order variance above demand
  variance here is manufactured entirely by the replenishment policy, so it is
  the clean control case.
* :class:`AR1` -- positively autocorrelated demand. This is the setting of
  Lee, Padmanabhan & Whang (1997) and Chen, Drezner, Ryan & Simchi-Levi (2000):
  when demand is correlated, a rational forecaster *should* extrapolate, and the
  bullwhip effect is the mathematically correct response of an optimising agent
  to its own information. It is not a mistake, which is why exhorting planners
  to "stop over-reacting" never works.
* :class:`SeasonalTrend` -- deterministic seasonality plus noise, the case where
  a naive smoother is structurally mis-specified and amplification is a
  forecasting defect rather than an information one.

Shocks are applied as a wrapper (:class:`ShockOverlay`) rather than baked into
each process, so a disruption experiment reuses exactly the same base stream and
the difference between the shocked and unshocked runs is the shock alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "DemandProcess",
    "IIDNormal",
    "AR1",
    "SeasonalTrend",
    "ShockOverlay",
    "demand_path",
]


class DemandProcess(ABC):
    """Generates one non-negative demand observation per period."""

    #: Name of the random stream this process draws from.
    stream_name: str = "customer-demand"

    @abstractmethod
    def sample(self, period: int, rng: np.random.Generator) -> float:
        """Return demand for ``period``. Called once per period, in order."""

    @abstractmethod
    def stationary_moments(self) -> tuple:
        """``(mean, std)`` of the marginal distribution, for initialisation."""

    def reset(self) -> None:
        """Clear any internal state before a replication."""


@dataclass
class IIDNormal(DemandProcess):
    """Independent normal demand, truncated at zero."""

    mean: float = 100.0
    std: float = 20.0
    stream_name: str = "customer-demand"

    def sample(self, period: int, rng: np.random.Generator) -> float:
        return max(0.0, float(rng.normal(self.mean, self.std)))

    def stationary_moments(self) -> tuple:
        return (self.mean, self.std)


@dataclass
class AR1(DemandProcess):
    """``d_t = mu + rho * (d_{t-1} - mu) + eps_t``.

    ``std`` is the *marginal* standard deviation; the innovation standard
    deviation is derived as ``std * sqrt(1 - rho^2)`` so that changing ``rho``
    does not also change how variable demand is. Comparing bullwhip across
    correlation levels is only meaningful if the marginal variance is held
    fixed -- otherwise the ratio moves for the wrong reason.
    """

    mean: float = 100.0
    std: float = 20.0
    rho: float = 0.6
    stream_name: str = "customer-demand"
    _last: Optional[float] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not -1.0 < self.rho < 1.0:
            raise ValueError("AR(1) requires |rho| < 1 for stationarity")

    @property
    def innovation_std(self) -> float:
        return self.std * float(np.sqrt(1.0 - self.rho ** 2))

    def reset(self) -> None:
        self._last = None

    def sample(self, period: int, rng: np.random.Generator) -> float:
        if self._last is None:
            # Start from the stationary distribution so there is no
            # correlation-induced transient on top of the inventory transient.
            value = float(rng.normal(self.mean, self.std))
        else:
            value = self.mean + self.rho * (self._last - self.mean) + float(
                rng.normal(0.0, self.innovation_std)
            )
        self._last = value
        return max(0.0, value)

    def stationary_moments(self) -> tuple:
        return (self.mean, self.std)


@dataclass
class SeasonalTrend(DemandProcess):
    """Mean plus a sinusoidal season plus a linear drift plus noise."""

    mean: float = 100.0
    std: float = 15.0
    amplitude: float = 30.0
    period_length: int = 12
    drift: float = 0.0
    stream_name: str = "customer-demand"

    def sample(self, period: int, rng: np.random.Generator) -> float:
        season = self.amplitude * float(np.sin(2.0 * np.pi * period / self.period_length))
        level = self.mean + self.drift * period + season
        return max(0.0, level + float(rng.normal(0.0, self.std)))

    def stationary_moments(self) -> tuple:
        marginal = float(np.sqrt(self.std ** 2 + 0.5 * self.amplitude ** 2))
        return (self.mean, marginal)


@dataclass
class ShockOverlay(DemandProcess):
    """Wrap a base process and add a step or pulse over a window of periods.

    ``multiplier`` scales the base draw; ``offset`` adds to it. A demand spike
    is ``multiplier=2.0`` over five periods; a permanent level shift is a
    multiplier applied to the end of the horizon.
    """

    base: DemandProcess = field(default_factory=IIDNormal)
    start: int = 0
    duration: int = 0
    multiplier: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        self.stream_name = self.base.stream_name

    def reset(self) -> None:
        self.base.reset()

    def active(self, period: int) -> bool:
        return self.start <= period < self.start + self.duration

    def sample(self, period: int, rng: np.random.Generator) -> float:
        value = self.base.sample(period, rng)
        if self.active(period):
            value = value * self.multiplier + self.offset
        return max(0.0, value)

    def stationary_moments(self) -> tuple:
        return self.base.stationary_moments()


def demand_path(process: DemandProcess, periods: int, rng: np.random.Generator) -> np.ndarray:
    """Materialise a demand path. Used by tests and by analytic cross-checks."""
    process.reset()
    return np.array([process.sample(t, rng) for t in range(periods)], dtype=float)
