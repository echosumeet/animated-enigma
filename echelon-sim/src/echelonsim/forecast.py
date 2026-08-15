"""Forecasters used inside the replenishment loop.

These are deliberately the *simple* forecasters, because the bullwhip question
is not "which forecaster is most accurate" -- it is "what does the act of
forecasting do to the order stream". Chen, Drezner, Ryan & Simchi-Levi (2000)
show that for an order-up-to policy with lead time ``L`` and an ``p``-period
moving average, the order variance satisfies

    Var(q)/Var(d) >= 1 + 2L/p + 2L^2/p^2

and for exponential smoothing with parameter ``alpha``

    Var(q)/Var(d) >= 1 + 2*alpha/(2 - alpha) + 2*alpha^2*L^2/(2 - alpha)

with equality under i.i.d. demand and unconstrained (signed) orders. Both bounds
are implemented in :mod:`echelonsim.bullwhip` and the simulator is checked
against them, which is the strongest validation available for an engine like
this: an independent closed form for a special case.

:class:`Oracle` is the ablation control. It returns the true parameters of the
demand process and never updates, so switching it in removes the
demand-signal-processing mechanism while leaving batching and lead time intact.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np

__all__ = [
    "Forecaster",
    "MovingAverage",
    "ExponentialSmoothing",
    "DampedTrend",
    "Oracle",
    "Forecast",
]


@dataclass(frozen=True)
class Forecast:
    """Per-period demand forecast: level and dispersion."""

    mean: float
    std: float
    observations: int


class Forecaster(ABC):
    """Consumes one demand observation per period, emits a per-period forecast."""

    @abstractmethod
    def update(self, observation: float) -> None:
        ...

    @abstractmethod
    def forecast(self) -> Forecast:
        ...

    @abstractmethod
    def reset(self, initial_mean: float, initial_std: float) -> None:
        ...

    def clone(self) -> "Forecaster":
        import copy

        return copy.deepcopy(self)


@dataclass
class MovingAverage(Forecaster):
    """Simple ``p``-period moving average with a moving-window standard deviation."""

    window: int = 10
    _values: Deque[float] = field(default_factory=deque, repr=False)
    _fallback_mean: float = field(default=0.0, repr=False)
    _fallback_std: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be >= 1")
        self._values = deque(maxlen=self.window)

    def reset(self, initial_mean: float, initial_std: float) -> None:
        self._values = deque(maxlen=self.window)
        self._fallback_mean = float(initial_mean)
        self._fallback_std = float(initial_std)

    def update(self, observation: float) -> None:
        self._values.append(float(observation))

    def forecast(self) -> Forecast:
        n = len(self._values)
        if n == 0:
            return Forecast(self._fallback_mean, self._fallback_std, 0)
        arr = np.fromiter(self._values, dtype=float, count=n)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if n > 1 else self._fallback_std
        return Forecast(mean, std, n)


@dataclass
class ExponentialSmoothing(Forecaster):
    """Single exponential smoothing of the level, with smoothed mean absolute error.

    Dispersion is tracked as a smoothed MAD converted with the normal-case
    factor ``sigma ~ 1.25 * MAD``. Practitioners use MAD rather than a rolling
    variance because it is what every planning system exposes and because it is
    far less sensitive to the single outlier that a promotion or a data error
    injects -- a rolling variance over a 10-period window doubles the safety
    stock for 10 periods after one bad record.
    """

    alpha: float = 0.3
    beta: float = 0.2  # smoothing constant for the error measure
    _level: float = field(default=0.0, repr=False)
    _mad: float = field(default=0.0, repr=False)
    _n: int = field(default=0, repr=False)

    MAD_TO_SIGMA = 1.2533141373155003  # sqrt(pi/2)

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")

    def reset(self, initial_mean: float, initial_std: float) -> None:
        self._level = float(initial_mean)
        self._mad = float(initial_std) / self.MAD_TO_SIGMA
        self._n = 0

    def update(self, observation: float) -> None:
        error = abs(float(observation) - self._level)
        self._mad = self.beta * error + (1.0 - self.beta) * self._mad
        self._level = self.alpha * float(observation) + (1.0 - self.alpha) * self._level
        self._n += 1

    def forecast(self) -> Forecast:
        return Forecast(max(0.0, self._level), max(0.0, self._mad * self.MAD_TO_SIGMA), self._n)


@dataclass
class DampedTrend(Forecaster):
    """Holt's linear method with a damping parameter ``phi``.

    Included because trend extrapolation is the sharpest amplifier in the set:
    an undamped trend projected over a long protection interval turns a two-period
    demand uptick into a large order, and the correction next period turns it into
    a cancellation. ``phi < 1`` is the standard fix (Gardner & McKenzie 1985).
    """

    alpha: float = 0.3
    beta_trend: float = 0.1
    phi: float = 0.85
    gamma_mad: float = 0.2
    _level: float = field(default=0.0, repr=False)
    _trend: float = field(default=0.0, repr=False)
    _mad: float = field(default=0.0, repr=False)
    _n: int = field(default=0, repr=False)

    MAD_TO_SIGMA = 1.2533141373155003

    def reset(self, initial_mean: float, initial_std: float) -> None:
        self._level = float(initial_mean)
        self._trend = 0.0
        self._mad = float(initial_std) / self.MAD_TO_SIGMA
        self._n = 0

    def update(self, observation: float) -> None:
        prediction = self._level + self.phi * self._trend
        error = abs(float(observation) - prediction)
        self._mad = self.gamma_mad * error + (1.0 - self.gamma_mad) * self._mad
        new_level = self.alpha * float(observation) + (1.0 - self.alpha) * prediction
        self._trend = (
            self.beta_trend * (new_level - self._level)
            + (1.0 - self.beta_trend) * self.phi * self._trend
        )
        self._level = new_level
        self._n += 1

    def forecast(self) -> Forecast:
        # One-step-ahead level; the policy multiplies by the protection interval,
        # so the trend is folded in as a per-period increment.
        point = self._level + self.phi * self._trend
        return Forecast(max(0.0, point), max(0.0, self._mad * self.MAD_TO_SIGMA), self._n)


@dataclass
class Oracle(Forecaster):
    """Knows the true stationary moments and never updates.

    The ablation control for demand signal processing. Any bullwhip that
    survives an Oracle forecaster is caused by batching, lead time or
    rationing -- not by forecast updating.
    """

    mean: Optional[float] = None
    std: Optional[float] = None

    def reset(self, initial_mean: float, initial_std: float) -> None:
        if self.mean is None:
            self.mean = float(initial_mean)
        if self.std is None:
            self.std = float(initial_std)

    def update(self, observation: float) -> None:
        return

    def forecast(self) -> Forecast:
        return Forecast(float(self.mean or 0.0), float(self.std or 0.0), 0)
