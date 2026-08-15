"""Baseline forecasters. These are the bar, not the strawman.

A benchmark that does not include seasonal naive is not a benchmark. In most
demand planning estates the incumbent statistical engine beats naive by a
smaller margin than anyone expects, and a meaningful fraction of SKUs are best
served by naive itself. Every method in this library is reported against these.

References
----------
Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and
Practice*, 3rd ed., Chapter 5 ("The forecaster's toolbox"). OTexts.
"""

from __future__ import annotations

import numpy as np

from .base import BaseLocalForecaster

__all__ = [
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "MovingAverageForecaster",
    "DriftForecaster",
    "MeanForecaster",
    "ZeroForecaster",
]


class NaiveForecaster(BaseLocalForecaster):
    """Random walk: every future period equals the last observation."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "naive"

    def _fit(self) -> None:
        y = self.y
        fitted = np.full_like(y, np.nan)
        fitted[1:] = y[:-1]
        self._fitted = fitted

    def _predict(self, h: int) -> np.ndarray:
        return np.full(h, self.y[-1])


class SeasonalNaiveForecaster(BaseLocalForecaster):
    """Repeat the most recent complete seasonal cycle.

    Falls back to the plain naive value for any lag the history cannot cover,
    which keeps it usable on short new-product series instead of erroring out.
    """

    def __init__(self, season_length: int = 52) -> None:
        super().__init__()
        self.season_length = int(season_length)
        if self.season_length < 1:
            raise ValueError("season_length must be >= 1")
        self.name = f"snaive[m={self.season_length}]"

    def _fit(self) -> None:
        y = self.y
        m = self.season_length
        fitted = np.full_like(y, np.nan)
        if y.size > m:
            fitted[m:] = y[:-m]
        self._fitted = fitted

    def _predict(self, h: int) -> np.ndarray:
        y = self.y
        m = self.season_length
        out = np.empty(h)
        for i in range(h):
            lag = m - (i % m)
            if lag <= y.size:
                out[i] = y[-lag]
            else:
                out[i] = y[-1]
        return out


class MovingAverageForecaster(BaseLocalForecaster):
    """Flat forecast at the mean of the last ``window`` observations."""

    def __init__(self, window: int = 8) -> None:
        super().__init__()
        self.window = int(window)
        if self.window < 1:
            raise ValueError("window must be >= 1")
        self.name = f"ma[w={self.window}]"

    def _fit(self) -> None:
        y = self.y
        w = min(self.window, y.size)
        fitted = np.full_like(y, np.nan)
        for t in range(1, y.size):
            lo = max(0, t - w)
            fitted[t] = float(np.mean(y[lo:t]))
        self._fitted = fitted

    def _predict(self, h: int) -> np.ndarray:
        w = min(self.window, self.y.size)
        return np.full(h, float(np.mean(self.y[-w:])))


class DriftForecaster(BaseLocalForecaster):
    """Random walk with drift: extrapolate the average per-period change."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "drift"
        self.slope_ = 0.0

    def _fit(self) -> None:
        y = self.y
        n = y.size
        self.slope_ = 0.0 if n < 2 else float((y[-1] - y[0]) / (n - 1))
        fitted = np.full_like(y, np.nan)
        if n >= 2:
            for t in range(1, n):
                local = (y[t - 1] - y[0]) / t if t >= 1 else 0.0
                fitted[t] = y[t - 1] + local
        self._fitted = fitted

    def _predict(self, h: int) -> np.ndarray:
        steps = np.arange(1, h + 1, dtype=float)
        return self.y[-1] + self.slope_ * steps


class MeanForecaster(BaseLocalForecaster):
    """Flat forecast at the full-history mean. The dumbest useful benchmark."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "mean"

    def _fit(self) -> None:
        y = self.y
        fitted = np.full_like(y, np.nan)
        for t in range(1, y.size):
            fitted[t] = float(np.mean(y[:t]))
        self._fitted = fitted

    def _predict(self, h: int) -> np.ndarray:
        return np.full(h, float(np.mean(self.y)))


class ZeroForecaster(BaseLocalForecaster):
    """Always forecast zero.

    Included on purpose. On a genuinely lumpy series with a long horizon, zero
    is a surprisingly strong point forecast under absolute-error metrics, and a
    method that cannot beat it has learned nothing. It is also the honest
    illustration of why point accuracy alone is the wrong objective for
    intermittent demand -- you would never actually plan to zero.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "zero"

    def _fit(self) -> None:
        self._fitted = np.zeros_like(self.y)

    def _predict(self, h: int) -> np.ndarray:
        return np.zeros(h)
