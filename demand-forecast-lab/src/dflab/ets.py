"""Exponential smoothing state-space recursions, implemented directly.

Nothing here wraps a library. The level/trend/seasonal recursions are written
out so the initialisation, the damping, and the multiplicative guard are all
visible and testable -- which is exactly where these models go wrong in
practice. Parameters are fitted by minimising in-sample one-step SSE with
``scipy.optimize.minimize`` (L-BFGS-B, box-constrained), with a multi-start to
avoid the flat local optima that plague seasonal fits.

Recursions (additive error form, following Hyndman et al.):

    SES     l_t = a*y_t + (1-a)*l_{t-1}
    Holt    l_t = a*y_t + (1-a)*(l_{t-1} + phi*b_{t-1})
            b_t = B*(l_t - l_{t-1}) + (1-B)*phi*b_{t-1}
    HW add  l_t = a*(y_t - s_{t-m}) + (1-a)*(l_{t-1} + phi*b_{t-1})
            s_t = g*(y_t - l_{t-1} - phi*b_{t-1}) + (1-g)*s_{t-m}
    HW mul  l_t = a*(y_t / s_{t-m}) + (1-a)*(l_{t-1} + phi*b_{t-1})
            s_t = g*(y_t / (l_{t-1} + phi*b_{t-1})) + (1-g)*s_{t-m}

References
----------
Holt, C.C. (1957/2004). "Forecasting seasonals and trends by exponentially
weighted moving averages." *International Journal of Forecasting* 20(1), 5-10.

Winters, P.R. (1960). "Forecasting sales by exponentially weighted moving
averages." *Management Science* 6(3), 324-342.

Hyndman, R.J., Koehler, A.B., Ord, J.K. & Snyder, R.D. (2008). *Forecasting
with Exponential Smoothing: The State Space Approach*. Springer.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .base import BaseLocalForecaster

__all__ = [
    "SimpleExponentialSmoothing",
    "HoltLinear",
    "HoltWinters",
    "ets_recursion",
]

_FLOOR = 1e-6


def ets_recursion(
    y: np.ndarray,
    alpha: float,
    beta: float | None,
    gamma: float | None,
    phi: float,
    m: int,
    seasonal: str,
    l0: float,
    b0: float,
    s0: np.ndarray | None,
) -> tuple[np.ndarray, float, float, np.ndarray | None]:
    """Run the smoothing recursion and return (fitted, l_T, b_T, s_state).

    ``fitted[t]`` is the one-step-ahead prediction of ``y[t]`` made from state
    at ``t-1``, so it is leakage-free by construction. ``s_state`` is the last
    ``m`` seasonal indices in chronological order.
    """
    n = y.size
    fitted = np.empty(n, dtype=float)
    level = float(l0)
    trend = float(b0)
    has_trend = beta is not None
    has_season = seasonal in ("add", "mul")
    b = float(beta) if has_trend else 0.0
    g = float(gamma) if has_season else 0.0

    if has_season:
        if s0 is None or len(s0) != m:
            raise ValueError("seasonal model requires m initial seasonal indices")
        season = np.array(s0, dtype=float)
    else:
        season = None

    for t in range(n):
        # ---- forecast for period t from state at t-1 --------------------
        trend_part = phi * trend if has_trend else 0.0
        base = level + trend_part
        if has_season:
            s_idx = t % m
            s_now = season[s_idx]
            pred = base * s_now if seasonal == "mul" else base + s_now
        else:
            s_now = 0.0
            pred = base
        fitted[t] = pred

        # ---- state update with the observation --------------------------
        prev_level = level
        if has_season:
            if seasonal == "mul":
                denom = s_now if abs(s_now) > _FLOOR else _FLOOR
                level = alpha * (y[t] / denom) + (1.0 - alpha) * base
                base_safe = base if abs(base) > _FLOOR else _FLOOR
                season[s_idx] = g * (y[t] / base_safe) + (1.0 - g) * s_now
            else:
                level = alpha * (y[t] - s_now) + (1.0 - alpha) * base
                season[s_idx] = g * (y[t] - base) + (1.0 - g) * s_now
        else:
            level = alpha * y[t] + (1.0 - alpha) * base

        if has_trend:
            trend = b * (level - prev_level) + (1.0 - b) * phi * trend

    return fitted, level, trend, season


def _sse(fitted: np.ndarray, y: np.ndarray, burn: int) -> float:
    err = y[burn:] - fitted[burn:]
    if not np.all(np.isfinite(err)):
        return 1e18
    return float(np.sum(err**2))


class _ETSBase(BaseLocalForecaster):
    """Shared fitting machinery for the exponential smoothing family."""

    seasonal = "none"
    m = 1

    def __init__(self, damped: bool = False, n_restarts: int = 3) -> None:
        super().__init__()
        self.damped = bool(damped)
        self.n_restarts = int(n_restarts)
        self.params_: dict[str, float] = {}
        self._state: tuple[float, float, np.ndarray | None] = (0.0, 0.0, None)

    # -- to be provided by subclasses -------------------------------------
    def _param_spec(self) -> list[tuple[str, tuple[float, float], float]]:
        raise NotImplementedError

    def _initial_state(self, y: np.ndarray) -> tuple[float, float, np.ndarray | None]:
        raise NotImplementedError

    # -- fitting -----------------------------------------------------------
    def _unpack(self, vec: np.ndarray) -> dict[str, float]:
        spec = self._param_spec()
        return {name: float(v) for (name, _, _), v in zip(spec, vec)}

    def _run(self, y: np.ndarray, p: dict[str, float]):
        l0, b0, s0 = self._initial_state(y)
        return ets_recursion(
            y,
            alpha=p["alpha"],
            beta=p.get("beta"),
            gamma=p.get("gamma"),
            phi=p.get("phi", 1.0),
            m=self.m,
            seasonal=self.seasonal,
            l0=l0,
            b0=b0,
            s0=s0,
        )

    def _fit(self) -> None:
        y = self.y
        spec = self._param_spec()
        bounds = [b for _, b, _ in spec]
        x0 = np.array([d for _, _, d in spec], dtype=float)
        burn = self.m if self.seasonal in ("add", "mul") else 1

        def objective(vec: np.ndarray) -> float:
            p = self._unpack(vec)
            try:
                fitted, *_ = self._run(y, p)
            except (ValueError, FloatingPointError):  # pragma: no cover
                return 1e18
            return _sse(fitted, y, burn)

        rng = np.random.default_rng(0)
        starts = [x0]
        for _ in range(max(0, self.n_restarts - 1)):
            starts.append(
                np.array(
                    [rng.uniform(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo))
                     for lo, hi in bounds]
                )
            )

        best_val = np.inf
        best_x = x0
        for start in starts:
            res = minimize(
                objective,
                start,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 200, "ftol": 1e-8},
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = float(res.fun)
                best_x = np.asarray(res.x, dtype=float)

        self.params_ = self._unpack(best_x)
        self.sse_ = best_val
        fitted, level, trend, season = self._run(y, self.params_)
        self._fitted = fitted
        self._state = (level, trend, season)

    # -- forecasting -------------------------------------------------------
    def _predict(self, h: int) -> np.ndarray:
        level, trend, season = self._state
        phi = self.params_.get("phi", 1.0)
        has_trend = "beta" in self.params_
        n = self.y.size
        out = np.empty(h, dtype=float)
        damp_sum = 0.0
        for i in range(1, h + 1):
            damp_sum += phi**i if has_trend else 0.0
            base = level + (damp_sum * trend if has_trend else 0.0)
            if self.seasonal in ("add", "mul"):
                s_val = season[(n + i - 1) % self.m]
                out[i - 1] = base * s_val if self.seasonal == "mul" else base + s_val
            else:
                out[i - 1] = base
        return np.maximum(out, 0.0)


class SimpleExponentialSmoothing(_ETSBase):
    """SES / ETS(A,N,N). One parameter, no trend, no seasonality."""

    seasonal = "none"

    def __init__(self, alpha: float | None = None, n_restarts: int = 2) -> None:
        super().__init__(damped=False, n_restarts=n_restarts)
        self.fixed_alpha = alpha
        self.name = "ses" if alpha is None else f"ses[a={alpha:g}]"

    def _param_spec(self):
        if self.fixed_alpha is not None:
            a = float(self.fixed_alpha)
            return [("alpha", (a, a), a)]
        return [("alpha", (0.01, 0.99), 0.2)]

    def _initial_state(self, y):
        k = min(y.size, 8)
        return float(np.mean(y[:k])), 0.0, None


class HoltLinear(_ETSBase):
    """Holt's linear trend, optionally damped: ETS(A,A,N) / ETS(A,Ad,N).

    Damping is on by default. Undamped linear trend extrapolated over a
    planning horizon is one of the most reliable ways to generate an absurd
    forecast for a slow-moving item, and Gardner & McKenzie (1985) has been the
    standard fix for forty years.
    """

    seasonal = "none"

    def __init__(self, damped: bool = True, n_restarts: int = 3) -> None:
        super().__init__(damped=damped, n_restarts=n_restarts)
        self.name = "holt_damped" if damped else "holt"

    def _param_spec(self):
        spec = [
            ("alpha", (0.01, 0.99), 0.3),
            ("beta", (0.001, 0.5), 0.05),
        ]
        if self.damped:
            spec.append(("phi", (0.80, 0.995), 0.95))
        return spec

    def _unpack(self, vec):
        p = super()._unpack(vec)
        p.setdefault("phi", 1.0)
        return p

    def _initial_state(self, y):
        k = min(y.size, 8)
        level = float(np.mean(y[:k]))
        if y.size >= 2 * k and k > 0:
            trend = float((np.mean(y[k : 2 * k]) - level) / k)
        else:
            trend = float((y[-1] - y[0]) / max(1, y.size - 1))
        return level, trend, None


class HoltWinters(_ETSBase):
    """Holt-Winters with additive or multiplicative seasonality.

    The multiplicative form divides by the seasonal index, so it is only
    defined on strictly positive data. A period with zero demand drives an
    index toward zero, the next division explodes, and the forecast leaves the
    solar system -- this is the single most common way a Holt-Winters engine
    embarrasses a planning team, and it always happens on the slow movers
    nobody was watching.

    Rather than clamp the recursion into something that is no longer the
    published model, this class detects the condition up front: if the history
    contains a non-positive value, it fits the **additive** form instead and
    sets ``degenerate_ = True``. Callers can count that flag to see how much of
    an assortment a multiplicative engine cannot actually serve.
    """

    def __init__(
        self,
        season_length: int = 52,
        seasonal_type: str = "add",
        damped: bool = True,
        n_restarts: int = 3,
    ) -> None:
        super().__init__(damped=damped, n_restarts=n_restarts)
        if seasonal_type not in ("add", "mul"):
            raise ValueError("seasonal_type must be 'add' or 'mul'")
        self.m = int(season_length)
        if self.m < 2:
            raise ValueError("season_length must be >= 2 for Holt-Winters")
        self.requested_seasonal = seasonal_type
        self.seasonal = seasonal_type
        self.degenerate_ = False
        self.name = f"hw_{seasonal_type}[m={self.m}]"

    def _fit(self) -> None:
        self.degenerate_ = self.requested_seasonal == "mul" and float(
            np.min(self.y)
        ) <= 0.0
        self.seasonal = "add" if self.degenerate_ else self.requested_seasonal
        super()._fit()

    def _param_spec(self):
        spec = [
            ("alpha", (0.01, 0.90), 0.2),
            ("beta", (0.001, 0.30), 0.02),
            ("gamma", (0.001, 0.60), 0.10),
        ]
        if self.damped:
            spec.append(("phi", (0.80, 0.995), 0.97))
        return spec

    def _unpack(self, vec):
        p = super()._unpack(vec)
        p.setdefault("phi", 1.0)
        return p

    def _initial_state(self, y):
        m = self.m
        n = y.size
        n_cycles = max(1, n // m)
        if n_cycles >= 2:
            usable = n_cycles * m
            block = y[:usable].reshape(n_cycles, m)
            cycle_means = block.mean(axis=1)
            level = float(cycle_means[0])
            trend = float((cycle_means[1] - cycle_means[0]) / m)
        else:
            block = y[: min(n, m)].reshape(1, -1)
            cycle_means = np.array([float(np.mean(y))])
            level = float(np.mean(y))
            trend = 0.0

        grand = float(np.mean(block))
        if self.seasonal == "mul":
            ratios = block / max(grand, _FLOOR)
            idx = ratios.mean(axis=0)
            idx = np.where(np.isfinite(idx) & (idx > 0.05), idx, 1.0)
            idx = idx / float(np.mean(idx))
            if block.shape[1] < m:
                idx = np.concatenate([idx, np.ones(m - idx.size)])
        else:
            idx = (block - grand).mean(axis=0)
            idx = idx - float(np.mean(idx))
            if block.shape[1] < m:
                idx = np.concatenate([idx, np.zeros(m - idx.size)])
        return level, trend, idx[:m]
