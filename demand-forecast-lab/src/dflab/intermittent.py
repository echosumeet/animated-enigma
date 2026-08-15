"""Intermittent demand methods: Croston, SBA, and TSB.

These are the methods that matter for the long tail of an assortment -- spare
parts, slow-moving finished goods, anything where most periods are zero. The
key structural idea, due to Croston, is to stop smoothing the raw series (which
drags the estimate toward zero right after a demand occurs and biases
replenishment timing) and instead smooth two separate processes: how big
demands are when they happen, and how often they happen.

Three known properties are implemented and tested here:

1. Croston's estimator is **positively biased**. Syntetos & Boylan showed the
   inversion ``E[z/p] != E[z]/E[p]`` inflates the forecast by roughly
   ``1/(1 - alpha/2)``; SBA applies the corresponding deflator.
2. Croston and SBA only update at demand epochs, so an item that has gone dead
   keeps its last forecast **forever**. This is the obsolescence failure mode
   that ruins service-level reporting on discontinued SKUs.
3. TSB replaces the interval process with a demand *probability* updated every
   period, so a dead item decays toward zero. That decay behaviour, not point
   accuracy, is usually the reason to pick TSB.

References
----------
Croston, J.D. (1972). "Forecasting and stock control for intermittent demands."
*Operational Research Quarterly* 23(3), 289-303.

Syntetos, A.A. & Boylan, J.E. (2005). "The accuracy of intermittent demand
estimates." *International Journal of Forecasting* 21(2), 303-314.

Teunter, R.H., Syntetos, A.A. & Babai, M.Z. (2011). "Intermittent demand:
Linking forecasting to inventory obsolescence." *European Journal of
Operational Research* 214(3), 606-615.
"""

from __future__ import annotations

import numpy as np

from .base import BaseLocalForecaster

__all__ = ["CrostonForecaster", "SBAForecaster", "TSBForecaster"]

_EPS = 1e-9


def _grid(lo: float, hi: float, n: int) -> np.ndarray:
    return np.linspace(lo, hi, n)


class _CrostonFamily(BaseLocalForecaster):
    """Croston's method and its Syntetos-Boylan bias-corrected variant.

    ``variant='croston'`` gives the 1972 estimator, ``variant='sba'`` multiplies
    by ``(1 - alpha/2)``.
    """

    def __init__(
        self,
        alpha: float | None = None,
        variant: str = "croston",
        grid_size: int = 19,
    ) -> None:
        super().__init__()
        if variant not in ("croston", "sba"):
            raise ValueError("variant must be 'croston' or 'sba'")
        self.variant = variant
        self.fixed_alpha = alpha
        self.grid_size = int(grid_size)
        self.name = variant if alpha is None else f"{variant}[a={alpha:g}]"
        self.alpha_ = float(alpha) if alpha is not None else 0.1
        self.z_ = 0.0
        self.p_ = 1.0

    # -- recursion ---------------------------------------------------------
    def _recursion(self, y: np.ndarray, alpha: float):
        """Return (per-period forecast series, final z, final p).

        ``forecast[t]`` is the rate forecast in force *entering* period t, so
        comparing it to ``y[t]`` gives a genuine one-step error.
        """
        n = y.size
        nz = np.flatnonzero(y > 0)
        forecast = np.full(n, np.nan)
        if nz.size == 0:
            return np.zeros(n), 0.0, float(max(n, 1))

        z = float(y[nz[0]])
        p = float(nz[0] + 1)
        gap = 0
        for t in range(n):
            forecast[t] = z / max(p, _EPS)
            gap += 1
            if t > nz[0] and y[t] > 0:
                z = alpha * y[t] + (1.0 - alpha) * z
                p = alpha * gap + (1.0 - alpha) * p
                gap = 0
            elif t == nz[0]:
                gap = 0
        return forecast, z, p

    def _deflator(self, alpha: float) -> float:
        return 1.0 - alpha / 2.0 if self.variant == "sba" else 1.0

    # -- fitting -----------------------------------------------------------
    def _fit(self) -> None:
        y = self.y
        if self.fixed_alpha is not None:
            best_alpha = float(self.fixed_alpha)
        else:
            best_alpha, best_sse = 0.1, np.inf
            for a in _grid(0.02, 0.4, self.grid_size):
                fc, _, _ = self._recursion(y, a)
                pred = fc * self._deflator(a)
                err = y[1:] - pred[1:]
                sse = float(np.sum(err**2))
                if sse < best_sse:
                    best_sse, best_alpha = sse, float(a)
        self.alpha_ = best_alpha
        fc, z, p = self._recursion(y, best_alpha)
        self.z_, self.p_ = float(z), float(p)
        self._fitted = fc * self._deflator(best_alpha)

    def _predict(self, h: int) -> np.ndarray:
        rate = self.z_ / max(self.p_, _EPS)
        return np.full(h, max(0.0, rate * self._deflator(self.alpha_)))


class CrostonForecaster(_CrostonFamily):
    """Croston (1972)."""

    def __init__(self, alpha: float | None = None, grid_size: int = 19) -> None:
        super().__init__(alpha=alpha, variant="croston", grid_size=grid_size)


class SBAForecaster(_CrostonFamily):
    """Syntetos-Boylan Approximation (2005): Croston x (1 - alpha/2)."""

    def __init__(self, alpha: float | None = None, grid_size: int = 19) -> None:
        super().__init__(alpha=alpha, variant="sba", grid_size=grid_size)


class TSBForecaster(BaseLocalForecaster):
    """Teunter-Syntetos-Babai (2011).

    Smooths demand *probability* every period and demand *size* only at demand
    epochs::

        if y_t > 0:  prob_t = prob_{t-1} + beta*(1 - prob_{t-1})
                     size_t = size_{t-1} + alpha*(y_t - size_{t-1})
        else:        prob_t = prob_{t-1} + beta*(0 - prob_{t-1})
                     size_t = size_{t-1}

        forecast = prob_t * size_t

    Because the probability decays on every zero period, a discontinued item's
    forecast tends to zero geometrically instead of freezing -- the property
    the paper was written to deliver.
    """

    def __init__(
        self,
        alpha: float | None = None,
        beta: float | None = None,
        grid_size: int = 11,
    ) -> None:
        super().__init__()
        self.fixed_alpha = alpha
        self.fixed_beta = beta
        self.grid_size = int(grid_size)
        self.name = "tsb"
        self.alpha_ = 0.1
        self.beta_ = 0.05
        self.prob_ = 0.0
        self.size_ = 0.0

    def _recursion(self, y: np.ndarray, alpha: float, beta: float):
        n = y.size
        nz = np.flatnonzero(y > 0)
        forecast = np.full(n, np.nan)
        if nz.size == 0:
            return np.zeros(n), 0.0, 0.0
        size = float(np.mean(y[nz]))
        prob = float(nz.size) / float(n)
        for t in range(n):
            forecast[t] = prob * size
            if y[t] > 0:
                size = size + alpha * (y[t] - size)
                prob = prob + beta * (1.0 - prob)
            else:
                prob = prob + beta * (0.0 - prob)
        return forecast, prob, size

    def _fit(self) -> None:
        y = self.y
        if self.fixed_alpha is not None and self.fixed_beta is not None:
            best = (float(self.fixed_alpha), float(self.fixed_beta))
        else:
            alphas = (
                [float(self.fixed_alpha)]
                if self.fixed_alpha is not None
                else _grid(0.02, 0.4, self.grid_size)
            )
            betas = (
                [float(self.fixed_beta)]
                if self.fixed_beta is not None
                else _grid(0.01, 0.3, self.grid_size)
            )
            best, best_sse = (0.1, 0.05), np.inf
            for a in alphas:
                for b in betas:
                    fc, _, _ = self._recursion(y, float(a), float(b))
                    sse = float(np.sum((y[1:] - fc[1:]) ** 2))
                    if sse < best_sse:
                        best_sse, best = sse, (float(a), float(b))
        self.alpha_, self.beta_ = best
        fc, prob, size = self._recursion(y, self.alpha_, self.beta_)
        self.prob_, self.size_ = float(prob), float(size)
        self._fitted = fc

    def _predict(self, h: int) -> np.ndarray:
        return np.full(h, max(0.0, self.prob_ * self.size_))
