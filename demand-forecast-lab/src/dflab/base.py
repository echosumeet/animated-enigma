"""Core forecaster interfaces.

Every method in this library implements one of two contracts:

``Forecaster``
    A *local* model: fitted independently on a single history ``y`` and asked
    for ``h`` steps ahead. Classical statistical methods live here.

``GlobalForecaster``
    A *cross-learning* model: fitted once on a panel of shape
    ``(n_series, T)`` and asked for ``h`` steps ahead for every series at
    once. Feature-based ML methods live here.

Keeping the two contracts explicit matters. A very common way to get a
forecasting benchmark quietly wrong is to hand a global model the same
train/test split loop written for local models and let it see the future of
series *B* while predicting series *A*. The backtester dispatches on the
``is_global`` flag so the split boundary is enforced in exactly one place.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["Forecaster", "GlobalForecaster", "as_float_1d", "as_float_2d"]


def as_float_1d(y) -> np.ndarray:
    """Coerce to a finite 1-D float array."""
    arr = np.asarray(y, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("empty series")
    if not np.all(np.isfinite(arr)):
        raise ValueError("series contains non-finite values")
    return arr


def as_float_2d(panel) -> np.ndarray:
    """Coerce to a finite 2-D float array of shape (n_series, T)."""
    arr = np.asarray(panel, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D panel, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("empty panel")
    if not np.all(np.isfinite(arr)):
        raise ValueError("panel contains non-finite values")
    return arr


@runtime_checkable
class Forecaster(Protocol):
    """Local forecaster contract."""

    name: str
    is_global: bool

    def fit(self, y: np.ndarray) -> "Forecaster": ...

    def predict(self, h: int) -> np.ndarray: ...


@runtime_checkable
class GlobalForecaster(Protocol):
    """Cross-learning forecaster contract."""

    name: str
    is_global: bool

    def fit_panel(self, panel: np.ndarray) -> "GlobalForecaster": ...

    def predict_panel(self, h: int) -> np.ndarray: ...


class BaseLocalForecaster:
    """Small shared implementation for local models.

    Handles the bookkeeping every local method needs: store the history,
    validate the horizon, and expose fitted values so residual-based
    reconciliation and scale estimation have something to work with.
    """

    name = "base"
    is_global = False

    def __init__(self) -> None:
        self._y: np.ndarray | None = None
        self._fitted: np.ndarray | None = None

    # -- lifecycle ---------------------------------------------------------
    def fit(self, y) -> "BaseLocalForecaster":
        self._y = as_float_1d(y)
        self._fit()
        return self

    def _fit(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def predict(self, h: int) -> np.ndarray:
        if self._y is None:
            raise RuntimeError(f"{self.name}: predict() called before fit()")
        h = int(h)
        if h < 1:
            raise ValueError("horizon must be >= 1")
        out = np.asarray(self._predict(h), dtype=float).ravel()
        if out.shape != (h,):
            raise RuntimeError(f"{self.name}: expected {h} points, got {out.shape}")
        return out

    def _predict(self, h: int) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    # -- introspection -----------------------------------------------------
    @property
    def y(self) -> np.ndarray:
        if self._y is None:
            raise RuntimeError(f"{self.name}: not fitted")
        return self._y

    @property
    def fitted_values(self) -> np.ndarray:
        """One-step-ahead in-sample predictions, NaN where undefined."""
        if self._fitted is None:
            raise RuntimeError(f"{self.name}: no fitted values available")
        return self._fitted

    def residuals(self) -> np.ndarray:
        """In-sample one-step residuals (NaN where the model had no history)."""
        return self.y - self.fitted_values

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r}>"
