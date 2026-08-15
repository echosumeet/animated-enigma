"""Point ETA model and split-conformal prediction intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

__all__ = ["ETAModel", "ConformalETA", "mae_by_horizon"]


@dataclass
class ETAModel:
    """Gradient-boosted point ETA, trained on absolute error.

    The target is transit hours. Absolute-error loss is deliberate: the transit
    distribution has a Pareto-ish tail, and a squared-error fit spends its
    capacity chasing a handful of disrupted shipments at the cost of the 95% of
    ordinary ones that the planning team actually acts on.
    """

    max_iter: int = 300
    learning_rate: float = 0.07
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 40
    seed: int = 7
    model_: HistGradientBoostingRegressor | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ETAModel":
        self.model_ = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=self.seed,
        ).fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("fit before predict")
        return self.model_.predict(np.asarray(X, dtype=float))


@dataclass
class ConformalETA:
    """Split-conformal intervals around a fitted point model.

    Calibration follows Lei et al. (2018): hold out a block the model never saw,
    take the ``ceil((n+1)(1-alpha))/n`` empirical quantile of the absolute
    residual, and add it symmetrically. The guarantee is marginal, so we also
    report coverage conditioned on transit horizon, where it is expected to be
    uneven -- short air lanes over-cover, long ocean lanes under-cover.

    Residuals are scaled by planned transit time before the quantile is taken.
    An unscaled interval is one width for the whole network, which is absurd
    when the network spans 20-hour road moves and 900-hour ocean moves.
    """

    model: ETAModel
    alpha: float = 0.1
    scaled: bool = True
    q_: float = 0.0

    def calibrate(self, X: pd.DataFrame, y: np.ndarray, scale: np.ndarray) -> "ConformalETA":
        resid = np.abs(np.asarray(y, dtype=float) - self.model.predict(X))
        s = self._scale(scale)
        n = resid.size
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_ = float(np.quantile(resid / s, level, method="higher"))
        return self

    def _scale(self, scale: np.ndarray) -> np.ndarray:
        if not self.scaled:
            return np.ones(len(scale))
        return np.clip(np.asarray(scale, dtype=float), 1.0, None) ** 0.5

    def predict_interval(self, X: pd.DataFrame, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        point = self.model.predict(X)
        half = self.q_ * self._scale(scale)
        return point, np.maximum(point - half, 0.0), point + half

    def coverage(self, X: pd.DataFrame, y: np.ndarray, scale: np.ndarray) -> dict[str, float]:
        _, lo, hi = self.predict_interval(X, scale)
        y = np.asarray(y, dtype=float)
        inside = (y >= lo) & (y <= hi)
        return {
            "nominal": 1 - self.alpha,
            "empirical": float(inside.mean()),
            "mean_width_h": float(np.mean(hi - lo)),
            "median_width_h": float(np.median(hi - lo)),
        }


def mae_by_horizon(y: np.ndarray, pred: np.ndarray, planned: np.ndarray) -> pd.DataFrame:
    """MAE and bias bucketed by planned transit time, the horizon that matters."""
    edges = [0, 48, 168, 336, 720, np.inf]
    labels = ["<2d", "2-7d", "7-14d", "14-30d", ">30d"]
    bucket = pd.cut(np.asarray(planned, dtype=float), bins=edges, labels=labels, right=False)
    frame = pd.DataFrame(
        {"bucket": bucket, "err": np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)}
    )
    out = frame.groupby("bucket", observed=True).agg(
        n=("err", "size"),
        mae_h=("err", lambda e: float(np.mean(np.abs(e)))),
        bias_h=("err", "mean"),
    )
    return out.reset_index()
