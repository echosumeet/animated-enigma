"""Delay-risk classifier with isotonic calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

__all__ = ["DelayRiskModel", "reliability_table", "brier_score", "expected_calibration_error"]


@dataclass
class DelayRiskModel:
    """P(late) with a post-hoc isotonic map fitted on a held-out later block.

    Boosted trees on a log-loss objective are usually close to calibrated in the
    middle of the range and badly off in the tails, which is precisely where an
    expedite decision gets made. Isotonic regression (Zadrozny & Elkan, 2002) is
    the right correction here because it is non-parametric and monotone, and the
    calibration block is large enough that overfitting it is not the binding
    concern. It is fitted on a *temporally later* block than the classifier, so
    the calibration also absorbs a slice of the drift.
    """

    max_iter: int = 250
    learning_rate: float = 0.07
    min_samples_leaf: int = 40
    seed: int = 11
    clf_: HistGradientBoostingClassifier | None = None
    iso_: IsotonicRegression | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "DelayRiskModel":
        self.clf_ = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=self.seed,
        ).fit(np.asarray(X, dtype=float), np.asarray(y).astype(int))
        return self

    def calibrate(self, X: pd.DataFrame, y: np.ndarray) -> "DelayRiskModel":
        raw = self.predict_raw(X)
        self.iso_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
            raw, np.asarray(y).astype(float)
        )
        return self

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        if self.clf_ is None:
            raise RuntimeError("fit before predict")
        return self.clf_.predict_proba(np.asarray(X, dtype=float))[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.predict_raw(X)
        return raw if self.iso_ is None else self.iso_.predict(raw)


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y).astype(float)) ** 2))


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 12) -> pd.DataFrame:
    """Equal-count reliability bins: predicted vs observed frequency."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(float)
    ranks = pd.qcut(pd.Series(p).rank(method="first"), n_bins, labels=False)
    frame = pd.DataFrame({"bin": ranks, "p": p, "y": y})
    out = frame.groupby("bin", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")
    )
    return out.reset_index(drop=True)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 12) -> float:
    """Weighted mean |predicted - observed| over equal-count bins."""
    tab = reliability_table(y, p, n_bins=n_bins)
    w = tab["n"].to_numpy(dtype=float)
    return float(np.sum(w * np.abs(tab["predicted"] - tab["observed"])) / w.sum())
