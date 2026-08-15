"""Rolling-origin backtesting and per-quadrant result aggregation.

A single holdout tells you almost nothing about a forecasting method. Demand
regimes shift, promotions cluster, and one unlucky quarter will reorder a
league table. Rolling origin evaluation (Tashman, 2000) re-fits at successive
cut-offs and averages over windows, which is both closer to how a planning
cycle actually runs and far harder to fool.

Two rules are enforced here rather than left to the caller:

1. **The model never sees past the cut-off.** Local models are re-fitted on
   ``y[:cut]``; the global model is re-fitted on ``panel[:, :cut]``. Nothing
   downstream of a cut-off can touch the training call.
2. **Scale denominators and demand classification come from the training
   window.** MASE and RMSSE scales are recomputed per window from
   ``y[:cut]``; quadrant labels likewise. Computing either on the full series
   is a subtle leak that changes the reported ranking.

Reference
---------
Tashman, L.J. (2000). "Out-of-sample tests of forecasting accuracy: an analysis
and review." *International Journal of Forecasting* 16(4), 437-450.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np

from . import metrics as M
from .base import Forecaster, GlobalForecaster, as_float_2d
from .classify import QUADRANTS, classify_series

__all__ = [
    "BacktestConfig",
    "WindowResult",
    "BacktestResult",
    "rolling_origins",
    "backtest",
]


@dataclass
class BacktestConfig:
    """Rolling-origin evaluation settings."""

    horizon: int = 13
    step: int = 13
    n_windows: int = 4
    min_train: int = 156
    season_length: int = 52
    quantiles: tuple[float, ...] = (0.5, 0.9, 0.95)


@dataclass
class WindowResult:
    """Per (method, series, window) metrics."""

    method: str
    series: int
    window: int
    cut: int
    quadrant: str
    metrics: dict[str, float]


@dataclass
class BacktestResult:
    """Everything a backtest produced, plus the aggregations worth reading."""

    rows: list[WindowResult]
    config: BacktestConfig
    cuts: list[int]
    quadrant_of: list[str]
    fit_seconds: dict[str, float] = field(default_factory=dict)
    forecasts: dict[str, np.ndarray] = field(default_factory=dict)

    # -- aggregation -------------------------------------------------------
    def methods(self) -> list[str]:
        seen: list[str] = []
        for r in self.rows:
            if r.method not in seen:
                seen.append(r.method)
        return seen

    def _pooled(self, rows: list[WindowResult]) -> dict[str, float]:
        """Volume-pooled WAPE plus simple means for the scaled metrics.

        WAPE is pooled as ``sum|e| / sum|a|`` across everything in the group,
        not averaged across series. Averaging per-series WAPE gives a
        1-unit-a-year SKU the same vote as the top seller, which is exactly the
        distortion WAPE exists to avoid.
        """
        keys = (
            "wape",
            "rmse",
            "pct_bias",
            "mase",
            "rmsse",
            "smape",
            "tracking_signal",
        )
        if not rows:
            # An empty group (a quadrant with no series) still gets a full row
            # of NaNs so downstream tables stay rectangular.
            out = {k: float("nan") for k in keys}
            out["n_obs"] = 0.0
            out["n_series_windows"] = 0.0
            return out
        abs_err = sum(r.metrics["abs_err"] for r in rows)
        abs_act = sum(r.metrics["abs_actual"] for r in rows)
        sq_err = sum(r.metrics["sq_err"] for r in rows)
        signed = sum(r.metrics["signed_err"] for r in rows)
        n = sum(r.metrics["n"] for r in rows)
        out = {
            "wape": abs_err / abs_act if abs_act > 0 else float("nan"),
            "rmse": float(np.sqrt(sq_err / n)) if n else float("nan"),
            "pct_bias": signed / abs_act if abs_act > 0 else float("nan"),
            "n_obs": float(n),
            "n_series_windows": float(len(rows)),
        }
        for key in ("mase", "rmsse", "smape", "tracking_signal"):
            vals = [r.metrics.get(key, float("nan")) for r in rows]
            vals = [v for v in vals if np.isfinite(v)]
            out[key] = float(np.mean(vals)) if vals else float("nan")
        pin = [
            r.metrics["pinball_mean"]
            for r in rows
            if np.isfinite(r.metrics.get("pinball_mean", float("nan")))
        ]
        if pin:
            out["pinball_mean"] = float(np.mean(pin))
        return out

    def overall(self) -> dict[str, dict[str, float]]:
        return {
            m: self._pooled([r for r in self.rows if r.method == m])
            for m in self.methods()
        }

    def by_quadrant(self) -> dict[str, dict[str, dict[str, float]]]:
        out: dict[str, dict[str, dict[str, float]]] = {}
        for m in self.methods():
            rows_m = [r for r in self.rows if r.method == m]
            out[m] = {
                q: self._pooled([r for r in rows_m if r.quadrant == q])
                for q in QUADRANTS
            }
        return out

    def quadrant_counts(self) -> dict[str, int]:
        counts = {q: 0 for q in QUADRANTS}
        for q in self.quadrant_of:
            counts[q] = counts.get(q, 0) + 1
        return counts

    def value_add(self, baseline: str, metric: str = "wape") -> dict[str, float]:
        """Forecast Value Add of every method versus ``baseline``."""
        agg = self.overall()
        if baseline not in agg:
            raise KeyError(f"baseline {baseline!r} not in results")
        base = agg[baseline][metric]
        return {
            m: M.forecast_value_add(agg[m][metric], base) for m in self.methods()
        }

    def best_by_quadrant(self, metric: str = "wape") -> dict[str, tuple[str, float]]:
        bq = self.by_quadrant()
        out: dict[str, tuple[str, float]] = {}
        for q in QUADRANTS:
            best, best_v = "", float("inf")
            for m in self.methods():
                v = bq[m].get(q, {}).get(metric, float("nan"))
                if np.isfinite(v) and v < best_v:
                    best, best_v = m, v
            if best:
                out[q] = (best, best_v)
        return out


def rolling_origins(
    T: int, horizon: int, step: int, n_windows: int, min_train: int
) -> list[int]:
    """Cut-off indices for rolling-origin evaluation.

    Returns the training lengths ``cut`` such that the test block is
    ``y[cut:cut + horizon]``. Windows are laid out backwards from the end of
    the series so the most recent data is always evaluated.
    """
    if horizon < 1 or step < 1 or n_windows < 1:
        raise ValueError("horizon, step and n_windows must all be >= 1")
    cuts: list[int] = []
    cut = T - horizon
    for _ in range(n_windows):
        if cut < min_train:
            break
        cuts.append(int(cut))
        cut -= step
    if not cuts:
        raise ValueError(
            f"series of length {T} cannot support {n_windows} windows of "
            f"horizon {horizon} with min_train {min_train}"
        )
    return sorted(cuts)


def _quantile_forecasts(
    quantile_models, cut: int, panel: np.ndarray, horizon: int
) -> dict[float, np.ndarray]:
    out: dict[float, np.ndarray] = {}
    for tau, model in quantile_models:
        mdl = copy.deepcopy(model)
        mdl.fit_panel(panel[:, :cut])
        out[float(tau)] = mdl.predict_panel(horizon)
    return out


def backtest(
    panel,
    local_models: list[Forecaster],
    config: BacktestConfig | None = None,
    *,
    global_models: list[GlobalForecaster] | None = None,
    quantile_models: list[tuple[float, GlobalForecaster]] | None = None,
    verbose: bool = False,
) -> BacktestResult:
    """Run a rolling-origin backtest over a panel of series.

    Parameters
    ----------
    panel
        ``(n_series, T)`` demand history.
    local_models
        forecasters implementing ``fit(y) / predict(h)``; deep-copied per fit.
    global_models
        forecasters implementing ``fit_panel(P) / predict_panel(h)``.
    quantile_models
        list of ``(tau, global_model)`` pairs; their pinball loss is attached
        to the method named by the model itself.
    """
    P = as_float_2d(panel)
    cfg = config or BacktestConfig()
    n_series, T = P.shape
    cuts = rolling_origins(T, cfg.horizon, cfg.step, cfg.n_windows, cfg.min_train)
    h = cfg.horizon
    m = cfg.season_length

    rows: list[WindowResult] = []
    fit_seconds: dict[str, float] = {}
    forecasts: dict[str, np.ndarray] = {}

    # Quadrant label per series, taken from the earliest training window so
    # every window of a given series is reported in one consistent bucket.
    label_cut = cuts[0]
    quadrant_of = [classify_series(P[i, :label_cut]).quadrant for i in range(n_series)]

    global_models = list(global_models or [])
    quantile_models = list(quantile_models or [])

    for w, cut in enumerate(cuts):
        train = P[:, :cut]
        actual = P[:, cut : cut + h]

        qf_by_series: dict[float, np.ndarray] = {}
        if quantile_models:
            t0 = time.perf_counter()
            qf_by_series = _quantile_forecasts(quantile_models, cut, P, h)
            fit_seconds["__quantiles__"] = fit_seconds.get("__quantiles__", 0.0) + (
                time.perf_counter() - t0
            )

        # ---- global (cross-learning) models -----------------------------
        for gm in global_models:
            t0 = time.perf_counter()
            mdl = copy.deepcopy(gm)
            mdl.fit_panel(train)
            pred = mdl.predict_panel(h)
            fit_seconds[mdl.name] = fit_seconds.get(mdl.name, 0.0) + (
                time.perf_counter() - t0
            )
            forecasts.setdefault(mdl.name, np.zeros((len(cuts), n_series, h)))
            forecasts[mdl.name][w] = pred
            # Quantile forecasts are attached to the first global model, which
            # is the one they are the predictive distribution of.
            attach = bool(qf_by_series) and gm is global_models[0]
            for i in range(n_series):
                qfs = (
                    {tau: qf_by_series[tau][i] for tau in qf_by_series}
                    if attach
                    else None
                )
                met = M.evaluate(
                    actual[i],
                    pred[i],
                    train=train[i],
                    season_length=m,
                    quantile_forecasts=qfs,
                )
                rows.append(
                    WindowResult(mdl.name, i, w, cut, quadrant_of[i], met)
                )

        # ---- local models ------------------------------------------------
        for lm in local_models:
            t0 = time.perf_counter()
            pred_all = np.zeros((n_series, h))
            for i in range(n_series):
                mdl = copy.deepcopy(lm)
                try:
                    mdl.fit(train[i])
                    pred_all[i] = mdl.predict(h)
                except (ValueError, np.linalg.LinAlgError):
                    # Degenerate series (all zero, too short for the season).
                    # Fall back to the training mean rather than dropping the
                    # series, so every method is scored on the same population.
                    pred_all[i] = float(np.mean(train[i]))
            fit_seconds[lm.name] = fit_seconds.get(lm.name, 0.0) + (
                time.perf_counter() - t0
            )
            forecasts.setdefault(lm.name, np.zeros((len(cuts), n_series, h)))
            forecasts[lm.name][w] = pred_all
            for i in range(n_series):
                met = M.evaluate(
                    actual[i], pred_all[i], train=train[i], season_length=m
                )
                rows.append(WindowResult(lm.name, i, w, cut, quadrant_of[i], met))

        if verbose:
            print(f"  window {w + 1}/{len(cuts)} cut={cut} done", flush=True)

    return BacktestResult(
        rows=rows,
        config=cfg,
        cuts=cuts,
        quadrant_of=quadrant_of,
        fit_seconds=fit_seconds,
        forecasts=forecasts,
    )
