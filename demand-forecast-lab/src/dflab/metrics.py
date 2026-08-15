"""Forecast accuracy metrics, written for intermittent demand.

The metric set here is deliberately opinionated. MAPE is not implemented as a
headline metric because it is undefined the moment an actual is zero, which for
a spare part or a slow-moving SKU is most periods. Everything below is either
scale-free by construction (sMAPE), normalised by total volume (WAPE) or
normalised by an explicit in-sample benchmark (MASE, RMSSE).

References
----------
Hyndman, R.J. & Koehler, A.B. (2006). "Another look at measures of forecast
accuracy." *International Journal of Forecasting* 22(4), 679-688.

Kolassa, S. & Schutz, W. (2007). "Advantages of the MAD/Mean ratio over the
MAPE." *Foresight* 6, 40-43.

Gneiting, T. & Raftery, A.E. (2007). "Strictly proper scoring rules,
prediction, and estimation." *JASA* 102(477), 359-378.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "wape",
    "mae",
    "rmse",
    "smape",
    "bias",
    "percent_bias",
    "tracking_signal",
    "seasonal_naive_scale",
    "rmsse_scale",
    "mase",
    "rmsse",
    "pinball_loss",
    "coverage",
    "forecast_value_add",
    "evaluate",
]

_EPS = 1e-12


def _pair(actual, forecast) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float).ravel()
    f = np.asarray(forecast, dtype=float).ravel()
    if a.shape != f.shape:
        raise ValueError(f"shape mismatch: actual {a.shape} vs forecast {f.shape}")
    if a.size == 0:
        raise ValueError("empty evaluation window")
    return a, f


def mae(actual, forecast) -> float:
    a, f = _pair(actual, forecast)
    return float(np.mean(np.abs(a - f)))


def rmse(actual, forecast) -> float:
    a, f = _pair(actual, forecast)
    return float(np.sqrt(np.mean((a - f) ** 2)))


def wape(actual, forecast) -> float:
    """Weighted absolute percentage error = sum|e| / sum|actual|.

    Also known as the MAD/Mean ratio. This is the metric that survives contact
    with a real assortment: it is volume-weighted, defined whenever the window
    has any demand at all, and it aggregates across SKUs without a denominator
    explosion on the slow movers.
    """
    a, f = _pair(actual, forecast)
    denom = float(np.sum(np.abs(a)))
    if denom < _EPS:
        # No demand in the window. Any non-zero forecast is pure over-forecast;
        # a zero forecast is perfect. Return NaN so the caller can drop it
        # rather than silently averaging in a 0 or an infinity.
        return float("nan")
    return float(np.sum(np.abs(a - f)) / denom)


def smape(actual, forecast) -> float:
    """Symmetric MAPE in the 0-2 range (Hyndman & Koehler convention, /2 form).

    Returned on a 0-1 scale (i.e. already divided by 2) so it reads like WAPE.
    Periods where both actual and forecast are zero contribute 0, not NaN.
    """
    a, f = _pair(actual, forecast)
    denom = np.abs(a) + np.abs(f)
    ratio = np.zeros_like(a)
    nz = denom > _EPS
    ratio[nz] = np.abs(a[nz] - f[nz]) / denom[nz]
    return float(np.mean(ratio))


def bias(actual, forecast) -> float:
    """Mean signed error (forecast - actual). Positive means over-forecast."""
    a, f = _pair(actual, forecast)
    return float(np.mean(f - a))


def percent_bias(actual, forecast) -> float:
    """Total signed error as a fraction of total actual demand."""
    a, f = _pair(actual, forecast)
    denom = float(np.sum(np.abs(a)))
    if denom < _EPS:
        return float("nan")
    return float(np.sum(f - a) / denom)


def tracking_signal(actual, forecast) -> float:
    """Cumulative signed error over the window divided by its MAD.

    The classic planner's control statistic (Brown, 1959): it is the tripwire
    that a series has broken its model -- a step change, a discontinued item, a
    cannibalising replacement -- and needs a human rather than a re-fit.

    One caveat worth stating, because it is routinely got wrong when this is
    wired into an exception report: under an unbiased model the statistic is a
    random walk, so its spread grows like ``sqrt(n)`` in the window length
    (roughly ``1.25 * sqrt(n)`` for symmetric errors). The conventional +/-4
    limit is calibrated for a short review window; applied to a long one it
    fires constantly on perfectly healthy series.
    """
    a, f = _pair(actual, forecast)
    err = f - a
    mad = float(np.mean(np.abs(err)))
    if mad < _EPS:
        return 0.0
    return float(np.sum(err) / mad)


def seasonal_naive_scale(train, season_length: int = 1) -> float:
    """In-sample MAE of the seasonal naive method: mean|y_t - y_{t-m}|.

    This is the MASE denominator. Using ``season_length=1`` on a seasonal
    series makes MASE far too easy to beat: the naive-1 denominator absorbs the
    whole seasonal swing, so a model that merely reproduces seasonality scores
    well below 1 and looks skilful when it has added nothing.
    """
    y = np.asarray(train, dtype=float).ravel()
    m = int(season_length)
    if m < 1:
        raise ValueError("season_length must be >= 1")
    if y.size <= m:
        raise ValueError(
            f"need more than season_length={m} training points, got {y.size}"
        )
    d = np.abs(y[m:] - y[:-m])
    scale = float(np.mean(d))
    return scale if scale > _EPS else float("nan")


def rmsse_scale(train, season_length: int = 1) -> float:
    """In-sample RMS of the seasonal naive method (the RMSSE denominator)."""
    y = np.asarray(train, dtype=float).ravel()
    m = int(season_length)
    if m < 1:
        raise ValueError("season_length must be >= 1")
    if y.size <= m:
        raise ValueError(
            f"need more than season_length={m} training points, got {y.size}"
        )
    scale = float(np.sqrt(np.mean((y[m:] - y[:-m]) ** 2)))
    return scale if scale > _EPS else float("nan")


def mase(actual, forecast, scale: float) -> float:
    """Mean absolute scaled error given a precomputed in-sample ``scale``."""
    if not np.isfinite(scale) or scale <= _EPS:
        return float("nan")
    return mae(actual, forecast) / float(scale)


def rmsse(actual, forecast, scale: float) -> float:
    """Root mean squared scaled error given a precomputed in-sample ``scale``."""
    if not np.isfinite(scale) or scale <= _EPS:
        return float("nan")
    return rmse(actual, forecast) / float(scale)


def pinball_loss(actual, forecast, quantile: float) -> float:
    """Quantile (pinball) loss.

    ``quantile`` is the nominal level tau in (0, 1). The loss is minimised in
    expectation by the true tau-quantile of the predictive distribution, which
    is what makes it the right scoring rule for a safety-stock decision: you
    are not asking "what will demand be" but "what level will demand stay
    below tau of the time".
    """
    a, f = _pair(actual, forecast)
    tau = float(quantile)
    if not 0.0 < tau < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    diff = a - f
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def coverage(actual, forecast, quantile: float) -> float:
    """Empirical fraction of actuals at or below the quantile forecast.

    Compare against the nominal ``quantile``. Note the caveat for intermittent
    demand: forecasts are non-negative and a large share of actuals are exactly
    zero, so every zero period counts as covered. A distribution with an atom
    at zero will over-cover at the lower quantiles, and that is arithmetic, not
    miscalibration.
    """
    a, f = _pair(actual, forecast)
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    return float(np.mean(a <= f))


def forecast_value_add(metric_model: float, metric_baseline: float) -> float:
    """Relative improvement of a model over a baseline on a lower-is-better metric.

    Positive means the model added value. Reported as a fraction, so 0.12 means
    the model cut the error by 12% versus the baseline. Forecast Value Add
    (Gilliland, 2010) is the only number that answers the question a planning
    organisation actually has: is this model worth the maintenance?
    """
    if not np.isfinite(metric_baseline) or abs(metric_baseline) < _EPS:
        return float("nan")
    if not np.isfinite(metric_model):
        return float("nan")
    return float(1.0 - metric_model / metric_baseline)


def evaluate(
    actual,
    forecast,
    *,
    train=None,
    season_length: int = 1,
    quantile_forecasts: dict[float, np.ndarray] | None = None,
) -> dict[str, float]:
    """Compute the full metric bundle for one forecast window.

    ``train`` is required for the scaled metrics (MASE, RMSSE); without it
    those come back as NaN rather than silently falling back to a naive-1
    denominator.
    """
    a, f = _pair(actual, forecast)
    out: dict[str, float] = {
        "mae": mae(a, f),
        "rmse": rmse(a, f),
        "wape": wape(a, f),
        "smape": smape(a, f),
        "bias": bias(a, f),
        "pct_bias": percent_bias(a, f),
        "tracking_signal": tracking_signal(a, f),
        "abs_err": float(np.sum(np.abs(a - f))),
        "abs_actual": float(np.sum(np.abs(a))),
        "sq_err": float(np.sum((a - f) ** 2)),
        "signed_err": float(np.sum(f - a)),
        "n": float(a.size),
    }
    if train is not None:
        try:
            s_mae = seasonal_naive_scale(train, season_length)
            s_rms = rmsse_scale(train, season_length)
        except ValueError:
            s_mae = float("nan")
            s_rms = float("nan")
        out["mase"] = mase(a, f, s_mae)
        out["rmsse"] = rmsse(a, f, s_rms)
        out["mase_scale"] = s_mae
        out["rmsse_scale"] = s_rms
    else:
        out["mase"] = float("nan")
        out["rmsse"] = float("nan")
        out["mase_scale"] = float("nan")
        out["rmsse_scale"] = float("nan")

    if quantile_forecasts:
        losses = []
        for tau, qf in sorted(quantile_forecasts.items()):
            pl = pinball_loss(a, qf, tau)
            out[f"pinball_{tau:g}"] = pl
            out[f"coverage_{tau:g}"] = coverage(a, qf, tau)
            losses.append(pl)
        out["pinball_mean"] = float(np.mean(losses))
    return out
