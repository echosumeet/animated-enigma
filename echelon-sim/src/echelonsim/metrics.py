"""Output analysis: the part that decides whether a simulation result means anything.

Three problems have to be solved before a simulated number can be quoted:

**Initial transient.** A simulation that starts with four periods of stock and an
empty pipeline is not in steady state, and averaging over the transient biases
every inventory and service statistic. :func:`mser5_truncation` implements MSER-5
(White 1997; Franklin & White 2008), which picks the truncation point minimising
the estimated standard error of the truncated mean. It is the best-performing of
the practical rules and, unlike Welch's method, needs no eyeballing.

**Autocorrelation within a run.** Consecutive periods of an inventory series are
strongly dependent, so the naive standard error ``s/sqrt(n)`` understates the
true one badly -- routinely by a factor of three or more, which turns a
"significant" difference into noise. :func:`batch_means_ci` blocks the series
into a small number of large batches whose means are approximately independent
(Schmeiser 1982 recommends 10-30 batches), and reports the lag-1 correlation of
those batch means so the assumption is checked rather than asserted.

**Comparing scenarios.** Under common random numbers the replications are paired,
so :func:`paired_difference_ci` is both correct and far tighter than an unpaired
interval. Using an unpaired interval on CRN data is not conservative -- it just
throws away the variance reduction you paid for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

__all__ = [
    "ConfidenceInterval",
    "mean_ci",
    "batch_means_ci",
    "mser5_truncation",
    "welch_moving_average",
    "paired_difference_ci",
    "lag1_autocorrelation",
    "bullwhip_ratio",
    "variance_ratio_ci",
]

_EPS = 1e-12


@dataclass(frozen=True)
class ConfidenceInterval:
    mean: float
    half_width: float
    low: float
    high: float
    n: int
    note: str = ""

    @property
    def relative_precision(self) -> float:
        """Half-width as a fraction of the mean -- the usual stopping rule."""
        return abs(self.half_width / self.mean) if abs(self.mean) > _EPS else float("inf")

    def format(self, digits: int = 3) -> str:
        return f"{self.mean:.{digits}f} +/- {self.half_width:.{digits}f}"

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.format()


def mean_ci(values: Sequence[float], confidence: float = 0.95) -> ConfidenceInterval:
    """Student-t interval for the mean of independent observations."""
    array = np.asarray(list(values), dtype=float)
    n = array.size
    if n < 2:
        value = float(array[0]) if n else float("nan")
        return ConfidenceInterval(value, float("nan"), value, value, n, "n<2")
    mean = float(array.mean())
    stderr = float(array.std(ddof=1) / np.sqrt(n))
    half = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1) * stderr)
    return ConfidenceInterval(mean, half, mean - half, mean + half, n)


def lag1_autocorrelation(values: Sequence[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size < 3:
        return float("nan")
    centred = array - array.mean()
    denominator = float(np.dot(centred, centred))
    if denominator <= _EPS:
        return 0.0
    return float(np.dot(centred[:-1], centred[1:]) / denominator)


def batch_means_ci(
    series: Sequence[float],
    n_batches: int = 20,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Batch-means confidence interval for the mean of one long, correlated run.

    Uses ``n_batches`` contiguous batches of equal length, discarding the
    remainder at the *front* of the series (the oldest, least steady-state data)
    rather than at the back.

    The returned ``note`` flags a lag-1 batch-mean correlation above 0.2, the
    conventional warning threshold: above it the batches are too short and the
    interval is too narrow, and the honest response is a longer run rather than
    more batches.
    """
    array = np.asarray(list(series), dtype=float)
    n = array.size
    if n_batches < 2:
        raise ValueError("need at least 2 batches")
    batch_size = n // n_batches
    if batch_size < 2:
        raise ValueError(
            f"series of {n} periods cannot support {n_batches} batches of >= 2 periods"
        )
    trimmed = array[n - batch_size * n_batches:]
    batches = trimmed.reshape(n_batches, batch_size).mean(axis=1)
    interval = mean_ci(batches, confidence)
    rho = lag1_autocorrelation(batches)
    note = ""
    if abs(rho) > 0.2:
        note = f"batch-mean lag-1 correlation {rho:.2f} > 0.2: batches too short"
    return ConfidenceInterval(
        interval.mean, interval.half_width, interval.low, interval.high, n_batches, note
    )


def mser5_truncation(
    series: Sequence[float],
    group: int = 5,
    max_fraction: float = 0.5,
) -> int:
    """MSER-5 warm-up truncation point, in periods.

    The series is first averaged into non-overlapping groups of ``group``
    periods (this is the "-5"; it damps the high-frequency noise that otherwise
    dominates the statistic). For each candidate truncation ``d`` the statistic

        ``MSER(d) = Var(Y_{d+1..n}) / (n - d)``

    estimates the squared standard error of the truncated mean; the minimising
    ``d`` is returned, scaled back to periods.

    Candidates are limited to the first half of the series. Without that limit
    the statistic is trivially minimised by keeping two observations, which is a
    known degeneracy rather than a warm-up estimate.
    """
    array = np.asarray(list(series), dtype=float)
    n_groups = array.size // group
    if n_groups < 4:
        return 0
    grouped = array[: n_groups * group].reshape(n_groups, group).mean(axis=1)
    best_d, best_value = 0, np.inf
    limit = max(1, int(n_groups * max_fraction))
    for d in range(limit):
        tail = grouped[d:]
        m = tail.size
        if m < 2:
            break
        value = float(tail.var(ddof=0) / m)
        if value < best_value:
            best_value, best_d = value, d
    return int(best_d * group)


def welch_moving_average(
    replications: Sequence[Sequence[float]],
    window: int = 10,
) -> np.ndarray:
    """Welch's procedure: average across replications, then smooth.

    Returned for plotting only. MSER-5 is what the code actually truncates on;
    Welch's plot is how you convince a sceptical reader that the truncation point
    is sane, and the two disagreeing is a signal worth chasing.
    """
    matrix = np.asarray([list(r) for r in replications], dtype=float)
    if matrix.ndim != 2:
        raise ValueError("replications must be rectangular")
    averaged = matrix.mean(axis=0)
    n = averaged.size
    out = np.empty(n, dtype=float)
    for i in range(n):
        half = min(window, i, n - i - 1)
        out[i] = averaged[i - half: i + half + 1].mean()
    return out


def paired_difference_ci(
    treatment: Sequence[float],
    control: Sequence[float],
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Paired-t interval on ``treatment - control``.

    Valid precisely because common random numbers make replication ``i`` of the
    two scenarios comparable. If the two sets of runs did not share seeds this
    is the wrong interval.
    """
    a = np.asarray(list(treatment), dtype=float)
    b = np.asarray(list(control), dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired comparison needs equal-length samples")
    return mean_ci(a - b, confidence)


def bullwhip_ratio(orders: Sequence[float], demand: Sequence[float]) -> float:
    """``Var(orders) / Var(demand)``.

    Variance, not standard deviation, and not the coefficient of variation.
    The CV ratio is the more common dashboard metric and it is misleading here:
    it confounds amplification with a change in mean throughput, which is
    exactly what happens during a disruption recovery.
    """
    demand_var = float(np.var(np.asarray(list(demand), dtype=float), ddof=1))
    if demand_var <= _EPS:
        return float("nan")
    return float(np.var(np.asarray(list(orders), dtype=float), ddof=1) / demand_var)


def variance_ratio_ci(
    numerator: Sequence[float],
    denominator: Sequence[float],
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Interval for a ratio of per-replication variances, on the log scale.

    Ratios are right-skewed and a symmetric interval on the raw scale can dip
    below zero. Taking logs, forming a t interval, and exponentiating gives an
    asymmetric interval that respects the positivity constraint -- the same
    reason bullwhip is decomposed additively in log space in
    :mod:`echelonsim.bullwhip`.
    """
    num = np.asarray(list(numerator), dtype=float)
    den = np.asarray(list(denominator), dtype=float)
    if num.shape != den.shape:
        raise ValueError("ratio components must be paired")
    valid = (num > _EPS) & (den > _EPS)
    if valid.sum() < 2:
        raise ValueError("not enough positive paired observations for a ratio interval")
    logs = np.log(num[valid] / den[valid])
    interval = mean_ci(logs, confidence)
    return ConfidenceInterval(
        mean=float(np.exp(interval.mean)),
        half_width=float(np.exp(interval.mean) * interval.half_width),
        low=float(np.exp(interval.low)),
        high=float(np.exp(interval.high)),
        n=int(valid.sum()),
        note="log-scale interval, half-width is a linearised approximation",
    )
