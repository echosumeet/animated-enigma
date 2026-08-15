"""Disruption experiments and recovery measurement.

Resilience gets discussed with adjectives and bought with capital, and the gap
between the two is a measurement problem. Three numbers make the conversation
concrete, and all three come from comparing a disrupted run against its own
undisrupted twin under common random numbers -- same demand, same transit draws,
only the disruption differs:

**Fill-rate trough** -- how bad it gets. The depth of the service dip is what
customers experience and it is almost never the number in the risk register,
which usually records the *duration of the supply outage* instead. Those are
different quantities and the ratio between them is the thing worth knowing.

**Time to recover** -- how long until service is back. Measured from the first
period of the disruption to the first period at which smoothed fill rate returns
to within a tolerance of its undisrupted level *and stays there*. The "and stays
there" clause matters: a chain that bounces back for two periods on a
panic-ordered surge and then dips again has not recovered, and a rule without a
hold requirement will score it as if it had.

**Recovery ratio** -- time to recover divided by the length of the disruption.
This is the number that surprises people. A four-period supplier outage does not
produce four periods of pain; the backlog has to be worked off on top of ongoing
demand, and every echelon between the outage and the customer is simultaneously
re-ordering against a depressed inventory position.

The comparison is always against the paired undisrupted baseline rather than
against a fixed target, because the undisrupted chain does not deliver 100% fill
rate either, and scoring recovery against 100% conflates the disruption with the
chain's ordinary service level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .experiments import (
    DEFAULT_CONFIG,
    ScenarioOutcome,
    estimate_warmup,
    merge_config,
    run_scenario,
)
from .metrics import ConfidenceInterval, mean_ci
from .simulation import SimulationResult

__all__ = [
    "RecoveryProfile",
    "DisruptionStudy",
    "retailer_fill_path",
    "measure_recovery",
    "run_disruption_study",
]


def retailer_fill_path(result: SimulationResult) -> np.ndarray:
    """Per-period fill rate aggregated across all retailers.

    Aggregated over units, not averaged over retailers: a 50% fill on a big
    store and a 100% fill on a small one is not 75% service, and reporting it
    that way is how a service problem gets hidden behind a store count.
    """
    numerator = np.zeros(result.periods, dtype=float)
    denominator = np.zeros(result.periods, dtype=float)
    for series in result.stocking_series():
        if series.level != 0:
            continue
        numerator += series.fill_numerator
        denominator += series.demand_received
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(denominator > 1e-9, numerator / np.maximum(denominator, 1e-9), 1.0)
    return np.clip(out, 0.0, 1.0)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / window
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, kernel, mode="valid")


def _recovery_offset(gap: np.ndarray, tolerance: float, hold: int, horizon: int) -> Tuple[float, bool]:
    """First offset after the worst point at which the gap stays inside tolerance.

    Returns ``(offset, censored)``. Measured from the *trough*, not from the
    first period in which anything went wrong: under a multi-period lead time an
    outage takes a while to reach the shelf, and the service gap in the interim
    can sit inside tolerance for long enough to score a spurious recovery before
    the real damage has arrived.
    """
    if gap.size == 0 or gap.max() <= tolerance:
        return 0.0, False
    trough_at = int(np.argmax(gap))
    run_length = 0
    for offset in range(trough_at, gap.size):
        if gap[offset] <= tolerance:
            run_length += 1
            if run_length >= hold:
                return float(offset - hold + 1), False
        else:
            run_length = 0
    return float(horizon), True


@dataclass
class RecoveryProfile:
    """Everything measured about one disruption scenario."""

    name: str
    start: int
    duration: int
    offsets: np.ndarray
    disrupted_fill: np.ndarray
    baseline_fill: np.ndarray
    trough: float
    trough_offset: int
    baseline_level: float
    recovery_offset: ConfidenceInterval
    lost_units: ConfidenceInterval
    peak_backlog: ConfidenceInterval
    censored_fraction: float

    @property
    def recovery_ratio(self) -> float:
        return self.recovery_offset.mean / self.duration if self.duration else float("nan")

    def row(self) -> Tuple[str, int, float, int, float, float, float]:
        """``(name, outage periods, trough %, trough offset, recovery, ratio, lost units)``."""
        return (
            self.name,
            self.duration,
            100.0 * self.trough,
            self.trough_offset,
            self.recovery_offset.mean,
            self.recovery_ratio,
            self.lost_units.mean,
        )


def measure_recovery(
    disrupted: Sequence[SimulationResult],
    baseline: Sequence[SimulationResult],
    start: int,
    duration: int,
    name: str = "disruption",
    tolerance: float = 0.01,
    hold: int = 5,
    smooth_window: int = 3,
    horizon: int = 80,
    bootstrap: int = 400,
    seed: int = 991,
) -> RecoveryProfile:
    """Compare paired disrupted/undisrupted replications and score the recovery.

    ``tolerance`` is in absolute fill-rate points: recovery is declared when the
    smoothed disrupted fill rate is within ``tolerance`` of the paired baseline
    for ``hold`` consecutive periods.

    Recovery time is measured on the **replication-averaged** service gap rather
    than per replication and then averaged. A single run's period fill rate is
    close to a Bernoulli sequence -- mostly 1.0, occasionally a deep dip -- and a
    hold-based rule applied to it fires on noise, producing an estimate whose
    mean sits *before* the trough of the average path. Averaging first and then
    applying the rule gives a stable statistic; its uncertainty comes from a
    nonparametric bootstrap over replications, which is the honest way to put an
    interval on a quantity that is a nonlinear functional of the whole path.

    Replications are resampled with replacement ``bootstrap`` times; the interval
    is the 2.5/97.5 percentile of the recomputed recovery time. A bootstrap
    sample that never recovers inside ``horizon`` is **censored**, counted at
    ``horizon``, and the censored fraction is reported, so "40 periods, 30%
    censored" cannot be mistaken for "40 periods".
    """
    if len(disrupted) != len(baseline):
        raise ValueError("recovery measurement needs paired replications")
    if not disrupted:
        raise ValueError("no replications supplied")

    periods = disrupted[0].periods
    window_end = min(periods, start + horizon)
    offsets = np.arange(0, window_end - start)
    span = window_end - start

    gaps, disrupted_paths, baseline_paths = [], [], []
    lost, peaks = [], []
    for run_d, run_b in zip(disrupted, baseline):
        fill_d = retailer_fill_path(run_d)
        fill_b = retailer_fill_path(run_b)
        disrupted_paths.append(fill_d[start:window_end])
        baseline_paths.append(fill_b[start:window_end])
        gap = _smooth(fill_b, smooth_window) - _smooth(fill_d, smooth_window)
        gaps.append(gap[start:window_end])

        shortfall = np.zeros(periods, dtype=float)
        backlog = np.zeros(periods, dtype=float)
        for series_d in run_d.stocking_series():
            if series_d.level != 0:
                continue
            series_b = run_b.nodes[series_d.name]
            shortfall += series_b.fill_numerator - series_d.fill_numerator
            backlog += series_d.backlog - series_b.backlog
        lost.append(float(shortfall[start:window_end].sum()))
        peaks.append(float(backlog[start:window_end].max()))

    gap_matrix = np.asarray(gaps)
    point, point_censored = _recovery_offset(
        gap_matrix.mean(axis=0), tolerance, hold, horizon
    )

    rng = np.random.default_rng(seed)
    n = gap_matrix.shape[0]
    draws = np.empty(bootstrap, dtype=float)
    censored = 0
    for index in range(bootstrap):
        picks = rng.integers(0, n, size=n)
        value, was_censored = _recovery_offset(
            gap_matrix[picks].mean(axis=0), tolerance, hold, horizon
        )
        draws[index] = value
        censored += int(was_censored)
    low, high = np.percentile(draws, [2.5, 97.5])
    recovery = ConfidenceInterval(
        mean=point,
        half_width=0.5 * float(high - low),
        low=float(low),
        high=float(high),
        n=n,
        note="bootstrap percentile interval over replications"
        + (" (point estimate censored at horizon)" if point_censored else ""),
    )

    disrupted_mean = np.mean(np.asarray(disrupted_paths), axis=0)
    baseline_mean = np.mean(np.asarray(baseline_paths), axis=0)
    trough_index = int(np.argmin(disrupted_mean))
    return RecoveryProfile(
        name=name,
        start=start,
        duration=duration,
        offsets=offsets,
        disrupted_fill=disrupted_mean,
        baseline_fill=baseline_mean,
        trough=float(disrupted_mean[trough_index]),
        trough_offset=trough_index,
        baseline_level=float(baseline_mean.mean()),
        recovery_offset=recovery,
        lost_units=mean_ci(lost),
        peak_backlog=mean_ci(peaks),
        censored_fraction=censored / max(1, bootstrap),
    )


@dataclass
class DisruptionStudy:
    baseline: ScenarioOutcome
    profiles: List[RecoveryProfile] = field(default_factory=list)
    warmup: int = 0

    def table(self) -> List[Tuple[str, int, float, int, float, float, float]]:
        return [profile.row() for profile in self.profiles]


def run_disruption_study(
    base_config: Optional[Dict[str, Any]] = None,
    scenarios: Optional[Sequence[Tuple[str, Dict[str, Any], int, int]]] = None,
    warmup: Optional[int] = None,
    horizon: int = 80,
) -> DisruptionStudy:
    """Run an undisrupted baseline plus a set of disruption scenarios.

    Each scenario is ``(name, config override, absolute start period, duration)``.
    ``start`` is in *pre-truncation* periods, which is the only frame the config
    can express; the returned profiles are indexed from the disruption onset.
    """
    base = merge_config(DEFAULT_CONFIG, base_config or {})
    if warmup is None:
        warmup = estimate_warmup(base, pilots=3)

    baseline = run_scenario(base, name="undisrupted", warmup=warmup, keep_results=True)
    study = DisruptionStudy(baseline=baseline, warmup=int(warmup))

    for name, override, start, duration in scenarios or ():
        if start <= warmup:
            raise ValueError(
                f"scenario {name!r} starts at period {start}, inside the "
                f"{warmup}-period warm-up truncation"
            )
        outcome = run_scenario(
            merge_config(base, override), name=name, warmup=warmup, keep_results=True
        )
        study.profiles.append(
            measure_recovery(
                disrupted=outcome.results,
                baseline=baseline.results,
                start=start - warmup,
                duration=duration,
                name=name,
                horizon=horizon,
            )
        )
    return study
