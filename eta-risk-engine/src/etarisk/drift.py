"""Population Stability Index over features.

One function and a thin table wrapper. PSI is crude -- it is a binned symmetric
KL-like divergence with no significance theory behind the 0.1/0.25 convention
(Yurdakul, 2018) -- but it is the number monitoring dashboards are built on, it
is cheap to compute nightly, and it flags the failure this repo cares about: the
lane and congestion distribution moving away from what the encoder was fitted on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["psi", "psi_table", "CALENDAR_COLUMNS", "PSI_MINOR", "PSI_MAJOR"]

PSI_MINOR = 0.10
PSI_MAJOR = 0.25


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10, eps: float = 1e-6) -> float:
    """PSI of ``actual`` against a reference sample ``expected``.

    Bin edges are the reference quantiles, so the reference is uniform by
    construction and all of the signal is in the actual-sample counts. Degenerate
    columns (fewer distinct values than bins) fall back to the distinct values.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    levels = np.unique(expected)
    if levels.size <= n_bins:
        # Discrete column (a target encoding over a handful of carriers, say).
        # Quantile edges land exactly on the mass points and the binning becomes
        # an artefact, so compare category frequencies directly instead.
        cats = np.union1d(levels, np.unique(actual))
        e = np.array([(expected == c).mean() for c in cats])
        a = np.array([(actual == c).mean() for c in cats])
    else:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
        if edges.size < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, bins=edges)[0] / expected.size
        a = np.histogram(actual, bins=edges)[0] / actual.size
    e = np.clip(e, eps, None)
    a = np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


#: Calendar features move by construction between two consecutive time blocks;
#: monitoring them tells you the calendar advanced, not that anything broke.
CALENDAR_COLUMNS = ("dow_sin", "dow_cos", "woy_sin", "woy_cos", "month")


def psi_table(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    n_bins: int = 10,
    skip_calendar: bool = True,
) -> pd.DataFrame:
    """PSI per shared numeric column, sorted worst first, with a verdict."""
    cols = [c for c in reference.columns if c in current.columns]
    if skip_calendar:
        cols = [c for c in cols if c not in CALENDAR_COLUMNS]
    rows = []
    for c in cols:
        value = psi(reference[c].to_numpy(dtype=float), current[c].to_numpy(dtype=float), n_bins)
        verdict = "stable" if value < PSI_MINOR else ("shifted" if value < PSI_MAJOR else "broken")
        rows.append({"feature": c, "psi": value, "verdict": verdict})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
