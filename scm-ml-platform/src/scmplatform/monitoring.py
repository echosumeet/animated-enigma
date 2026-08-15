"""Drift and slice monitoring.

Population Stability Index and the two-sample Kolmogorov-Smirnov statistic answer
different questions. PSI is binned, bounded and comparable across features, which is
what an on-call engineer needs on a dashboard; KS is distribution-free with a p-value,
which is what a review needs before anyone retrains. Both are reported.

Aggregate accuracy hides the failure that matters: a model that is fine overall and
badly wrong on one region or one category. Slice analysis surfaces it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PSI_MODERATE = 0.10
PSI_MAJOR = 0.25


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index with quantile bins fixed on the reference sample."""
    ref = np.asarray(reference, float)
    cur = np.asarray(current, float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) < bins or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, bins=edges)[0] / len(ref)
    q = np.histogram(cur, bins=edges)[0] / len(cur)
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def psi_band(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value >= PSI_MAJOR:
        return "major"
    if value >= PSI_MODERATE:
        return "moderate"
    return "stable"


def drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, features: list[str], bins: int = 10
) -> pd.DataFrame:
    """Per-feature PSI, KS statistic and KS p-value, ordered by severity."""
    rows = []
    for f in features:
        ref = reference[f].to_numpy(float)
        cur = current[f].to_numpy(float)
        ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
        stat, pvalue = (ks_2samp(ref, cur) if len(ref) and len(cur) else (np.nan, np.nan))[:2]
        value = psi(ref, cur, bins)
        rows.append(
            {
                "feature": f,
                "psi": value,
                "band": psi_band(value),
                "ks_stat": float(stat),
                "ks_pvalue": float(pvalue),
                "ref_mean": float(np.mean(ref)) if len(ref) else np.nan,
                "cur_mean": float(np.mean(cur)) if len(cur) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)


def prediction_drift(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> dict[str, float]:
    """Drift in the model's own output -- the first thing to check when inputs look clean."""
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    stat, pvalue = ks_2samp(ref, cur)[:2]
    value = psi(ref, cur, bins)
    return {
        "psi": value,
        "ks_stat": float(stat),
        "ks_pvalue": float(pvalue),
        "mean_shift": float(np.mean(cur) - np.mean(ref)),
        "relative_mean_shift": float((np.mean(cur) - np.mean(ref)) / max(abs(np.mean(ref)), 1e-9)),
    }


def drift_over_time(
    frame: pd.DataFrame,
    features: list[str],
    reference_end: pd.Timestamp,
    ts: str = "date",
    window_days: int = 30,
) -> pd.DataFrame:
    """PSI per feature per rolling window against a fixed reference period."""
    ref = frame[frame[ts] <= reference_end]
    later = frame[frame[ts] > reference_end]
    if later.empty:
        return pd.DataFrame(columns=["window_start", "feature", "psi"])
    rows = []
    start = later[ts].min()
    end = later[ts].max()
    while start <= end:
        stop = start + pd.Timedelta(days=window_days)
        chunk = later[(later[ts] >= start) & (later[ts] < stop)]
        if len(chunk) >= 30:
            for f in features:
                rows.append(
                    {
                        "window_start": start,
                        "feature": f,
                        "psi": psi(ref[f].to_numpy(float), chunk[f].to_numpy(float)),
                    }
                )
        start = stop
    return pd.DataFrame(rows)


@dataclass
class SliceReport:
    table: pd.DataFrame
    overall_wmape: float
    tolerance: float

    @property
    def degraded(self) -> pd.DataFrame:
        return self.table[self.table["flag"]].reset_index(drop=True)

    def summary(self) -> str:
        return (
            f"overall wMAPE {self.overall_wmape:.4f}; "
            f"{len(self.degraded)} of {len(self.table)} slices exceed "
            f"{self.tolerance:.2f}x that level"
        )


def slice_performance(
    predictions: pd.DataFrame,
    by: list[str],
    tolerance: float = 1.25,
    y: str = "y",
    yhat: str = "yhat",
    min_rows: int = 30,
) -> SliceReport:
    """wMAPE and bias per slice, flagging slices materially worse than the aggregate."""
    err = (predictions[yhat] - predictions[y]).abs()
    overall = float(err.sum() / max(predictions[y].abs().sum(), 1e-9))
    g = predictions.assign(_abs_err=err, _signed=predictions[yhat] - predictions[y]).groupby(
        by, observed=True
    )
    table = g.apply(
        lambda d: pd.Series(
            {
                "rows": float(len(d)),
                "units": float(d[y].sum()),
                "wmape": float(d["_abs_err"].sum() / max(d[y].abs().sum(), 1e-9)),
                "bias": float(d["_signed"].sum() / max(d[y].abs().sum(), 1e-9)),
            }
        ),
        include_groups=False,
    ).reset_index()
    table["ratio_to_overall"] = table["wmape"] / max(overall, 1e-9)
    table["flag"] = (table["ratio_to_overall"] > tolerance) & (table["rows"] >= min_rows)
    return SliceReport(table.sort_values("wmape", ascending=False, ignore_index=True), overall, tolerance)
