"""Feature specifications with point-in-time correctness and skew detection.

Two independent checks run against a feature set, because they catch different bugs:

1. **Knowledge-time check** (declarative). Every source column carries a knowledge
   delay -- how long after the event timestamp the value is actually settled in the
   source system. A spec that reads a column at a lag shorter than its delay is
   leaky even though it looks perfectly causal in the dataframe.
2. **Truncation check** (empirical). Recompute each feature from history truncated at
   the row's own timestamp and compare with the batch-computed value. Any feature
   whose value moves is a function of the future, whatever the spec claims.

Neither check subsumes the other. (1) misses full-sample transforms that read only
"past" columns; (2) misses columns that are backfilled late in the source system.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .datagen import KNOWLEDGE_DELAY_DAYS

TRANSFORMS = ("lag", "rolling_mean", "rolling_std", "ewm_mean", "entity_mean")


@dataclass(frozen=True)
class FeatureSpec:
    """One derived column.

    ``lag`` is in periods and is applied *before* any window, so a spec with
    ``lag=1, window=7`` is the trailing 7-period mean ending one period before the
    label. ``entity_mean`` deliberately has no causal form -- it is the classic
    full-sample group statistic that leaks, and the truncation check exists to find it.
    """

    name: str
    source: str
    transform: str = "lag"
    lag: int = 1
    window: int = 1

    def __post_init__(self) -> None:
        if self.transform not in TRANSFORMS:
            raise ValueError(f"unknown transform {self.transform!r}, expected one of {TRANSFORMS}")
        if self.lag < 0:
            raise ValueError("negative lag reads the future; use lag >= 0")


def _apply(spec: FeatureSpec, g: pd.Series) -> pd.Series:
    if spec.transform == "entity_mean":
        return pd.Series(np.full(len(g), g.mean()), index=g.index)
    s = g.shift(spec.lag)
    if spec.transform == "lag":
        return s
    if spec.transform == "rolling_mean":
        return s.rolling(spec.window, min_periods=1).mean()
    if spec.transform == "rolling_std":
        return s.rolling(spec.window, min_periods=2).std()
    return s.ewm(span=max(2, spec.window), adjust=False).mean()


def build_features(
    panel: pd.DataFrame,
    specs: list[FeatureSpec],
    entity: str = "sku",
    ts: str = "date",
) -> pd.DataFrame:
    """Batch (training-path) feature computation over the whole panel."""
    df = panel.sort_values([entity, ts])
    out = df[[entity, ts]].copy()
    for spec in specs:
        out[spec.name] = df.groupby(entity, observed=True)[spec.source].transform(
            lambda g, s=spec: _apply(s, g)
        )
    return out.reset_index(drop=True)


# ------------------------------------------------------------------ leakage checks


@dataclass(frozen=True)
class LeakageFinding:
    feature: str
    check: str
    detail: str


def knowledge_time_check(
    specs: list[FeatureSpec], delays: dict[str, int] | None = None
) -> list[LeakageFinding]:
    """Flag specs that read a source column before that column is knowable."""
    delays = delays if delays is not None else KNOWLEDGE_DELAY_DAYS
    findings = []
    for spec in specs:
        need = delays.get(spec.source, 0)
        if spec.transform == "entity_mean":
            continue  # handled by the truncation check
        if spec.lag < need:
            findings.append(
                LeakageFinding(
                    spec.name,
                    "knowledge_time",
                    f"{spec.source} settles {need}d after the event but the spec uses lag={spec.lag}",
                )
            )
    return findings


def truncation_check(
    panel: pd.DataFrame,
    specs: list[FeatureSpec],
    as_of_dates: list[pd.Timestamp] | None = None,
    entity: str = "sku",
    ts: str = "date",
    tol: float = 1e-9,
) -> list[LeakageFinding]:
    """Recompute features from truncated history and report any value that moves.

    For each probe date, features are rebuilt from ``panel[date <= probe]`` and the
    rows at ``probe`` are compared with the batch-computed rows. A correct
    point-in-time feature is invariant to data that did not exist yet.
    """
    batch = build_features(panel, specs, entity, ts).set_index([entity, ts])
    dates = np.sort(panel[ts].unique())
    if as_of_dates is None:
        as_of_dates = [pd.Timestamp(d) for d in dates[len(dates) // 2 :: max(1, len(dates) // 6)]]

    mismatch: dict[str, tuple[int, float]] = {}
    for probe in as_of_dates:
        truncated = panel[panel[ts] <= probe]
        online = build_features(truncated, specs, entity, ts)
        online = online[online[ts] == probe].set_index([entity, ts])
        if online.empty:
            continue
        ref = batch.loc[online.index]
        for spec in specs:
            a, b = ref[spec.name].to_numpy(float), online[spec.name].to_numpy(float)
            diff = np.abs(np.nan_to_num(a) - np.nan_to_num(b))
            bad = int((diff > tol).sum())
            if bad:
                n, m = mismatch.get(spec.name, (0, 0.0))
                mismatch[spec.name] = (n + bad, max(m, float(diff.max())))

    return [
        LeakageFinding(
            name,
            "truncation",
            f"{n} row(s) changed when history was truncated to the label date "
            f"(max abs diff {m:.4f}); the feature reads future data",
        )
        for name, (n, m) in sorted(mismatch.items())
    ]


def audit_features(
    panel: pd.DataFrame, specs: list[FeatureSpec], **kwargs
) -> list[LeakageFinding]:
    """Run both checks. An empty list is the only acceptable CI result."""
    return knowledge_time_check(specs) + truncation_check(panel, specs, **kwargs)


# --------------------------------------------------------------- training/serving skew


@dataclass
class SkewReport:
    rows: int
    per_feature: pd.DataFrame

    @property
    def worst_mismatch_rate(self) -> float:
        return float(self.per_feature["mismatch_rate"].max()) if len(self.per_feature) else 0.0

    def failing(self, threshold: float = 0.01) -> list[str]:
        p = self.per_feature
        return sorted(p.loc[p["mismatch_rate"] > threshold, "feature"].tolist())

    def summary(self) -> str:
        return (
            f"skew over {self.rows} rows: worst mismatch rate "
            f"{self.worst_mismatch_rate:.2%} on "
            f"{self.per_feature.sort_values('mismatch_rate').iloc[-1]['feature'] if len(self.per_feature) else 'n/a'}"
        )


def detect_skew(
    offline: pd.DataFrame,
    online: pd.DataFrame,
    keys: list[str],
    features: list[str],
    rtol: float = 1e-6,
) -> SkewReport:
    """Compare the training-path and serving-path feature values on shared keys.

    Real skew is rarely a modelling problem; it is a null-handling or default-value
    difference between two codebases. Reporting per-feature mismatch rate and the mean
    signed gap points straight at which transform diverged.
    """
    merged = offline.merge(online, on=keys, suffixes=("_off", "_on"), how="inner")
    rows = []
    for f in features:
        a = merged[f"{f}_off"].to_numpy(float)
        b = merged[f"{f}_on"].to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, rtol=rtol, atol=1e-9) | both_nan
        gap = np.nan_to_num(b - a)
        rows.append(
            {
                "feature": f,
                "mismatch_rate": float(1.0 - close.mean()),
                "mean_signed_gap": float(gap.mean()),
                "max_abs_gap": float(np.abs(gap).max()) if len(gap) else 0.0,
            }
        )
    return SkewReport(len(merged), pd.DataFrame(rows))


def default_specs() -> list[FeatureSpec]:
    """The production feature set: every spec is point-in-time safe."""
    return [
        FeatureSpec("units_lag_1", "units", "lag", lag=1),
        FeatureSpec("units_ma_7", "units", "rolling_mean", lag=1, window=7),
        FeatureSpec("units_ma_28", "units", "rolling_mean", lag=1, window=28),
        FeatureSpec("units_sd_28", "units", "rolling_std", lag=1, window=28),
        FeatureSpec("price_lag_0", "price", "lag", lag=0),
        FeatureSpec("promo_lag_0", "promo_flag", "lag", lag=0),
        FeatureSpec("on_hand_lag_1", "on_hand", "lag", lag=1),
    ]


def leaky_specs() -> list[FeatureSpec]:
    """The same set with two bugs a reviewer would plausibly wave through."""
    return default_specs() + [
        FeatureSpec("returns_lag_1", "returns_units", "lag", lag=1),
        FeatureSpec("sku_mean_units", "units", "entity_mean"),
    ]
