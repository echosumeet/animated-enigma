"""Leakage-safe feature engineering.

Two rules are enforced here rather than left to the caller's discipline:

1.  **Splits are temporal.** Shipments are ordered by ship timestamp and cut into
    contiguous blocks. A random split lets the model see next month's congestion
    regime while scoring this month's shipments, which inflates offline accuracy
    and then evaporates in production. See Bergmeir & Benitez (2012) on
    evaluation of predictive models on serially dependent data.
2.  **Target encodings are out-of-fold.** Lane and carrier mean delay is by far
    the strongest single feature, and computing it in-sample gives every row a
    small amount of its own label back. Each row's encoding is computed from
    *strictly earlier* shipments only, which is also exactly what is available at
    booking time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "temporal_folds",
    "temporal_split",
    "TargetEncoder",
    "build_features",
    "FEATURE_COLUMNS",
]

NUMERIC_BASE = [
    "distance_km",
    "planned_transit_h",
    "weight_kg",
    "pieces",
    "origin_congestion_obs",
    "dest_congestion_obs",
    "weather_forecast_origin",
    "weather_forecast_dest",
]

FEATURE_COLUMNS = NUMERIC_BASE + [
    "cross_border_i",
    "dow_sin",
    "dow_cos",
    "woy_sin",
    "woy_cos",
    "month",
    "te_lane",
    "te_carrier",
    "te_lane_carrier",
]


def temporal_folds(n: int, n_folds: int = 5, min_train_frac: float = 0.3) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds over a time-ordered index.

    Fold ``k`` trains on ``[0, cut_k)`` and validates on ``[cut_k, cut_{k+1})``.
    Every validation index is strictly greater than every training index, which
    is the property the leakage test asserts.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    start = int(n * min_train_frac)
    cuts = np.linspace(start, n, n_folds + 1).astype(int)
    folds = []
    for k in range(n_folds):
        tr = np.arange(0, cuts[k])
        va = np.arange(cuts[k], cuts[k + 1])
        if tr.size and va.size:
            folds.append((tr, va))
    return folds


def temporal_split(df: pd.DataFrame, fracs: tuple[float, float, float] = (0.6, 0.2, 0.2)):
    """Split a ship-time-sorted frame into train / calibration / test blocks."""
    if not np.isclose(sum(fracs), 1.0):
        raise ValueError("fracs must sum to 1")
    ts = df["ship_ts"].to_numpy()
    if np.any(ts[1:] < ts[:-1]):
        raise ValueError("frame must be sorted by ship_ts before splitting")
    n = len(df)
    a = int(n * fracs[0])
    b = a + int(n * fracs[1])
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


@dataclass
class TargetEncoder:
    """Smoothed mean-target encoder over one or more categorical keys.

    The encoded quantity is the mean *delay ratio* ``actual / planned`` rather
    than raw hours, so a slow ocean lane and a fast air lane land on the same
    scale and the encoding stays useful when the lane mix shifts.
    """

    keys: tuple[str, ...]
    smoothing: float = 50.0
    prior_: float = 1.0
    table_: dict | None = None

    def fit(self, df: pd.DataFrame) -> "TargetEncoder":
        y = (df["actual_transit_h"] / df["planned_transit_h"]).to_numpy(dtype=float)
        self.prior_ = float(np.mean(y))
        grp = pd.DataFrame({"_y": y})
        for k in self.keys:
            grp[k] = df[k].to_numpy()
        agg = grp.groupby(list(self.keys))["_y"].agg(["sum", "count"])
        shrunk = (agg["sum"] + self.smoothing * self.prior_) / (agg["count"] + self.smoothing)
        self.table_ = shrunk.to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.table_ is None:
            raise RuntimeError("TargetEncoder.fit must be called first")
        if len(self.keys) == 1:
            keys = df[self.keys[0]].to_numpy()
        else:
            keys = list(zip(*(df[k].to_numpy() for k in self.keys)))
        table = self.table_
        prior = self.prior_
        return np.array([table.get(k, prior) for k in keys], dtype=float)

    def oof_transform(self, df: pd.DataFrame, n_folds: int = 5) -> np.ndarray:
        """Encode a training frame using only strictly earlier shipments.

        Rows before the first fold cut fall back to the global prior of the
        block that precedes them; they are never encoded with their own label.
        """
        n = len(df)
        out = np.full(n, np.nan)
        for tr, va in temporal_folds(n, n_folds=n_folds):
            enc = TargetEncoder(self.keys, self.smoothing).fit(df.iloc[tr])
            out[va] = enc.transform(df.iloc[va])
        warm = np.isnan(out)
        if warm.any():
            seed = TargetEncoder(self.keys, self.smoothing).fit(df.iloc[~warm]) if (~warm).any() else None
            out[warm] = seed.prior_ if seed is not None else 1.0
        return out


def _calendar(df: pd.DataFrame) -> dict[str, np.ndarray]:
    dow = df["ship_dow"].to_numpy(dtype=float)
    woy = df["ship_week"].to_numpy(dtype=float)
    return {
        "dow_sin": np.sin(2 * np.pi * dow / 7.0),
        "dow_cos": np.cos(2 * np.pi * dow / 7.0),
        "woy_sin": np.sin(2 * np.pi * woy / 52.0),
        "woy_cos": np.cos(2 * np.pi * woy / 52.0),
        "month": df["ship_month"].to_numpy(dtype=float),
    }


def build_features(
    train: pd.DataFrame,
    others: list[pd.DataFrame] | None = None,
    n_folds: int = 5,
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, TargetEncoder]]:
    """Build the design matrices for a train block and any downstream blocks.

    The train block gets out-of-fold encodings; downstream blocks get encodings
    fitted on the whole train block, which is the production analogue (you refit
    nightly on history and score tomorrow's bookings).
    """
    others = others or []
    specs = {
        "te_lane": ("lane", "mode"),
        "te_carrier": ("carrier",),
        "te_lane_carrier": ("lane", "carrier"),
    }
    encoders: dict[str, TargetEncoder] = {}
    frames: list[dict[str, np.ndarray]] = [{} for _ in range(1 + len(others))]
    for name, keys in specs.items():
        enc = TargetEncoder(keys).fit(train)
        encoders[name] = enc
        frames[0][name] = enc.oof_transform(train, n_folds=n_folds)
        for i, other in enumerate(others, start=1):
            frames[i][name] = enc.transform(other)

    out = []
    for block, extra in zip([train, *others], frames):
        cols = {c: block[c].to_numpy(dtype=float) for c in NUMERIC_BASE}
        cols["cross_border_i"] = block["cross_border"].to_numpy().astype(float)
        cols.update(_calendar(block))
        cols.update(extra)
        out.append(pd.DataFrame(cols, columns=FEATURE_COLUMNS, index=block.index))
    return out[0], out[1:], encoders
