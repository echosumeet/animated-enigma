"""End-to-end experiment wiring: the model zoo and the reconciliation study.

Both the benchmark script and the example scripts call into here, so there is
exactly one definition of "the standard comparison" and the README numbers
cannot drift from what an example prints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import metrics as M
from .backtest import BacktestConfig, BacktestResult, backtest, rolling_origins
from .base import Forecaster, GlobalForecaster
from .baselines import (
    DriftForecaster,
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    ZeroForecaster,
)
from .classify import classify_panel
from .datagen import DemandPanel
from .ets import HoltLinear, HoltWinters, SimpleExponentialSmoothing
from .hierarchy import Hierarchy, coherency_error, reconcile, shrink_covariance
from .intermittent import CrostonForecaster, SBAForecaster, TSBForecaster
from .ml import FeatureConfig, GlobalGBTForecaster

__all__ = [
    "static_codes",
    "build_model_zoo",
    "run_backtest",
    "ReconciliationReport",
    "run_reconciliation_study",
    "select_base_model",
    "quadrant_profile_table",
    "pinball_table",
    "value_add_table",
    "RECONCILIATION_METHODS",
]

RECONCILIATION_METHODS = ("base", "bottom_up", "top_down", "ols", "mint")


def static_codes(keys: list[tuple[str, ...]]) -> np.ndarray:
    """Integer-code the hierarchy coordinates for use as categorical features."""
    n_dims = len(keys[0])
    codes = np.zeros((len(keys), n_dims))
    for d in range(n_dims):
        levels = sorted({k[d] for k in keys})
        lookup = {v: i for i, v in enumerate(levels)}
        for i, k in enumerate(keys):
            codes[i, d] = lookup[k[d]]
    return codes


def build_model_zoo(
    panel: DemandPanel,
    horizon: int = 13,
    *,
    season_length: int = 52,
    include_ml: bool = True,
    quantiles: tuple[float, ...] = (0.5, 0.9, 0.95),
):
    """The standard comparison set: baselines, ETS, intermittent, ML.

    Returns ``(local_models, global_models, quantile_models)``.
    """
    local: list[Forecaster] = [
        NaiveForecaster(),
        SeasonalNaiveForecaster(season_length),
        MovingAverageForecaster(8),
        DriftForecaster(),
        ZeroForecaster(),
        SimpleExponentialSmoothing(),
        HoltLinear(damped=True),
        HoltWinters(season_length, "add", damped=True),
        HoltWinters(season_length, "mul", damped=True),
        CrostonForecaster(),
        SBAForecaster(),
        TSBForecaster(),
    ]

    global_models: list[GlobalForecaster] = []
    quantile_models: list[tuple[float, GlobalForecaster]] = []
    if include_ml:
        cfg = FeatureConfig(season_length=season_length)
        static = static_codes(panel.keys)
        common = dict(
            horizon=horizon,
            cfg=cfg,
            promo=panel.promo,
            static=static,
        )
        global_models = [
            GlobalGBTForecaster(loss="squared_error", name="gbt_mean", **common),
            GlobalGBTForecaster(loss="absolute_error", name="gbt_median", **common),
        ]
        quantile_models = [
            (
                q,
                GlobalGBTForecaster(
                    loss="quantile", quantile=q, name=f"gbt_q{q:g}", **common
                ),
            )
            for q in quantiles
        ]
    return local, global_models, quantile_models


def run_backtest(
    panel: DemandPanel,
    *,
    horizon: int = 13,
    step: int = 13,
    n_windows: int = 4,
    min_train: int = 156,
    include_ml: bool = True,
    verbose: bool = False,
) -> BacktestResult:
    """Run the standard rolling-origin backtest on the bottom-level panel."""
    m = panel.config.season_length
    local, glob, quants = build_model_zoo(
        panel, horizon=horizon, season_length=m, include_ml=include_ml
    )
    cfg = BacktestConfig(
        horizon=horizon,
        step=step,
        n_windows=n_windows,
        min_train=min_train,
        season_length=m,
    )
    return backtest(
        panel.y,
        local,
        cfg,
        global_models=glob,
        quantile_models=quants,
        verbose=verbose,
    )


def select_base_model(y: np.ndarray, season_length: int):
    """Pick a base model per hierarchy node with a simple, defensible rule.

    Seasonal exponential smoothing needs two full cycles of reasonably dense
    history to estimate anything; below that it overfits noise into seasonal
    indices and forecasts garbage. Nodes that fail the test fall back to SES,
    and near-dead nodes to Croston. Automated model selection in a live estate
    is mostly rules like this one, not information criteria.
    """
    n = y.size
    nonzero_share = float(np.mean(y > 0))
    if n >= 2 * season_length and nonzero_share >= 0.6 and float(np.mean(y)) > 0:
        return HoltWinters(season_length, "add", damped=True, n_restarts=2)
    if nonzero_share < 0.35:
        return SBAForecaster()
    return SimpleExponentialSmoothing()


@dataclass
class ReconciliationReport:
    """WAPE by hierarchy level for each reconciliation method."""

    table: dict[str, dict[str, float]]
    coherency: dict[str, float]
    levels: list[str]
    shrinkage: float
    n_nodes: int
    n_bottom: int
    cuts: list[int]
    horizon: int

    def as_markdown(self) -> str:
        head = "| method | " + " | ".join(self.levels) + " | max coherency error |"
        sep = "|---" * (len(self.levels) + 2) + "|"
        lines = [head, sep]
        for mth, row in self.table.items():
            cells = " | ".join(f"{row.get(lv, float('nan')):.4f}" for lv in self.levels)
            lines.append(f"| {mth} | {cells} | {self.coherency[mth]:.2e} |")
        return "\n".join(lines)


def run_reconciliation_study(
    panel: DemandPanel,
    *,
    horizon: int = 13,
    n_windows: int = 2,
    step: int = 13,
    min_train: int = 156,
    residual_window: int = 104,
    verbose: bool = False,
) -> ReconciliationReport:
    """Fit base forecasts at every node, then compare reconciliation methods.

    The base forecasts are produced independently per node, so they are
    incoherent -- which is exactly the situation reconciliation exists for and
    exactly what a multi-level planning process produces if left alone.
    """
    hier: Hierarchy = panel.hierarchy
    Y = panel.node_series()
    n_nodes, T = Y.shape
    m = panel.config.season_length
    cuts = rolling_origins(T, horizon, step, n_windows, min_train)

    levels: list[str] = []
    for lv in hier.node_levels:
        if lv not in levels:
            levels.append(lv)

    accum = {
        mth: {lv: [0.0, 0.0] for lv in levels} for mth in RECONCILIATION_METHODS
    }
    coherency = {mth: 0.0 for mth in RECONCILIATION_METHODS}
    last_lambda = 0.0

    for w, cut in enumerate(cuts):
        train = Y[:, :cut]
        actual = Y[:, cut : cut + horizon]

        base = np.zeros((n_nodes, horizon))
        resid = np.full((cut, n_nodes), np.nan)
        for i in range(n_nodes):
            mdl = select_base_model(train[i], m)
            mdl.fit(train[i])
            base[i] = mdl.predict(horizon)
            resid[:, i] = mdl.residuals()

        # keep the most recent fully-observed residual block
        tail = resid[-residual_window:, :]
        good = np.all(np.isfinite(tail), axis=1)
        R = tail[good]
        if R.shape[0] < 12:
            R = np.nan_to_num(resid[-residual_window:, :], nan=0.0)

        _, last_lambda = shrink_covariance(R)

        props = train[hier.S.shape[0] - hier.n_bottom :].sum(axis=1)

        for mth in RECONCILIATION_METHODS:
            if mth == "base":
                rec = base.copy()
            else:
                rec = reconcile(
                    base,
                    hier,
                    method=mth,
                    residuals=R,
                    proportions=props,
                    nonnegative=True,
                )
            coherency[mth] = max(coherency[mth], coherency_error(rec, hier))
            for lv in levels:
                idx = hier.level_index(lv)
                a = actual[idx].ravel()
                f = rec[idx].ravel()
                accum[mth][lv][0] += float(np.sum(np.abs(a - f)))
                accum[mth][lv][1] += float(np.sum(np.abs(a)))
        if verbose:
            print(f"  reconciliation window {w + 1}/{len(cuts)} cut={cut}", flush=True)

    table = {
        mth: {
            lv: (accum[mth][lv][0] / accum[mth][lv][1])
            if accum[mth][lv][1] > 0
            else float("nan")
            for lv in levels
        }
        for mth in RECONCILIATION_METHODS
    }
    return ReconciliationReport(
        table=table,
        coherency=coherency,
        levels=levels,
        shrinkage=float(last_lambda),
        n_nodes=n_nodes,
        n_bottom=hier.n_bottom,
        cuts=cuts,
        horizon=horizon,
    )


def quadrant_profile_table(panel: DemandPanel, cut: int) -> list:
    """Classify the panel on the training window used for reporting."""
    return classify_panel(panel.y[:, :cut])


def pinball_table(result: BacktestResult, method: str) -> dict[str, float]:
    """Pinball loss and empirical coverage per quantile for one method."""
    rows = [r for r in result.rows if r.method == method]
    out: dict[str, float] = {}
    if not rows:
        return out
    for key in sorted(rows[0].metrics):
        if key.startswith(("pinball_", "coverage_")):
            vals = [r.metrics[key] for r in rows if np.isfinite(r.metrics[key])]
            if vals:
                out[key] = float(np.mean(vals))
    return out


def value_add_table(result: BacktestResult, baseline: str) -> dict[str, float]:
    """Forecast Value Add versus a named baseline, on pooled WAPE."""
    agg = result.overall()
    base = agg[baseline]["wape"]
    return {m: M.forecast_value_add(agg[m]["wape"], base) for m in result.methods()}
