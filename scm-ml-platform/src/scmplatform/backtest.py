"""Rolling-origin backtest with a CI-enforceable release gate.

The gate has two arms on purpose. An accuracy-only gate lets through models that shave
wMAPE by systematically under-forecasting, which reads as an improvement and costs
money in stockouts. Pairing accuracy with realised inventory cost (see
:mod:`scmplatform.decisions`) makes that trade explicit at merge time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .decisions import DecisionQuality, InventoryEconomics, evaluate_decisions
from .features import FeatureSpec, build_features


def wmape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Weighted MAPE -- the demand-weighted error planners actually track."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.abs(y - yhat).sum() / max(np.abs(y).sum(), 1e-9))


def bias(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float((yhat - y).sum() / max(np.abs(y).sum(), 1e-9))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


@dataclass
class BacktestResult:
    label: str
    predictions: pd.DataFrame
    folds: int

    @property
    def wmape(self) -> float:
        return wmape(self.predictions["y"], self.predictions["yhat"])

    @property
    def bias(self) -> float:
        return bias(self.predictions["y"], self.predictions["yhat"])

    @property
    def rmse(self) -> float:
        return rmse(self.predictions["y"], self.predictions["yhat"])

    def decisions(self, econ: InventoryEconomics | None = None) -> DecisionQuality:
        return evaluate_decisions(self.predictions, econ)

    def metrics(self, econ: InventoryEconomics | None = None) -> dict[str, float]:
        dq = self.decisions(econ)
        return {"wmape": self.wmape, "bias": self.bias, "rmse": self.rmse, **dq.as_dict()}


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], seed: int) -> np.ndarray:
    model = HistGradientBoostingRegressor(
        max_depth=6, max_iter=150, learning_rate=0.08, random_state=seed
    )
    model.fit(train[cols], train["y"])
    return model.predict(test[cols])


def rolling_backtest(
    panel: pd.DataFrame,
    specs: list[FeatureSpec],
    label: str = "model",
    horizon: int = 14,
    folds: int = 4,
    min_train_days: int = 240,
    seed: int = 0,
    target: str = "units",
    bias_adjustment: float = 1.0,
) -> BacktestResult:
    """Expanding-window backtest: fit on history, score the next ``horizon`` days.

    ``bias_adjustment`` scales predictions and exists so that a challenger which buys
    accuracy by under-forecasting can be simulated and caught by the cost arm of the gate.
    """
    feats = build_features(panel, specs)
    df = feats.merge(panel[["sku", "date", target, "category", "region"]], on=["sku", "date"])
    df = df.rename(columns={target: "y"}).dropna().sort_values("date").reset_index(drop=True)
    cols = [s.name for s in specs]

    dates = np.sort(df["date"].unique())
    start = np.searchsorted(dates, dates[0] + np.timedelta64(min_train_days, "D"))
    if start >= len(dates) - horizon:
        raise ValueError("panel is too short for the requested min_train_days/horizon")
    cutoffs = np.linspace(start, len(dates) - horizon - 1, folds).astype(int)

    out = []
    for i, ci in enumerate(cutoffs):
        cutoff = dates[ci]
        end = dates[min(ci + horizon, len(dates) - 1)]
        train = df[df["date"] <= cutoff]
        test = df[(df["date"] > cutoff) & (df["date"] <= end)]
        if train.empty or test.empty:
            continue
        pred = _fit_predict(train, test, cols, seed) * bias_adjustment
        resid = train["y"].to_numpy(float) - _fit_predict(train, train, cols, seed)
        chunk = test[["sku", "date", "y", "category", "region"]].copy()
        chunk["yhat"] = np.maximum(0.0, pred)
        chunk["sigma"] = float(np.std(resid))
        chunk["fold"] = i
        out.append(chunk)
    return BacktestResult(label, pd.concat(out, ignore_index=True), len(out))


# ------------------------------------------------------------------------- the gate


@dataclass
class Gate:
    """Release thresholds, checked in CI before a promotion is allowed."""

    max_wmape: float = 0.25
    max_abs_bias: float = 0.08
    max_cost_per_unit: float = 0.25


@dataclass
class GateResult:
    label: str
    passed: bool
    metrics: dict[str, float]
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        head = f"{'PASS' if self.passed else 'FAIL'} gate for {self.label}"
        return head if self.passed else head + ": " + "; ".join(self.reasons)


def check_gate(result: BacktestResult, gate: Gate, econ: InventoryEconomics | None = None) -> GateResult:
    m = result.metrics(econ)
    reasons = []
    if m["wmape"] > gate.max_wmape:
        reasons.append(f"wMAPE {m['wmape']:.4f} > {gate.max_wmape:.4f}")
    if abs(m["bias"]) > gate.max_abs_bias:
        reasons.append(f"|bias| {abs(m['bias']):.4f} > {gate.max_abs_bias:.4f}")
    if m["cost_per_unit"] > gate.max_cost_per_unit:
        reasons.append(f"cost/unit {m['cost_per_unit']:.4f} > {gate.max_cost_per_unit:.4f}")
    return GateResult(result.label, not reasons, m, reasons)


@dataclass
class ChallengerDecision:
    promote: bool
    rationale: str
    champion: dict[str, float]
    challenger: dict[str, float]

    def as_frame(self) -> pd.DataFrame:
        keys = ["wmape", "bias", "cost_per_unit", "regret", "fill_rate"]
        return pd.DataFrame(
            {
                "metric": keys,
                "champion": [self.champion[k] for k in keys],
                "challenger": [self.challenger[k] for k in keys],
            }
        )


def champion_challenger(
    champion: BacktestResult,
    challenger: BacktestResult,
    gate: Gate,
    econ: InventoryEconomics | None = None,
    min_cost_improvement: float = 0.02,
) -> ChallengerDecision:
    """Promote only if the challenger clears the gate *and* lowers realised cost.

    Cost is the tiebreaker rather than accuracy because two models within a point of
    each other on wMAPE can differ materially in what they cost to hold and expedite.
    """
    c_res = check_gate(challenger, gate, econ)
    champ_m = champion.metrics(econ)
    chall_m = challenger.metrics(econ)
    if not c_res.passed:
        return ChallengerDecision(False, "challenger failed the gate: " + "; ".join(c_res.reasons), champ_m, chall_m)
    lift = (champ_m["cost_per_unit"] - chall_m["cost_per_unit"]) / max(champ_m["cost_per_unit"], 1e-9)
    if lift < min_cost_improvement:
        return ChallengerDecision(
            False,
            f"cost improvement {lift:.2%} below the {min_cost_improvement:.0%} promotion bar",
            champ_m,
            chall_m,
        )
    return ChallengerDecision(True, f"cost per unit improved {lift:.2%} and the gate passed", champ_m, chall_m)
