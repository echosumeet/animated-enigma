"""End-to-end wiring: generate -> features -> ETA -> conformal -> risk -> decision.

Kept in one place so the benchmarks, the example and the CLI all exercise the
identical code path and cannot quietly disagree about a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .decision import CostMatrix, compare_to_baseline, fixed_rule_actions
from .features import build_features, temporal_split
from .generate import GeneratorConfig, generate_shipments
from .model import ConformalETA, ETAModel, mae_by_horizon
from .risk import DelayRiskModel, brier_score, expected_calibration_error, reliability_table

__all__ = ["PipelineResult", "run_pipeline"]


@dataclass
class PipelineResult:
    data: pd.DataFrame
    splits: dict[str, pd.DataFrame]
    X: dict[str, pd.DataFrame]
    eta: ETAModel
    conformal: dict[float, ConformalETA]
    risk: DelayRiskModel
    metrics: dict[str, Any] = field(default_factory=dict)


def run_pipeline(
    cfg: GeneratorConfig | None = None,
    alphas: tuple[float, ...] = (0.2, 0.1, 0.05),
    costs: CostMatrix | None = None,
    max_iter: int = 300,
) -> PipelineResult:
    cfg = cfg or GeneratorConfig()
    costs = costs or CostMatrix()
    df = generate_shipments(cfg)
    train, calib, test = temporal_split(df)
    # ``X_ref`` is the train block encoded the way production encodes it (full-fit
    # encoder, not out-of-fold). Drift monitoring has to compare like with like:
    # PSI between an out-of-fold training column and a fully-fitted scoring column
    # measures the encoding scheme, not the network.
    X_tr, (X_ref, X_cal, X_te), _ = build_features(train, [train, calib, test])

    y_tr = train["actual_transit_h"].to_numpy(dtype=float)
    y_cal = calib["actual_transit_h"].to_numpy(dtype=float)
    y_te = test["actual_transit_h"].to_numpy(dtype=float)

    eta = ETAModel(max_iter=max_iter).fit(X_tr, y_tr)
    pred_te = eta.predict(X_te)

    conformal: dict[float, ConformalETA] = {}
    coverage_rows = []
    for a in alphas:
        cp = ConformalETA(eta, alpha=a).calibrate(X_cal, y_cal, calib["planned_transit_h"].to_numpy())
        conformal[a] = cp
        cov = cp.coverage(X_te, y_te, test["planned_transit_h"].to_numpy())
        coverage_rows.append({"alpha": a, **cov})

    risk = DelayRiskModel(max_iter=max_iter).fit(X_tr, train["is_late"].to_numpy())
    risk.calibrate(X_cal, calib["is_late"].to_numpy())
    y_late = test["is_late"].to_numpy().astype(int)
    p_raw = risk.predict_raw(X_te)
    p_cal = risk.predict_proba(X_te)

    late_rate = train.groupby("carrier")["is_late"].mean()
    baseline = fixed_rule_actions(test, late_rate)

    naive = test["planned_transit_h"].to_numpy(dtype=float)
    metrics = {
        "n_shipments": int(len(df)),
        "n_train": int(len(train)),
        "n_calib": int(len(calib)),
        "n_test": int(len(test)),
        "eta_mae_h": float(np.mean(np.abs(pred_te - y_te))),
        "eta_rmse_h": float(np.sqrt(np.mean((pred_te - y_te) ** 2))),
        "quoted_mae_h": float(np.mean(np.abs(naive - y_te))),
        "eta_bias_h": float(np.mean(pred_te - y_te)),
        "quoted_bias_h": float(np.mean(naive - y_te)),
        "coverage": pd.DataFrame(coverage_rows),
        "mae_by_horizon": mae_by_horizon(y_te, pred_te, naive),
        "late_rate_test": float(y_late.mean()),
        "brier_raw": brier_score(y_late, p_raw),
        "brier_calibrated": brier_score(y_late, p_cal),
        "ece_raw": expected_calibration_error(y_late, p_raw),
        "ece_calibrated": expected_calibration_error(y_late, p_cal),
        "reliability_raw": reliability_table(y_late, p_raw),
        "reliability_calibrated": reliability_table(y_late, p_cal),
        "p_test_calibrated": p_cal,
        "decision": compare_to_baseline(p_cal, y_late, baseline, costs),
        "decision_uncalibrated": compare_to_baseline(p_raw, y_late, baseline, costs),
        "cost_thresholds": costs.thresholds(),
    }
    return PipelineResult(
        data=df,
        splits={"train": train, "calib": calib, "test": test},
        X={"train": X_tr, "train_scoring": X_ref, "calib": X_cal, "test": X_te},
        eta=eta,
        conformal=conformal,
        risk=risk,
        metrics=metrics,
    )
