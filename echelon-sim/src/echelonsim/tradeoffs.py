"""Lead-time length versus lead-time variability, at a fixed service target.

The question this module answers is the one that decides sourcing arguments:
given a fixed fill-rate commitment, is it worth more to make the supplier
*faster* or to make it *more consistent*?

The theory says consistency, and says it loudly. The standard deviation of
lead-time demand is

    ``sigma_DL = sqrt(L * sigma_d^2 + d_bar^2 * sigma_L^2)``

(Silver, Pyke & Thomas 2016, S6.7; Eppen & Martin 1988). The first term is
linear in the mean lead time; the second is quadratic in mean demand and
quadratic in lead-time standard deviation. For any item with a demand
coefficient of variation below about 0.5 -- which is most items -- the second
term dominates as soon as the lead time is even mildly unreliable.

Practitioners rarely see this, for a structural reason: mean lead time is on
every supplier scorecard and lead-time variance is on almost none. Suppliers are
therefore optimised on the term that matters less, and "we cut lead time from 6
weeks to 4" arrives alongside a *widened* delivery window that eats the gain.

Comparing the two fairly requires holding service constant, not holding the
policy constant. A longer or more erratic lead time under a fixed ``z`` degrades
service *and* raises inventory, and reading off the inventory alone flatters it.
So every cell in the grid is calibrated: ``z`` is solved by bisection until the
simulated fill rate hits the target, and only then is inventory compared.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .experiments import DEFAULT_CONFIG, ScenarioOutcome, merge_config, run_scenario
from .metrics import ConfidenceInterval

__all__ = [
    "lead_time_demand_sigma",
    "CalibrationResult",
    "calibrate_service",
    "LeadTimeCell",
    "lead_time_grid",
]


def lead_time_demand_sigma(
    mean_lead: float, demand_mean: float, demand_std: float, lead_std: float
) -> float:
    """``sqrt(L*sigma_d^2 + d_bar^2*sigma_L^2)`` -- the convolution formula."""
    return math.sqrt(mean_lead * demand_std ** 2 + demand_mean ** 2 * lead_std ** 2)


@dataclass
class CalibrationResult:
    z: float
    achieved: ConfidenceInterval
    iterations: int
    converged: bool
    outcome: ScenarioOutcome


def calibrate_service(
    config: Dict[str, Any],
    target_fill: float = 0.95,
    metric: str = "service_fill_rate",
    low: float = -1.0,
    high: float = 8.0,
    tolerance: float = 5e-4,
    max_iterations: int = 24,
    warmup: Optional[int] = None,
) -> CalibrationResult:
    """Bisect on the safety factor ``z`` until the simulated fill rate hits target.

    Bisection rather than a stochastic root finder because every evaluation uses
    the *same* seed: the simulated fill rate is a deterministic, monotone
    function of ``z`` here, so the noisy-optimisation machinery is unnecessary
    and would only add its own variance.

    The lower bracket is negative on purpose. On a short, reliable lead time the
    ``z`` needed for a 95% fill rate can be below zero -- fill rate is a
    unit-weighted measure and a policy can miss a cycle and still ship 95% of
    units. Bracketing at zero would silently peg those cells at the boundary and
    understate how cheap a reliable supplier is.
    """
    config = merge_config(DEFAULT_CONFIG, config)

    def evaluate(z: float) -> ScenarioOutcome:
        return run_scenario(
            merge_config(config, {"policy": {"z": z}}),
            name=f"z={z:.4f}",
            warmup=warmup,
            keep_results=True,
        )

    lo_outcome = evaluate(low)
    if lo_outcome.ci(metric).mean >= target_fill:
        return CalibrationResult(low, lo_outcome.ci(metric), 0, True, lo_outcome)
    hi_outcome = evaluate(high)
    if hi_outcome.ci(metric).mean < target_fill:
        return CalibrationResult(high, hi_outcome.ci(metric), 0, False, hi_outcome)

    outcome = hi_outcome
    mid = 0.5 * (low + high)
    iterations = 0
    converged = False
    for iterations in range(1, max_iterations + 1):
        mid = 0.5 * (low + high)
        outcome = evaluate(mid)
        achieved = outcome.ci(metric).mean
        if abs(achieved - target_fill) <= tolerance:
            converged = True
            break
        if achieved < target_fill:
            low = mid
        else:
            high = mid
    return CalibrationResult(
        z=mid,
        achieved=outcome.ci(metric),
        iterations=iterations,
        converged=converged,
        outcome=outcome,
    )


@dataclass
class LeadTimeCell:
    mean_lead: float
    cv_lead: float
    z: float
    fill_rate: ConfidenceInterval
    inventory: ConfidenceInterval
    cost: ConfidenceInterval
    analytic_sigma: float
    converged: bool

    def row(self) -> Tuple[float, float, float, float, float, float]:
        return (
            self.mean_lead,
            self.cv_lead,
            self.z,
            100.0 * self.fill_rate.mean,
            self.inventory.mean,
            self.analytic_sigma,
        )


def lead_time_grid(
    means: Sequence[float] = (2.0, 4.0, 8.0),
    cvs: Sequence[float] = (0.0, 0.25, 0.5),
    target_fill: float = 0.95,
    base_config: Optional[Dict[str, Any]] = None,
    warmup: Optional[int] = None,
) -> List[LeadTimeCell]:
    """Calibrate every (mean, CV) cell to the same fill rate and report inventory.

    A single stocking echelon fed by an always-available source: the point is to
    isolate the lead-time effect, and in a multi-echelon chain the upstream
    stockouts would contaminate it with a second, unrelated mechanism.
    """
    base = merge_config(
        merge_config(
            DEFAULT_CONFIG,
            {
                "topology": {"kind": "serial", "levels": 1},
                "leadtime": {"kind": "gamma", "order_lead_time": 1},
                "run": {"periods": 1040, "replications": 12, "warmup": 60},
            },
        ),
        base_config or {},
    )
    demand_mean = float(base["demand"]["mean"])
    demand_std = float(base["demand"]["std"])

    cells: List[LeadTimeCell] = []
    for mean_lead in means:
        for cv_lead in cvs:
            config = merge_config(
                base, {"leadtime": {"mean": mean_lead, "cv": cv_lead}}
            )
            calibration = calibrate_service(
                config, target_fill=target_fill, warmup=warmup
            )
            outcome = calibration.outcome
            protection = mean_lead + float(base["leadtime"]["order_lead_time"]) + \
                float(base["review_period"]) - 1.0
            cells.append(
                LeadTimeCell(
                    mean_lead=float(mean_lead),
                    cv_lead=float(cv_lead),
                    z=calibration.z,
                    fill_rate=outcome.ci("service_fill_rate"),
                    inventory=outcome.ci("avg_inventory"),
                    cost=outcome.ci("avg_cost"),
                    analytic_sigma=lead_time_demand_sigma(
                        protection, demand_mean, demand_std, mean_lead * cv_lead
                    ),
                    converged=calibration.converged,
                )
            )
    return cells
