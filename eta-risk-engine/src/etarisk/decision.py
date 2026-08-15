"""Cost-matrix decision layer.

A calibrated probability is not a decision. What the control tower actually
needs is: for this shipment, do nothing, notify the consignee, or expedite. The
cost matrix below is explicit and editable because in every real deployment the
argument is about these numbers, not about the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["CostMatrix", "optimal_actions", "evaluate_policy", "compare_to_baseline", "ACTIONS"]

ACTIONS = ("nothing", "notify", "expedite")


@dataclass(frozen=True)
class CostMatrix:
    """Cost in currency units of taking each action, per realised outcome.

    Defaults are ordinary mid-value freight: a missed delivery costs a service
    credit plus downstream expediting at the receiving site; a notification is
    cheap but not free (it consumes planner attention and erodes trust if it
    fires on nothing); an expedite has a real freight bill whether or not the
    shipment was going to be late, and only partially recovers the lateness.
    """

    late: dict[str, float] = field(
        default_factory=lambda: {"nothing": 600.0, "notify": 430.0, "expedite": 150.0}
    )
    on_time: dict[str, float] = field(
        default_factory=lambda: {"nothing": 0.0, "notify": 80.0, "expedite": 360.0}
    )

    def expected_cost(self, p_late: np.ndarray) -> np.ndarray:
        """(n, 3) matrix of expected cost per action."""
        p = np.asarray(p_late, dtype=float).reshape(-1, 1)
        late = np.array([[self.late[a] for a in ACTIONS]])
        ok = np.array([[self.on_time[a] for a in ACTIONS]])
        return p * late + (1 - p) * ok

    def realised(self, action_idx: np.ndarray, is_late: np.ndarray) -> np.ndarray:
        late = np.array([self.late[a] for a in ACTIONS])
        ok = np.array([self.on_time[a] for a in ACTIONS])
        y = np.asarray(is_late).astype(bool)
        return np.where(y, late[action_idx], ok[action_idx])

    def thresholds(self) -> dict[str, float]:
        """Indifference probabilities between adjacent actions."""
        def cross(a: str, b: str) -> float:
            num = self.on_time[a] - self.on_time[b]
            den = num + self.late[b] - self.late[a]
            return float(num / den) if den else float("nan")

        return {"nothing_to_notify": cross("nothing", "notify"), "notify_to_expedite": cross("notify", "expedite")}


def optimal_actions(p_late: np.ndarray, costs: CostMatrix) -> np.ndarray:
    """Index into ACTIONS minimising expected cost for each shipment."""
    return np.asarray(costs.expected_cost(p_late).argmin(axis=1), dtype=int)


def evaluate_policy(action_idx: np.ndarray, is_late: np.ndarray, costs: CostMatrix) -> dict[str, float]:
    realised = costs.realised(action_idx, is_late)
    out = {
        "mean_cost": float(realised.mean()),
        "total_cost": float(realised.sum()),
    }
    for i, name in enumerate(ACTIONS):
        out[f"share_{name}"] = float(np.mean(action_idx == i))
    return out


def fixed_rule_actions(
    df: pd.DataFrame, late_rate_by_carrier: pd.Series, threshold: float | None = None
) -> np.ndarray:
    """The rule most control towers actually run: segment by carrier scorecard.

    Expedite anything on a carrier whose historical late rate is above the
    scorecard cut (the median carrier by default), notify everything else. It is
    a per-carrier constant, so it cannot separate the bad week on a good carrier
    from the good week on a bad one -- which is the whole point of the model.
    """
    if threshold is None:
        threshold = float(late_rate_by_carrier.median())
    rate = df["carrier"].map(late_rate_by_carrier).fillna(late_rate_by_carrier.mean()).to_numpy()
    return np.where(rate > threshold, ACTIONS.index("expedite"), ACTIONS.index("notify")).astype(int)


def compare_to_baseline(
    p_late: np.ndarray,
    is_late: np.ndarray,
    baseline_actions: np.ndarray,
    costs: CostMatrix | None = None,
) -> dict[str, float]:
    """Model policy vs a fixed rule, on the same shipments and cost matrix."""
    costs = costs or CostMatrix()
    model = evaluate_policy(optimal_actions(p_late, costs), is_late, costs)
    base = evaluate_policy(np.asarray(baseline_actions, dtype=int), is_late, costs)
    oracle = evaluate_policy(optimal_actions(np.asarray(is_late).astype(float), costs), is_late, costs)
    saved = base["mean_cost"] - model["mean_cost"]
    head = base["mean_cost"] - oracle["mean_cost"]
    return {
        "model_cost_per_shipment": model["mean_cost"],
        "baseline_cost_per_shipment": base["mean_cost"],
        "oracle_cost_per_shipment": oracle["mean_cost"],
        "saved_per_shipment": saved,
        "saved_pct": 100.0 * saved / base["mean_cost"] if base["mean_cost"] else 0.0,
        "pct_of_oracle_gap_captured": 100.0 * saved / head if head else float("nan"),
        "model_share_expedite": model["share_expedite"],
        "baseline_share_expedite": base["share_expedite"],
    }
