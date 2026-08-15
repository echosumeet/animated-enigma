"""Decision quality: mapping forecast error onto inventory cost.

Accuracy metrics are a proxy. The thing the business feels is the cost of the order
quantity the forecast produced. This module closes that loop with a periodic-review
order-up-to policy and a newsvendor critical ratio (Silver, Pyke & Peterson, 1998,
ch. 7; Chopra & Meindl, ch. 12), then reports realised cost against the irreducible
cost a perfect forecast would still incur under the same service target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class InventoryEconomics:
    """Cost parameters for one product family."""

    unit_cost: float = 12.0
    holding_rate_annual: float = 0.25
    shortage_multiple: float = 3.0
    review_period_days: int = 7

    @property
    def holding_cost_per_unit(self) -> float:
        """Carrying cost of one leftover unit over one review period."""
        return self.unit_cost * self.holding_rate_annual * self.review_period_days / 365.0

    @property
    def shortage_cost_per_unit(self) -> float:
        """Lost-margin plus expedite penalty for one unit short."""
        return self.unit_cost * self.shortage_multiple

    @property
    def critical_ratio(self) -> float:
        ce, co = self.shortage_cost_per_unit, self.holding_cost_per_unit
        return ce / (ce + co)

    @property
    def safety_factor(self) -> float:
        return float(norm.ppf(self.critical_ratio))


def order_quantity(forecast: np.ndarray, sigma: np.ndarray, econ: InventoryEconomics) -> np.ndarray:
    """Order-up-to level = forecast over the review period + safety stock."""
    return np.maximum(0.0, np.asarray(forecast, float) + econ.safety_factor * np.asarray(sigma, float))


def period_cost(actual: np.ndarray, order: np.ndarray, econ: InventoryEconomics) -> np.ndarray:
    actual = np.asarray(actual, float)
    order = np.asarray(order, float)
    over = np.maximum(0.0, order - actual)
    under = np.maximum(0.0, actual - order)
    return over * econ.holding_cost_per_unit + under * econ.shortage_cost_per_unit


@dataclass
class DecisionQuality:
    units: float
    realised_cost: float
    irreducible_cost: float
    fill_rate: float
    over_order_units: float
    under_order_units: float

    @property
    def regret(self) -> float:
        """Cost attributable to forecast error alone."""
        return self.realised_cost - self.irreducible_cost

    @property
    def cost_per_unit(self) -> float:
        return self.realised_cost / max(self.units, 1e-9)

    @property
    def regret_per_unit(self) -> float:
        return self.regret / max(self.units, 1e-9)

    def as_dict(self) -> dict[str, float]:
        return {
            "units": self.units,
            "realised_cost": self.realised_cost,
            "irreducible_cost": self.irreducible_cost,
            "regret": self.regret,
            "cost_per_unit": self.cost_per_unit,
            "regret_per_unit": self.regret_per_unit,
            "fill_rate": self.fill_rate,
        }


def evaluate_decisions(
    predictions: pd.DataFrame,
    econ: InventoryEconomics | None = None,
    y: str = "y",
    yhat: str = "yhat",
    sigma_col: str = "sigma",
) -> DecisionQuality:
    """Convert a prediction frame into inventory cost and service outcomes.

    ``irreducible_cost`` is the cost of running the same safety-stock policy with a
    perfect point forecast; the gap between it and ``realised_cost`` is the money the
    forecast error is actually costing, which is the number to put in a business case.
    """
    econ = econ or InventoryEconomics()
    actual = predictions[y].to_numpy(float)
    sigma = (
        predictions[sigma_col].to_numpy(float)
        if sigma_col in predictions
        else np.full(len(predictions), float(np.std(actual - predictions[yhat].to_numpy(float))))
    )
    order = order_quantity(predictions[yhat].to_numpy(float), sigma, econ)
    oracle = order_quantity(actual, sigma, econ)
    cost = period_cost(actual, order, econ).sum()
    under = float(np.maximum(0.0, actual - order).sum())
    return DecisionQuality(
        units=float(actual.sum()),
        realised_cost=float(cost),
        irreducible_cost=float(period_cost(actual, oracle, econ).sum()),
        fill_rate=float(1.0 - under / max(actual.sum(), 1e-9)),
        over_order_units=float(np.maximum(0.0, order - actual).sum()),
        under_order_units=under,
    )
