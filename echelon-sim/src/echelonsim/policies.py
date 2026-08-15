"""Replenishment policies.

Same policy shapes as classical single-echelon inventory theory -- base-stock,
``(s,S)``, ``(R,S)`` -- because that is what planning systems actually run, and
because the interesting multi-echelon result is what those familiar policies do
when you stack four of them in series rather than what an exotic policy would do.

Every policy takes an inventory position and a forecast and returns an order
quantity. The protection interval is passed in by the node (``R + L``, review
period plus lead time -- not ``L``, which is the single most common
off-by-one-interval error in production systems and understates safety stock by
``z*sigma*(sqrt(R+L) - sqrt(L))``).

:class:`Batched` is a decorator rather than a parameter on each policy. Order
batching is one of Lee, Padmanabhan & Whang's (1997) four causes of bullwhip and
it needs to be switchable independently of the policy shape for the variance
decomposition to attribute anything.

References
----------
Clark & Scarf (1960), "Optimal policies for a multi-echelon inventory problem".
Lee, Padmanabhan & Whang (1997), "Information distortion in a supply chain: the
bullwhip effect", Management Science 43(4).
Silver, Pyke & Thomas (2016), "Inventory and Production Management in Supply
Chains", 3e, Ch. 6-8.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .forecast import Forecast

__all__ = [
    "Policy",
    "BaseStock",
    "SsPolicy",
    "RSPolicy",
    "Batched",
    "OrderDecision",
]


@dataclass(frozen=True)
class OrderDecision:
    """What the policy decided, and the intermediate quantities behind it.

    The target level is carried out of the policy deliberately: when a chain
    misbehaves the first diagnostic question is always "did the target move or
    did the position move", and a policy that only returns a quantity cannot
    answer it.
    """

    quantity: float
    target_level: float
    reorder_point: Optional[float] = None
    raw_quantity: float = 0.0


class Policy(ABC):
    """Maps (inventory position, forecast, protection interval) to an order."""

    #: Whether the node may skip an order when the policy declines to buy.
    can_decline: bool = True

    @abstractmethod
    def decide(self, position: float, forecast: Forecast, protection: float) -> OrderDecision:
        ...

    def target(self, forecast: Forecast, protection: float) -> float:
        return self.decide(math.inf, forecast, protection).target_level


@dataclass
class BaseStock(Policy):
    """Order-up-to ``S`` every review, with ``S`` rebuilt from the live forecast.

    ``S = mu_hat * (R + L) + z * sigma_hat * sqrt(R + L)``

    This is the policy that produces textbook bullwhip: because ``S`` is a
    function of the forecast, a demand observation moves both the position *and*
    the target, and the order absorbs both moves. The amplification is a
    property of an optimising agent, not of a careless one.

    ``allow_returns`` controls whether a negative order (a return to the
    supplier) is permitted. The analytic bullwhip results assume it is; every
    real chain forbids it. Keeping it as a flag lets the test suite check the
    simulator against the closed form and the experiments run the realistic case.
    """

    z: float = 1.645
    allow_returns: bool = False
    fixed_target: Optional[float] = None

    def decide(self, position: float, forecast: Forecast, protection: float) -> OrderDecision:
        if self.fixed_target is not None:
            target = float(self.fixed_target)
        else:
            target = forecast.mean * protection + self.z * forecast.std * math.sqrt(max(protection, 0.0))
        raw = target - position
        quantity = raw if self.allow_returns else max(0.0, raw)
        return OrderDecision(quantity=quantity, target_level=target, raw_quantity=raw)


@dataclass
class SsPolicy(Policy):
    """``(s, S)``: order up to ``S`` only when the position has fallen to ``s``.

    ``s`` is the same forecast-driven reorder point as :class:`BaseStock`;
    ``S = s + Q`` where ``Q`` is either given or sized from a demand multiple.
    The gap between ``s`` and ``S`` is what makes this a batching policy: orders
    become lumpy even when demand is smooth, which is a second, independent
    route to bullwhip.
    """

    z: float = 1.645
    order_quantity: Optional[float] = None
    quantity_periods: float = 4.0

    def decide(self, position: float, forecast: Forecast, protection: float) -> OrderDecision:
        reorder = forecast.mean * protection + self.z * forecast.std * math.sqrt(max(protection, 0.0))
        gap = self.order_quantity if self.order_quantity is not None else forecast.mean * self.quantity_periods
        gap = max(gap, 1e-9)
        target = reorder + gap
        if position <= reorder:
            # Order up to S; if the position has undershot far below s the
            # single order can exceed Q, which is correct.
            raw = target - position
            return OrderDecision(max(0.0, raw), target, reorder, raw)
        return OrderDecision(0.0, target, reorder, 0.0)


@dataclass
class RSPolicy(Policy):
    """``(R, S)`` periodic review order-up-to.

    Mechanically identical to :class:`BaseStock` at a review epoch; it exists as
    a separate class because the review period belongs to the policy in most
    planning systems and because naming it ``(R,S)`` is how a practitioner will
    look for it. ``review_period`` is advisory: the node reads it when the
    network is built.
    """

    z: float = 1.645
    review_period: int = 4
    allow_returns: bool = False

    def decide(self, position: float, forecast: Forecast, protection: float) -> OrderDecision:
        target = forecast.mean * protection + self.z * forecast.std * math.sqrt(max(protection, 0.0))
        raw = target - position
        quantity = raw if self.allow_returns else max(0.0, raw)
        return OrderDecision(quantity=quantity, target_level=target, raw_quantity=raw)


@dataclass
class Batched(Policy):
    """Decorator: round the inner policy's order to a shippable quantity.

    ``multiple`` is the case/pallet/truck quantity -- orders are rounded *up* to
    a multiple of it. ``minimum`` is a minimum order quantity below which the
    order is suppressed entirely (not rounded up), which is the behaviour of
    almost every purchasing system and is what turns a steady trickle of demand
    into a periodic burst.

    Rounding up is a bias, not just a variance change: expected inventory rises
    by roughly ``multiple / 2``. Both effects are measured separately in the
    decomposition.
    """

    inner: Policy = None  # type: ignore[assignment]
    multiple: float = 1.0
    minimum: float = 0.0

    def __post_init__(self) -> None:
        if self.inner is None:
            raise ValueError("Batched needs an inner policy")
        if self.multiple <= 0:
            raise ValueError("multiple must be positive")

    def decide(self, position: float, forecast: Forecast, protection: float) -> OrderDecision:
        decision = self.inner.decide(position, forecast, protection)
        raw = decision.quantity
        if raw <= 0:
            return OrderDecision(0.0, decision.target_level, decision.reorder_point, raw)
        if raw < self.minimum:
            return OrderDecision(0.0, decision.target_level, decision.reorder_point, raw)
        rounded = math.ceil(raw / self.multiple - 1e-9) * self.multiple
        rounded = max(rounded, self.minimum)
        return OrderDecision(rounded, decision.target_level, decision.reorder_point, raw)
