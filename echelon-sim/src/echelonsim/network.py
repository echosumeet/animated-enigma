"""Nodes and the multi-echelon network they form.

A node is a stocking point with:

* on-hand inventory,
* a FIFO backlog of orders it has received and not yet shipped,
* an inbound pipeline of shipments in transit to it,
* a replenishment policy and a forecaster,
* a review period, an order (information) lead time, and a transit-time
  distribution on the arc from its supplier,
* optionally a per-period throughput capacity.

Inventory position is ``on hand - backlog + on order``. The backlog term is the
one that gets dropped in practice, and dropping it is catastrophic: a node that
ignores its own unfilled customer orders when computing its position will
re-order the same shortfall every review until the goods arrive, which is a
textbook self-inflicted amplifier.

Echelon inventory position -- installation stock plus everything downstream,
less end-customer backorders -- is the Clark & Scarf (1960) state variable and
is what the information-sharing modes in :mod:`echelonsim.information` switch
the policy over to.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .forecast import Forecaster, MovingAverage
from .leadtime import Deterministic, LeadTime
from .policies import BaseStock, Policy

__all__ = [
    "InfoMode",
    "Allocation",
    "Node",
    "SupplyNetwork",
    "OrderLine",
    "serial_chain",
    "divergent_network",
]


class InfoMode(str, Enum):
    """What each node is allowed to see when it places an order.

    ``DECENTRALIZED``
        The classic arrangement. A node observes only the orders its immediate
        customers place, forecasts from that stream, and manages its own
        installation stock. Every echelon is forecasting a signal that has
        already been distorted by the echelon below it.

    ``POS_SHARED``
        Point-of-sale demand is broadcast upstream. Every node forecasts from
        true end-customer demand but still manages installation stock and still
        has its own review period and lead time. This removes the
        *signal-processing* cascade only.

    ``VMI``
        Vendor-managed inventory, modelled as centralised echelon base-stock:
        POS forecasting, echelon inventory position (so a node sees the stock it
        already pushed downstream), and continuous review at the retailer. This
        is the Clark-Scarf state variable, and it is the arrangement Cachon &
        Fisher (2000) and Disney & Towill (2003) find delivers most of the
        benefit attributed to "information sharing".
    """

    DECENTRALIZED = "decentralized"
    POS_SHARED = "pos_shared"
    VMI = "vmi"


class Allocation(str, Enum):
    """How a node splits short supply across competing customer orders.

    ``FIFO`` fills the oldest order first. ``PROPORTIONAL`` gives every waiting
    customer the same fraction of what they asked for -- which is the rule that
    creates Lee, Padmanabhan & Whang's rationing game, because a customer who
    knows allocation is pro rata has a strict incentive to inflate its order.
    """

    FIFO = "fifo"
    PROPORTIONAL = "proportional"


@dataclass
class OrderLine:
    """One customer order sitting in a node's backlog."""

    customer: Optional[str]
    quantity: float
    remaining: float
    period: int


@dataclass
class Node:
    """A stocking location. Configuration and mutable state in one object.

    The network is deep-copied per replication, so mixing configuration and
    state costs nothing and keeps the model readable.
    """

    name: str
    level: int
    supplier: Optional[str] = None
    policy: Policy = field(default_factory=BaseStock)
    forecaster: Forecaster = field(default_factory=lambda: MovingAverage(10))
    review_period: int = 1
    order_lead_time: int = 1
    transit: LeadTime = field(default_factory=lambda: Deterministic(2.0))
    capacity: Optional[float] = None
    allocation: Allocation = Allocation.FIFO
    holding_cost: float = 1.0
    backorder_cost: float = 10.0
    is_source: bool = False
    initial_periods_of_stock: float = 0.0
    review_offset: int = 0

    # -- mutable state --------------------------------------------------
    on_hand: float = field(default=0.0, repr=False)
    #: Ordered from the supplier but not yet dispatched. Still physically the
    #: supplier's stock (or nobody's, if the supplier is short).
    on_order: float = field(default=0.0, repr=False)
    #: Dispatched and moving. Physically real, owned by nobody's shelf.
    in_transit: float = field(default=0.0, repr=False)
    backlog: Deque[OrderLine] = field(default_factory=deque, repr=False)
    customers: List[str] = field(default_factory=list, repr=False)
    capacity_factor: float = field(default=1.0, repr=False)
    supply_available: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        if self.order_lead_time < 1:
            raise ValueError(
                f"{self.name}: order_lead_time must be at least 1 period. "
                "A zero information delay would make the order visible only "
                "after the supplier's allocation step anyway, so encoding it as "
                "zero would misstate the model rather than speed it up."
            )
        if self.review_period < 1:
            raise ValueError(f"{self.name}: review_period must be at least 1")

    # -- state helpers ---------------------------------------------------
    @property
    def backlog_units(self) -> float:
        return math.fsum(line.remaining for line in self.backlog)

    @property
    def outstanding(self) -> float:
        """Everything ordered and not yet received, dispatched or not."""
        return self.on_order + self.in_transit

    def inventory_position(self) -> float:
        return self.on_hand - self.backlog_units + self.outstanding

    @property
    def expected_lead_time(self) -> float:
        """Order lead time plus mean transit -- what the target must cover."""
        return float(self.order_lead_time) + float(self.transit.mean)

    @property
    def protection_interval(self) -> float:
        """Periods of demand a replenishment target must cover: ``R + L - 1``.

        Not ``L`` -- that is the classic understatement, and it costs
        ``z*sigma*(sqrt(R+L-1) - sqrt(L))`` of missing safety stock.

        But also not ``R + L``, which is what the textbook expression gives and
        what most planning systems implement. The extra period is real only if
        the review happens *before* the current period's demand is served. In
        this model (see the phase table in ``simulation.py``) allocation runs at
        priority 30 and review at priority 50, so by the time a node reviews, it
        has already shipped today's orders and today's demand is not at risk.

        Derivation for ``R = 1``: an order placed at the end of ``t`` arrives at
        the start of ``t + L``. Net inventory at the end of any period is then
        ``S`` minus the last ``L`` demands, so the risk period is ``L``, which is
        ``R + L - 1``. For general ``R`` the low point sits at the end of period
        ``t + R + L - 1`` and the same algebra gives ``R + L - 1``.

        This is not pedantry. Using ``R + L`` here inflates the target by a full
        period of mean demand -- 100 units on a 100-unit-per-period item, which
        at four echelons is 400 units of system inventory bought to fix an
        arithmetic convention.
        """
        return max(1.0, float(self.review_period) + self.expected_lead_time - 1.0)

    def is_review_period(self, period: int) -> bool:
        return (period - self.review_offset) % self.review_period == 0 and period >= self.review_offset

    def receive_order(self, customer: Optional[str], quantity: float, period: int) -> None:
        if quantity <= 0:
            return
        self.backlog.append(OrderLine(customer, quantity, quantity, period))

    def capacity_this_period(self) -> float:
        if self.capacity is None:
            return math.inf
        return max(0.0, self.capacity * self.capacity_factor)

    def reset_state(self, mean_demand: float) -> None:
        self.on_hand = self.initial_periods_of_stock * mean_demand
        self.on_order = 0.0
        self.in_transit = 0.0
        self.backlog = deque()
        self.capacity_factor = 1.0
        self.supply_available = True
        self.transit.reset()


class SupplyNetwork:
    """A single-sourcing tree: retailers at level 0, an external source at the top."""

    def __init__(self, nodes: Sequence[Node], info_mode: InfoMode = InfoMode.DECENTRALIZED) -> None:
        self.nodes: Dict[str, Node] = {n.name: n for n in nodes}
        if len(self.nodes) != len(nodes):
            raise ValueError("duplicate node names")
        self.info_mode = InfoMode(info_mode)
        self._wire()
        self._validate()

    def _wire(self) -> None:
        for node in self.nodes.values():
            node.customers = []
        for node in self.nodes.values():
            if node.supplier is not None:
                if node.supplier not in self.nodes:
                    raise ValueError(f"{node.name}: unknown supplier {node.supplier!r}")
                self.nodes[node.supplier].customers.append(node.name)
        for node in self.nodes.values():
            node.customers.sort()

    def _validate(self) -> None:
        roots = [n for n in self.nodes.values() if n.supplier is None]
        if len(roots) != 1:
            raise ValueError(f"expected exactly one source node, found {len(roots)}")
        if not roots[0].is_source:
            raise ValueError(f"root node {roots[0].name!r} must have is_source=True")
        # No cycles: walking up from any node must terminate at the root.
        for node in self.nodes.values():
            seen = {node.name}
            cursor = node
            while cursor.supplier is not None:
                cursor = self.nodes[cursor.supplier]
                if cursor.name in seen:
                    raise ValueError("supply graph contains a cycle")
                seen.add(cursor.name)
        for node in self.nodes.values():
            if node.level == 0 and node.is_source:
                raise ValueError("a retailer cannot also be the external source")

    # -- topology --------------------------------------------------------
    @property
    def retailers(self) -> List[Node]:
        return [n for n in self.ordered_nodes if n.level == 0]

    @property
    def source(self) -> Node:
        return next(n for n in self.nodes.values() if n.supplier is None)

    @property
    def ordered_nodes(self) -> List[Node]:
        """Deterministic order: by level, then by name. Reproducibility depends on it."""
        return sorted(self.nodes.values(), key=lambda n: (n.level, n.name))

    @property
    def max_level(self) -> int:
        return max(n.level for n in self.nodes.values())

    def subtree(self, name: str) -> List[Node]:
        """The node and everything it ultimately supplies."""
        out: List[Node] = []
        stack = [name]
        while stack:
            current = self.nodes[stack.pop()]
            out.append(current)
            stack.extend(current.customers)
        return out

    def stocking_nodes(self) -> List[Node]:
        return [n for n in self.ordered_nodes if not n.is_source]

    # -- state -----------------------------------------------------------
    def echelon_inventory_position(self, name: str) -> float:
        """Clark-Scarf echelon inventory position for ``name``.

        ``sum over the subtree of (on hand + in transit)``, plus **this node's**
        outstanding order on its own supplier, minus end-customer backorders.

        The subtlety is which pipeline terms count. Units a downstream node has
        *ordered but not been shipped* are not in the echelon twice -- they are
        still sitting in this node's on-hand, or they do not exist at all
        because this node is short. Counting them (the obvious implementation,
        using a single ``outstanding`` field) inflates the echelon position by
        the internal order book and deadlocks the chain: every node sees a
        position already at target and nobody ever orders. Units already
        dispatched between two nodes of the subtree are different -- they are
        physically real and on nobody's shelf, so they count exactly once.

        Internal backlogs are not subtracted: a distributor owing a retailer has
        not lost anything to the echelon. Only end-customer backorders have.
        """
        total = 0.0
        for node in self.subtree(name):
            if node.is_source:
                continue
            total += node.on_hand + node.in_transit
            if node.level == 0:
                total -= math.fsum(
                    line.remaining for line in node.backlog if line.customer is None
                )
        return total + self.nodes[name].on_order

    def cumulative_protection(self, name: str) -> float:
        """Protection interval for an echelon target: this node's plus everything below.

        An echelon target is compared against echelon stock, which already
        contains the downstream inventory -- so it has to be large enough to
        *fund* that downstream inventory as well as cover this node's own lead
        time. Setting every echelon target to its local ``R + L - 1`` makes the
        upstream targets equal to the downstream ones, which means the upstream
        nodes carry no installation stock at all and the chain starves.

        The longest downstream path is used in a divergent network: the target
        has to be adequate for the slowest branch.
        """
        node = self.nodes[name]
        downstream = max(
            (self.cumulative_protection(child) for child in node.customers),
            default=0.0,
        )
        return node.protection_interval + downstream

    def position_for_policy(self, name: str) -> float:
        node = self.nodes[name]
        if self.info_mode is InfoMode.VMI:
            return self.echelon_inventory_position(name)
        return node.inventory_position()

    def protection_for_policy(self, name: str) -> float:
        if self.info_mode is InfoMode.VMI:
            return self.cumulative_protection(name)
        return self.nodes[name].protection_interval

    def reset(self, mean_demand: float) -> None:
        for node in self.nodes.values():
            node.reset_state(mean_demand)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SupplyNetwork({len(self.nodes)} nodes, mode={self.info_mode.value})"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def serial_chain(
    levels: int = 4,
    policy_factory=None,
    forecaster_factory=None,
    review_period: int = 1,
    order_lead_time: int = 1,
    transit_factory=None,
    capacity: Optional[float] = None,
    initial_periods_of_stock: float = 4.0,
    info_mode: InfoMode = InfoMode.DECENTRALIZED,
    names: Optional[Sequence[str]] = None,
) -> SupplyNetwork:
    """The canonical serial chain: retailer -> distributor -> factory -> supplier.

    ``levels`` counts the stocking echelons; an external source is appended on
    top of them, so ``levels=3`` yields four nodes of which three hold stock.
    """
    if levels < 1:
        raise ValueError("need at least one echelon")
    default_names = ["retailer", "distributor", "factory", "tier2", "tier3", "tier4"]
    labels = list(names) if names is not None else default_names[:levels]
    if len(labels) < levels:
        labels += [f"echelon{i}" for i in range(len(labels), levels)]

    policy_factory = policy_factory or (lambda level: BaseStock())
    forecaster_factory = forecaster_factory or (lambda level: MovingAverage(10))
    transit_factory = transit_factory or (lambda level: Deterministic(2.0))

    nodes: List[Node] = []
    for level in range(levels):
        nodes.append(
            Node(
                name=labels[level],
                level=level,
                supplier=labels[level + 1] if level + 1 < levels else "source",
                policy=policy_factory(level),
                forecaster=forecaster_factory(level),
                review_period=review_period,
                order_lead_time=order_lead_time,
                transit=transit_factory(level),
                capacity=capacity if level == levels - 1 else None,
                initial_periods_of_stock=initial_periods_of_stock,
            )
        )
    nodes.append(
        Node(
            name="source",
            level=levels,
            supplier=None,
            is_source=True,
            transit=Deterministic(0.0),
            order_lead_time=1,
            holding_cost=0.0,
            backorder_cost=0.0,
        )
    )
    return SupplyNetwork(nodes, info_mode=info_mode)


def divergent_network(
    n_retailers: int = 3,
    policy_factory=None,
    forecaster_factory=None,
    review_period: int = 1,
    order_lead_time: int = 1,
    transit_factory=None,
    factory_capacity: Optional[float] = None,
    initial_periods_of_stock: float = 4.0,
    info_mode: InfoMode = InfoMode.DECENTRALIZED,
) -> SupplyNetwork:
    """``n`` retailers -> one distributor -> one factory -> external supplier.

    The divergent case is where allocation policy starts to matter: when the
    distributor is short it must choose between retailers, and the choice is a
    real decision with a real incentive attached to it.
    """
    if n_retailers < 1:
        raise ValueError("need at least one retailer")
    policy_factory = policy_factory or (lambda level: BaseStock())
    forecaster_factory = forecaster_factory or (lambda level: MovingAverage(10))
    transit_factory = transit_factory or (lambda level: Deterministic(2.0))

    nodes: List[Node] = []
    for index in range(n_retailers):
        nodes.append(
            Node(
                name=f"retailer{index + 1}",
                level=0,
                supplier="distributor",
                policy=policy_factory(0),
                forecaster=forecaster_factory(0),
                review_period=review_period,
                order_lead_time=order_lead_time,
                transit=transit_factory(0),
                initial_periods_of_stock=initial_periods_of_stock,
            )
        )
    nodes.append(
        Node(
            name="distributor",
            level=1,
            supplier="factory",
            policy=policy_factory(1),
            forecaster=forecaster_factory(1),
            review_period=review_period,
            order_lead_time=order_lead_time,
            transit=transit_factory(1),
            allocation=Allocation.PROPORTIONAL,
            initial_periods_of_stock=initial_periods_of_stock,
        )
    )
    nodes.append(
        Node(
            name="factory",
            level=2,
            supplier="source",
            policy=policy_factory(2),
            forecaster=forecaster_factory(2),
            review_period=review_period,
            order_lead_time=order_lead_time,
            transit=transit_factory(2),
            capacity=factory_capacity,
            initial_periods_of_stock=initial_periods_of_stock,
        )
    )
    nodes.append(
        Node(
            name="source",
            level=3,
            supplier=None,
            is_source=True,
            transit=Deterministic(0.0),
            order_lead_time=1,
            holding_cost=0.0,
            backorder_cost=0.0,
        )
    )
    return SupplyNetwork(nodes, info_mode=info_mode)
