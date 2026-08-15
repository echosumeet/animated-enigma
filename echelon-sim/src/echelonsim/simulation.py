"""The multi-echelon simulation model, wired onto the event engine.

Period structure
----------------
Everything below happens at the same integer timestamp. The order is a
modelling decision, encoded as event priorities so it is visible and testable
rather than emergent:

============  ========  ==================================================
priority      phase     what happens
============  ========  ==================================================
``-100``      control   disruptions start and end
``-50``       receive   shipments in transit arrive (any real-valued time)
``10``        demand    end customers place orders on retailers
``20``        orders    downstream replenishment orders reach their supplier
``30..``      allocate  each node ships against its backlog, upstream first
``50..``      review    each node forecasts and orders, downstream first
``90``        record    end-of-period state is written to the series
============  ========  ==================================================

A receipt therefore *is* available to ship in the period it lands, and an order
placed in period ``t`` reaches the supplier's allocation step in period
``t + order_lead_time``. Transit times are real-valued: a shipment dispatched at
``t`` with a 3.4-period transit arrives inside period ``t+3``, before that
period's allocation, and order crossing is possible unless the transit
distribution forbids it.

Why the phases are not "obvious"
--------------------------------
Move the receive phase after allocation and measured fill rate drops by roughly
one period of lead time with no other change; move the review phase before
allocation and every node orders against a position that ignores the shipment
it is about to make. Both are defensible-sounding and both are wrong, and
neither shows up as an error -- only as a number that is quietly off. Making the
sequence explicit is the point.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .demand import DemandProcess, IIDNormal
from .engine import URGENT, Environment, Interrupt, Process
from .network import Allocation, InfoMode, Node, OrderLine, SupplyNetwork
from .rng import StreamBank

__all__ = [
    "PH_RECEIVE",
    "PH_DEMAND",
    "PH_ORDER_IN",
    "PH_FULFILL",
    "PH_REVIEW",
    "PH_RECORD",
    "SupplyOutage",
    "CapacityLoss",
    "DisruptionPlan",
    "NodeSeries",
    "SimulationResult",
    "Simulator",
    "run_simulation",
]

PH_RECEIVE = -50.0
PH_DEMAND = 10.0
PH_ORDER_IN = 20.0
PH_FULFILL = 30.0
PH_REVIEW = 50.0
PH_RECORD = 90.0

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Disruptions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SupplyOutage:
    """``node`` ships nothing for ``duration`` periods starting at ``start``.

    Orders keep arriving and keep queueing; nothing is lost, everything is late.
    That is the realistic failure mode for an upstream outage and it is what
    makes the recovery interesting -- the backlog has to be worked off *on top
    of* ongoing demand, against a capacity that was sized for ongoing demand.
    """

    node: str
    start: int
    duration: int


@dataclass(frozen=True)
class CapacityLoss:
    """``node``'s throughput is multiplied by ``factor`` for ``duration`` periods."""

    node: str
    start: int
    duration: int
    factor: float = 0.5


@dataclass
class DisruptionPlan:
    outages: Sequence[SupplyOutage] = field(default_factory=tuple)
    capacity_losses: Sequence[CapacityLoss] = field(default_factory=tuple)

    def __bool__(self) -> bool:
        return bool(self.outages) or bool(self.capacity_losses)


# ---------------------------------------------------------------------------
# Output series
# ---------------------------------------------------------------------------

@dataclass
class NodeSeries:
    """Per-period record for one node. All arrays have length ``periods``."""

    name: str
    level: int
    periods: int

    def __post_init__(self) -> None:
        zeros = lambda: np.zeros(self.periods, dtype=float)  # noqa: E731
        self.demand_received = zeros()
        self.orders_placed = zeros()
        self.shipped = zeros()
        self.on_hand = zeros()
        self.backlog = zeros()
        self.position = zeros()
        self.echelon_position = zeros()
        self.fill_numerator = zeros()
        self.target = np.full(self.periods, np.nan, dtype=float)

    def slice(self, start: int, stop: Optional[int] = None) -> "NodeSeries":
        stop = self.periods if stop is None else stop
        out = NodeSeries(self.name, self.level, stop - start)
        for attr in (
            "demand_received", "orders_placed", "shipped", "on_hand", "backlog",
            "position", "echelon_position", "fill_numerator", "target",
        ):
            setattr(out, attr, getattr(self, attr)[start:stop].copy())
        return out

    # -- derived metrics -------------------------------------------------
    def fill_rate(self) -> float:
        demand = self.demand_received.sum()
        if demand <= 0:
            return 1.0
        return float(self.fill_numerator.sum() / demand)

    def period_fill_rate(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(
                self.demand_received > _EPS,
                self.fill_numerator / np.maximum(self.demand_received, _EPS),
                1.0,
            )
        return np.clip(out, 0.0, 1.0)

    def bullwhip(self) -> float:
        """``Var(orders placed) / Var(demand received)`` -- local amplification."""
        demand_var = float(np.var(self.demand_received, ddof=1))
        if demand_var <= _EPS:
            return float("nan")
        return float(np.var(self.orders_placed, ddof=1) / demand_var)


@dataclass
class SimulationResult:
    periods: int
    customer_demand: np.ndarray
    nodes: Dict[str, NodeSeries]
    node_levels: Dict[str, int]
    events_processed: int
    holding_cost: Dict[str, float]
    backorder_cost: Dict[str, float]

    def trim(self, warmup: int) -> "SimulationResult":
        """Drop the first ``warmup`` periods from every series."""
        if warmup <= 0:
            return self
        if warmup >= self.periods:
            raise ValueError("warm-up would consume the entire run")
        return SimulationResult(
            periods=self.periods - warmup,
            customer_demand=self.customer_demand[warmup:].copy(),
            nodes={k: v.slice(warmup) for k, v in self.nodes.items()},
            node_levels=dict(self.node_levels),
            events_processed=self.events_processed,
            holding_cost=dict(self.holding_cost),
            backorder_cost=dict(self.backorder_cost),
        )

    def stocking_series(self) -> List[NodeSeries]:
        return [
            self.nodes[name]
            for name in sorted(self.nodes, key=lambda n: (self.node_levels[n], n))
            if name in self.holding_cost
        ]

    def chain_bullwhip(self) -> float:
        """Variance of the most upstream stocking node's orders vs end demand."""
        top = max(self.stocking_series(), key=lambda s: s.level)
        demand_var = float(np.var(self.customer_demand, ddof=1))
        if demand_var <= _EPS:
            return float("nan")
        return float(np.var(top.orders_placed, ddof=1) / demand_var)

    def cumulative_bullwhip(self) -> Dict[str, float]:
        """Per node: ``Var(orders placed) / Var(end-customer demand)``."""
        demand_var = float(np.var(self.customer_demand, ddof=1))
        return {
            s.name: float(np.var(s.orders_placed, ddof=1) / demand_var)
            for s in self.stocking_series()
        }

    def local_bullwhip(self) -> Dict[str, float]:
        return {s.name: s.bullwhip() for s in self.stocking_series()}

    def fill_rates(self) -> Dict[str, float]:
        return {s.name: s.fill_rate() for s in self.stocking_series()}

    def average_cost(self) -> float:
        total = 0.0
        for series in self.stocking_series():
            total += self.holding_cost[series.name] * float(series.on_hand.mean())
            total += self.backorder_cost[series.name] * float(series.backlog.mean())
        return total

    def average_inventory(self) -> float:
        return float(sum(s.on_hand.mean() for s in self.stocking_series()))


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class Simulator:
    """Runs one replication of one configuration."""

    def __init__(
        self,
        network: SupplyNetwork,
        demand: DemandProcess,
        periods: int = 520,
        streams: Optional[StreamBank] = None,
        disruptions: Optional[DisruptionPlan] = None,
        copy_inputs: bool = True,
    ) -> None:
        if periods < 2:
            raise ValueError("need at least 2 periods")
        self.network = copy.deepcopy(network) if copy_inputs else network
        self.demand_spec = demand
        self.periods = int(periods)
        self.streams = streams or StreamBank()
        self.disruptions = disruptions or DisruptionPlan()
        self.env = Environment()

        self.series: Dict[str, NodeSeries] = {
            name: NodeSeries(name, node.level, self.periods)
            for name, node in self.network.nodes.items()
        }
        self.customer_demand = np.zeros(self.periods, dtype=float)
        self._pos: Dict[str, np.ndarray] = {
            name: np.zeros(self.periods, dtype=float) for name in self.network.nodes
        }
        self._retailer_demand: Dict[str, DemandProcess] = {}
        self._ancestors: Dict[str, List[str]] = {}
        self._fulfil_processes: Dict[str, Process] = {}
        self._validate_disruptions()

    # -- setup -----------------------------------------------------------
    def _validate_disruptions(self) -> None:
        for outage in self.disruptions.outages:
            if outage.node not in self.network.nodes:
                raise ValueError(f"outage refers to unknown node {outage.node!r}")
            if outage.duration < 1 or outage.start < 0:
                raise ValueError("outage needs start >= 0 and duration >= 1")
        by_node: Dict[str, List[SupplyOutage]] = {}
        for outage in self.disruptions.outages:
            by_node.setdefault(outage.node, []).append(outage)
        for node_name, events in by_node.items():
            events = sorted(events, key=lambda e: e.start)
            for first, second in zip(events, events[1:]):
                if first.start + first.duration > second.start:
                    raise ValueError(f"overlapping outages on {node_name!r}")
        for loss in self.disruptions.capacity_losses:
            if loss.node not in self.network.nodes:
                raise ValueError(f"capacity loss refers to unknown node {loss.node!r}")
            if self.network.nodes[loss.node].capacity is None:
                raise ValueError(
                    f"node {loss.node!r} has no capacity, so it cannot lose any"
                )

    def _throughput_moments(self, node: Node) -> tuple:
        """Mean and std of the demand stream this node ultimately serves.

        Retailers in a divergent network are independent, so variances add and
        the standard deviation grows with ``sqrt(n)`` -- this is exactly the
        risk-pooling effect, and getting it right matters because it sets the
        upstream nodes' initial forecast and opening stock.
        """
        retailers = [n for n in self.network.subtree(node.name) if n.level == 0]
        if not retailers:
            retailers = self.network.retailers
        mean, std = self.demand_spec.stationary_moments()
        count = max(1, len(retailers))
        return mean * count, std * math.sqrt(count)

    def _initialise(self) -> None:
        mean, _std = self.demand_spec.stationary_moments()
        for node in self.network.nodes.values():
            throughput_mean, throughput_std = self._throughput_moments(node)
            node.reset_state(throughput_mean)
            if not node.is_source:
                node.forecaster.reset(throughput_mean, throughput_std)
            ancestors: List[str] = []
            cursor = node
            while cursor.supplier is not None:
                ancestors.append(cursor.supplier)
                cursor = self.network.nodes[cursor.supplier]
            self._ancestors[node.name] = ancestors
        for retailer in self.network.retailers:
            process = copy.deepcopy(self.demand_spec)
            process.reset()
            self._retailer_demand[retailer.name] = process

    # -- phases ----------------------------------------------------------
    def _demand_phase(self, period: int) -> None:
        total = 0.0
        for retailer in self.network.retailers:
            process = self._retailer_demand[retailer.name]
            rng = self.streams.stream(f"{process.stream_name}:{retailer.name}")
            quantity = process.sample(period, rng)
            retailer.receive_order(None, quantity, period)
            self.series[retailer.name].demand_received[period] += quantity
            self._pos[retailer.name][period] += quantity
            for ancestor in self._ancestors[retailer.name]:
                self._pos[ancestor][period] += quantity
            total += quantity
        self.customer_demand[period] = total

    def _allocate(self, available: float, lines: Sequence[OrderLine],
                  rule: Allocation) -> List[float]:
        if rule is Allocation.FIFO:
            out = []
            remaining = available
            for line in lines:
                take = min(line.remaining, remaining)
                out.append(max(0.0, take))
                remaining -= take
                if remaining <= _EPS:
                    remaining = 0.0
            return out
        requested = math.fsum(line.remaining for line in lines)
        if requested <= _EPS:
            return [0.0] * len(lines)
        share = min(1.0, available / requested)
        return [line.remaining * share for line in lines]

    def _fulfil(self, node: Node, period: int) -> None:
        series = self.series[node.name]
        if not node.supply_available:
            return
        available = math.inf if node.is_source else node.on_hand
        available = min(available, node.capacity_this_period())
        if available <= _EPS or not node.backlog:
            return
        lines = list(node.backlog)
        for line, quantity in zip(lines, self._allocate(available, lines, node.allocation)):
            if quantity <= _EPS:
                continue
            line.remaining -= quantity
            if not node.is_source:
                node.on_hand -= quantity
            series.shipped[period] += quantity
            if line.period == period:
                series.fill_numerator[period] += quantity
            self._dispatch(node, line.customer, quantity)
        node.backlog = deque(line for line in lines if line.remaining > _EPS)
        if not node.is_source:
            node.on_hand = max(0.0, node.on_hand)

    def _dispatch(self, node: Node, customer: Optional[str], quantity: float) -> None:
        if customer is None:
            return  # end customer takes delivery at the counter
        receiver = self.network.nodes[customer]
        # The units leave the order book and join the physical pipeline. Echelon
        # accounting depends on the distinction (see
        # SupplyNetwork.echelon_inventory_position).
        receiver.on_order = max(0.0, receiver.on_order - quantity)
        receiver.in_transit += quantity
        transit = receiver.transit
        observe = getattr(transit, "observe_dispatch", None)
        if observe is not None:
            observe(self.env.now)
        duration = max(0.0, transit.sample(self.streams.stream(f"transit:{customer}")))
        arrival = self.env.timeout(duration, priority=PH_RECEIVE, name=f"arrive:{customer}")
        arrival.add_callback(lambda _event, r=receiver, q=quantity: self._receive(r, q))

    def _receive(self, node: Node, quantity: float) -> None:
        node.on_hand += quantity
        node.in_transit = max(0.0, node.in_transit - quantity)

    def _forecast_signal(self, node: Node, period: int) -> float:
        if self.network.info_mode is InfoMode.DECENTRALIZED:
            return float(self.series[node.name].demand_received[period])
        return float(self._pos[node.name][period])

    def _review(self, node: Node, period: int) -> None:
        node.forecaster.update(self._forecast_signal(node, period))
        if not node.is_review_period(period):
            return
        forecast = node.forecaster.forecast()
        position = self.network.position_for_policy(node.name)
        protection = self.network.protection_for_policy(node.name)
        decision = node.policy.decide(position, forecast, protection)
        series = self.series[node.name]
        series.target[period] = decision.target_level
        quantity = decision.quantity
        series.orders_placed[period] = quantity
        if abs(quantity) <= _EPS:
            return
        node.on_order += quantity
        supplier = self.network.nodes[node.supplier]
        order = self.env.timeout(
            float(node.order_lead_time), priority=PH_ORDER_IN, name=f"order:{node.name}"
        )
        order.add_callback(
            lambda _event, s=supplier, c=node.name, q=quantity: self._deliver_order(s, c, q)
        )

    def _deliver_order(self, supplier: Node, customer: str, quantity: float) -> None:
        period = int(round(self.env.now))
        if period >= self.periods:
            return
        # The supplier's forecaster sees the order that was placed, whether or
        # not the physical consequence can be carried out in full.
        self.series[supplier.name].demand_received[period] += quantity
        if quantity >= 0:
            supplier.receive_order(customer, quantity, period)
            return
        self._cancel_order(supplier, customer, -quantity)

    def _cancel_order(self, supplier: Node, customer: str, quantity: float) -> None:
        """Withdraw ``quantity`` units of unshipped order from ``supplier``.

        Only reachable when a policy is configured with ``allow_returns``. Real
        buyers cancel purchase orders constantly, and the constraint that makes
        it interesting is that you cannot recall a truck: cancellation bites
        against the *unshipped* backlog, newest line first, and whatever has
        already been dispatched stays dispatched and stays on the customer's
        pipeline.
        """
        residual = quantity
        lines = list(supplier.backlog)
        for line in reversed(lines):
            if line.customer != customer or residual <= _EPS:
                continue
            take = min(line.remaining, residual)
            line.remaining -= take
            residual -= take
        supplier.backlog = deque(line for line in lines if line.remaining > _EPS)
        if residual > _EPS:
            self.network.nodes[customer].on_order += residual

    def _record(self, period: int) -> None:
        for name, node in self.network.nodes.items():
            series = self.series[name]
            series.on_hand[period] = node.on_hand
            series.backlog[period] = node.backlog_units
            series.position[period] = node.inventory_position()
            series.echelon_position[period] = self.network.echelon_inventory_position(name)

    # -- processes -------------------------------------------------------
    def _phase_loop(self, action: Callable[[int], None], priority: float):
        delay = 0.0
        while True:
            yield self.env.timeout(delay, priority=priority)
            action(int(round(self.env.now)))
            delay = 1.0

    def _fulfil_loop(self, node: Node, priority: float):
        """Allocation loop, interruptible by a supply outage.

        The ``Interrupt`` path is the reason this is a process rather than a
        callback: an outage is something that happens *to* a node that is in the
        middle of its normal cycle, and modelling it as an exception thrown into
        that cycle keeps the normal path free of outage bookkeeping.
        """
        delay = 0.0
        while True:
            try:
                yield self.env.timeout(delay, priority=priority)
            except Interrupt as interrupt:
                release = float(interrupt.cause)
                node.supply_available = False
                while self.env.now < release - 1.0:
                    yield self.env.timeout(1.0, priority=priority)
                node.supply_available = True
                delay = 1.0
                continue
            self._fulfil(node, int(round(self.env.now)))
            delay = 1.0

    def _disruption_loop(self):
        timeline: List[tuple] = []
        for outage in self.disruptions.outages:
            timeline.append((outage.start, 0, ("outage", outage)))
        for loss in self.disruptions.capacity_losses:
            timeline.append((loss.start, 1, ("capacity_on", loss)))
            timeline.append((loss.start + loss.duration, 2, ("capacity_off", loss)))
        timeline.sort(key=lambda item: (item[0], item[1]))
        for time, _order, (kind, spec) in timeline:
            delay = max(0.0, time - self.env.now)
            yield self.env.timeout(delay, priority=URGENT)
            if kind == "outage":
                self._fulfil_processes[spec.node].interrupt(spec.start + spec.duration)
            elif kind == "capacity_on":
                self.network.nodes[spec.node].capacity_factor = spec.factor
            else:
                self.network.nodes[spec.node].capacity_factor = 1.0

    # -- driver ----------------------------------------------------------
    def run(self) -> SimulationResult:
        self._initialise()
        env = self.env
        env.process(self._phase_loop(self._demand_phase, PH_DEMAND), "demand")
        top = self.network.max_level
        for node in self.network.ordered_nodes:
            priority = PH_FULFILL + (top - node.level) * 0.1
            self._fulfil_processes[node.name] = env.process(
                self._fulfil_loop(node, priority), f"fulfil:{node.name}"
            )
        for node in self.network.ordered_nodes:
            if node.is_source:
                continue
            priority = PH_REVIEW + node.level * 0.1
            env.process(
                self._phase_loop(lambda p, n=node: self._review(n, p), priority),
                f"review:{node.name}",
            )
        env.process(self._phase_loop(self._record, PH_RECORD), "record")
        if self.disruptions:
            env.process(self._disruption_loop(), "disruption")
        env.run(until=self.periods)

        stocking = {n.name for n in self.network.stocking_nodes()}
        return SimulationResult(
            periods=self.periods,
            customer_demand=self.customer_demand,
            nodes=self.series,
            node_levels={n.name: n.level for n in self.network.nodes.values()},
            events_processed=env.events_processed,
            holding_cost={n: self.network.nodes[n].holding_cost for n in stocking},
            backorder_cost={n: self.network.nodes[n].backorder_cost for n in stocking},
        )


def run_simulation(
    network: SupplyNetwork,
    demand: Optional[DemandProcess] = None,
    periods: int = 520,
    seed: int = 12345,
    replication: int = 0,
    disruptions: Optional[DisruptionPlan] = None,
    warmup: int = 0,
) -> SimulationResult:
    """Convenience wrapper: build a simulator, run it, optionally truncate."""
    demand = demand or IIDNormal()
    streams = StreamBank(seed=seed, replication=replication)
    result = Simulator(network, demand, periods, streams, disruptions).run()
    return result.trim(warmup) if warmup else result
