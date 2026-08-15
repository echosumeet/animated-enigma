"""Graves-Willems guaranteed-service model: safety stock placement on a BOM tree.

The stochastic-service view (Clark-Scarf and its descendants) asks how much
inventory to hold everywhere.  The guaranteed-service view asks a different and,
for network design, more useful question: *where* should inventory sit at all?

Each stage quotes an outbound service time ``S_j`` - a promise to fill its
customer's order within ``S_j`` periods - and receives an inbound service time
``SI_j`` from its suppliers.  Its net replenishment time is
``tau_j = SI_j + T_j - S_j``, where ``T_j`` is its own processing time.  A stage
holds safety stock proportional to ``sqrt(tau_j)``; a stage that quotes a service
time equal to its inbound time plus processing holds none at all.  The model then
chooses service times to minimise total safety-stock cost subject to the external
service commitment.

The result is almost always a small number of *decoupling points*: most stages
end up with ``tau_j = 0`` and no safety stock, and a handful carry the whole
buffer.  That is a genuinely different answer from "every stage gets a service
level", and it is the reason this model is what network design teams actually
run.  It is also why the output is worth reading as a placement recommendation
rather than a set of parameters - the interesting content is *which* stages get
non-zero ``tau``.

The bounded-demand assumption is the model's price of admission: demand over
``t`` periods is assumed never to exceed ``mu*t + z*sigma*sqrt(t)``, and anything
above that bound is handled outside the model (expediting, allocation, a
conversation with the customer).  That is a real assumption and it is why this
model understates the buffer needed on genuinely fat-tailed items.  It is stated
here rather than buried, because the failure mode - a stage with zero safety
stock that was supposed to be a pass-through and turns out to be the constraint -
is a well-known one.

Algorithm
---------
For a tree, the problem is solved exactly by dynamic programming over integer
service times, which is Graves & Willems' contribution.  This implementation
roots the tree arbitrarily and computes, for every node and every integer service
time, the minimum cost of that node's subtree:

* ``f_j(S)``  - node ``j`` supplies its parent and quotes outbound service ``S``.
* ``g_j(SI)`` - node ``j`` is supplied by its parent, which quotes ``SI``.

Both reduce to minimising ``c_j(SI + T_j - S)`` plus the subtree terms, over the
integer grid ``[0, max_service]``.  Complexity is ``O(N * M^2)`` with ``M`` the
service-time grid size.

References
----------
Graves, S.C. and Willems, S.P. (2000) 'Optimizing strategic safety stock placement
in supply chains', *Manufacturing & Service Operations Management* 2(1), 68-83.
Graves, S.C. and Willems, S.P. (2003) 'Supply chain design: safety stock placement
and supply chain configuration', in *Handbooks in OR & MS*, Vol. 11.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import stats

__all__ = [
    "Stage",
    "SupplyChainTree",
    "PlacementResult",
    "solve_guaranteed_service",
    "example_bom_tree",
    "enumerate_optimal_cost",
]

INF = float("inf")


@dataclass(frozen=True)
class Stage:
    """One node of the supply chain tree.

    ``processing_time``
        Periods from receiving all inputs to being ready to ship (``T_j``).
    ``cost_added``
        Cumulative unit cost at this stage, used with ``holding_rate`` to price a
        unit of safety stock per period.
    ``demand_mean`` / ``demand_sd``
        External per-period demand at this stage.  Zero for internal stages;
        internal demand is derived by rolling up downstream demand through the
        BOM.
    ``max_service_time``
        Only meaningful for demand nodes: the service time promised to the
        external customer.  ``None`` for internal stages.
    """

    name: str
    processing_time: int
    cost_added: float
    demand_mean: float = 0.0
    demand_sd: float = 0.0
    max_service_time: int | None = None


@dataclass
class SupplyChainTree:
    """A tree of stages with supplier -> customer arcs and usage multipliers.

    ``arcs`` maps ``(supplier, customer)`` to the number of units of the supplier
    item consumed per unit of the customer item.  The graph must be a tree when
    the arc directions are ignored; that is the condition under which the Graves-
    Willems DP is exact.
    """

    stages: dict[str, Stage]
    arcs: dict[tuple[str, str], float] = field(default_factory=dict)

    def add_stage(self, stage: Stage) -> None:
        self.stages[stage.name] = stage

    def add_arc(self, supplier: str, customer: str, usage: float = 1.0) -> None:
        if supplier not in self.stages or customer not in self.stages:
            raise KeyError("both endpoints must exist before adding an arc")
        self.arcs[(supplier, customer)] = float(usage)

    def customers(self, node: str) -> list[str]:
        return [c for (s, c) in self.arcs if s == node]

    def suppliers(self, node: str) -> list[str]:
        return [s for (s, c) in self.arcs if c == node]

    def neighbours(self, node: str) -> list[tuple[str, str]]:
        """``(neighbour, relation)`` with relation in {'supplier', 'customer'}."""
        out = [(s, "supplier") for s in self.suppliers(node)]
        out += [(c, "customer") for c in self.customers(node)]
        return out

    def validate_tree(self) -> None:
        n_nodes = len(self.stages)
        n_edges = len(self.arcs)
        if n_edges != n_nodes - 1:
            raise ValueError(
                f"not a tree: {n_nodes} nodes and {n_edges} arcs (expected {n_nodes - 1})"
            )
        seen: set[str] = set()
        stack = [next(iter(self.stages))]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(nb for nb, _ in self.neighbours(node) if nb not in seen)
        if len(seen) != n_nodes:
            raise ValueError("not a tree: the graph is disconnected")

    def propagate_demand(self) -> dict[str, tuple[float, float]]:
        """Roll external demand up through the BOM to give each stage its own.

        A stage's demand is the usage-weighted sum of its customers' demands plus
        its own external demand.  Variances are added assuming the downstream
        demand streams are independent - stated here because it is the assumption
        most likely to be wrong in a real BOM (two finished goods sharing a
        component usually share a demand driver too) and it understates the
        component-level standard deviation when it is.
        """
        self.validate_tree()
        order = self._topological_order()
        mean: dict[str, float] = {}
        var: dict[str, float] = {}
        for node in reversed(order):  # customers before suppliers
            st = self.stages[node]
            m = st.demand_mean
            v = st.demand_sd ** 2
            for c in self.customers(node):
                usage = self.arcs[(node, c)]
                m += usage * mean[c]
                v += (usage ** 2) * var[c]
            mean[node] = m
            var[node] = v
        return {k: (mean[k], math.sqrt(var[k])) for k in mean}

    def _topological_order(self) -> list[str]:
        indeg = {n: len(self.suppliers(n)) for n in self.stages}
        queue = [n for n, d in indeg.items() if d == 0]
        order: list[str] = []
        while queue:
            node = queue.pop()
            order.append(node)
            for c in self.customers(node):
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
        if len(order) != len(self.stages):
            raise ValueError("the arc set contains a directed cycle")
        return order


@dataclass
class PlacementResult:
    service_times: dict[str, int]
    inbound_service_times: dict[str, int]
    net_replenishment_times: dict[str, int]
    safety_stock: dict[str, float]
    safety_stock_cost: dict[str, float]
    total_cost: float
    z: float

    @property
    def decoupling_points(self) -> list[str]:
        """Stages that actually hold safety stock."""
        return sorted(k for k, v in self.net_replenishment_times.items() if v > 0)

    def table(self) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []
        for name in sorted(self.service_times):
            rows.append(
                {
                    "stage": name,
                    "inbound_service": float(self.inbound_service_times[name]),
                    "outbound_service": float(self.service_times[name]),
                    "net_repl_time": float(self.net_replenishment_times[name]),
                    "safety_stock": float(self.safety_stock[name]),
                    "safety_stock_cost": float(self.safety_stock_cost[name]),
                }
            )
        return rows


def solve_guaranteed_service(
    tree: SupplyChainTree,
    service_level: float = 0.95,
    holding_rate: float = 0.25,
    max_service: int | None = None,
) -> PlacementResult:
    """Minimise safety-stock holding cost over service times, exactly, by DP.

    ``holding_rate`` is per period, applied to ``cost_added`` to price one unit of
    safety stock.  ``max_service`` bounds the service-time grid; it defaults to
    the sum of all processing times, which is the longest service time that can
    ever be useful.
    """
    tree.validate_tree()
    z = float(stats.norm.ppf(service_level))
    demand = tree.propagate_demand()
    if max_service is None:
        max_service = int(sum(st.processing_time for st in tree.stages.values()))
    grid = np.arange(max_service + 1)

    # Per-stage cost of a net replenishment time, c_j(tau) = h_j * z * sigma_j * sqrt(tau)
    cost_curve: dict[str, np.ndarray] = {}
    for name, st in tree.stages.items():
        _, sd = demand[name]
        unit_holding = holding_rate * st.cost_added
        cost_curve[name] = unit_holding * z * sd * np.sqrt(grid.astype(float))

    root = next(iter(tree.stages))
    # memo[(node, parent, mode)] -> array over the parent-facing service grid
    memo: dict[tuple[str, str | None, str], np.ndarray] = {}
    choice: dict[tuple[str, str | None, str, int], tuple[int, int]] = {}

    def node_table(node: str, parent: str | None) -> np.ndarray:
        """``h[S, SI]`` - cost of this node plus all subtree children."""
        st = tree.stages[node]
        n = max_service + 1
        table = np.full((n, n), INF)
        S_idx = np.arange(n)[:, None]
        SI_idx = np.arange(n)[None, :]
        tau = SI_idx + st.processing_time - S_idx
        feasible = tau >= 0
        tau_clipped = np.clip(tau, 0, max_service)
        own = np.where(feasible, cost_curve[node][tau_clipped], INF)
        table = own.copy()

        if st.max_service_time is not None:
            table[st.max_service_time + 1 :, :] = INF

        for nb, relation in tree.neighbours(node):
            if nb == parent:
                continue
            if relation == "supplier":
                # nb supplies node: node's SI must be >= nb's outbound service
                f = subtree(nb, node, "supplies_parent")
                cum = np.minimum.accumulate(f)  # best cost with service <= SI
                table = table + cum[None, :]
            else:
                # nb is a customer of node: nb's inbound service is node's S
                g = subtree(nb, node, "supplied_by_parent")
                table = table + g[:, None]
        return table

    def subtree(node: str, parent: str | None, mode: str) -> np.ndarray:
        key = (node, parent, mode)
        if key in memo:
            return memo[key]
        table = node_table(node, parent)
        n = max_service + 1
        if mode == "supplies_parent":
            # indexed by this node's outbound service S; minimise over SI
            best = np.full(n, INF)
            for s in range(n):
                col = table[s, :]
                j = int(np.argmin(col))
                best[s] = col[j]
                choice[(node, parent, mode, s)] = (s, j)
        else:
            # indexed by the inbound service the parent quotes; SI >= that value
            best = np.full(n, INF)
            flat_min = np.full(n, INF)
            arg = np.zeros((n, 2), dtype=int)
            for si in range(n):
                col = table[:, si]
                s = int(np.argmin(col))
                flat_min[si] = col[s]
                arg[si] = (s, si)
            # cost is non-increasing in the allowed SI floor: take suffix minima
            best_idx = np.zeros(n, dtype=int)
            run = INF
            run_idx = 0
            for si in range(n - 1, -1, -1):
                if flat_min[si] <= run:
                    run = flat_min[si]
                    run_idx = si
                best[si] = run
                best_idx[si] = run_idx
            for si in range(n):
                s, chosen_si = arg[best_idx[si]]
                choice[(node, parent, mode, si)] = (int(s), int(chosen_si))
        memo[key] = best
        return best

    # Root: free choice of both S and SI
    root_table = node_table(root, None)
    flat = int(np.argmin(root_table))
    root_S, root_SI = divmod(flat, max_service + 1)
    total = float(root_table[root_S, root_SI])
    if not math.isfinite(total):
        raise ValueError("no feasible service-time assignment; relax max_service_time")

    service: dict[str, int] = {}
    inbound: dict[str, int] = {}

    def traceback(node: str, parent: str | None, S: int, SI: int) -> None:
        service[node] = S
        inbound[node] = SI
        for nb, relation in tree.neighbours(node):
            if nb == parent:
                continue
            if relation == "supplier":
                f = subtree(nb, node, "supplies_parent")
                cum_best = int(np.argmin(f[: SI + 1])) if SI >= 0 else 0
                s_nb, si_nb = choice[(nb, node, "supplies_parent", cum_best)]
                traceback(nb, node, s_nb, si_nb)
            else:
                subtree(nb, node, "supplied_by_parent")
                s_nb, si_nb = choice[(nb, node, "supplied_by_parent", S)]
                traceback(nb, node, s_nb, si_nb)

    traceback(root, None, root_S, root_SI)

    tau_map: dict[str, int] = {}
    ss: dict[str, float] = {}
    ss_cost: dict[str, float] = {}
    for name, st in tree.stages.items():
        tau = inbound[name] + st.processing_time - service[name]
        tau_map[name] = int(tau)
        _, sd = demand[name]
        ss[name] = float(z * sd * math.sqrt(max(tau, 0)))
        ss_cost[name] = float(holding_rate * st.cost_added * ss[name])

    return PlacementResult(
        service_times=service,
        inbound_service_times=inbound,
        net_replenishment_times=tau_map,
        safety_stock=ss,
        safety_stock_cost=ss_cost,
        total_cost=float(sum(ss_cost.values())),
        z=z,
    )


def example_bom_tree() -> SupplyChainTree:
    """A nine-stage assembly BOM used in the examples, benchmarks and tests.

    Shape: two raw materials feed a sub-assembly; that sub-assembly plus a
    purchased part feed a build stage; the build feeds a pack stage which feeds
    two regional distribution centres that carry the external demand.  Long,
    cheap upstream processing and short, expensive downstream processing is the
    configuration where guaranteed-service placement earns its keep - it will push
    the buffer upstream to where a unit is cheap to hold and pull it out of the
    DCs, subject to the customer service promise.
    """
    tree = SupplyChainTree(stages={})
    tree.add_stage(Stage("raw_A", processing_time=10, cost_added=6.0))
    tree.add_stage(Stage("raw_B", processing_time=14, cost_added=4.0))
    tree.add_stage(Stage("subassembly", processing_time=6, cost_added=18.0))
    tree.add_stage(Stage("purchased_part", processing_time=21, cost_added=9.0))
    tree.add_stage(Stage("build", processing_time=5, cost_added=42.0))
    tree.add_stage(Stage("pack", processing_time=2, cost_added=48.0))
    tree.add_stage(
        Stage("dc_north", processing_time=3, cost_added=55.0, demand_mean=120.0,
              demand_sd=40.0, max_service_time=0)
    )
    tree.add_stage(
        Stage("dc_south", processing_time=4, cost_added=55.0, demand_mean=80.0,
              demand_sd=35.0, max_service_time=1)
    )
    tree.add_stage(Stage("service_parts", processing_time=2, cost_added=50.0,
                         demand_mean=15.0, demand_sd=12.0, max_service_time=5))

    tree.add_arc("raw_A", "subassembly", usage=2.0)
    tree.add_arc("raw_B", "subassembly", usage=1.0)
    tree.add_arc("subassembly", "build", usage=1.0)
    tree.add_arc("purchased_part", "build", usage=1.0)
    tree.add_arc("build", "pack", usage=1.0)
    tree.add_arc("pack", "dc_north", usage=1.0)
    tree.add_arc("pack", "dc_south", usage=1.0)
    tree.add_arc("build", "service_parts", usage=1.0)
    return tree


def enumerate_optimal_cost(
    tree: SupplyChainTree, service_level: float = 0.95, holding_rate: float = 0.25
) -> float:
    """Brute-force minimum cost over all integer service-time vectors.

    Exponential, and only usable on small trees - which is exactly what makes it
    a good oracle for the DP in the test suite.
    """
    from itertools import product

    tree.validate_tree()
    z = float(stats.norm.ppf(service_level))
    demand = tree.propagate_demand()
    names = sorted(tree.stages)
    max_service = int(sum(st.processing_time for st in tree.stages.values()))
    ranges = []
    for n in names:
        st = tree.stages[n]
        hi = st.max_service_time if st.max_service_time is not None else max_service
        ranges.append(range(0, int(hi) + 1))

    best = INF
    for combo in product(*ranges):
        S = dict(zip(names, combo))
        total = 0.0
        ok = True
        for n in names:
            st = tree.stages[n]
            sups = tree.suppliers(n)
            si = max((S[s] for s in sups), default=0)
            tau = si + st.processing_time - S[n]
            if tau < 0:
                ok = False
                break
            _, sd = demand[n]
            total += holding_rate * st.cost_added * z * sd * math.sqrt(tau)
        if ok and total < best:
            best = total
    return float(best)
