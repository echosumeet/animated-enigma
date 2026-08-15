"""Local search on measured travel, not on a proxy for it.

Every slotting tool I have seen optimises a surrogate: fast movers near the
dock, co-ordered items near each other, weighted somehow. The surrogate exists
because evaluating a candidate layout looks expensive - you have to route the
whole order stream - and a swap-based search needs tens of thousands of
evaluations.

That reasoning is wrong, and :mod:`slotting.calibration` documents how badly.
Swapping two SKUs changes the routed distance of exactly the orders that
contain one of them. With an inverted index from SKU to orders, the delta costs
``O(orders containing a) + O(orders containing b)`` routing calls - about twenty
on this instance, roughly half a millisecond. Forty thousand moves is under a
minute, in pure Python, and the objective being optimised is the number that
goes in the report.

What that buys, measured on the benchmark instance:

* the surrogate objective and measured travel disagree at the margin badly
  enough that annealing on a *calibrated* surrogate improves the surrogate by
  6% while making measured travel 9% worse;
* annealing on measured travel improves measured travel, which is the entire
  point, and the improvement transfers to the held-out period.

The cost is that the objective is tied to a routing policy. That is not a
defect - it is the truth surfacing. A layout tuned for return routing is not
the layout you want if your pickers walk S-shape, and the benchmark reports the
cross-policy transfer so the size of that effect is visible.

The ergonomic term stays additive and decomposable: it is a per-line penalty
that depends only on the level a SKU sits at, so it costs nothing to carry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .assignment import Assignment
from .ergonomics import ErgonomicModel
from .layout import Warehouse
from .orders import OrderStream
from .skus import Catalog

__all__ = ["TravelObjective"]


class TravelObjective:
    """Total routed travel over an order stream, with O(affected orders) swap deltas.

    Exposes the same surface as :class:`slotting.objective.SlottingObjective`
    (``total``, ``components``, ``swap_delta``, ``move_delta``, ``pick_rate``)
    plus a ``commit`` hook the search calls when it accepts a move, so the
    per-order distance cache stays in step without a recompute.
    """

    def __init__(
        self,
        warehouse: Warehouse,
        catalog: Catalog,
        stream: OrderStream,
        policy: str = "s_shape",
        ergonomics: ErgonomicModel | None = None,
        ergonomic_weight: float = 1.0,
        sample: int | None = None,
        seed: int = 13,
    ) -> None:
        from .routing import ROUTERS

        self.warehouse = warehouse
        self.catalog = catalog
        self.policy = policy
        self._router = ROUTERS[policy]
        self.ergo = ergonomics or ErgonomicModel()
        self._gamma = ergonomic_weight

        orders = list(stream)
        if sample is not None and sample < len(orders):
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(len(orders), size=sample, replace=False))
            orders = [orders[int(k)] for k in idx]
        self.n_orders = len(orders)
        self.order_lines: list[tuple[int, ...]] = [
            tuple(sorted(set(o.lines))) for o in orders
        ]
        self.scale = len(stream) / max(self.n_orders, 1)

        n_skus = catalog.n_skus
        self.orders_of_sku: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * n_skus
        buckets: dict[int, list[int]] = {}
        for k, lines in enumerate(self.order_lines):
            for sku in lines:
                buckets.setdefault(sku, []).append(k)
        for sku, ks in buckets.items():
            self.orders_of_sku[sku] = np.asarray(ks, dtype=np.int64)

        # Line counts on the indexed orders drive both the ergonomic term and
        # the search's proposal bias.
        self.pick_rate = np.zeros(n_skus, dtype=float)
        for sku, ks in buckets.items():
            self.pick_rate[sku] = float(len(ks))
        self.line_rate = self.pick_rate

        c = warehouse.config
        heights = [self.ergo.height_of_level(l, c.level_height_m) for l in range(c.n_levels)]
        self._ergo_metres = np.asarray(
            [[self.ergo.metres(h, float(w)) for h in heights] for w in catalog.case_weight],
            dtype=float,
        )
        self._loc_level = np.asarray([loc.level for loc in warehouse.locations])

        self._dist = np.zeros(self.n_orders, dtype=float)
        self._cached_for: Assignment | None = None
        self._pending: tuple[np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------------
    def _route(self, assignment: Assignment, k: int) -> float:
        loc = assignment.location_of
        picks = sorted({int(loc[s]) for s in self.order_lines[k]})
        return self._router(self.warehouse, picks).distance

    def refresh(self, assignment: Assignment) -> None:
        """Recompute the whole per-order distance cache."""
        for k in range(self.n_orders):
            self._dist[k] = self._route(assignment, k)
        self._cached_for = assignment
        self._pending = None

    def travel(self, assignment: Assignment) -> float:
        if self._cached_for is not assignment:
            self.refresh(assignment)
        return float(self._dist.sum())

    def ergonomic_metres(self, assignment: Assignment) -> float:
        loc = assignment.location_of
        return float(
            np.dot(
                self.line_rate,
                self._ergo_metres[np.arange(len(loc)), self._loc_level[loc]],
            )
        )

    def components(self, assignment: Assignment) -> dict[str, float]:
        travel = self.travel(assignment)
        ergo = self._gamma * self.ergonomic_metres(assignment)
        return {"travel": travel, "ergonomics": ergo, "total": travel + ergo}

    def total(self, assignment: Assignment) -> float:
        return self.components(assignment)["total"]

    # ------------------------------------------------------------------
    def _ergo_cost(self, sku: int, loc: int) -> float:
        return self._gamma * self.line_rate[sku] * self._ergo_metres[sku, self._loc_level[loc]]

    def swap_delta(self, assignment: Assignment, sku_a: int, sku_b: int) -> float:
        """Exact change in the objective from exchanging two SKUs' locations."""
        if self._cached_for is not assignment:
            self.refresh(assignment)
        a_orders = self.orders_of_sku[sku_a]
        b_orders = self.orders_of_sku[sku_b]
        affected = np.union1d(a_orders, b_orders) if b_orders.size else a_orders
        la = int(assignment.location_of[sku_a])
        lb = int(assignment.location_of[sku_b])

        ergo_delta = (
            self._ergo_cost(sku_a, lb)
            + self._ergo_cost(sku_b, la)
            - self._ergo_cost(sku_a, la)
            - self._ergo_cost(sku_b, lb)
        )
        if affected.size == 0:
            self._pending = (affected, np.empty(0, dtype=float))
            return ergo_delta

        before = float(self._dist[affected].sum())
        assignment.swap(sku_a, sku_b)
        new = np.asarray([self._route(assignment, int(k)) for k in affected], dtype=float)
        assignment.swap(sku_a, sku_b)
        self._pending = (affected, new)
        return float(new.sum()) - before + ergo_delta

    def move_delta(self, assignment: Assignment, sku: int, new_loc: int) -> float:
        """Exact change in the objective from moving one SKU to an empty location."""
        if self._cached_for is not assignment:
            self.refresh(assignment)
        affected = self.orders_of_sku[sku]
        old = int(assignment.location_of[sku])
        ergo_delta = self._ergo_cost(sku, new_loc) - self._ergo_cost(sku, old)
        if affected.size == 0:
            self._pending = (affected, np.empty(0, dtype=float))
            return ergo_delta
        before = float(self._dist[affected].sum())
        assignment.relocate(sku, new_loc)
        new = np.asarray([self._route(assignment, int(k)) for k in affected], dtype=float)
        assignment.relocate(sku, old)
        self._pending = (affected, new)
        return float(new.sum()) - before + ergo_delta

    def commit(self) -> None:
        """Fold the last evaluated move's distances into the cache."""
        if self._pending is None:
            return
        affected, new = self._pending
        if affected.size:
            self._dist[affected] = new
        self._pending = None

    # ------------------------------------------------------------------
    def golden_zone_share(self, assignment: Assignment) -> float:
        c = self.warehouse.config
        good = np.asarray(
            [self.ergo.is_golden(l, c.level_height_m) for l in range(c.n_levels)]
        )
        levels = self._loc_level[assignment.location_of]
        return float(np.dot(self.line_rate, good[levels]) / max(self.line_rate.sum(), 1e-9))

    def mean_affected_orders(self) -> float:
        """Average number of orders a swap has to re-route. The cost of exactness."""
        sizes = [len(o) for o in self.orders_of_sku if len(o)]
        return 2.0 * float(np.mean(sizes)) if sizes else 0.0
