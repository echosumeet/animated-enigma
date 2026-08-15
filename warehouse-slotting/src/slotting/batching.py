"""Order batching under a cart capacity, and the interaction nobody prices.

Batching and routing are usually optimised by different people at different
times, and the interaction between them is larger than either effect alone.

Pick density per tour is the hinge. A single-order tour in a well-slotted
warehouse touches two or three aisles, so the return route is nearly optimal
and S-shape is terrible - it walks whole aisles for one carton. Batch eight
orders together and the same tour touches most aisles, at which point S-shape
becomes competitive and return becomes terrible. The benchmark reports the
routing comparison at two batch sizes for exactly this reason: quoting "policy
X is 12% better" without stating the batch size is quoting a number that flips.

The second thing batching does is blunt slotting. Slotting reduces travel by
concentrating picks near the dock; batching reduces travel by amortising the
walk across more lines. They compete for the same savings, so the slotting
benefit measured at batch size 1 overstates the benefit you will realise at
batch size 8. That is measured here rather than asserted.

Implemented:

* **Seed batching** (Gibson & Sharp 1992; de Koster, Van der Poort & Wolters
  1999): choose a seed order by a seed rule, then add the order that is most
  "congruent" with the partial batch until the cart is full.
* **Savings batching**: Clarke & Wright (1964) applied to batches - the saving
  from picking two orders together instead of separately, greedily merged
  subject to capacity. Consistently the better of the two here, at O(n^2)
  routing calls to build the savings list, which is the reason people ship
  seed batching instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from .assignment import Assignment
from .layout import Warehouse
from .orders import Order, OrderStream
from .routing import ROUTERS, Route
from .skus import Catalog

__all__ = [
    "CartCapacity",
    "Batch",
    "single_order_batches",
    "seed_batching",
    "savings_batching",
    "evaluate_batches",
    "BatchingResult",
]


@dataclass(frozen=True)
class CartCapacity:
    """What one picker can take out in one trip."""

    max_orders: int = 6
    max_lines: int = 24
    max_cube_m3: float = 0.75

    def fits(self, n_orders: int, n_lines: int, cube: float) -> bool:
        return (
            n_orders <= self.max_orders
            and n_lines <= self.max_lines
            and cube <= self.max_cube_m3 + 1e-9
        )


@dataclass
class Batch:
    """A set of orders picked in one tour."""

    order_ids: list[int]
    picks: list[int]  # location indices, one per line
    lines: int
    cube: float

    def location_set(self) -> list[int]:
        return sorted(set(self.picks))


def _order_picks(order: Order, assignment: Assignment) -> list[int]:
    return [int(assignment.location_of[s]) for s in order.lines]


def _order_cube(order: Order, catalog: Catalog) -> float:
    return float(
        sum(catalog.case_cube[s] * q for s, q in zip(order.lines, order.quantities))
    )


def single_order_batches(
    stream: OrderStream, assignment: Assignment, catalog: Catalog
) -> list[Batch]:
    """One order per tour: strict order picking, the batching baseline."""
    return [
        Batch([o.order_id], _order_picks(o, assignment), o.n_lines, _order_cube(o, catalog))
        for o in stream
    ]


def seed_batching(
    stream: OrderStream,
    assignment: Assignment,
    catalog: Catalog,
    warehouse: Warehouse,
    capacity: CartCapacity | None = None,
    seed_rule: str = "largest",
    congruency: str = "shared_aisles",
    candidate_window: int = 60,
) -> list[Batch]:
    """Seed-and-add batching.

    ``seed_rule``:
      ``largest``  the order with the most lines - fills the cart fastest;
      ``farthest`` the order whose deepest pick is furthest from the dock -
                   the classical choice, on the logic that the expensive walk
                   should be shared by as many orders as possible.

    ``congruency``:
      ``shared_aisles``   maximise the number of aisles already being visited;
      ``added_aisles``    minimise the number of *new* aisles introduced;
      ``added_distance``  minimise the true increase in routed distance. Exact
                          and much slower - a routing call per candidate per
                          addition. Included because it is the honest reference
                          the cheap rules should be judged against.

    ``candidate_window`` caps how many unbatched orders are scored per addition.
    Without it this is O(n^2) in the order count and the benchmark stops being
    runnable; with it the batches are indistinguishable, which is itself worth
    knowing.
    """
    cap = capacity or CartCapacity()
    remaining = {o.order_id: o for o in stream}
    picks = {o.order_id: _order_picks(o, assignment) for o in stream}
    cubes = {o.order_id: _order_cube(o, catalog) for o in stream}
    aisles = {
        oid: {warehouse.locations[p].aisle for p in pk} for oid, pk in picks.items()
    }
    depth = {
        oid: max((warehouse.distance_from_depot(p) for p in pk), default=0.0)
        for oid, pk in picks.items()
    }

    batches: list[Batch] = []
    while remaining:
        if seed_rule == "largest":
            seed_id = max(remaining, key=lambda oid: (remaining[oid].n_lines, -oid))
        elif seed_rule == "farthest":
            seed_id = max(remaining, key=lambda oid: (depth[oid], -oid))
        else:
            raise ValueError(f"unknown seed rule {seed_rule!r}")

        seed = remaining.pop(seed_id)
        batch = Batch([seed_id], list(picks[seed_id]), seed.n_lines, cubes[seed_id])
        batch_aisles = set(aisles[seed_id])

        while remaining:
            window = list(remaining)[:candidate_window]
            best_id, best_score = None, None
            for oid in window:
                o = remaining[oid]
                if not cap.fits(
                    len(batch.order_ids) + 1, batch.lines + o.n_lines, batch.cube + cubes[oid]
                ):
                    continue
                if congruency == "shared_aisles":
                    score = -len(batch_aisles & aisles[oid])
                elif congruency == "added_aisles":
                    score = len(aisles[oid] - batch_aisles)
                elif congruency == "added_distance":
                    base = ROUTERS["s_shape"](warehouse, batch.location_set()).distance
                    merged = ROUTERS["s_shape"](
                        warehouse, sorted(set(batch.picks) | set(picks[oid]))
                    ).distance
                    score = merged - base
                else:
                    raise ValueError(f"unknown congruency rule {congruency!r}")
                if best_score is None or score < best_score:
                    best_id, best_score = oid, score
            if best_id is None:
                break
            o = remaining.pop(best_id)
            batch.order_ids.append(best_id)
            batch.picks.extend(picks[best_id])
            batch.lines += o.n_lines
            batch.cube += cubes[best_id]
            batch_aisles |= aisles[best_id]
        batches.append(batch)
    return batches


def savings_batching(
    stream: OrderStream,
    assignment: Assignment,
    catalog: Catalog,
    warehouse: Warehouse,
    capacity: CartCapacity | None = None,
    policy: str = "s_shape",
    neighbours: int = 25,
) -> list[Batch]:
    """Clarke-Wright savings applied to order batching.

    ``s_ij = d(i) + d(j) - d(i and j together)`` is the metres saved by picking
    two orders in one tour. Savings are computed on singleton pairs and merged
    greedily; batches are tracked with union-find and capacity is re-checked at
    every merge.

    Computing all n^2 savings costs n^2 routing calls, which is not viable on a
    4,000-order stream. The pairs are restricted to the ``neighbours`` orders
    with the most overlapping aisles, which is where all the positive savings
    are anyway - two orders in different halves of the building have a saving
    of roughly zero and merging them is exactly the mistake full savings makes
    when it is truncated by capacity instead of by geography.
    """
    cap = capacity or CartCapacity()
    router = ROUTERS[policy]
    orders = list(stream)
    n = len(orders)
    if n == 0:
        return []
    picks = [_order_picks(o, assignment) for o in orders]
    cubes = [_order_cube(o, catalog) for o in orders]
    aisle_sets = [{warehouse.locations[p].aisle for p in pk} for pk in picks]
    solo = [router(warehouse, sorted(set(pk))).distance for pk in picks]

    # Candidate pairs: for each order, the orders sharing the most aisles.
    by_aisle: dict[int, list[int]] = {}
    for i, aset in enumerate(aisle_sets):
        for a in aset:
            by_aisle.setdefault(a, []).append(i)
    pair_scores: dict[tuple[int, int], int] = {}
    for a, members in by_aisle.items():
        for x in range(len(members)):
            for y in range(x + 1, min(len(members), x + 1 + neighbours)):
                i, j = members[x], members[y]
                key = (i, j) if i < j else (j, i)
                pair_scores[key] = pair_scores.get(key, 0) + 1

    savings: list[tuple[float, int, int]] = []
    for (i, j), _ in pair_scores.items():
        merged = router(warehouse, sorted(set(picks[i]) | set(picks[j]))).distance
        s = solo[i] + solo[j] - merged
        if s > 0:
            savings.append((s, i, j))
    savings.sort(reverse=True)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    b_lines = [o.n_lines for o in orders]
    b_cube = list(cubes)

    for _, i, j in savings:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        n_o = len(members[ri]) + len(members[rj])
        n_l = b_lines[ri] + b_lines[rj]
        cu = b_cube[ri] + b_cube[rj]
        if not cap.fits(n_o, n_l, cu):
            continue
        parent[rj] = ri
        members[ri].extend(members[rj])
        del members[rj]
        b_lines[ri] = n_l
        b_cube[ri] = cu

    batches: list[Batch] = []
    for root, group in members.items():
        pk: list[int] = []
        for i in group:
            pk.extend(picks[i])
        batches.append(
            Batch(
                [orders[i].order_id for i in group],
                pk,
                sum(orders[i].n_lines for i in group),
                sum(cubes[i] for i in group),
            )
        )
    return batches


@dataclass(frozen=True)
class BatchingResult:
    policy: str
    n_batches: int
    total_distance_m: float
    distance_per_order_m: float
    distance_per_line_m: float
    mean_lines_per_batch: float
    mean_aisles_per_batch: float

    def as_row(self) -> tuple:
        return (
            self.policy,
            self.n_batches,
            self.total_distance_m,
            self.distance_per_order_m,
            self.distance_per_line_m,
            self.mean_lines_per_batch,
            self.mean_aisles_per_batch,
        )


def evaluate_batches(
    warehouse: Warehouse,
    batches: Sequence[Batch],
    policy: str = "s_shape",
    label: str | None = None,
) -> BatchingResult:
    """Route every batch and total the metres."""
    router = ROUTERS[policy]
    total = 0.0
    n_orders = n_lines = 0
    aisle_counts: list[int] = []
    for b in batches:
        locs = b.location_set()
        total += router(warehouse, locs).distance
        n_orders += len(b.order_ids)
        n_lines += b.lines
        aisle_counts.append(len({warehouse.locations[p].aisle for p in locs}))
    n_batches = max(len(batches), 1)
    return BatchingResult(
        policy=label or policy,
        n_batches=len(batches),
        total_distance_m=total,
        distance_per_order_m=total / max(n_orders, 1),
        distance_per_line_m=total / max(n_lines, 1),
        mean_lines_per_batch=n_lines / n_batches,
        mean_aisles_per_batch=float(np.mean(aisle_counts)) if aisle_counts else 0.0,
    )
