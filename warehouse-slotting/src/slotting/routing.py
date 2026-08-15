"""Pick-path routing: four classical heuristics, two exact methods, one gap.

A slotting change is only worth what the router converts it into. The same
assignment can look 20% better or 3% better depending on how the picker walks,
because the heuristics differ in how much they care about *where* in an aisle a
pick is. S-shape traverses whole aisles and is therefore almost indifferent to
along-aisle position; return and largest-gap are highly sensitive to it. Any
slotting benefit quoted without naming the routing policy is not a number.

Implemented:

* ``s_shape``      traversal: enter each aisle containing picks, traverse it
                   end to end, alternate direction. If the count of visited
                   aisles is odd the last one is entered and left from the same
                   end. (Petersen 1997; Hall 1993.)
* ``return_route`` enter and leave every aisle from the front cross aisle.
* ``midpoint``     picks past the aisle midpoint are served from the back cross
                   aisle, picks before it from the front. (Hall 1993.)
* ``largest_gap``  each intermediate aisle is entered from both ends and the
                   largest gap between consecutive picks is never walked. The
                   best of the simple heuristics, and the closest in structure
                   to the optimal route. (Hall 1993.)
* ``exact_aisle_dp``  the Ratliff-Rosenthal (1983) dynamic program over aisles.
                   Linear in the number of aisles, exact for a single-block
                   layout, and the reason optimality gaps in this repo are
                   measured rather than assumed.
* ``held_karp``    exact TSP over the pick set under the warehouse metric.
                   Exponential, capped at 14 picks, and used to *validate* the
                   aisle DP: ``tests/test_routing_exact.py`` asserts the two
                   agree on hundreds of random instances. Two independent exact
                   methods agreeing is the only reason to believe either.
* ``nearest_neighbour`` + ``two_opt``: geometry-agnostic fallbacks that work on
                   multi-block layouts, where the four classical heuristics are
                   not defined (Roodbergen & de Koster 2001 extend them; that
                   extension is not implemented here and is listed as such).

All routes are returned as an ordered list of waypoints in the corridor
network, so a route can be drawn as well as measured. ``validate_route``
asserts every leg is a legal corridor move, which caught three sign errors
during development that a distance-only check would have hidden.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from .layout import Location, Point, Warehouse

__all__ = [
    "Route",
    "s_shape",
    "return_route",
    "midpoint",
    "largest_gap",
    "nearest_neighbour",
    "two_opt",
    "held_karp",
    "exact_aisle_dp",
    "ROUTERS",
    "route_picks",
    "validate_route",
]


@dataclass(frozen=True)
class Route:
    """A closed picker tour: waypoints in order, starting and ending at the depot."""

    waypoints: tuple[Point, ...]
    distance: float
    policy: str

    def __len__(self) -> int:
        return len(self.waypoints)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _pick_points(warehouse: Warehouse, picks: Sequence[Location | int]) -> list[Point]:
    out = []
    for p in picks:
        loc = warehouse.locations[p] if isinstance(p, (int, np.integer)) else p
        out.append(Point(loc.aisle, warehouse.bay_y(loc.bay)))
    return out


def _by_aisle(points: Sequence[Point]) -> dict[int, list[float]]:
    grouped: dict[int, list[float]] = {}
    for p in points:
        grouped.setdefault(p.aisle, []).append(p.y)
    for a in grouped:
        grouped[a] = sorted(set(grouped[a]))
    return grouped


def _path_distance(warehouse: Warehouse, waypoints: Sequence[Point]) -> float:
    return float(
        sum(warehouse.travel_distance(p, q) for p, q in zip(waypoints, waypoints[1:]))
    )


def _append(path: list[Point], point: Point) -> None:
    if not path or path[-1] != point:
        path.append(point)


def _require_single_block(warehouse: Warehouse, policy: str) -> None:
    if warehouse.config.n_blocks != 1:
        raise ValueError(
            f"{policy} is defined for a single-block layout; this warehouse has "
            f"{warehouse.config.n_blocks} blocks. Use nearest_neighbour/two_opt, or "
            "see Roodbergen & de Koster (2001) for the multi-block extension."
        )


def validate_route(warehouse: Warehouse, route: Route, picks: Sequence[Location | int]) -> None:
    """Assert the route is a legal corridor walk that visits every pick.

    Legality: every leg either stays in one aisle (walking along it) or stays on
    one cross aisle (walking across). Anything else is a picker walking through
    a rack.
    """
    wps = route.waypoints
    if not wps:
        raise AssertionError("empty route")
    if wps[0] != warehouse.depot or wps[-1] != warehouse.depot:
        raise AssertionError("route must start and end at the depot")
    cross = set(warehouse.cross_aisle_ys)
    for p, q in zip(wps, wps[1:]):
        if p.aisle == q.aisle:
            continue
        if abs(p.y - q.y) < 1e-9 and any(abs(p.y - c) < 1e-9 for c in cross):
            continue
        raise AssertionError(f"illegal leg {p} -> {q}: not along an aisle or a cross aisle")
    needed = set((pt.aisle, round(pt.y, 6)) for pt in _pick_points(warehouse, picks))
    covered: set[tuple[int, float]] = set()
    for p, q in zip(wps, wps[1:]):
        if p.aisle != q.aisle:
            continue
        lo, hi = sorted((p.y, q.y))
        for a, y in needed:
            if a == p.aisle and lo - 1e-9 <= y <= hi + 1e-9:
                covered.add((a, y))
    missing = needed - covered
    if missing:
        raise AssertionError(f"route misses {len(missing)} pick position(s): {sorted(missing)[:3]}")


# ----------------------------------------------------------------------
# classical heuristics
# ----------------------------------------------------------------------
def s_shape(warehouse: Warehouse, picks: Sequence[Location | int]) -> Route:
    """Traversal routing. The policy most WMS products ship, and rightly so.

    It is trivially trainable ("go up one, down the next"), congestion-friendly
    because everyone moves the same way, and never much worse than the
    alternatives when pick density is high. It is poor at low density, where it
    walks whole aisles for one carton - which is exactly the regime a good
    slotting creates, and one of the reasons slotting and routing have to be
    evaluated together.
    """
    _require_single_block(warehouse, "s_shape")
    points = _pick_points(warehouse, picks)
    if not points:
        return Route((warehouse.depot,), 0.0, "s_shape")
    grouped = _by_aisle(points)
    aisles = sorted(grouped)
    L = warehouse.aisle_length_m

    path: list[Point] = [warehouse.depot]
    at_front = True
    for k, a in enumerate(aisles):
        last = k == len(aisles) - 1
        if last and at_front:
            deepest = max(grouped[a])
            _append(path, Point(a, 0.0))
            _append(path, Point(a, deepest))
            _append(path, Point(a, 0.0))
        else:
            y_in = 0.0 if at_front else L
            y_out = L if at_front else 0.0
            _append(path, Point(a, y_in))
            _append(path, Point(a, y_out))
            at_front = not at_front
    if not at_front:
        # Ended at the back of the last aisle: come down it to the front.
        _append(path, Point(aisles[-1], 0.0))
    _append(path, warehouse.depot)
    return Route(tuple(path), _path_distance(warehouse, path), "s_shape")


def return_route(warehouse: Warehouse, picks: Sequence[Location | int]) -> Route:
    """Enter and leave every aisle from the front. Optimal when picks are shallow."""
    _require_single_block(warehouse, "return_route")
    points = _pick_points(warehouse, picks)
    if not points:
        return Route((warehouse.depot,), 0.0, "return")
    grouped = _by_aisle(points)
    path: list[Point] = [warehouse.depot]
    for a in sorted(grouped):
        _append(path, Point(a, 0.0))
        _append(path, Point(a, max(grouped[a])))
        _append(path, Point(a, 0.0))
    _append(path, warehouse.depot)
    return Route(tuple(path), _path_distance(warehouse, path), "return")


def midpoint(warehouse: Warehouse, picks: Sequence[Location | int]) -> Route:
    """Front half from the front cross aisle, back half from the back one.

    The first and last aisles containing picks are traversed end to end; every
    aisle between them is entered from whichever cross aisle is nearer to each
    of its picks. Midpoint is the weakest of the four classical policies and is
    included because it is the one that shows most clearly how routing and
    slotting interact: once velocity slotting pulls the picks forward, the back
    half empties out and midpoint collapses onto the return route.
    """
    _require_single_block(warehouse, "midpoint")
    points = _pick_points(warehouse, picks)
    if not points:
        return Route((warehouse.depot,), 0.0, "midpoint")
    L = warehouse.aisle_length_m
    mid = L / 2.0
    grouped = _by_aisle(points)
    aisles = sorted(grouped)
    front = {a: [y for y in ys if y <= mid] for a, ys in grouped.items()}
    back = {a: [y for y in ys if y > mid] for a, ys in grouped.items()}

    if len(aisles) == 1 or not any(back.values()):
        r = return_route(warehouse, picks)
        return Route(r.waypoints, r.distance, "midpoint")

    first, last = aisles[0], aisles[-1]
    path: list[Point] = [warehouse.depot]
    for a in aisles[1:-1]:
        if front[a]:
            _append(path, Point(a, 0.0))
            _append(path, Point(a, max(front[a])))
            _append(path, Point(a, 0.0))
    _append(path, Point(last, 0.0))
    _append(path, Point(last, L))
    for a in reversed(aisles[1:-1]):
        if back[a]:
            _append(path, Point(a, L))
            _append(path, Point(a, min(back[a])))
            _append(path, Point(a, L))
    _append(path, Point(first, L))
    _append(path, Point(first, 0.0))
    _append(path, warehouse.depot)
    return Route(tuple(path), _path_distance(warehouse, path), "midpoint")


def largest_gap(warehouse: Warehouse, picks: Sequence[Location | int]) -> Route:
    """Never walk the largest gap between consecutive picks in an aisle.

    The first and last visited aisles are traversed; every aisle between them is
    entered from both ends, going only as deep as the picks on each side of its
    largest gap. This is the strongest of the classical heuristics and is
    usually within a few percent of optimal (Hall 1993; Petersen 1997), which
    the gap table in the README confirms on this instance.
    """
    _require_single_block(warehouse, "largest_gap")
    points = _pick_points(warehouse, picks)
    if not points:
        return Route((warehouse.depot,), 0.0, "largest_gap")
    grouped = _by_aisle(points)
    aisles = sorted(grouped)
    L = warehouse.aisle_length_m

    if len(aisles) == 1:
        r = return_route(warehouse, picks)
        return Route(r.waypoints, r.distance, "largest_gap")

    path: list[Point] = [warehouse.depot]
    first, last = aisles[0], aisles[-1]
    # Up the first aisle.
    _append(path, Point(first, 0.0))
    _append(path, Point(first, L))
    # Back sweep: cover the far side of each intermediate aisle's largest gap.
    for a in aisles[1:-1]:
        ys = grouped[a]
        _, back_start = _largest_gap_split(ys, L)
        if back_start is not None:
            _append(path, Point(a, L))
            _append(path, Point(a, back_start))
            _append(path, Point(a, L))
    # Down the last aisle.
    _append(path, Point(last, L))
    _append(path, Point(last, 0.0))
    # Front sweep back towards the depot.
    for a in reversed(aisles[1:-1]):
        ys = grouped[a]
        front_end, _ = _largest_gap_split(ys, L)
        if front_end is not None:
            _append(path, Point(a, 0.0))
            _append(path, Point(a, front_end))
            _append(path, Point(a, 0.0))
    _append(path, warehouse.depot)
    return Route(tuple(path), _path_distance(warehouse, path), "largest_gap")


def _largest_gap_split(ys: Sequence[float], L: float) -> tuple[float | None, float | None]:
    """Split an aisle's picks at its largest gap.

    Returns ``(deepest pick reached from the front, shallowest reached from the
    back)``; either can be ``None`` when the largest gap is at an end, in which
    case the whole aisle is served from the other end.
    """
    ys = sorted(ys)
    edges = [0.0, *ys, L]
    gaps = [(edges[i + 1] - edges[i], i) for i in range(len(edges) - 1)]
    best_gap, best_i = max(gaps)
    if best_i == 0:  # gap is between the front and the first pick
        return None, ys[0]
    if best_i == len(edges) - 2:  # gap is between the last pick and the back
        return ys[-1], None
    return edges[best_i], edges[best_i + 1]


# ----------------------------------------------------------------------
# geometry-agnostic heuristics
# ----------------------------------------------------------------------
def _expand(warehouse: Warehouse, p: Point, q: Point) -> list[Point]:
    """Waypoints of the shortest corridor path between two points."""
    if p.aisle == q.aisle:
        return [q]
    best_c, best_d = None, math.inf
    for c in warehouse.cross_aisle_ys:
        d = abs(p.y - c) + abs(c - q.y)
        if d < best_d:
            best_c, best_d = c, d
    return [Point(p.aisle, best_c), Point(q.aisle, best_c), q]


def _tour_to_route(warehouse: Warehouse, order: Sequence[Point], policy: str) -> Route:
    path: list[Point] = [warehouse.depot]
    prev = warehouse.depot
    for pt in list(order) + [warehouse.depot]:
        for step in _expand(warehouse, prev, pt):
            _append(path, step)
        prev = pt
    return Route(tuple(path), _path_distance(warehouse, path), policy)


def _unique_points(warehouse: Warehouse, picks: Sequence[Location | int]) -> list[Point]:
    uniq = sorted({(p.aisle, p.y) for p in _pick_points(warehouse, picks)})
    return [Point(a, y) for a, y in uniq]


def _nn_order(warehouse: Warehouse, nodes: list[Point]) -> list[Point]:
    remaining = list(nodes)
    order: list[Point] = []
    cur = warehouse.depot
    while remaining:
        k = min(
            range(len(remaining)),
            key=lambda i: (warehouse.travel_distance(cur, remaining[i]), i),
        )
        cur = remaining.pop(k)
        order.append(cur)
    return order


def nearest_neighbour(warehouse: Warehouse, picks: Sequence[Location | int]) -> Route:
    """Greedy nearest-unvisited-pick tour under the warehouse metric."""
    nodes = _unique_points(warehouse, picks)
    if not nodes:
        return Route((warehouse.depot,), 0.0, "nearest_neighbour")
    return _tour_to_route(warehouse, _nn_order(warehouse, nodes), "nearest_neighbour")


def two_opt(
    warehouse: Warehouse, picks: Sequence[Location | int], max_passes: int = 40
) -> Route:
    """2-opt improvement on the nearest-neighbour tour.

    Works on any geometry including multi-block, at the cost of being a general
    TSP heuristic that knows nothing about aisles. On single-block instances it
    lands close to the aisle DP, which is a useful sanity check on both.
    """
    picks_pts = _unique_points(warehouse, picks)
    if not picks_pts:
        return Route((warehouse.depot,), 0.0, "two_opt")
    if len(picks_pts) <= 2:
        return _tour_to_route(warehouse, _nn_order(warehouse, picks_pts), "two_opt")
    nodes = [warehouse.depot] + _nn_order(warehouse, picks_pts)
    n = len(nodes)
    d = np.asarray(
        [[warehouse.travel_distance(a, b) for b in nodes] for a in nodes], dtype=float
    )
    idx = list(range(n))
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = idx[i - 1], idx[i]
                c, e = idx[j], idx[(j + 1) % n]
                delta = d[a, c] + d[b, e] - d[a, b] - d[c, e]
                if delta < -1e-9:
                    idx[i : j + 1] = idx[i : j + 1][::-1]
                    improved = True
    ordered = [nodes[k] for k in idx[1:]]
    return _tour_to_route(warehouse, ordered, "two_opt")


# ----------------------------------------------------------------------
# exact
# ----------------------------------------------------------------------
def held_karp(
    warehouse: Warehouse, picks: Sequence[Location | int], max_picks: int = 14
) -> Route:
    """Exact optimal tour by Held-Karp DP over subsets.

    The picker's optimal route is exactly the TSP tour over ``{depot} u picks``
    under the shortest-path metric, because the metric already encodes every
    legal way of getting between two points. That equivalence is what lets a
    generic TSP solver certify a warehouse-specific aisle DP.
    """
    nodes = [warehouse.depot] + _unique_points(warehouse, picks)
    n = len(nodes) - 1
    if n == 0:
        return Route((warehouse.depot,), 0.0, "exact_held_karp")
    if n > max_picks:
        raise ValueError(
            f"held_karp is exponential; {n} distinct pick positions exceeds the "
            f"cap of {max_picks}. Use exact_aisle_dp for large single-block instances."
        )
    d = np.asarray(
        [[warehouse.travel_distance(a, b) for b in nodes] for a in nodes], dtype=float
    )
    size = 1 << n
    dp = np.full((size, n), np.inf)
    parent = np.full((size, n), -1, dtype=np.int64)
    for j in range(n):
        dp[1 << j, j] = d[0, j + 1]
    for mask in range(size):
        row = dp[mask]
        for j in range(n):
            cost = row[j]
            if not np.isfinite(cost) or not (mask >> j) & 1:
                continue
            for k in range(n):
                if (mask >> k) & 1:
                    continue
                nmask = mask | (1 << k)
                cand = cost + d[j + 1, k + 1]
                if cand < dp[nmask, k] - 1e-12:
                    dp[nmask, k] = cand
                    parent[nmask, k] = j
    full = size - 1
    totals = dp[full] + d[1:, 0]
    end = int(np.argmin(totals))
    best = float(totals[end])

    order_idx: list[int] = []
    mask, j = full, end
    while j >= 0:
        order_idx.append(j)
        pj = int(parent[mask, j])
        mask ^= 1 << j
        j = pj
    order_idx.reverse()
    order = [nodes[i + 1] for i in order_idx]
    route = _tour_to_route(warehouse, order, "exact_held_karp")
    return Route(route.waypoints, best, "exact_held_karp")


# --- Ratliff-Rosenthal dynamic program ---------------------------------
#
# State after processing aisle j is the equivalence class of the partial
# multigraph built so far, described by:
#   * the degree of the aisle's back vertex a_j and front vertex b_j, reduced
#     to (touched?, parity),
#   * the connectivity: 0 = no edges yet, 1 = one component, 2 = two
#     components (necessarily one holding a_j and one holding b_j).
#
# Any component containing neither active vertex can never be joined to the
# rest, so it is pruned - which is what keeps the class count tiny. The
# transition adds cross-aisle edges (closing a_j and b_j, whose degrees must
# then be even) and then the next aisle's vertical configuration.

_CFG_NONE = ("none", 0, 0, "none")
_VerticalCfg = tuple[str, int, int, str]


def _vertical_configs(ys: Sequence[float], L: float) -> list[tuple[float, _VerticalCfg]]:
    """Cost and effect of every way the picker can service one aisle."""
    cfgs: list[tuple[float, _VerticalCfg]] = []
    if not ys:
        cfgs.append((0.0, _CFG_NONE))
    cfgs.append((L, ("one_pass", 1, 1, "ab")))
    cfgs.append((2.0 * L, ("two_pass", 2, 2, "ab")))
    if ys:
        ys = sorted(ys)
        cfgs.append((2.0 * max(ys), ("front_return", 0, 2, "b")))
        cfgs.append((2.0 * (L - min(ys)), ("back_return", 2, 0, "a")))
        front_end, back_start = _largest_gap_split(ys, L)
        if front_end is not None and back_start is not None:
            cfgs.append(
                (2.0 * front_end + 2.0 * (L - back_start), ("gap", 2, 2, "a|b"))
            )
    return cfgs


class _DSU:
    __slots__ = ("p",)

    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _apply_vertical(state: tuple, link: str, dda: int, ddb: int) -> tuple | None:
    """Fold one aisle's vertical configuration into the partial-tour class."""
    a_t, a_p, b_t, b_p, conn = state
    dsu = _DSU(2)  # 0 = back vertex a, 1 = front vertex b
    if conn == 1 and a_t and b_t:
        dsu.union(0, 1)
    if link == "ab":
        dsu.union(0, 1)
    a_in = bool(a_t) or link in ("ab", "a", "a|b")
    b_in = bool(b_t) or link in ("ab", "b", "a|b")
    if a_in and b_in:
        new_conn = 1 if dsu.find(0) == dsu.find(1) else 2
    elif a_in or b_in:
        new_conn = 1
    else:
        new_conn = 0
    return (
        bool(a_t or dda > 0),
        (a_p + dda) % 2,
        bool(b_t or ddb > 0),
        (b_p + ddb) % 2,
        new_conn,
    )


def _apply_cross(state: tuple, t: int, u: int) -> tuple | None:
    """Close the current aisle's vertices and step to the next aisle."""
    a_t, a_p, b_t, b_p, conn = state
    if (a_p + t) % 2 or (b_p + u) % 2:
        return None
    has_edges = conn != 0
    if not has_edges and t == 0 and u == 0:
        return (False, 0, False, 0, 0)
    # every existing component must reach a vertex of the next aisle
    if conn == 1:
        alive = (a_t and t > 0) or (b_t and u > 0)
        if not alive:
            return None
    elif conn == 2:
        if not (t > 0 and u > 0):
            return None
    dsu = _DSU(4)  # 0 = a_old, 1 = b_old, 2 = a_new, 3 = b_new
    if conn == 1 and a_t and b_t:
        dsu.union(0, 1)
    if t > 0:
        dsu.union(0, 2)
    if u > 0:
        dsu.union(1, 3)
    roots = set()
    if t > 0:
        roots.add(dsu.find(2))
    if u > 0:
        roots.add(dsu.find(3))
    new_conn = len(roots)
    return (t > 0, t % 2, u > 0, u % 2, new_conn)


def exact_aisle_dp(warehouse: Warehouse, picks: Sequence[Location | int]) -> float:
    """Minimum picker tour length by the Ratliff-Rosenthal DP. Single block only.

    Returns the distance. The waypoint reconstruction is not implemented -
    the DP is used to measure heuristic gaps, and for drawing a route the
    heuristics or Held-Karp give a path directly.
    """
    _require_single_block(warehouse, "exact_aisle_dp")
    points = _pick_points(warehouse, picks)
    if not points:
        return 0.0
    grouped = _by_aisle(points)
    depot_aisle = warehouse.config.depot_aisle
    lo = min(min(grouped), depot_aisle)
    hi = max(max(grouped), depot_aisle)
    L = warehouse.aisle_length_m
    pitch = warehouse.config.aisle_pitch_m

    states: dict[tuple, float] = {(False, 0, False, 0, 0): 0.0}
    for a in range(lo, hi + 1):
        ys = grouped.get(a, [])
        nxt: dict[tuple, float] = {}
        for cost, (_, dda, ddb, link) in _vertical_configs(ys, L):
            for state, base in states.items():
                new_state = _apply_vertical(state, link, dda, ddb)
                if new_state is None:
                    continue
                total = base + cost
                if total < nxt.get(new_state, math.inf) - 1e-12:
                    nxt[new_state] = total
        states = nxt
        if not states:
            return math.inf

        if a < hi:
            stepped: dict[tuple, float] = {}
            for state, base in states.items():
                for t in range(3):
                    for u in range(3):
                        if a == depot_aisle and not (state[2] or u > 0):
                            continue  # depot vertex would end up isolated
                        new_state = _apply_cross(state, t, u)
                        if new_state is None:
                            continue
                        total = base + (t + u) * pitch
                        if total < stepped.get(new_state, math.inf) - 1e-12:
                            stepped[new_state] = total
            states = stepped
            if not states:
                return math.inf

    best = math.inf
    for (a_t, a_p, b_t, b_p, conn), cost in states.items():
        if a_p or b_p or conn != 1:
            continue
        if hi == depot_aisle and not b_t:
            continue
        best = min(best, cost)
    return best


def route_picks(
    warehouse: Warehouse, picks: Sequence[Location | int], policy: str = "s_shape"
) -> Route:
    """Route by policy name."""
    try:
        fn = ROUTERS[policy]
    except KeyError:  # pragma: no cover - defensive
        raise ValueError(f"unknown routing policy {policy!r}; have {sorted(ROUTERS)}") from None
    return fn(warehouse, picks)


ROUTERS: dict[str, Callable[[Warehouse, Sequence[Location | int]], Route]] = {
    "s_shape": s_shape,
    "return": return_route,
    "midpoint": midpoint,
    "largest_gap": largest_gap,
    "nearest_neighbour": nearest_neighbour,
    "two_opt": two_opt,
    "held_karp": held_karp,
}
