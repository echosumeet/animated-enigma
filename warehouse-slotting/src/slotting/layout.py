"""Warehouse geometry and the travel-distance metric.

The single most consequential modelling decision in a slotting study is the
distance function. Euclidean distance between two pallet positions is wrong by
a factor that is not constant: two locations facing each other across a rack
are 2.5 m apart in the plane and 60 m apart for a picker, because the picker
has to walk to a cross aisle and back. Every downstream number - ABC savings,
affinity clusters, route length - inherits that error.

So the metric here is the shortest path on the actual aisle graph:

    same aisle              -> |y_p - y_q|
    different aisle         -> |x_p - x_q| + min_c (|y_p - c| + |c - y_q|)

where ``c`` ranges over the y-coordinates of the cross aisles. The second term
collapses to |y_p - y_q| whenever a cross aisle lies between the two points,
which is exactly the effect a mid-warehouse cross aisle is bought for.

``tests/test_layout.py`` checks this closed form against a Dijkstra shortest
path on an explicit networkx graph of the corridor network, on every pair of
locations in a small warehouse. The closed form is used everywhere else
because it is ~400x faster and the slotting search calls it millions of times.

Vertical travel is deliberately *not* in the distance metric. A picker does not
walk up the rack; reaching level 3 costs time and ergonomic risk, not metres.
That cost lives in :mod:`slotting.ergonomics` and is converted to
metre-equivalents at a stated walking speed so that one objective can carry
both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

import numpy as np

__all__ = [
    "Location",
    "Point",
    "WarehouseConfig",
    "Warehouse",
]


@dataclass(frozen=True, slots=True, order=True)
class Location:
    """A single pick face: one aisle, one side of that aisle, one bay, one level."""

    aisle: int
    side: int  # 0 = left-hand rack, 1 = right-hand rack
    bay: int
    level: int  # 0 = floor

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"A{self.aisle:02d}-{'LR'[self.side]}{self.bay:02d}-{self.level}"


@dataclass(frozen=True, slots=True)
class Point:
    """A position on the floor of the warehouse (an access point or the depot)."""

    aisle: int
    y: float


@dataclass(frozen=True)
class WarehouseConfig:
    """Physical description of a rectangular, block-layout warehouse.

    Defaults are a mid-size case-pick forward area: 10 aisles at 4 m centres,
    24 bays of 1.2 m, 4 levels. That is 1,920 pick faces over a 28.8 m aisle
    run, which is the scale at which manual slotting stops being possible and
    people start asking for a tool.
    """

    n_aisles: int = 10
    n_bays: int = 24
    n_levels: int = 4
    n_blocks: int = 1  # 1 -> front and back cross aisles only
    aisle_pitch_m: float = 4.0  # centre-to-centre spacing of picking aisles
    bay_depth_m: float = 1.2  # along-aisle length of one bay
    level_height_m: float = 1.0
    depot_aisle: int = 0  # depot sits at the front end of this aisle
    # Slot capability. Weight capacity falls with level: a picker lifting to
    # level 3 off a ladder cannot handle what they can handle at knee height,
    # and the rack beam rating is not the binding constraint - the human is.
    base_weight_capacity_kg: float = 1200.0
    weight_capacity_decay: float = 0.50  # multiplier applied per level above 0
    slot_cube_m3: float = 1.35
    # A pick face holds a bounded number of cases regardless of how small they
    # are: replenishment is by pallet or tote, not by grain of sand. Without
    # this cap a 3-litre case implies 450 cases in a slot and a slot weight
    # nothing could ever lift, and the level constraint stops meaning anything.
    max_cases_per_slot: int = 60

    def __post_init__(self) -> None:
        if self.n_aisles < 1 or self.n_bays < 1 or self.n_levels < 1:
            raise ValueError("warehouse must have at least one aisle, bay and level")
        if self.n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")
        if self.n_bays % self.n_blocks:
            raise ValueError(
                f"n_bays ({self.n_bays}) must divide evenly into n_blocks ({self.n_blocks})"
            )
        if not 0 <= self.depot_aisle < self.n_aisles:
            raise ValueError("depot_aisle must index an existing aisle")
        if not 0.0 < self.weight_capacity_decay <= 1.0:
            raise ValueError("weight_capacity_decay must be in (0, 1]")


class Warehouse:
    """Geometry, the location universe, and the travel metric over it."""

    def __init__(self, config: WarehouseConfig | None = None) -> None:
        self.config = config or WarehouseConfig()
        c = self.config

        self.aisle_length_m: float = c.n_bays * c.bay_depth_m
        self.cross_aisle_ys: tuple[float, ...] = tuple(
            self.aisle_length_m * i / c.n_blocks for i in range(c.n_blocks + 1)
        )
        self._cross = np.asarray(self.cross_aisle_ys, dtype=float)

        self.locations: tuple[Location, ...] = tuple(
            Location(a, s, b, l)
            for a in range(c.n_aisles)
            for s in range(2)
            for b in range(c.n_bays)
            for l in range(c.n_levels)
        )
        self.index: dict[Location, int] = {loc: i for i, loc in enumerate(self.locations)}

        # Distance depends only on (aisle, bay): both sides of an aisle and all
        # levels of a bay share one floor access point. Collapsing to access
        # points shrinks the distance matrix by 2 * n_levels in each dimension.
        self._n_access = c.n_aisles * c.n_bays
        self._loc_access = np.asarray(
            [loc.aisle * c.n_bays + loc.bay for loc in self.locations], dtype=np.int64
        )
        self._access_matrix = self._build_access_matrix()

        self.depot = Point(c.depot_aisle, 0.0)
        self._depot_access_dist = np.asarray(
            [
                self.travel_distance(self.depot, Point(a, self.bay_y(b)))
                for a in range(c.n_aisles)
                for b in range(c.n_bays)
            ],
            dtype=float,
        )

    # ------------------------------------------------------------------
    # basic geometry
    # ------------------------------------------------------------------
    @property
    def n_locations(self) -> int:
        return len(self.locations)

    def aisle_x(self, aisle: int) -> float:
        """Centreline x of a picking aisle, in metres."""
        return aisle * self.config.aisle_pitch_m

    def bay_y(self, bay: int) -> float:
        """Along-aisle y of the centre of a bay, in metres."""
        return (bay + 0.5) * self.config.bay_depth_m

    def block_of_bay(self, bay: int) -> int:
        """Which block (between which pair of cross aisles) a bay falls in."""
        per_block = self.config.n_bays // self.config.n_blocks
        return bay // per_block

    def location_point(self, loc: Location) -> Point:
        return Point(loc.aisle, self.bay_y(loc.bay))

    def location_xy(self, loc: Location) -> tuple[float, float]:
        """Cartesian position of the *face* of a location, used for drawing."""
        offset = self.config.aisle_pitch_m * 0.28
        return (self.aisle_x(loc.aisle) + (offset if loc.side else -offset), self.bay_y(loc.bay))

    # ------------------------------------------------------------------
    # the metric
    # ------------------------------------------------------------------
    def travel_distance(self, p: Point, q: Point) -> float:
        """Shortest walking distance between two floor positions, in metres."""
        if p.aisle == q.aisle:
            return abs(p.y - q.y)
        horizontal = abs(self.aisle_x(p.aisle) - self.aisle_x(q.aisle))
        lo, hi = (p.y, q.y) if p.y <= q.y else (q.y, p.y)
        best = math.inf
        for c in self.cross_aisle_ys:
            if lo <= c <= hi:
                return horizontal + (hi - lo)
            d = abs(p.y - c) + abs(c - q.y)
            if d < best:
                best = d
        return horizontal + best

    def distance(self, a: Location | int, b: Location | int) -> float:
        """Travel distance between two locations, by object or by index."""
        ia = a if isinstance(a, int) else self.index[a]
        ib = b if isinstance(b, int) else self.index[b]
        return float(self._access_matrix[self._loc_access[ia], self._loc_access[ib]])

    def distance_from_depot(self, a: Location | int) -> float:
        ia = a if isinstance(a, int) else self.index[a]
        return float(self._depot_access_dist[self._loc_access[ia]])

    def depot_distance_vector(self) -> np.ndarray:
        """One-way depot distance for every location index."""
        return self._depot_access_dist[self._loc_access]

    def access_index(self, loc: Location | int) -> int:
        ia = loc if isinstance(loc, int) else self.index[loc]
        return int(self._loc_access[ia])

    def access_distance_matrix(self) -> np.ndarray:
        """The (n_aisles * n_bays) square distance matrix over access points."""
        return self._access_matrix

    def _build_access_matrix(self) -> np.ndarray:
        c = self.config
        n = self._n_access
        aisles = np.repeat(np.arange(c.n_aisles), c.n_bays)
        ys = np.tile(np.asarray([self.bay_y(b) for b in range(c.n_bays)]), c.n_aisles)
        xs = aisles * c.aisle_pitch_m

        same_aisle = aisles[:, None] == aisles[None, :]
        dy = np.abs(ys[:, None] - ys[None, :])
        dx = np.abs(xs[:, None] - xs[None, :])

        # Cost of getting from y_i to y_j via the best cross aisle.
        via = np.abs(ys[:, None, None] - self._cross[None, None, :]) + np.abs(
            self._cross[None, None, :] - ys[None, :, None]
        )
        via_best = via.min(axis=2)

        out = np.where(same_aisle, dy, dx + via_best)
        np.fill_diagonal(out, 0.0)
        return out.astype(float)

    # ------------------------------------------------------------------
    # slot capability
    # ------------------------------------------------------------------
    def weight_capacity(self, level: int) -> float:
        """Practical weight a picker can handle at this level, in kg per slot."""
        return self.config.base_weight_capacity_kg * (self.config.weight_capacity_decay**level)

    def cube_capacity(self, level: int) -> float:
        return self.config.slot_cube_m3

    # ------------------------------------------------------------------
    # convenience selectors
    # ------------------------------------------------------------------
    def locations_by_depot_distance(self) -> list[int]:
        """Location indices sorted by walking distance from the depot, nearest first.

        Ties (which are everywhere - four levels share one access point) are
        broken by level using the golden-zone order, so that a caller filling
        locations in this sequence gets the ergonomically good faces of the
        near bays before the bad faces of the near bays. That single tiebreak
        is worth more than it looks; see :mod:`slotting.ergonomics`.
        """
        from .ergonomics import golden_zone_order

        order = golden_zone_order(self.config.n_levels, self.config.level_height_m)
        rank = {lvl: i for i, lvl in enumerate(order)}
        d = self.depot_distance_vector()
        return sorted(range(self.n_locations), key=lambda i: (d[i], rank[self.locations[i].level]))

    def iter_aisle(self, aisle: int) -> Iterator[Location]:
        for loc in self.locations:
            if loc.aisle == aisle:
                yield loc

    def to_networkx(self):
        """Explicit corridor graph. Used by the tests to validate the closed form.

        Nodes are (aisle, y) for every cross-aisle intersection and every bay
        access point; edges run only along aisles and along cross aisles.
        """
        import networkx as nx

        c = self.config
        g = nx.Graph()
        ys_by_aisle: dict[int, list[float]] = {}
        for a in range(c.n_aisles):
            ys = sorted({*self.cross_aisle_ys, *(self.bay_y(b) for b in range(c.n_bays))})
            ys_by_aisle[a] = ys
            for y0, y1 in zip(ys, ys[1:]):
                g.add_edge((a, y0), (a, y1), weight=y1 - y0)
        for cy in self.cross_aisle_ys:
            for a in range(c.n_aisles - 1):
                g.add_edge((a, cy), (a + 1, cy), weight=c.aisle_pitch_m)
        return g


def total_path_distance(warehouse: Warehouse, waypoints: Sequence[Point]) -> float:
    """Sum the metric along an ordered list of waypoints."""
    return float(
        sum(
            warehouse.travel_distance(p, q)
            for p, q in zip(waypoints, waypoints[1:])
        )
    )
