"""The assignment itself, and the constraints that make it hard.

A slotting assignment is a partial injection from SKUs to pick faces: one SKU
per face, one face per SKU. Everything downstream - objective, local search,
routing - reads it through this class.

The constraints are the part that separates a slotting tool from a sorting
exercise:

* **Unary (SKU x level).** A full slot of a dense SKU weighs more than a picker
  should handle at height, and a case that does not fit in the slot cube does
  not fit. Both reduce to a boolean ``allowed[sku, level]`` table, which is
  small (n_skus x n_levels) and can be checked in constant time inside the
  search loop.
* **Hazmat segregation (binary).** Incompatible classes may not be stored
  within a separation distance of each other. This is a genuine pairwise
  constraint and cannot be folded into a level table. It is checked
  incrementally: only the pairs touching a proposed move are re-examined, which
  is what makes 100k-move simulated annealing feasible in pure Python.
* **Family grouping.** Modelled as a *soft* term in the objective rather than a
  hard constraint. Hard family zones are easy to write and almost always wrong:
  they forbid the trade the optimiser should be allowed to make, which is
  putting one very fast SKU from a slow family in a near slot.

The design rule here: hard constraints are for things that are illegal or
impossible, soft terms are for things that are merely undesirable. Warehouse
optimisers that encode preferences as hard constraints end up infeasible, and
then someone relaxes them by hand and the answer is nobody's model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .layout import Location, Warehouse
from .skus import Catalog, hazmat_incompatible

__all__ = [
    "ConstraintConfig",
    "ConstraintModel",
    "Assignment",
    "random_assignment",
    "greedy_place",
]


@dataclass(frozen=True)
class ConstraintConfig:
    """Knobs for the hard-constraint model."""

    hazmat_separation_bays: int = 3
    hazmat_max_level: int = 0  # flammables and friends stay on the floor
    enforce_weight_by_level: bool = True
    enforce_cube_fit: bool = True
    enforce_hazmat: bool = True


class ConstraintModel:
    """Precomputed feasibility structure for one (warehouse, catalog) pair."""

    def __init__(
        self,
        warehouse: Warehouse,
        catalog: Catalog,
        config: ConstraintConfig | None = None,
    ) -> None:
        self.warehouse = warehouse
        self.catalog = catalog
        self.config = config or ConstraintConfig()
        c = warehouse.config

        n_levels = c.n_levels
        slot_weight = catalog.slot_weight(
            c.slot_cube_m3, c.max_cases_per_slot, warehouse.weight_capacity(0)
        )
        allowed = np.ones((catalog.n_skus, n_levels), dtype=bool)

        if self.config.enforce_weight_by_level:
            caps = np.asarray([warehouse.weight_capacity(l) for l in range(n_levels)])
            allowed &= slot_weight[:, None] <= caps[None, :]

        if self.config.enforce_cube_fit:
            cube_ok = catalog.case_cube <= warehouse.cube_capacity(0)
            allowed &= cube_ok[:, None]

        if self.config.enforce_hazmat:
            is_haz = catalog.hazmat != "none"
            lvl = np.arange(n_levels)
            allowed &= ~(is_haz[:, None] & (lvl[None, :] > self.config.hazmat_max_level))

        # A SKU with no feasible level is a data problem, not a search problem.
        dead = np.flatnonzero(~allowed.any(axis=1))
        if len(dead):
            raise ValueError(
                f"{len(dead)} SKU(s) have no feasible level; first is sku {int(dead[0])} "
                f"(slot weight {slot_weight[dead[0]]:.0f} kg, case cube "
                f"{catalog.case_cube[dead[0]]:.3f} m3)"
            )

        self.allowed_level = allowed
        self.slot_weight = slot_weight
        self.location_level = np.asarray([loc.level for loc in warehouse.locations])
        self.location_aisle = np.asarray([loc.aisle for loc in warehouse.locations])
        self.location_bay = np.asarray([loc.bay for loc in warehouse.locations])

        haz_idx = catalog.hazmat_indices()
        self.hazmat_skus = haz_idx
        self._is_hazmat = np.zeros(catalog.n_skus, dtype=bool)
        self._is_hazmat[haz_idx] = True
        self._haz_class = catalog.hazmat
        # Precompute which hazmat SKU pairs conflict at all; the spatial test is
        # only run for pairs that are chemically incompatible.
        self._conflicts: dict[int, list[int]] = {int(i): [] for i in haz_idx}
        for a_pos in range(len(haz_idx)):
            for b_pos in range(a_pos + 1, len(haz_idx)):
                i, j = int(haz_idx[a_pos]), int(haz_idx[b_pos])
                if hazmat_incompatible(str(self._haz_class[i]), str(self._haz_class[j])):
                    self._conflicts[i].append(j)
                    self._conflicts[j].append(i)

    # ------------------------------------------------------------------
    def level_ok(self, sku: int, loc: int) -> bool:
        return bool(self.allowed_level[sku, self.location_level[loc]])

    def feasible_locations(self, sku: int) -> np.ndarray:
        """Indices of every location this SKU may occupy."""
        return np.flatnonzero(self.allowed_level[sku, self.location_level])

    def hazmat_conflict(self, loc_a: int, loc_b: int) -> bool:
        """True if two locations are too close for incompatible classes."""
        if self.location_aisle[loc_a] != self.location_aisle[loc_b]:
            return False
        return (
            abs(int(self.location_bay[loc_a]) - int(self.location_bay[loc_b]))
            < self.config.hazmat_separation_bays
        )

    def hazmat_ok_for(self, sku: int, loc: int, assignment: "Assignment") -> bool:
        """Would placing ``sku`` at ``loc`` violate segregation against the rest?"""
        if not self.config.enforce_hazmat or not self._is_hazmat[sku]:
            return True
        for other in self._conflicts[int(sku)]:
            other_loc = assignment.location_of[other]
            if other_loc < 0 or other == sku:
                continue
            if self.hazmat_conflict(loc, int(other_loc)):
                return False
        return True

    def violations(self, assignment: "Assignment") -> list[str]:
        """Full audit of an assignment. Used by tests and by the CLI report."""
        problems: list[str] = []
        for sku, loc in enumerate(assignment.location_of):
            if loc < 0:
                problems.append(f"sku {sku} unslotted")
                continue
            if not self.level_ok(sku, int(loc)):
                problems.append(
                    f"sku {sku} at level {int(self.location_level[loc])} violates level capability"
                )
        occupied = assignment.location_of[assignment.location_of >= 0]
        if len(np.unique(occupied)) != len(occupied):
            problems.append("two SKUs share a location")
        if self.config.enforce_hazmat:
            for i, others in self._conflicts.items():
                li = assignment.location_of[i]
                if li < 0:
                    continue
                for j in others:
                    if j <= i:
                        continue
                    lj = assignment.location_of[j]
                    if lj >= 0 and self.hazmat_conflict(int(li), int(lj)):
                        problems.append(
                            f"hazmat separation violated between sku {i} and sku {j}"
                        )
        return problems


class Assignment:
    """Mutable SKU -> location mapping with an inverse index kept in step."""

    __slots__ = ("n_skus", "n_locations", "location_of", "sku_at")

    def __init__(self, n_skus: int, n_locations: int) -> None:
        if n_locations < n_skus:
            raise ValueError(
                f"{n_skus} SKUs do not fit in {n_locations} locations; "
                "the forward area is undersized for this catalogue"
            )
        self.n_skus = n_skus
        self.n_locations = n_locations
        self.location_of = np.full(n_skus, -1, dtype=np.int64)
        self.sku_at = np.full(n_locations, -1, dtype=np.int64)

    # ------------------------------------------------------------------
    def place(self, sku: int, loc: int) -> None:
        old = self.location_of[sku]
        if old >= 0:
            self.sku_at[old] = -1
        occupant = self.sku_at[loc]
        if occupant >= 0 and occupant != sku:
            raise ValueError(f"location {loc} already holds sku {int(occupant)}")
        self.location_of[sku] = loc
        self.sku_at[loc] = sku

    def swap(self, sku_a: int, sku_b: int) -> None:
        """Exchange the locations of two slotted SKUs."""
        la, lb = int(self.location_of[sku_a]), int(self.location_of[sku_b])
        self.location_of[sku_a], self.location_of[sku_b] = lb, la
        self.sku_at[la], self.sku_at[lb] = sku_b, sku_a

    def relocate(self, sku: int, new_loc: int) -> None:
        """Move a SKU to an empty location."""
        if self.sku_at[new_loc] >= 0:
            raise ValueError(f"location {new_loc} is occupied")
        old = int(self.location_of[sku])
        if old >= 0:
            self.sku_at[old] = -1
        self.location_of[sku] = new_loc
        self.sku_at[new_loc] = sku

    def copy(self) -> "Assignment":
        other = Assignment.__new__(Assignment)
        other.n_skus = self.n_skus
        other.n_locations = self.n_locations
        other.location_of = self.location_of.copy()
        other.sku_at = self.sku_at.copy()
        return other

    def is_complete(self) -> bool:
        return bool((self.location_of >= 0).all())

    def empty_locations(self) -> np.ndarray:
        return np.flatnonzero(self.sku_at < 0)

    def locations(self, warehouse: Warehouse) -> list[Location]:
        return [warehouse.locations[int(i)] for i in self.location_of]

    def occupancy(self) -> float:
        return float((self.sku_at >= 0).mean())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Assignment):
            return NotImplemented
        return bool(np.array_equal(self.location_of, other.location_of))


def greedy_place(
    constraints: ConstraintModel,
    sku_order: list[int],
    location_order: list[int],
) -> Assignment:
    """Place SKUs into locations in the two given priority orders.

    Every construction heuristic in this library reduces to this function: rank
    the SKUs, rank the locations, then zip them together skipping infeasible
    pairs. Keeping one implementation means the constraint handling - and in
    particular the hazmat check, which is order-dependent - is identical across
    methods, so a comparison between them measures the ranking and nothing else.

    The frontier cursor makes this O(n_locations) amortised rather than
    O(n_skus * n_locations): once every location before the cursor is taken it
    is never rescanned. The inner scan can still walk past infeasible-but-free
    locations, which is what the level constraint costs.
    """
    n_skus = constraints.catalog.n_skus
    n_locs = constraints.warehouse.n_locations
    assignment = Assignment(n_skus, n_locs)
    taken = np.zeros(n_locs, dtype=bool)
    cursor = 0
    for sku in sku_order:
        sku = int(sku)
        placed = False
        i = cursor
        while i < len(location_order):
            loc = int(location_order[i])
            if not taken[loc] and constraints.level_ok(sku, loc):
                if constraints.hazmat_ok_for(sku, loc, assignment):
                    assignment.place(sku, loc)
                    taken[loc] = True
                    placed = True
                    break
            i += 1
        if not placed:
            raise RuntimeError(
                f"no feasible free location for sku {sku}; the forward area cannot "
                "hold this catalogue under the current constraints"
            )
        while cursor < len(location_order) and taken[int(location_order[cursor])]:
            cursor += 1
    return assignment


def random_assignment(
    constraints: ConstraintModel,
    seed: int = 0,
    order: np.ndarray | None = None,
) -> Assignment:
    """A feasible but velocity-blind assignment: the honest 'before' picture.

    This is what a warehouse that has never been slotted actually looks like.
    Nobody assigns locations at random on purpose; they assign the first empty
    location that will hold the pallet when the first receipt lands, which with
    respect to pick velocity is random. Benchmarking against a *sorted-by-id*
    baseline flatters the optimiser, so the baseline here is a real random
    feasible fill, averaged over seeds in the benchmark.
    """
    rng = np.random.default_rng(seed)
    n_skus = constraints.catalog.n_skus
    assignment = Assignment(n_skus, constraints.warehouse.n_locations)

    sku_order = np.arange(n_skus) if order is None else np.asarray(order)
    # Hardest-to-place first: SKUs with the fewest feasible levels go down
    # before the easy ones eat their slots. Without this the fill fails on
    # instances where heavy SKUs are a meaningful share of the catalogue.
    n_levels_ok = constraints.allowed_level.sum(axis=1)
    sku_order = sku_order[np.argsort(n_levels_ok[sku_order], kind="stable")]

    free_by_level: dict[int, list[int]] = {}
    for loc_idx, loc in enumerate(constraints.warehouse.locations):
        free_by_level.setdefault(loc.level, []).append(loc_idx)
    for lvl in free_by_level:
        arr = np.asarray(free_by_level[lvl])
        rng.shuffle(arr)
        free_by_level[lvl] = list(arr)

    taken = np.zeros(constraints.warehouse.n_locations, dtype=bool)
    for sku in sku_order:
        sku = int(sku)
        levels = np.flatnonzero(constraints.allowed_level[sku])
        rng.shuffle(levels)
        placed = False
        for lvl in levels:
            bucket = free_by_level[int(lvl)]
            for pos in range(len(bucket)):
                loc = bucket[pos]
                if taken[loc]:
                    continue
                if not constraints.hazmat_ok_for(sku, loc, assignment):
                    continue
                assignment.place(sku, loc)
                taken[loc] = True
                bucket[pos] = bucket[-1]
                bucket.pop()
                placed = True
                break
            if placed:
                break
        if not placed:
            raise RuntimeError(
                f"could not place sku {sku}: no feasible free location remains"
            )
    return assignment
