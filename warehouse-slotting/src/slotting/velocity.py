"""Velocity-based (ABC) slotting, and the three ranking keys worth arguing about.

The construction is the same in every case - rank SKUs by a key, rank locations
by cost, zip them together respecting feasibility - and the whole content of
the method is in the key:

``picks``
    Pick lines per period. Right when the constraint is picker labour and every
    pick costs about the same to execute. This is the default and it is right
    more often than the alternatives.

``cube``
    Cube shipped per period. Right when the binding constraint is
    replenishment, not picking: a fast-moving bulky SKU in a far slot generates
    a replenishment trip per case-and-a-half, and those trips are longer than
    picks because they come from reserve.

``coi``
    Heskett's cube-per-order index (1963): storage cube divided by order
    frequency, ranked ascending. The classical result is that COI ordering is
    optimal for the single-command, one-item-per-trip warehouse. That
    assumption is almost never true in a picking operation, which is why COI
    usually loses to plain pick frequency here - but it wins whenever slot
    scarcity, rather than distance, is what is really being allocated.

The location ranking deserves as much attention as the SKU ranking and
generally gets none. Ranking by depot distance alone leaves the level
completely undetermined, because four levels of a bay share one floor access
point. The tiebreak in :meth:`Warehouse.locations_by_depot_distance` resolves it
by ergonomic quality, which is free: it changes nothing about travel and moves
a large share of picks out of the ladder zone.

References: Heskett (1963); Hausman, Schwarz & Graves (1976) on class-based vs
full turnover storage; Petersen & Aase (2004) for the interaction with routing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .assignment import Assignment, ConstraintModel, greedy_place
from .layout import Warehouse
from .skus import Catalog

__all__ = ["velocity_slotting", "class_based_slotting", "abc_summary"]


def velocity_slotting(
    constraints: ConstraintModel,
    metric: str = "picks",
    pick_rate: np.ndarray | None = None,
    location_order: list[int] | None = None,
) -> Assignment:
    """Full-turnover slotting: SKUs by descending velocity into locations by cost.

    Greedy with a feasibility filter. The greedy is exact for the unconstrained
    problem (rearrangement inequality: pairing the largest rate with the
    smallest distance minimises the sum of products) and only approximate once
    level capability and hazmat segregation are in play - which is the honest
    reason a local-search pass afterwards still finds something.
    """
    warehouse = constraints.warehouse
    catalog = constraints.catalog
    values = (
        np.asarray(pick_rate, dtype=float)
        if pick_rate is not None and metric == "picks"
        else catalog.metric(metric, warehouse.config.slot_cube_m3)
    )

    sku_order = [int(i) for i in np.argsort(-values, kind="stable")]
    locs = location_order or warehouse.locations_by_depot_distance()
    return greedy_place(constraints, sku_order, locs)


def class_based_slotting(
    constraints: ConstraintModel,
    metric: str = "picks",
    thresholds: tuple[float, float] = (0.80, 0.95),
    pick_rate: np.ndarray | None = None,
) -> Assignment:
    """Class-based (A/B/C zone) storage: rank by class, random within class.

    This is what most warehouses actually run, because it survives contact with
    receiving: a new SKU only needs a class, not a rank, and a slot inside the
    right zone is interchangeable. Hausman, Schwarz & Graves (1976) show
    class-based capture most of the full-turnover benefit with a small number of
    classes, and reproducing that result is one of the tests.
    """
    warehouse = constraints.warehouse
    catalog = constraints.catalog
    if pick_rate is not None and metric == "picks":
        values = np.asarray(pick_rate, dtype=float)
        order = np.argsort(-values)
        cum = np.cumsum(values[order]) / values.sum()
        classes = np.empty(len(values), dtype="<U1")
        for rank, idx in enumerate(order):
            classes[idx] = "A" if cum[rank] <= thresholds[0] else (
                "B" if cum[rank] <= thresholds[1] else "C"
            )
    else:
        classes = catalog.abc_classes(metric, thresholds)

    rng = np.random.default_rng(4242)
    sku_order: list[int] = []
    for cls in ("A", "B", "C"):
        members = np.flatnonzero(classes == cls)
        rng.shuffle(members)
        sku_order.extend(int(m) for m in members)

    return greedy_place(constraints, sku_order, warehouse.locations_by_depot_distance())


@dataclass(frozen=True)
class ABCSummary:
    counts: dict[str, int]
    pick_share: dict[str, float]
    cube_share: dict[str, float]
    mean_depot_distance: dict[str, float]

    def as_rows(self) -> list[tuple[str, int, float, float, float]]:
        return [
            (
                cls,
                self.counts[cls],
                self.pick_share[cls],
                self.cube_share[cls],
                self.mean_depot_distance[cls],
            )
            for cls in ("A", "B", "C")
        ]


def abc_summary(
    warehouse: Warehouse,
    catalog: Catalog,
    assignment: Assignment,
    metric: str = "picks",
    thresholds: tuple[float, float] = (0.80, 0.95),
) -> ABCSummary:
    """Class counts, volume shares, and where each class actually ended up."""
    classes = catalog.abc_classes(metric, thresholds)
    depot_d = warehouse.depot_distance_vector()[assignment.location_of]
    counts, pick_share, cube_share, mean_d = {}, {}, {}, {}
    total_picks = catalog.picks.sum()
    total_cube = catalog.cube_velocity.sum()
    for cls in ("A", "B", "C"):
        m = classes == cls
        counts[cls] = int(m.sum())
        pick_share[cls] = float(catalog.picks[m].sum() / total_picks) if m.any() else 0.0
        cube_share[cls] = float(catalog.cube_velocity[m].sum() / total_cube) if m.any() else 0.0
        mean_d[cls] = float(depot_d[m].mean()) if m.any() else 0.0
    return ABCSummary(counts, pick_share, cube_share, mean_d)
