"""Fitting the surrogate objective's weights to measured travel.

This module exists because of a mistake that is easy to make and expensive to
miss. The surrogate objective is a weighted sum of a depot-distance term and an
inter-pick-distance term; those weights have to come from somewhere; and every
obvious way of choosing them is wrong in a way that only shows up when you
measure the thing you were supposedly optimising.

Three attempts, in the order they failed here.

**1. Hand-set weights.** Weight the quadratic term by raw co-occurrence counts,
which looks natural. The term then scales like a complete graph rather than
like a path, overstating a ten-line order's structure by a factor of five.
Simulated annealing improved that objective by 5% while measured travel got 3%
worse.

**2. Regress tour length on the features across reference layouts.** Fits the
average level, not the margin. On this instance it returns a linear coefficient
around 1.0, and the search - which only ever sees margins - then under-prices
displacement and buys ergonomic improvement with travel it thinks is free.

**3. Regress the *change* in travel against the *change* in features, using
random swaps as the perturbation.** Closer, and the R2 looks superb (0.99), but
the coefficient is still wrong by a factor of three. The reason is that the
marginal cost of displacement saturates: shifting a SKU 30 m costs far less
than 30 times what shifting it 1 m costs, because the picker was going to walk
that aisle anyway. Random swaps produce huge displacements; local search
produces small ones. Fitting on the wrong displacement scale measures the wrong
slope.

**What is done here.** The perturbations are generated *by the search itself*.
Short annealing runs with a grid of provisional weight ratios produce move sets
drawn from the same distribution the real search will draw from, and at the
same displacement scale; the weight grid varies which term each run attacks, so
the two features are not collinear in the design. A handful of random-swap
perturbations are added for range. Then the deltas are regressed through the
origin by non-negative least squares, and the whole thing is repeated once with
the fitted weights so the design ends up centred on the region the final search
actually explores.

The reported R2 is on the deltas and is the licence to optimise the surrogate.
If it is low, a swap that improves the objective does not reliably improve
metres, and the honest response is to fix the objective rather than to run the
search for longer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .assignment import Assignment, ConstraintModel
from .ergonomics import ErgonomicModel
from .layout import Warehouse
from .objective import ObjectiveWeights, SlottingObjective
from .orders import OrderStream
from .skus import Catalog

__all__ = ["Calibration", "calibrate_weights"]


@dataclass(frozen=True)
class Calibration:
    """Fitted surrogate weights plus the evidence that they mean something."""

    velocity: float
    affinity: float
    r_squared: float
    n_observations: int
    sample_orders: int
    policy: str

    def weights(self, ergonomics: float = 1.0, grouping: float = 0.0) -> ObjectiveWeights:
        return ObjectiveWeights(
            velocity=self.velocity,
            affinity=self.affinity,
            ergonomics=ergonomics,
            grouping=grouping,
            depot_trip_factor=1.0,
            lines_per_tour=1.0,
        )

    def summary(self) -> str:
        return (
            f"surrogate calibrated on {self.n_observations} move sets over "
            f"{self.sample_orders} {self.policy} tours: velocity {self.velocity:.3f}, "
            f"affinity {self.affinity:.3f}, marginal R2 {self.r_squared:.3f}"
        )


class _SampleModel:
    """Feature and travel measurement over a fixed sample of orders."""

    def __init__(
        self,
        warehouse: Warehouse,
        sampled: Sequence,
        affinity_pairs: Sequence[tuple[int, int, float]],
        n_skus: int,
        policy: str,
    ) -> None:
        from .routing import ROUTERS

        self.warehouse = warehouse
        self.sampled = list(sampled)
        self.router = ROUTERS[policy]

        kept = {(i, j) for i, j, _ in affinity_pairs}
        w_lin = np.zeros(n_skus, dtype=float)
        pair_weight: dict[tuple[int, int], float] = {}
        for order in self.sampled:
            lines = sorted(set(order.lines))
            n = len(lines)
            if n == 0:
                continue
            share = 2.0 / n
            for sku in lines:
                w_lin[sku] += share
            for a in range(n):
                for b in range(a + 1, n):
                    key = (lines[a], lines[b])
                    if key in kept:
                        pair_weight[key] = pair_weight.get(key, 0.0) + share
        self.w_lin = w_lin
        self.pair_i = np.asarray([k[0] for k in pair_weight] or [0], dtype=np.int64)
        self.pair_j = np.asarray([k[1] for k in pair_weight] or [0], dtype=np.int64)
        self.pair_w = np.asarray(list(pair_weight.values()) or [0.0], dtype=float)

        self.depot_d = warehouse.depot_distance_vector()
        self.access = np.asarray(
            [warehouse.access_index(i) for i in range(warehouse.n_locations)], dtype=np.int64
        )
        self.access_matrix = warehouse.access_distance_matrix()
        self.order_lines = [sorted(set(o.lines)) for o in self.sampled]

    def features(self, assignment: Assignment) -> tuple[float, float]:
        loc = assignment.location_of
        f1 = float(np.dot(self.w_lin, self.depot_d[loc]))
        acc = self.access[loc]
        f2 = float(
            np.dot(self.pair_w, self.access_matrix[acc[self.pair_i], acc[self.pair_j]])
        )
        return f1, f2

    def travel(self, assignment: Assignment) -> float:
        loc = assignment.location_of
        total = 0.0
        for lines in self.order_lines:
            total += self.router(self.warehouse, sorted({int(loc[s]) for s in lines})).distance
        return total


def _random_perturbation(
    reference: Assignment,
    constraints: ConstraintModel,
    n_swaps: int,
    rng: np.random.Generator,
) -> Assignment:
    candidate = reference.copy()
    n_skus = candidate.n_skus
    for _ in range(n_swaps):
        a, b = int(rng.integers(n_skus)), int(rng.integers(n_skus))
        if a == b:
            continue
        la, lb = int(candidate.location_of[a]), int(candidate.location_of[b])
        if not (constraints.level_ok(a, lb) and constraints.level_ok(b, la)):
            continue
        candidate.swap(a, b)
        if not (
            constraints.hazmat_ok_for(a, lb, candidate)
            and constraints.hazmat_ok_for(b, la, candidate)
        ):
            candidate.swap(a, b)
    return candidate


def _biased_perturbation(
    reference: Assignment,
    constraints: ConstraintModel,
    warehouse: Warehouse,
    anchor_rate: np.ndarray,
    n_swaps: int,
    max_shift_m: float,
    rng: np.random.Generator,
) -> Assignment:
    """Small displacements of busy SKUs - the regime local search actually explores.

    The marginal metre cost of displacement saturates with distance: a SKU
    pushed 30 m down an aisle the picker was walking anyway costs almost
    nothing extra, while the same SKU pushed 3 m out of the first bay costs
    close to a full round trip per line. A design built only from large random
    swaps measures the saturated slope and under-prices displacement by a
    factor of three. This sampler generates the other end of the curve.
    """
    candidate = reference.copy()
    depot_d = warehouse.depot_distance_vector()
    rate = np.maximum(anchor_rate, 1e-9)
    cdf = np.cumsum(rate / rate.sum())
    cdf[-1] = 1.0
    n_skus = candidate.n_skus
    for _ in range(n_swaps):
        a = int(np.searchsorted(cdf, rng.random()))
        la = int(candidate.location_of[a])
        target = depot_d[la] + rng.uniform(-max_shift_m, max_shift_m)
        for _ in range(12):
            b = int(rng.integers(n_skus))
            lb = int(candidate.location_of[b])
            if a == b or abs(depot_d[lb] - target) > max_shift_m:
                continue
            if not (constraints.level_ok(a, lb) and constraints.level_ok(b, la)):
                continue
            candidate.swap(a, b)
            if not (
                constraints.hazmat_ok_for(a, lb, candidate)
                and constraints.hazmat_ok_for(b, la, candidate)
            ):
                candidate.swap(a, b)
            break
    return candidate


def _iso_distance_perturbation(
    reference: Assignment,
    constraints: ConstraintModel,
    warehouse: Warehouse,
    affinity_degree: np.ndarray,
    n_swaps: int,
    tolerance_m: float,
    rng: np.random.Generator,
) -> Assignment:
    """Swaps that hold depot distance fixed and move only the affinity feature.

    Without this the design is rank-deficient in practice. Every other
    perturbation moves both features together - push a SKU further from the dock
    and you also push it away from whatever it is co-ordered with - so
    non-negative least squares can put the entire coefficient on the linear term
    and set the quadratic one to zero. It then costs nothing, in the model, to
    scatter co-ordered SKUs across the building, and the search does exactly
    that: a run calibrated on a collinear design moved the affinity feature up
    by 700 units and paid 1,285 real metres for it.

    Two locations equidistant from the dock but in different aisles are far
    apart from each other. Swapping between them changes the quadratic term and
    leaves the linear term almost untouched, which is what identifies the
    coefficient.
    """
    candidate = reference.copy()
    depot_d = warehouse.depot_distance_vector()
    order = np.argsort(depot_d, kind="stable")
    sorted_d = depot_d[order]
    deg = np.maximum(affinity_degree, 1e-9)
    cdf = np.cumsum(deg / deg.sum())
    cdf[-1] = 1.0
    aisle_of = np.asarray([loc.aisle for loc in warehouse.locations])

    for _ in range(n_swaps):
        a = int(np.searchsorted(cdf, rng.random()))
        la = int(candidate.location_of[a])
        lo = int(np.searchsorted(sorted_d, depot_d[la] - tolerance_m, side="left"))
        hi = int(np.searchsorted(sorted_d, depot_d[la] + tolerance_m, side="right"))
        if hi - lo < 2:
            continue
        # Take the *most* distant admissible partner in the band, not a random
        # one. A random partner moves the quadratic feature by a few units and
        # the regression cannot see it above the noise; the extreme one moves it
        # by hundreds, which is what makes the coefficient identifiable.
        best_lb, best_sep = -1, -1
        for _ in range(30):
            lb = int(order[rng.integers(lo, hi)])
            b = int(candidate.sku_at[lb])
            if b < 0 or b == a:
                continue
            sep = abs(int(aisle_of[lb]) - int(aisle_of[la]))
            if sep <= best_sep:
                continue
            if not (constraints.level_ok(a, lb) and constraints.level_ok(b, la)):
                continue
            best_lb, best_sep = lb, sep
        if best_lb < 0 or best_sep == 0:
            continue
        b = int(candidate.sku_at[best_lb])
        candidate.swap(a, b)
        if not (
            constraints.hazmat_ok_for(a, best_lb, candidate)
            and constraints.hazmat_ok_for(b, la, candidate)
        ):
            candidate.swap(a, b)
    return candidate


# Provisional (velocity, affinity, ergonomics) ratios used to generate the
# design. Each one produces a search that attacks a different mix of the two
# features, which is what makes them separately identifiable in the regression.
_DESIGN_GRID: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),
    (1.0, 0.0, 1.0),
    (1.0, 2.0, 0.0),
    (1.0, 8.0, 0.0),
    (0.25, 1.0, 0.0),
    (0.05, 1.0, 1.0),
    (1.0, 1.0, 3.0),
)


def calibrate_weights(
    warehouse: Warehouse,
    catalog: Catalog,
    stream: OrderStream,
    constraints: ConstraintModel,
    reference: Assignment,
    anchor_rate: np.ndarray,
    line_rate: np.ndarray,
    affinity_pairs: list[tuple[int, int, float]],
    ergonomics: ErgonomicModel | None = None,
    policy: str = "two_opt",
    sample: int = 500,
    sa_iterations: int = 6_000,
    random_swap_sizes: Sequence[int] = (10, 40, 160),
    biased_designs: Sequence[tuple[int, float]] = (
        (25, 2.0),
        (60, 2.0),
        (25, 6.0),
        (80, 6.0),
        (40, 15.0),
        (120, 15.0),
        (60, 40.0),
    ),
    iso_distance_designs: Sequence[tuple[int, float]] = (
        (60, 1.0),
        (200, 1.0),
        (500, 1.0),
        (900, 1.0),
        (300, 2.5),
    ),
    rounds: int = 2,
    seed: int = 2,
) -> Calibration:
    """Fit ``(velocity, affinity)`` so the surrogate predicts marginal metres."""
    from .localsearch import AnnealingConfig, simulated_annealing

    rng = np.random.default_rng(seed)
    orders = list(stream)
    idx = rng.choice(len(orders), size=min(sample, len(orders)), replace=False)
    model = _SampleModel(
        warehouse, [orders[int(k)] for k in idx], affinity_pairs, catalog.n_skus, policy
    )

    base_f = model.features(reference)
    base_y = model.travel(reference)

    rows: list[tuple[float, float]] = []
    targets: list[float] = []

    def observe(candidate: Assignment) -> None:
        f = model.features(candidate)
        d1, d2 = f[0] - base_f[0], f[1] - base_f[1]
        if abs(d1) < 1e-9 and abs(d2) < 1e-9:
            return
        rows.append((d1, d2))
        targets.append(model.travel(candidate) - base_y)

    for n_swaps in random_swap_sizes:
        observe(_random_perturbation(reference, constraints, n_swaps, rng))
    for n_swaps, band in biased_designs:
        observe(
            _biased_perturbation(
                reference, constraints, warehouse, anchor_rate, n_swaps, band, rng
            )
        )
    degree = np.zeros(catalog.n_skus, dtype=float)
    for i, j, w in affinity_pairs:
        degree[i] += w
        degree[j] += w
    if degree.sum() > 0:
        for n_swaps, tol in iso_distance_designs:
            observe(
                _iso_distance_perturbation(
                    reference, constraints, warehouse, degree, n_swaps, tol, rng
                )
            )

    def run_grid(grid: Sequence[tuple[float, float, float]], tag: int) -> None:
        for k, (v, a, e) in enumerate(grid):
            obj = SlottingObjective(
                warehouse,
                catalog,
                pick_rate=anchor_rate,
                line_rate=line_rate,
                affinity_pairs=affinity_pairs,
                weights=ObjectiveWeights(
                    velocity=v,
                    affinity=a,
                    ergonomics=e,
                    depot_trip_factor=1.0,
                    lines_per_tour=1.0,
                ),
                ergonomics=ergonomics,
            )
            result = simulated_annealing(
                obj,
                reference,
                constraints,
                AnnealingConfig(iterations=sa_iterations, seed=1000 * tag + k, log_every=0),
            )
            observe(result.assignment)

    run_grid(_DESIGN_GRID, 0)

    from scipy.optimize import nnls

    coef = np.zeros(2)
    r2 = 0.0
    for r in range(rounds):
        x = np.asarray(rows, dtype=float)
        y = np.asarray(targets, dtype=float)
        coef, _ = nnls(x, y)
        pred = x @ coef
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float((y**2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if r + 1 < rounds:
            # Re-centre the design on the region the fitted weights explore.
            v, a = float(coef[0]), float(coef[1])
            spread = max(a, 0.25 * v)
            run_grid(((v, a, 1.0), (v, a, 0.0), (v, 4.0 * spread, 1.0)), r + 1)

    return Calibration(
        velocity=float(coef[0]),
        affinity=float(coef[1]),
        r_squared=r2,
        n_observations=len(rows),
        sample_orders=len(model.sampled),
        policy=policy,
    )
