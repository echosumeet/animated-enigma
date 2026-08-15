"""The benchmark harness: build an instance, slot it eight ways, and measure.

Two rules make the numbers here worth reading.

**The slotting is fitted on one period and scored on the next.** Every slotting
policy sees only the first 60% of the order stream. Travel is measured on the
last 40%, which contains SKUs whose rank has drifted and pairs whose affinity
has decayed. Slotting studies that fit and score on the same demand overstate
the benefit by a wide margin, and the gap between in-sample and out-of-sample
here is reported so you can see how wide.

**The baseline is a real random feasible fill, averaged over seeds.** Not
"slotted by SKU id", which correlates with nothing and therefore looks worse
than random; not "worst case", which is not a thing anybody has. The number
being claimed is what you get from re-slotting an unslotted warehouse, so the
before-picture has to be an unslotted warehouse.

Everything is metres of picker travel. Converting to hours or currency needs a
walking speed and a labour rate that belong to a specific site, and putting a
made-up one in a benchmark is how a real number becomes a fake one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .affinity import AffinityGraph, affinity_slotting, cluster_quality
from .assignment import Assignment, ConstraintConfig, ConstraintModel, random_assignment
from .batching import (
    Batch,
    BatchingResult,
    CartCapacity,
    evaluate_batches,
    savings_batching,
    seed_batching,
    single_order_batches,
)
from .ergonomics import ErgonomicModel
from .layout import Warehouse, WarehouseConfig
from .localsearch import AnnealingConfig, simulated_annealing, steepest_descent
from .calibration import Calibration, calibrate_weights
from .objective import (
    ObjectiveWeights,
    SlottingObjective,
    build_affinity_pairs,
    tour_pick_weights,
)
from .orders import OrderConfig, OrderStream, generate_orders
from .routing import ROUTERS, exact_aisle_dp, held_karp
from .skus import Catalog, CatalogConfig, generate_catalog
from .velocity import class_based_slotting, velocity_slotting

__all__ = [
    "Instance",
    "build_instance",
    "SlottingRun",
    "evaluate_travel",
    "optimality_gaps",
    "run_benchmark",
    "BenchmarkReport",
]

DEFAULT_POLICIES = ("s_shape", "return", "midpoint", "largest_gap", "two_opt")


@dataclass
class Instance:
    """Everything one benchmark run needs, generated from a single seed."""

    warehouse: Warehouse
    catalog: Catalog
    stream: OrderStream
    constraints: ConstraintModel
    fit_stream: OrderStream
    score_stream: OrderStream

    @property
    def fit_rate(self) -> np.ndarray:
        """Observed pick lines per SKU in the fitting period. What slotting may use."""
        return self.fit_stream.line_counts()

    def describe(self) -> dict[str, float]:
        c = self.warehouse.config
        return {
            "aisles": c.n_aisles,
            "bays": c.n_bays,
            "levels": c.n_levels,
            "locations": self.warehouse.n_locations,
            "aisle_length_m": self.warehouse.aisle_length_m,
            "skus": self.catalog.n_skus,
            "orders": len(self.stream),
            "mean_lines": self.stream.mean_lines(),
            "single_line_share": self.stream.single_line_share(),
            "top20pct_pick_share": self.catalog.concentration(0.20),
            "top5pct_pick_share": self.catalog.concentration(0.05),
            "hazmat_skus": int((self.catalog.hazmat != "none").sum()),
        }


def build_instance(
    seed: int = 7,
    warehouse_config: WarehouseConfig | None = None,
    catalog_config: CatalogConfig | None = None,
    order_config: OrderConfig | None = None,
    constraint_config: ConstraintConfig | None = None,
    fit_fraction: float = 0.6,
) -> Instance:
    warehouse = Warehouse(warehouse_config or WarehouseConfig())
    catalog = generate_catalog(catalog_config or CatalogConfig(), seed=seed)
    stream = generate_orders(catalog, order_config or OrderConfig(), seed=seed + 100)
    constraints = ConstraintModel(warehouse, catalog, constraint_config or ConstraintConfig())
    fit, score = stream.split(fit_fraction)
    return Instance(warehouse, catalog, stream, constraints, fit, score)


# ----------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------
def evaluate_travel(
    instance: Instance,
    assignment: Assignment,
    stream: OrderStream | None = None,
    policies: Sequence[str] = DEFAULT_POLICIES,
    batches: Sequence[Batch] | None = None,
) -> dict[str, float]:
    """Total picker travel in metres under each routing policy."""
    stream = stream if stream is not None else instance.score_stream
    if batches is None:
        batches = single_order_batches(stream, assignment, instance.catalog)
    out: dict[str, float] = {}
    for policy in policies:
        router = ROUTERS[policy]
        total = 0.0
        for b in batches:
            total += router(instance.warehouse, b.location_set()).distance
        out[policy] = total
    return out


def optimality_gaps(
    instance: Instance,
    assignment: Assignment,
    stream: OrderStream | None = None,
    policies: Sequence[str] = DEFAULT_POLICIES,
    sample: int = 400,
    max_picks: int = 14,
    seed: int = 1,
) -> dict[str, dict[str, float]]:
    """Mean and p90 optimality gap of each routing heuristic against the exact DP.

    The exact reference is :func:`exact_aisle_dp`, which is polynomial and
    therefore usable on the whole sample; Held-Karp is run on the subset small
    enough for it and asserted to agree, so the gap table is backed by two
    independent exact methods rather than one.
    """
    stream = stream if stream is not None else instance.score_stream
    rng = np.random.default_rng(seed)
    orders = list(stream)
    idx = rng.choice(len(orders), size=min(sample, len(orders)), replace=False)

    per_policy: dict[str, list[float]] = {p: [] for p in policies}
    hk_checked = 0
    for i in idx:
        order = orders[int(i)]
        picks = sorted({int(assignment.location_of[s]) for s in order.lines})
        opt = exact_aisle_dp(instance.warehouse, picks)
        if not np.isfinite(opt) or opt <= 0:
            continue
        n_positions = len(
            {(instance.warehouse.locations[p].aisle, instance.warehouse.locations[p].bay) for p in picks}
        )
        if n_positions <= max_picks and hk_checked < 60:
            hk = held_karp(instance.warehouse, picks, max_picks=max_picks).distance
            if abs(hk - opt) > 1e-6:  # pragma: no cover - would be a real bug
                raise AssertionError(
                    f"aisle DP {opt:.4f} disagrees with Held-Karp {hk:.4f}"
                )
            hk_checked += 1
        for p in policies:
            d = ROUTERS[p](instance.warehouse, picks).distance
            per_policy[p].append(d / opt - 1.0)

    return {
        p: {
            "mean_gap": float(np.mean(v)) if v else 0.0,
            "p90_gap": float(np.percentile(v, 90)) if v else 0.0,
            "max_gap": float(np.max(v)) if v else 0.0,
            "n": float(len(v)),
        }
        for p, v in per_policy.items()
    }


# ----------------------------------------------------------------------
# slotting policies under test
# ----------------------------------------------------------------------
@dataclass
class SlottingRun:
    name: str
    assignment: Assignment
    seconds: float
    objective: dict[str, float] = field(default_factory=dict)
    travel: dict[str, float] = field(default_factory=dict)
    golden_zone_share: float = 0.0
    notes: str = ""


def make_objective(
    instance: Instance,
    weights: ObjectiveWeights | None = None,
    top_k: int = 12,
    normalise: str = "tour",
    ergonomics_weight: float = 1.0,
    calibration_policy: str = "two_opt",
    calibration_sample: int = 500,
) -> SlottingObjective:
    """Surrogate objective, fitted on the fitting period only.

    The linear and quadratic weights are fitted differentially against measured
    travel around a velocity-slotted reference - see
    :func:`slotting.objective.calibrate_weights` for why that has to be done at
    the margin rather than on levels. Passing ``weights`` explicitly skips the
    fit, which is what the tests do when they need a deterministic objective.
    """
    line_rate = instance.fit_rate
    anchor_rate = tour_pick_weights(instance.fit_stream)
    pairs = build_affinity_pairs(
        instance.fit_stream, instance.catalog.n_skus, top_k=top_k, normalise=normalise
    )
    calibration = None
    if weights is None:
        reference = velocity_slotting(
            instance.constraints, metric="picks", pick_rate=line_rate
        )
        calibration = calibrate_weights(
            instance.warehouse,
            instance.catalog,
            instance.fit_stream,
            instance.constraints,
            reference,
            anchor_rate,
            line_rate,
            pairs,
            policy=calibration_policy,
            sample=calibration_sample,
        )
        weights = calibration.weights(ergonomics=ergonomics_weight)
    return SlottingObjective(
        instance.warehouse,
        instance.catalog,
        pick_rate=anchor_rate,
        line_rate=line_rate,
        affinity_pairs=pairs,
        weights=weights,
        calibration=calibration,
    )


def build_slotting_runs(
    instance: Instance,
    objective: SlottingObjective,
    sa_iterations: int = 60_000,
    include_spectral: bool = True,
    baseline_seeds: Sequence[int] = (0, 1, 2),
) -> list[SlottingRun]:
    """Construct every slotting policy under test, in the order the README reports them."""
    runs: list[SlottingRun] = []
    rate = instance.fit_rate
    con = instance.constraints

    t0 = time.perf_counter()
    baselines = [random_assignment(con, seed=s) for s in baseline_seeds]
    base_costs = [objective.total(a) for a in baselines]
    # Report the median-cost draw so the baseline is not a lucky or unlucky one.
    baseline = baselines[int(np.argsort(base_costs)[len(base_costs) // 2])]
    runs.append(
        SlottingRun(
            "as-received (random feasible)",
            baseline,
            time.perf_counter() - t0,
            notes=f"median of {len(baseline_seeds)} feasible random fills",
        )
    )

    for label, metric in (
        ("ABC by pick frequency", "picks"),
        ("ABC by cube movement", "cube"),
        ("COI (Heskett 1963)", "coi"),
    ):
        t0 = time.perf_counter()
        a = velocity_slotting(con, metric=metric, pick_rate=rate if metric == "picks" else None)
        runs.append(SlottingRun(label, a, time.perf_counter() - t0))

    t0 = time.perf_counter()
    a = class_based_slotting(con, metric="picks", pick_rate=rate)
    runs.append(
        SlottingRun("class-based A/B/C zones", a, time.perf_counter() - t0,
                    notes="random within class")
    )

    t0 = time.perf_counter()
    a, graph, clusters = affinity_slotting(
        con, instance.fit_stream, method="greedy", pick_rate=rate
    )
    q = cluster_quality(graph, clusters)
    runs.append(
        SlottingRun(
            "affinity (greedy clusters)",
            a,
            time.perf_counter() - t0,
            notes=(
                f"{int(q['n_clusters'])} clusters, "
                f"{100 * q['captured_weight_share']:.0f}% of affinity weight captured"
            ),
        )
    )

    if include_spectral:
        t0 = time.perf_counter()
        a_sp, graph_sp, clusters_sp = affinity_slotting(
            con, instance.fit_stream, method="spectral", pick_rate=rate, graph=graph
        )
        q_sp = cluster_quality(graph_sp, clusters_sp)
        runs.append(
            SlottingRun(
                "affinity (spectral clusters)",
                a_sp,
                time.perf_counter() - t0,
                notes=(
                    f"{int(q_sp['n_clusters'])} clusters, "
                    f"{100 * q_sp['captured_weight_share']:.0f}% captured"
                ),
            )
        )

    velocity_start = runs[1].assignment
    t0 = time.perf_counter()
    descent = steepest_descent(objective, velocity_start, con)
    runs.append(
        SlottingRun(
            "ABC + steepest descent",
            descent.assignment,
            time.perf_counter() - t0,
            notes=f"{descent.accepted} improving swaps",
        )
    )

    t0 = time.perf_counter()
    sa = simulated_annealing(
        objective,
        velocity_start,
        con,
        AnnealingConfig(iterations=sa_iterations),
    )
    runs.append(
        SlottingRun(
            "ABC + simulated annealing",
            sa.assignment,
            time.perf_counter() - t0,
            notes=sa.summary(),
        )
    )

    affinity_start = runs[5].assignment
    t0 = time.perf_counter()
    sa2 = simulated_annealing(
        objective,
        affinity_start,
        con,
        AnnealingConfig(iterations=sa_iterations, seed=99),
    )
    runs.append(
        SlottingRun(
            "affinity + simulated annealing",
            sa2.assignment,
            time.perf_counter() - t0,
            notes=sa2.summary(),
        )
    )

    for run in runs:
        run.objective = objective.components(run.assignment)
        run.golden_zone_share = objective.golden_zone_share(run.assignment)
    return runs


# ----------------------------------------------------------------------
@dataclass
class BenchmarkReport:
    instance: dict[str, float]
    lift: dict[str, float]
    runs: list[SlottingRun]
    travel_out_of_sample: dict[str, dict[str, float]]
    travel_in_sample: dict[str, dict[str, float]]
    gaps_baseline: dict[str, dict[str, float]]
    gaps_best: dict[str, dict[str, float]]
    batching: list[BatchingResult]
    batching_baseline: list[BatchingResult]
    best_run: str
    seconds: float

    def reduction(self, run_name: str, policy: str, in_sample: bool = False) -> float:
        table = self.travel_in_sample if in_sample else self.travel_out_of_sample
        base = table[self.runs[0].name][policy]
        return 1.0 - table[run_name][policy] / base


def run_benchmark(
    instance: Instance | None = None,
    seed: int = 7,
    policies: Sequence[str] = DEFAULT_POLICIES,
    sa_iterations: int = 60_000,
    gap_sample: int = 300,
    quick: bool = False,
) -> BenchmarkReport:
    """Run everything the README reports. Returns a structured result."""
    t_start = time.perf_counter()
    if instance is None:
        if quick:
            instance = build_instance(
                seed=seed,
                warehouse_config=WarehouseConfig(n_aisles=6, n_bays=12, n_levels=3),
                catalog_config=CatalogConfig(n_skus=200, n_families=8),
                order_config=OrderConfig(n_orders=500),
            )
        else:
            instance = build_instance(seed=seed)

    objective = make_objective(instance)
    runs = build_slotting_runs(
        instance,
        objective,
        sa_iterations=2_000 if quick else sa_iterations,
        include_spectral=not quick,
    )

    travel_out: dict[str, dict[str, float]] = {}
    travel_in: dict[str, dict[str, float]] = {}
    for run in runs:
        run.travel = evaluate_travel(instance, run.assignment, instance.score_stream, policies)
        travel_out[run.name] = run.travel
        travel_in[run.name] = evaluate_travel(
            instance, run.assignment, instance.fit_stream, policies
        )

    best_policy = "largest_gap" if "largest_gap" in policies else policies[0]
    best_run = min(runs[1:], key=lambda r: r.travel[best_policy])

    gaps_baseline = optimality_gaps(
        instance, runs[0].assignment, policies=policies, sample=gap_sample
    )
    gaps_best = optimality_gaps(
        instance, best_run.assignment, policies=policies, sample=gap_sample
    )

    cap = CartCapacity()
    batching: list[BatchingResult] = []
    batching_baseline: list[BatchingResult] = []
    for label, assignment, sink in (
        ("best", best_run.assignment, batching),
        ("baseline", runs[0].assignment, batching_baseline),
    ):
        single = single_order_batches(instance.score_stream, assignment, instance.catalog)
        sink.append(evaluate_batches(instance.warehouse, single, "s_shape", "single order"))
        seeded = seed_batching(
            instance.score_stream, assignment, instance.catalog, instance.warehouse, cap
        )
        sink.append(evaluate_batches(instance.warehouse, seeded, "s_shape", "seed batching"))
        saved = savings_batching(
            instance.score_stream, assignment, instance.catalog, instance.warehouse, cap
        )
        sink.append(evaluate_batches(instance.warehouse, saved, "s_shape", "savings batching"))
        sink.append(
            evaluate_batches(instance.warehouse, saved, "largest_gap", "savings + largest gap")
        )

    return BenchmarkReport(
        instance=instance.describe(),
        lift=instance.stream.lift_summary(instance.catalog),
        runs=runs,
        travel_out_of_sample=travel_out,
        travel_in_sample=travel_in,
        gaps_baseline=gaps_baseline,
        gaps_best=gaps_best,
        batching=batching,
        batching_baseline=batching_baseline,
        best_run=best_run.name,
        seconds=time.perf_counter() - t_start,
    )
