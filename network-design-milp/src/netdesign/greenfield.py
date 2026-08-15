"""Greenfield (center-of-gravity) heuristics, and what they are actually for.

A greenfield study answers "if geography were the only constraint, where would
the DCs go?" It is a *continuous* location problem: no candidate sites, no
fixed costs, no capacity. Every planning team runs one, and the honest reading
of the output is that it produces a **search region and an upper bound**, not a
network.

This module implements the alternating heuristic properly - assign zones to the
nearest center under great-circle distance, recompute each center as the
weighted geometric median (Weiszfeld), repeat - and then does the part that is
usually skipped: it prices the snapped-to-candidates network by solving the
flow problem with those DCs fixed open, so the heuristic can be compared with
the MILP optimum in currency rather than in kilometres.

``scipy.optimize.milp`` exposes no warm-start hook, so the heuristic's value as
a starting point is realised differently here: its cost is a valid upper bound,
added to the model as an objective cutoff row. That is a legitimate and
measurable use of a heuristic incumbent, and the benchmark reports whether it
actually helps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .geometry import haversine_matrix, weiszfeld
from .instances import Instance
from .network_flow import NetworkDesignModel, NetworkOptions, NetworkSolution

__all__ = [
    "GreenfieldResult",
    "greenfield_centers",
    "greenfield_open_set",
    "evaluate_open_set",
    "greenfield_vs_milp",
    "solve_with_cutoff",
]


@dataclass
class GreenfieldResult:
    p: int
    centers: list[tuple[float, float]]
    assignment: dict[str, int]
    weighted_km: float
    snapped_dcs: list[str] = field(default_factory=list)
    snap_distance_km: list[float] = field(default_factory=list)
    iterations: int = 0


def _cluster_center(points: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Best 1-median for one cluster: Weiszfeld, floored by the best demand point.

    Weiszfeld's update is a weighted average in the lat/lon *plane* while the
    objective is measured along great circles. At continental scale the
    approximation is good but not exact, and it can land marginally worse than
    simply siting on the largest demand point. Taking the better of the two
    costs one small matrix product and makes the heuristic monotone, which the
    test suite then asserts.
    """
    cand = weiszfeld(points, weights)
    cand_cost = float(haversine_matrix([cand], points)[0] @ weights)
    within = haversine_matrix(points, points) @ weights
    j = int(np.argmin(within))
    if float(within[j]) < cand_cost:
        return (float(points[j][0]), float(points[j][1]))
    return cand


def greenfield_centers(
    points: Sequence[tuple[float, float]],
    weights: Sequence[float],
    p: int,
    *,
    seed: int = 3,
    max_iter: int = 100,
    restarts: int = 8,
) -> tuple[list[tuple[float, float]], dict[int, int], float, int]:
    """Weighted p-median on the sphere by alternating assignment / Weiszfeld.

    Multi-start because the alternating heuristic is a local method and the
    weighted demand surface is lumpy; a single run from a random seeding is
    routinely 5-10% worse than the best of eight, which is exactly the sort of
    difference that gets mistaken for a real network insight.
    """
    pts = np.asarray(points, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = len(pts)
    if p < 1:
        raise ValueError("p must be >= 1")
    p = min(p, n)
    rng = np.random.default_rng(seed)

    best: tuple[float, list[tuple[float, float]], dict[int, int], int] | None = None
    for _ in range(restarts):
        # k-means++-style weighted seeding: spread the initial centers out
        idx = [int(rng.choice(n, p=w / w.sum()))]
        while len(idx) < p:
            d = haversine_matrix(pts, pts[idx]).min(axis=1)
            score = w * d**2
            total = score.sum()
            idx.append(int(rng.choice(n, p=score / total)) if total > 0 else int(rng.integers(n)))
        centers = [tuple(pts[i]) for i in idx]

        assignment: dict[int, int] = {}
        iterations = 0
        prev = float("inf")
        for iterations in range(1, max_iter + 1):
            dmat = haversine_matrix(pts, centers)
            labels = dmat.argmin(axis=1)
            cost = float((w * dmat[np.arange(n), labels]).sum())
            assignment = {i: int(labels[i]) for i in range(n)}
            new_centers: list[tuple[float, float]] = []
            for j in range(len(centers)):
                members = np.flatnonzero(labels == j)
                if len(members) == 0:
                    # re-seed an empty cluster onto the worst-served zone
                    worst = int(np.argmax(w * dmat[np.arange(n), labels]))
                    new_centers.append(tuple(pts[worst]))
                    continue
                new_centers.append(_cluster_center(pts[members], w[members]))
            centers = new_centers
            if prev - cost <= 1e-6 * max(1.0, prev):
                break
            prev = cost
        dmat = haversine_matrix(pts, centers)
        labels = dmat.argmin(axis=1)
        cost = float((w * dmat[np.arange(n), labels]).sum())
        assignment = {i: int(labels[i]) for i in range(n)}
        if best is None or cost < best[0]:
            best = (cost, [tuple(c) for c in centers], assignment, iterations)

    assert best is not None
    return best[1], best[2], best[0], best[3]


def greenfield_open_set(
    instance: Instance, p: int, *, seed: int = 3, restarts: int = 8
) -> GreenfieldResult:
    """Run the greenfield heuristic on demand zones and snap to DC candidates.

    Zones are weighted by shipped **kilograms**, not units: the transport cost
    a DC location is trying to minimise is weight-distance, and a network sited
    on unit volume quietly over-serves the light, high-count SKUs.
    """
    zones = instance.zones
    pts = [z.coords for z in zones]
    weights = [
        sum(
            instance.demand.get((z.id, c.id), 0.0) * c.kg_per_unit for c in instance.commodities
        )
        for z in zones
    ]
    centers, assignment, cost, iters = greenfield_centers(
        pts, weights, p, seed=seed, restarts=restarts
    )
    total_weight = float(sum(weights))

    # snap each center to its nearest not-yet-used candidate DC
    cand = instance.dcs
    cand_pts = [d.coords for d in cand]
    dmat = haversine_matrix(centers, cand_pts)
    used: set[int] = set()
    pairs: list[tuple[str, float]] = []
    for i in np.argsort(dmat.min(axis=1)):  # snap the least ambiguous centers first
        for j in np.argsort(dmat[i]):
            if int(j) not in used:
                used.add(int(j))
                pairs.append((cand[int(j)].id, float(dmat[i, j])))
                break
    pairs.sort()
    return GreenfieldResult(
        p=p,
        centers=centers,
        assignment={zones[i].id: lab for i, lab in assignment.items()},
        weighted_km=cost / total_weight if total_weight else 0.0,
        snapped_dcs=[dc for dc, _ in pairs],
        snap_distance_km=[km for _, km in pairs],
        iterations=iters,
    )


def evaluate_open_set(
    instance: Instance,
    dc_ids: Sequence[str],
    options: NetworkOptions | None = None,
    *,
    free_plants: bool = True,
    **solve_kwargs: Any,
) -> NetworkSolution:
    """Cost a specific DC set by solving the flow problem with it fixed open."""
    opt = options or NetworkOptions()
    chosen = set(dc_ids)
    fixed = {d.id: int(d.id in chosen) for d in instance.dcs}
    evaluated = NetworkOptions(**{**opt.__dict__, "fixed_open_dcs": fixed})
    if not free_plants:
        evaluated.fixed_open_plants = {p.id: 1 for p in instance.plants}
    builder = NetworkDesignModel(instance, evaluated, name="evaluate-open-set")
    sol, _ = builder.solve(**solve_kwargs)
    return sol


def greenfield_vs_milp(
    instance: Instance,
    p_values: Sequence[int],
    options: NetworkOptions | None = None,
    *,
    seed: int = 3,
    **solve_kwargs: Any,
) -> list[dict[str, Any]]:
    """Price the greenfield answer at each p and compare with the free MILP.

    The output is the table worth showing a steering committee: geography alone
    gets you most of the way, and the remaining gap is what fixed costs and
    capacity - the things a center-of-gravity study cannot see - are worth.
    """
    opt = options or NetworkOptions()
    milp = NetworkDesignModel(instance, opt, name="reference-milp")
    ref, _ = milp.solve(**solve_kwargs)
    rows: list[dict[str, Any]] = []
    for p in p_values:
        gf = greenfield_open_set(instance, p, seed=seed)
        sol = evaluate_open_set(instance, gf.snapped_dcs, opt, **solve_kwargs)
        obj = sol.objective if sol.is_optimal else None
        rows.append(
            {
                "p": p,
                "greenfield_dcs": gf.snapped_dcs,
                "weighted_km": gf.weighted_km,
                "status": sol.status,
                "cost": obj,
                "gap_pct": (
                    100.0 * (obj - ref.objective) / ref.objective
                    if obj is not None and ref.objective
                    else None
                ),
                "milp_cost": ref.objective,
                "milp_dcs": ref.open_dcs,
            }
        )
    return rows


def solve_with_cutoff(
    instance: Instance,
    options: NetworkOptions | None = None,
    *,
    upper_bound: float | None = None,
    slack: float = 1e-6,
    **solve_kwargs: Any,
) -> tuple[NetworkSolution, float | None]:
    """Solve the MILP with a heuristic incumbent imposed as an objective cutoff.

    ``objective <= upper_bound`` is a valid inequality whenever ``upper_bound``
    is the cost of a feasible solution, so it cannot cut off the optimum. It is
    the closest thing to a warm start that ``scipy.optimize.milp`` supports.
    Whether it *helps* is an empirical question - HiGHS usually finds a good
    incumbent quickly on its own - so the benchmark measures it instead of
    assuming.
    """
    opt = options or NetworkOptions()
    builder = NetworkDesignModel(instance, opt, name="cutoff")
    if upper_bound is not None:
        builder.model.add(
            builder.model.objective <= upper_bound * (1.0 + slack),
            name="incumbent_cutoff",
            tag="cutoff",
        )
    sol, _ = builder.solve(**solve_kwargs)
    return sol, upper_bound


def snap_report(instance: Instance, result: GreenfieldResult) -> str:
    """Human-readable summary of how far the snap moved each center."""
    lines = [f"greenfield p={result.p}: demand-weighted {result.weighted_km:,.0f} km to center"]
    for dc_id, km in zip(result.snapped_dcs, result.snap_distance_km):
        site = instance.site(dc_id)
        lines.append(f"  center -> {dc_id} ({site.name}) snap distance {km:,.0f} km")
    return "\n".join(lines)
