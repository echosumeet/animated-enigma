"""Pairwise-swap local search with simulated annealing over the surrogate objective.

Construction heuristics get the structure right and the details wrong. Velocity
slotting puts fast movers near the dock but cannot see affinity; affinity
slotting groups co-ordered SKUs but treats every slot inside a cluster region as
interchangeable; both are greedy against hard constraints that block their first
choice and never revisit the decision. A swap-based improver fixes all three
because it optimises the objective that was actually written down, whatever went
into the starting point.

Why simulated annealing rather than steepest descent: the slotting landscape is
a QAP with a strong linear term, and pure descent from a velocity start reaches
a local optimum within a few hundred moves and stops - it cannot pay the
short-term cost of moving a fast SKU further out to bring a cluster together.
The temperature schedule is the mechanism for buying that trade. Kirkpatrick,
Gelatt & Vecchi (1983); Burkard et al. (2009) on QAP metaheuristics.

Three implementation choices carry the run time:

* **O(neighbours) deltas.** Never re-evaluate the objective; ask it for the
  change. That is the difference between 300 moves a second and 30,000.
* **Distance-biased proposals.** A uniformly random pair of SKUs is almost
  always a pair of C movers in far slots, and swapping them changes nothing.
  A configurable share of proposals is drawn instead from a fast SKU and a
  near-to-it location, which is where the improvements live.
* **Feasibility inside the proposal.** A move that breaks a level or hazmat
  constraint is rejected before it is scored, so the incumbent is feasible at
  every iteration and the search can be stopped at any time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .assignment import Assignment, ConstraintModel
from .objective import SlottingObjective
from .travel import TravelObjective

__all__ = ["AnnealingConfig", "AnnealingResult", "simulated_annealing", "steepest_descent"]


@dataclass(frozen=True)
class AnnealingConfig:
    iterations: int = 60_000
    t_start: float | None = None  # None -> calibrated from sampled move sizes
    t_end_ratio: float = 1e-3
    biased_share: float = 0.7  # share of proposals drawn from the hot end
    relocate_share: float = 0.15  # share of proposals that move to an empty slot
    seed: int = 17
    log_every: int = 2_000


@dataclass
class AnnealingResult:
    assignment: Assignment
    best_objective: float
    start_objective: float
    accepted: int
    improved: int
    proposed: int
    trace: list[tuple[int, float, float]] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        if self.start_objective == 0:
            return 0.0
        return 1.0 - self.best_objective / self.start_objective

    def summary(self) -> str:
        return (
            f"SA {self.proposed:,} proposals, {self.accepted:,} accepted "
            f"({self.improved:,} improving); objective {self.start_objective:,.0f} -> "
            f"{self.best_objective:,.0f} ({100 * self.improvement:.2f}% better)"
        )


def _calibrate_temperature(
    objective: SlottingObjective | TravelObjective,
    assignment: Assignment,
    rng: np.random.Generator,
    n: int = 400,
) -> float:
    """Start hot enough to accept most uphill moves, no hotter.

    Setting ``t_start`` by hand is how annealing gets a reputation for being
    fiddly. Sampling the move-size distribution and setting the temperature so
    that a typical worsening move is accepted with probability ~0.5 makes the
    schedule scale-free, which matters here because the objective is in metres
    and a different warehouse has a different order of magnitude.
    """
    n_skus = assignment.n_skus
    deltas = []
    for _ in range(n):
        a, b = int(rng.integers(n_skus)), int(rng.integers(n_skus))
        if a == b:
            continue
        d = objective.swap_delta(assignment, a, b)
        if d > 0:
            deltas.append(d)
    if not deltas:
        return 1.0
    return float(np.median(deltas) / math.log(2.0))


def _feasible_swap(constraints: ConstraintModel, assignment: Assignment, a: int, b: int) -> bool:
    la = int(assignment.location_of[a])
    lb = int(assignment.location_of[b])
    if not (constraints.level_ok(a, lb) and constraints.level_ok(b, la)):
        return False
    if not constraints.config.enforce_hazmat:
        return True
    assignment.swap(a, b)
    ok = constraints.hazmat_ok_for(a, lb, assignment) and constraints.hazmat_ok_for(
        b, la, assignment
    )
    assignment.swap(a, b)
    return ok


def simulated_annealing(
    objective: SlottingObjective | TravelObjective,
    assignment: Assignment,
    constraints: ConstraintModel,
    config: AnnealingConfig | None = None,
) -> AnnealingResult:
    """Anneal over swaps and relocations. Returns the best assignment seen."""
    cfg = config or AnnealingConfig()
    rng = np.random.default_rng(cfg.seed)
    current = assignment.copy()
    n_skus = current.n_skus

    start_obj = objective.total(current)
    cur_obj = start_obj
    best_obj = start_obj
    best = current.copy()

    t0 = cfg.t_start if cfg.t_start is not None else _calibrate_temperature(
        objective, current, rng
    )
    t0 = max(t0, 1e-9)
    t1 = t0 * cfg.t_end_ratio
    decay = (t1 / t0) ** (1.0 / max(cfg.iterations, 1))

    # Proposal biasing: sample the first SKU proportional to pick rate, so hot
    # SKUs are considered far more often than the long tail they sit above.
    rate = np.maximum(objective.pick_rate, 1e-9)
    hot_cdf = np.cumsum(rate / rate.sum())
    hot_cdf[-1] = 1.0

    empty = list(current.empty_locations())

    accepted = improved = proposed = 0
    trace: list[tuple[int, float, float]] = []
    temperature = t0

    for it in range(cfg.iterations):
        temperature *= decay
        proposed += 1

        use_relocate = bool(empty) and rng.random() < cfg.relocate_share
        if use_relocate:
            sku = (
                int(np.searchsorted(hot_cdf, rng.random()))
                if rng.random() < cfg.biased_share
                else int(rng.integers(n_skus))
            )
            loc = int(empty[int(rng.integers(len(empty)))])
            if not constraints.level_ok(sku, loc):
                continue
            if not constraints.hazmat_ok_for(sku, loc, current):
                continue
            delta = objective.move_delta(current, sku, loc)
            if delta < 0 or rng.random() < math.exp(-min(delta / temperature, 700.0)):
                old = int(current.location_of[sku])
                current.relocate(sku, loc)
                objective.commit()
                empty[empty.index(loc)] = old
                cur_obj += delta
                accepted += 1
                if delta < 0:
                    improved += 1
        else:
            if rng.random() < cfg.biased_share:
                a = int(np.searchsorted(hot_cdf, rng.random()))
            else:
                a = int(rng.integers(n_skus))
            b = int(rng.integers(n_skus))
            if a == b:
                continue
            if not _feasible_swap(constraints, current, a, b):
                continue
            delta = objective.swap_delta(current, a, b)
            if delta < 0 or rng.random() < math.exp(-min(delta / temperature, 700.0)):
                current.swap(a, b)
                objective.commit()
                cur_obj += delta
                accepted += 1
                if delta < 0:
                    improved += 1

        if cur_obj < best_obj - 1e-9:
            best_obj = cur_obj
            best = current.copy()
        if cfg.log_every and it % cfg.log_every == 0:
            trace.append((it, cur_obj, temperature))

    # Guard against accumulated floating point drift in the incremental score.
    # With the exact travel objective this also re-syncs its per-order cache
    # onto the returned assignment, which is a different object from the
    # incumbent the search was mutating.
    recomputed = objective.total(best)
    if abs(recomputed - best_obj) > 1e-6 * max(1.0, abs(recomputed)):
        best_obj = recomputed

    trace.append((cfg.iterations, best_obj, temperature))
    return AnnealingResult(best, best_obj, start_obj, accepted, improved, proposed, trace)


def steepest_descent(
    objective: SlottingObjective | TravelObjective,
    assignment: Assignment,
    constraints: ConstraintModel,
    max_rounds: int = 6,
    candidates_per_sku: int = 40,
    seed: int = 3,
) -> AnnealingResult:
    """First-improvement descent over swaps. The baseline annealing has to beat.

    Included so the benchmark can report what the temperature schedule is
    actually buying. If annealing does not beat descent by a clear margin on
    your instance, the honest conclusion is to ship descent - it is faster,
    deterministic, and much easier to explain to the people who have to trust
    the output.
    """
    rng = np.random.default_rng(seed)
    current = assignment.copy()
    start_obj = objective.total(current)
    cur_obj = start_obj
    accepted = proposed = 0

    order = np.argsort(-objective.pick_rate)
    for _ in range(max_rounds):
        moved = False
        for a in order:
            a = int(a)
            partners = rng.integers(0, current.n_skus, size=candidates_per_sku)
            for b in partners:
                b = int(b)
                if a == b:
                    continue
                proposed += 1
                if not _feasible_swap(constraints, current, a, b):
                    continue
                delta = objective.swap_delta(current, a, b)
                if delta < -1e-9:
                    current.swap(a, b)
                    objective.commit()
                    cur_obj += delta
                    accepted += 1
                    moved = True
                    break
        if not moved:
            break

    cur_obj = objective.total(current)
    return AnnealingResult(current, cur_obj, start_obj, accepted, accepted, proposed, [])
