"""Feasibility diagnostics for network design models.

"Model status: infeasible" is the least useful sentence in this field. A
planner asked "can we serve the new region from the existing network?" and got
back a status code. The answer they need is *what has to change, and by how
much*.

Two layers here:

1. **A structural ledger.** Before solving, compare supply, plant capacity, DC
   capacity and demand echelon by echelon, per commodity. Most infeasibilities
   are a single number in that ledger and are found in milliseconds.
2. **Elastic relaxation.** When the ledger balances but the model is still
   infeasible, allow the constraints in selected groups to be violated at unit
   cost and minimise total violation. The rows that come back non-zero are a
   near-minimal explanation - not a certified IIS, but the same practical
   payload, and it works with any LP/MILP backend.

The second layer is also how you answer the more valuable version of the
question: not "is it feasible" but "what is the cheapest constraint to buy your
way out of".
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from .instances import Instance
from .modeling import Model
from .network_flow import NetworkDesignModel, NetworkOptions

__all__ = ["Violation", "Diagnosis", "capacity_ledger", "diagnose", "diagnose_model"]

#: Constraint groups worth relaxing. Balance and linking rows are excluded on
#: purpose: relaxing conservation of flow always "fixes" the model and explains
#: nothing, which is the classic way an elastic analysis wastes an afternoon.
RELAXABLE_PREFIXES = (
    "supply",
    "plant_capacity",
    "dc_throughput",
    "dc_min_volume",
    "demand",
    "dual_source",
    "single_source",
    "count",
    "fix",
)


@dataclass(frozen=True)
class Violation:
    name: str
    group: str
    amount: float
    direction: str  # "shortfall" if a >= row had to give, "excess" if a <= row did


@dataclass
class Diagnosis:
    feasible: bool
    ledger: dict[str, dict[str, float]] = field(default_factory=dict)
    structural_problems: list[str] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    total_violation: float = 0.0
    by_group: dict[str, float] = field(default_factory=dict)
    status: str = ""

    def summary(self, max_rows: int = 12) -> str:
        lines = []
        if self.feasible:
            lines.append("model is feasible")
        else:
            lines.append(f"model is INFEASIBLE (solver status: {self.status})")
        if self.structural_problems:
            lines.append("structural problems found before solving:")
            lines += [f"  - {p}" for p in self.structural_problems]
        if self.ledger:
            lines.append("")
            lines.append(
                f"{'commodity':<10}{'supply':>14}{'plant cap':>14}{'dc cap':>14}{'demand':>14}"
            )
            for c, row in self.ledger.items():
                lines.append(
                    f"{c:<10}{row['supply']:>14,.0f}{row['plant_capacity']:>14,.0f}"
                    f"{row['dc_capacity']:>14,.0f}{row['demand']:>14,.0f}"
                )
        if self.violations:
            lines.append("")
            lines.append(f"minimum total violation: {self.total_violation:,.1f}")
            lines.append("by constraint group:")
            for g, v in sorted(self.by_group.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {g:<20}{v:>14,.1f}")
            lines.append("largest individual violations:")
            for v in self.violations[:max_rows]:
                lines.append(f"  {v.name:<44}{v.direction:>10}{v.amount:>14,.1f}")
        return "\n".join(lines)


def capacity_ledger(instance: Instance) -> dict[str, dict[str, float]]:
    """Per-commodity supply / plant capacity / DC capacity / demand."""
    ledger: dict[str, dict[str, float]] = {}
    for c in instance.commodities:
        ledger[c.id] = {
            "supply": float(sum(v for (_, k), v in instance.supply.items() if k == c.id)),
            # capacity is shared across commodities, so it is reported pro-rata
            # by this commodity's share of total demand
            "plant_capacity": float(sum(p.capacity for p in instance.plants))
            * _share(instance, c.id),
            "dc_capacity": float(sum(d.capacity for d in instance.dcs)) * _share(instance, c.id),
            "demand": instance.total_demand(c.id),
        }
    ledger["TOTAL"] = {
        "supply": float(sum(instance.supply.values())),
        "plant_capacity": float(sum(p.capacity for p in instance.plants)),
        "dc_capacity": float(sum(d.capacity for d in instance.dcs)),
        "demand": instance.total_demand(),
    }
    return ledger


def _share(instance: Instance, commodity_id: str) -> float:
    total = instance.total_demand()
    return instance.total_demand(commodity_id) / total if total > 0 else 0.0


def structural_problems(instance: Instance, options: NetworkOptions | None = None) -> list[str]:
    problems = list(instance.validate())
    ledger = capacity_ledger(instance)["TOTAL"]
    if ledger["dc_capacity"] < ledger["demand"]:
        problems.append(
            f"total DC throughput capacity {ledger['dc_capacity']:,.0f} "
            f"< total demand {ledger['demand']:,.0f}"
        )
    opt = options or NetworkOptions()
    if opt.max_open_dcs is not None:
        best = sorted((d.capacity for d in instance.dcs), reverse=True)[: opt.max_open_dcs]
        if sum(best) < instance.total_demand():
            problems.append(
                f"max_open_dcs={opt.max_open_dcs} caps reachable throughput at {sum(best):,.0f} "
                f"< demand {instance.total_demand():,.0f}"
            )
    if opt.min_second_source_share > 0:
        # every zone needs at least two DCs with a lane to it
        for z in instance.zone_ids:
            n = len({ln.origin for ln in instance.lanes if ln.dest == z})
            if n < 2:
                problems.append(
                    f"zone {z} has {n} inbound DC lane(s) but dual sourcing requires 2"
                )
    if opt.single_source:
        for d in instance.dcs:
            if d.min_volume > 0 and d.min_volume > max(
                (instance.zone_demand(z) for z in instance.zone_ids), default=0.0
            ) * len(instance.zone_ids):
                problems.append(f"DC {d.id} min_volume unreachable under single sourcing")
    return problems


def diagnose_model(
    model: Model,
    *,
    relaxable: Iterable[str] | None = None,
    penalty: float = 1.0,
    tol: float = 1e-6,
    **solve_kwargs: Any,
) -> Diagnosis:
    """Solve, and if infeasible, run the elastic relaxation and report."""
    raw = model.solve(**solve_kwargs)
    diag = Diagnosis(feasible=raw.is_optimal, status=raw.status)
    if raw.is_optimal:
        return diag

    prefixes = tuple(relaxable) if relaxable is not None else RELAXABLE_PREFIXES
    tags = [t for t in model.tags() if t.split("|")[0] in prefixes]
    elastic, rows, slacks = model.elastic_copy(tags, penalty=penalty)
    relaxed = elastic.solve(**solve_kwargs)
    if not relaxed.is_optimal:
        diag.structural_problems.append(
            "even the elastic relaxation did not solve; the model is malformed "
            "rather than merely over-constrained"
        )
        return diag

    values = relaxed.values(slacks, nonzero=True, tol=tol)
    by_group: dict[str, float] = defaultdict(float)
    violations: list[Violation] = []
    for (row_idx, side), amount in values.items():
        row = rows[row_idx]
        group = row.tag.split("|")[0] or "untagged"
        by_group[group] += amount
        violations.append(
            Violation(
                name=row.name or f"row[{row_idx}]",
                group=group,
                amount=float(amount),
                direction="shortfall" if side == "under" else "excess",
            )
        )
    violations.sort(key=lambda v: -v.amount)
    diag.violations = violations
    diag.by_group = dict(by_group)
    diag.total_violation = float(sum(v.amount for v in violations))
    return diag


def diagnose(
    instance: Instance,
    options: NetworkOptions | None = None,
    **solve_kwargs: Any,
) -> Diagnosis:
    """Full diagnosis of a network design instance: ledger, then elastic solve."""
    opt = options or NetworkOptions()
    builder = NetworkDesignModel(instance, opt, name="diagnostic")
    diag = diagnose_model(builder.model, **solve_kwargs)
    diag.ledger = capacity_ledger(instance)
    diag.structural_problems = structural_problems(instance, opt) + diag.structural_problems
    return diag
