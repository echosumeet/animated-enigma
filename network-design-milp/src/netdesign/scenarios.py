"""Scenario running and sensitivity analysis.

A single optimal network is a fragile artefact. It is optimal for one demand
vector, one fuel price and one set of fixed costs, all of which are estimates,
and the difference between the best and second-best network is routinely
smaller than the error in those estimates.

So the deliverable from a network study is not the optimal set - it is the
partition of candidate sites into three groups:

* **core** - open in essentially every run; commit to these;
* **swing** - open in some; these are the real decision, and they are where the
  qualitative arguments (labour market, tax, customer perception) belong;
* **out** - never open; stop discussing them.

:func:`stability_profile` computes exactly that partition from a sweep, and it
is usually the only output of a network study that survives contact with the
business.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .instances import Instance
from .network_flow import NetworkDesignModel, NetworkOptions

__all__ = [
    "ScenarioRun",
    "run_scenarios",
    "sensitivity_sweep",
    "stability_profile",
    "sweep_table",
]


@dataclass
class ScenarioRun:
    label: str
    status: str
    objective: float | None
    open_dcs: list[str]
    open_plants: list[str]
    demand_weighted_km: float
    unmet_units: float
    runtime: float
    costs: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def n_dcs(self) -> int:
        return len(self.open_dcs)


def _run(
    label: str,
    instance: Instance,
    options: NetworkOptions,
    params: dict[str, Any],
    **solve_kwargs: Any,
) -> ScenarioRun:
    builder = NetworkDesignModel(instance, options, name=label)
    sol, _ = builder.solve(**solve_kwargs)
    return ScenarioRun(
        label=label,
        status=sol.status,
        objective=sol.objective,
        open_dcs=list(sol.open_dcs),
        open_plants=list(sol.open_plants),
        demand_weighted_km=sol.demand_weighted_km,
        unmet_units=sol.unmet_units,
        runtime=sol.runtime,
        costs=dict(sol.costs),
        params=params,
    )


def run_scenarios(
    instance: Instance,
    scenarios: dict[str, Instance] | dict[str, dict[tuple[str, str], float]],
    options: NetworkOptions | None = None,
    **solve_kwargs: Any,
) -> list[ScenarioRun]:
    """Solve a named set of scenarios (instances or demand vectors)."""
    opt = options or NetworkOptions()
    runs: list[ScenarioRun] = []
    for label, payload in scenarios.items():
        inst = payload if isinstance(payload, Instance) else instance.with_demand(payload)
        runs.append(_run(label, inst, opt, {"scenario": label}, **solve_kwargs))
    return runs


def sensitivity_sweep(
    instance: Instance,
    demand_multipliers: Sequence[float] = (0.8, 0.9, 1.0, 1.1, 1.25),
    cost_multipliers: Sequence[float] = (0.7, 1.0, 1.4),
    options: NetworkOptions | None = None,
    **solve_kwargs: Any,
) -> list[ScenarioRun]:
    """Grid sweep over total demand and transport rates.

    These two are swept together deliberately: they move the optimum in
    *opposite* directions. More demand justifies more DCs (fixed cost is spread
    over more volume); cheaper transport justifies fewer (centralise and ship).
    A study that sweeps one at a time will report a stable network that is not.
    """
    opt = options or NetworkOptions()
    runs: list[ScenarioRun] = []
    for dm in demand_multipliers:
        scaled = instance.with_demand(instance.scaled_demand(float(dm)))
        for cm in cost_multipliers:
            inst = scaled.with_lane_cost_multiplier(float(cm))
            runs.append(
                _run(
                    f"d{dm:g}xc{cm:g}",
                    inst,
                    opt,
                    {"demand_multiplier": float(dm), "cost_multiplier": float(cm)},
                    **solve_kwargs,
                )
            )
    return runs


def stability_profile(
    runs: Iterable[ScenarioRun], *, core_threshold: float = 0.9, out_threshold: float = 0.05
) -> dict[str, Any]:
    """Partition candidate DCs into core / swing / out by open frequency."""
    solved = [r for r in runs if r.objective is not None]
    if not solved:
        return {"core": [], "swing": [], "out": [], "frequency": {}, "n_runs": 0}
    counts: Counter[str] = Counter()
    for r in solved:
        counts.update(r.open_dcs)
    n = len(solved)
    freq = {dc: c / n for dc, c in counts.items()}
    core = sorted([d for d, f in freq.items() if f >= core_threshold])
    swing = sorted([d for d, f in freq.items() if out_threshold < f < core_threshold])
    opened_at_least_once = sorted(freq)
    return {
        "core": core,
        "swing": swing,
        "out": sorted([d for d, f in freq.items() if f <= out_threshold]),
        "opened_at_least_once": opened_at_least_once,
        "frequency": dict(sorted(freq.items(), key=lambda kv: -kv[1])),
        "n_runs": n,
        "distinct_networks": len({frozenset(r.open_dcs) for r in solved}),
        "mean_dcs": float(np.mean([r.n_dcs for r in solved])),
    }


def sweep_table(runs: Sequence[ScenarioRun]) -> tuple[list[str], list[list[object]]]:
    headers = ["run", "demand x", "cost x", "cost/period", "DCs", "open set", "wtd km"]
    rows: list[list[object]] = []
    for r in runs:
        rows.append(
            [
                r.label,
                r.params.get("demand_multiplier", ""),
                r.params.get("cost_multiplier", ""),
                f"{r.objective:,.0f}" if r.objective is not None else r.status,
                r.n_dcs,
                " ".join(r.open_dcs),
                f"{r.demand_weighted_km:,.0f}",
            ]
        )
    return headers, rows


def elasticity(
    runs: Sequence[ScenarioRun], key: str = "demand_multiplier"
) -> float | None:
    """Log-log slope of total cost against a swept parameter.

    A cost elasticity below 1 against demand is the economies-of-scale claim
    made quantitative: it says the network absorbs growth without proportional
    cost. It is also the number to check before believing a savings case built
    on a volume assumption.
    """
    xs, ys = [], []
    for r in runs:
        v = r.params.get(key)
        if v and r.objective:
            xs.append(np.log(float(v)))
            ys.append(np.log(float(r.objective)))
    if len(set(xs)) < 2:
        return None
    slope = float(np.polyfit(np.asarray(xs), np.asarray(ys), 1)[0])
    return slope
