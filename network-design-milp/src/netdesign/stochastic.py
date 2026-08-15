"""Two-stage stochastic network design and the value of modelling uncertainty.

The deterministic equivalent is built by the same
:class:`~netdesign.network_flow.NetworkDesignModel` used for the deterministic
case: first-stage variables (open/close, sourcing assignment) are shared, and
the flow block is replicated per scenario and weighted by probability.

Four objective values matter, and only two of them are worth reporting to a
steering committee:

``RP``   the here-and-now stochastic optimum - open the network that is best in
         expectation across scenarios.
``EEV``  expected result of using the expected-value solution - design for
         average demand, then live with it in every scenario.
``WS``   wait-and-see - the average of per-scenario optima; unattainable,
         because it requires knowing demand before choosing the network.

``VSS = EEV - RP`` is what stochastic modelling is worth. ``EVPI = RP - WS`` is
what perfect information would be worth on top of that - the ceiling on any
forecasting investment.

That last sentence is the one that changes decisions. A network team that
cannot say "a perfect forecast is worth at most X per period" will keep
funding forecast accuracy projects that cannot pay back.

Reference: Birge & Louveaux, *Introduction to Stochastic Programming* (2011),
Ch. 4 for VSS/EVPI; Santoso et al. (2005) for the stochastic supply chain
network design application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .instances import Instance
from .network_flow import NetworkDesignModel, NetworkOptions, NetworkSolution

__all__ = ["StochasticResult", "solve_two_stage", "expected_value_solution", "wait_and_see"]


@dataclass
class StochasticResult:
    """RP / EEV / WS plus the two numbers that get quoted."""

    rp: float
    eev: float
    ws: float
    rp_solution: NetworkSolution
    ev_solution: NetworkSolution
    eev_solution: NetworkSolution
    ws_objectives: list[float] = field(default_factory=list)
    ws_open_sets: list[list[str]] = field(default_factory=list)
    n_scenarios: int = 0
    runtime: float = 0.0

    @property
    def vss(self) -> float:
        """Value of the stochastic solution, ``EEV - RP`` (>= 0)."""
        return self.eev - self.rp

    @property
    def evpi(self) -> float:
        """Expected value of perfect information, ``RP - WS`` (>= 0)."""
        return self.rp - self.ws

    @property
    def vss_pct(self) -> float:
        return 100.0 * self.vss / self.rp if self.rp else 0.0

    @property
    def evpi_pct(self) -> float:
        return 100.0 * self.evpi / self.rp if self.rp else 0.0

    @property
    def network_stability(self) -> float:
        """Share of wait-and-see solutions whose DC set equals the RP set.

        A high number means the stochastic answer is structurally robust and the
        scenarios are arguing about flow, not about geography. A low number is
        the honest signal that the site decision is genuinely uncertain.
        """
        if not self.ws_open_sets:
            return 0.0
        target = set(self.rp_solution.open_dcs)
        return sum(1 for s in self.ws_open_sets if set(s) == target) / len(self.ws_open_sets)

    def check_bounds(self, tol: float = 1e-6) -> None:
        """Assert the theory: ``WS <= RP <= EEV``. Cheap and catches real bugs."""
        if self.ws > self.rp + tol * max(1.0, abs(self.rp)):
            raise AssertionError(f"WS ({self.ws:,.2f}) must not exceed RP ({self.rp:,.2f})")
        if self.rp > self.eev + tol * max(1.0, abs(self.rp)):
            raise AssertionError(f"RP ({self.rp:,.2f}) must not exceed EEV ({self.eev:,.2f})")

    def to_table(self) -> str:
        rows = [
            ("WS  (wait-and-see, unattainable)", self.ws),
            ("RP  (stochastic here-and-now)", self.rp),
            ("EEV (expected-value design, lived with)", self.eev),
        ]
        lines = [f"{'measure':<42}{'cost/period':>16}"]
        lines += [f"{k:<42}{v:>16,.0f}" for k, v in rows]
        lines.append("-" * 58)
        lines.append(f"{'VSS  = EEV - RP':<42}{self.vss:>16,.0f}")
        lines.append(f"{'EVPI = RP - WS':<42}{self.evpi:>16,.0f}")
        return "\n".join(lines)


def _mean_demand(
    scenarios: Sequence[dict[tuple[str, str], float]], probs: np.ndarray
) -> dict[tuple[str, str], float]:
    keys = set()
    for s in scenarios:
        keys |= set(s)
    return {k: float(sum(p * s.get(k, 0.0) for p, s in zip(probs, scenarios))) for k in keys}


def expected_value_solution(
    instance: Instance,
    scenarios: Sequence[dict[tuple[str, str], float]],
    probs: np.ndarray,
    options: NetworkOptions,
    **solve_kwargs: Any,
) -> NetworkSolution:
    """Solve the mean-value problem - the design most teams actually ship."""
    mean_inst = instance.with_demand(_mean_demand(scenarios, probs))
    builder = NetworkDesignModel(mean_inst, options, name="mean-value")
    sol, _ = builder.solve(**solve_kwargs)
    return sol


def wait_and_see(
    instance: Instance,
    scenarios: Sequence[dict[tuple[str, str], float]],
    probs: np.ndarray,
    options: NetworkOptions,
    **solve_kwargs: Any,
) -> tuple[float, list[float], list[list[str]]]:
    """Solve every scenario as if it were known in advance."""
    objectives: list[float] = []
    open_sets: list[list[str]] = []
    for i, demand in enumerate(scenarios):
        builder = NetworkDesignModel(
            instance.with_demand(demand), options, name=f"wait-and-see[{i}]"
        )
        sol, _ = builder.solve(**solve_kwargs)
        if sol.objective is None:
            raise RuntimeError(f"wait-and-see scenario {i} did not solve: {sol.status}")
        objectives.append(float(sol.objective))
        open_sets.append(list(sol.open_dcs))
    ws = float(np.dot(np.asarray(probs, dtype=float), np.asarray(objectives)))
    return ws, objectives, open_sets


def solve_two_stage(
    instance: Instance,
    scenarios: Sequence[dict[tuple[str, str], float]],
    probs: Sequence[float] | np.ndarray | None = None,
    options: NetworkOptions | None = None,
    **solve_kwargs: Any,
) -> StochasticResult:
    """Full VSS/EVPI study.

    ``options.allow_unmet`` is forced on: without a recourse action that is
    always available, the expected-value design is simply infeasible in the
    high-demand scenarios and EEV is ``+inf``. That is technically true and
    practically useless - "your plan fails" is not a number anyone can trade
    off. Pricing the shortfall at the lost margin turns it into one.
    """
    import time

    t0 = time.perf_counter()
    p = (
        np.full(len(scenarios), 1.0 / len(scenarios))
        if probs is None
        else np.asarray(probs, dtype=float)
    )
    base = options or NetworkOptions()
    if not base.allow_unmet:
        base = NetworkOptions(**{**base.__dict__, "allow_unmet": True})

    # RP - the here-and-now stochastic programme
    rp_builder = NetworkDesignModel(
        instance, base, scenarios=scenarios, probabilities=p, name="two-stage-RP"
    )
    rp_sol, _ = rp_builder.solve(**solve_kwargs)
    if rp_sol.objective is None:
        raise RuntimeError(f"stochastic model did not solve: {rp_sol.status}")

    # EV - design on mean demand, then EEV - evaluate that design everywhere
    ev_sol = expected_value_solution(instance, scenarios, p, base, **solve_kwargs)
    fixed_dcs = {d.id: int(d.id in ev_sol.open_dcs) for d in instance.dcs}
    fixed_plants = {pl.id: int(pl.id in ev_sol.open_plants) for pl in instance.plants}
    eev_options = NetworkOptions(
        **{**base.__dict__, "fixed_open_dcs": fixed_dcs, "fixed_open_plants": fixed_plants}
    )
    eev_builder = NetworkDesignModel(
        instance, eev_options, scenarios=scenarios, probabilities=p, name="two-stage-EEV"
    )
    eev_sol, _ = eev_builder.solve(**solve_kwargs)
    if eev_sol.objective is None:
        raise RuntimeError(f"EEV evaluation did not solve: {eev_sol.status}")

    ws, ws_objs, ws_sets = wait_and_see(instance, scenarios, p, base, **solve_kwargs)

    result = StochasticResult(
        rp=float(rp_sol.objective),
        eev=float(eev_sol.objective),
        ws=ws,
        rp_solution=rp_sol,
        ev_solution=ev_sol,
        eev_solution=eev_sol,
        ws_objectives=ws_objs,
        ws_open_sets=ws_sets,
        n_scenarios=len(scenarios),
        runtime=time.perf_counter() - t0,
    )
    result.check_bounds()
    return result
