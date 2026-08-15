"""Multi-echelon, multi-commodity network design.

The model is a capacitated fixed-charge network flow over four echelons:

``suppliers -> plants -> DCs -> demand zones``

with binary open/close decisions on plants and DCs, per-lane per-mode transport
cost, throughput ceilings, minimum-volume-if-open thresholds, optional
single-sourcing, and an optional dual-sourcing floor. The same builder handles
the two-stage stochastic case by replicating the flow block per demand scenario
while keeping the structural decisions (open/close, and the sourcing
assignment) in the first stage - which is the correct split, because you cannot
re-site a DC once demand arrives.

References: Geoffrion & Graves (1974) for the multicommodity distribution
design formulation; Chopra & Meindl (2015) Ch. 5 for the practitioner framing;
Melo, Nickel & Saldanha-da-Gama (2009) for the survey of what modern variants
add.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .instances import Facility, Instance
from .modeling import ANY, LinExpr, Model, Solution, VarGroup, quicksum

__all__ = [
    "NetworkOptions",
    "NetworkDesignModel",
    "NetworkSolution",
    "FlowRecord",
    "throughput_big_m",
    "solve_network_design",
]


@dataclass
class NetworkOptions:
    """Switches that change the *structure* of the model, not just its data."""

    #: Each zone is served by exactly one DC (all commodities together).
    single_source: bool = False
    #: Minimum share of a zone's volume that must come from a second DC.
    #: Mutually exclusive with ``single_source``.
    min_second_source_share: float = 0.0
    #: Enforce ``throughput >= min_volume`` when a DC is opened.
    enforce_min_volume: bool = True
    #: Disaggregated linking ``flow[d,z,c] <= demand[z,c] * open[d]`` in addition
    #: to the aggregate throughput link. Valid either way; the disaggregated
    #: form is what makes the LP relaxation tight.
    disaggregated_link: bool = True
    #: Multiplier applied to the tight big-M on the facility linking rows.
    #: 1.0 is the tight value; larger values reproduce the "just use a big
    #: number" formulation so its cost can be measured rather than asserted.
    big_m_scale: float = 1.0
    max_open_dcs: int | None = None
    min_open_dcs: int | None = None
    #: Fix first-stage decisions (id -> 0/1). Used to evaluate a candidate
    #: network - the mean-value solution, or a greenfield heuristic's answer.
    fixed_open_dcs: dict[str, int] | None = None
    fixed_open_plants: dict[str, int] | None = None
    #: Permit unmet demand at a penalty. Required for the stochastic recourse
    #: problem to be well defined under capacity-binding scenarios.
    allow_unmet: bool = False
    unmet_penalty: float = 45.0

    def validate(self) -> None:
        if self.single_source and self.min_second_source_share > 0:
            raise ValueError(
                "single_source and min_second_source_share are contradictory: "
                "one forces exactly one DC per zone, the other forces at least two"
            )
        if self.big_m_scale < 1.0:
            raise ValueError(
                "big_m_scale < 1 would cut off feasible solutions: the tight M is "
                "min(capacity, demand) and anything smaller is an invalid formulation"
            )
        if not 0.0 <= self.min_second_source_share < 0.5:
            raise ValueError(
                "min_second_source_share must be in [0, 0.5); a second source cannot "
                "be required to carry more than half the volume"
            )


@dataclass(frozen=True)
class FlowRecord:
    origin: str
    dest: str
    mode: str
    commodity: str
    units: float
    distance_km: float
    cost: float

    @property
    def echelon(self) -> str:
        return f"{self.origin[:2]}->{self.dest[:2]}"


@dataclass
class NetworkSolution:
    """Everything a planner needs to argue about the answer."""

    status: str
    objective: float | None
    runtime: float
    mip_gap: float | None
    open_plants: list[str] = field(default_factory=list)
    open_dcs: list[str] = field(default_factory=list)
    flows: list[FlowRecord] = field(default_factory=list)
    costs: dict[str, float] = field(default_factory=dict)
    utilization: dict[str, dict[str, float]] = field(default_factory=dict)
    zone_service: dict[str, dict[str, float]] = field(default_factory=dict)
    unmet_units: float = 0.0
    demand_weighted_km: float = 0.0
    model_stats: dict[str, int] = field(default_factory=dict)
    scenario_costs: list[float] = field(default_factory=list)

    @property
    def is_optimal(self) -> bool:
        return self.status == "optimal"

    def cost_total(self) -> float:
        return float(sum(self.costs.values()))

    def open_dc_set(self) -> frozenset[str]:
        return frozenset(self.open_dcs)


def throughput_big_m(facility: Facility, total_relevant_demand: float) -> float:
    """The tight big-M for "no flow unless open" at a facility.

    ``min(capacity, demand that could conceivably route here)``. Using an
    arbitrary large constant instead is formally valid and practically
    disastrous: the LP relaxation lets ``y = throughput / M`` become
    near-zero, the bound collapses to roughly the transport-only cost, and
    branch and bound has to close the entire fixed-cost gap by enumeration.
    On this repository's default instance the loose-M variant is measured in
    ``benchmarks/results.md``.
    """
    return float(min(facility.capacity, total_relevant_demand))


class NetworkDesignModel:
    """Builds (and solves) the network design MILP for one or many scenarios."""

    def __init__(
        self,
        instance: Instance,
        options: NetworkOptions | None = None,
        *,
        scenarios: Sequence[dict[tuple[str, str], float]] | None = None,
        probabilities: Sequence[float] | np.ndarray | None = None,
        name: str = "network-design",
    ) -> None:
        self.instance = instance
        self.options = options or NetworkOptions()
        self.options.validate()
        self.scenarios: list[dict[tuple[str, str], float]] = (
            [dict(instance.demand)] if scenarios is None else [dict(s) for s in scenarios]
        )
        if probabilities is None:
            self.probs = np.full(len(self.scenarios), 1.0 / len(self.scenarios))
        else:
            p = np.asarray(probabilities, dtype=float)
            if len(p) != len(self.scenarios):
                raise ValueError("probabilities and scenarios have different lengths")
            if abs(p.sum() - 1.0) > 1e-9:
                raise ValueError(f"probabilities must sum to 1, got {p.sum()!r}")
            self.probs = p

        self.model = Model(name, sense="min")
        self.flow: list[VarGroup] = []
        self.unmet: list[VarGroup] = []
        self.cost_terms: dict[str, LinExpr] = {}
        self._build()

    # -- construction --------------------------------------------------------

    @property
    def is_stochastic(self) -> bool:
        return len(self.scenarios) > 1

    def _build(self) -> None:
        inst = self.instance
        opt = self.options
        m = self.model
        commodities = inst.commodity_ids

        # ---- first stage: what exists ------------------------------------
        self.open_plant = m.add_vars(
            inst.plant_ids,
            name="open_plant",
            vtype="binary",
            obj={p.id: p.fixed_cost for p in inst.plants},
        )
        self.open_dc = m.add_vars(
            inst.dc_ids,
            name="open_dc",
            vtype="binary",
            obj={d.id: d.fixed_cost for d in inst.dcs},
        )
        self.cost_terms["plant_fixed"] = quicksum(
            p.fixed_cost * self.open_plant[p.id] for p in inst.plants
        )
        self.cost_terms["dc_fixed"] = quicksum(d.fixed_cost * self.open_dc[d.id] for d in inst.dcs)

        if opt.fixed_open_plants:
            for pid, val in opt.fixed_open_plants.items():
                m.fix(self.open_plant[pid], val, name=f"fix_plant[{pid}]")
        if opt.fixed_open_dcs:
            for did, val in opt.fixed_open_dcs.items():
                m.fix(self.open_dc[did], val, name=f"fix_dc[{did}]")

        if opt.max_open_dcs is not None:
            m.add(self.open_dc.sum() <= opt.max_open_dcs, name="max_open_dcs", tag="count")
        if opt.min_open_dcs is not None:
            m.add(self.open_dc.sum() >= opt.min_open_dcs, name="min_open_dcs", tag="count")

        # ---- first stage: who serves whom (structural, not per-scenario) --
        self.assign: VarGroup | None = None
        if opt.single_source:
            zone_lane_pairs = sorted(
                {(ln.origin, ln.dest) for ln in inst.lanes_by_echelon("dc", "zone")}
            )
            self.assign = m.add_vars(zone_lane_pairs, name="assign", vtype="binary")
            for z in inst.zone_ids:
                pairs = self.assign.select(ANY, z)
                m.add(
                    quicksum(self.assign[k] for k in pairs) == 1.0,
                    name=f"single_source[{z}]",
                    tag="single_source",
                )
            for d, z in self.assign:
                m.add(
                    self.assign[(d, z)] <= self.open_dc[d],
                    name=f"assign_open[{d},{z}]",
                    tag="assign_open",
                )

        # ---- second stage: flows, one block per scenario ------------------
        lanes_sp = inst.lanes_by_echelon("supplier", "plant")
        lanes_pd = inst.lanes_by_echelon("plant", "dc")
        lanes_dz = inst.lanes_by_echelon("dc", "zone")
        all_lanes = lanes_sp + lanes_pd + lanes_dz

        transport_terms = {"inbound": LinExpr(), "primary": LinExpr(), "outbound": LinExpr()}
        production_term = LinExpr()
        handling_term = LinExpr()
        unmet_term = LinExpr()
        self._scenario_variable_cost: list[LinExpr] = []

        for s, demand in enumerate(self.scenarios):
            w = float(self.probs[s])
            keys = [(ln.origin, ln.dest, ln.mode, c) for ln in all_lanes for c in commodities]
            cost_of = {
                (ln.origin, ln.dest, ln.mode, c): inst.unit_cost(ln, c)
                for ln in all_lanes
                for c in commodities
            }
            flow = self.model.add_vars(
                keys,
                name=f"flow[s{s}]",
                lb=0.0,
                obj={k: w * v for k, v in cost_of.items()},
            )
            self.flow.append(flow)

            scen_cost = quicksum(cost_of[k] * flow[k] for k in keys)
            for group, lanes in (
                ("inbound", lanes_sp),
                ("primary", lanes_pd),
                ("outbound", lanes_dz),
            ):
                sub = [(ln.origin, ln.dest, ln.mode, c) for ln in lanes for c in commodities]
                transport_terms[group] = transport_terms[group] + w * quicksum(
                    cost_of[k] * flow[k] for k in sub
                )

            # production cost, charged on plant outbound volume
            prod = quicksum(
                inst.production_cost[(p.id, c)] * flow.sum(p.id, ANY, ANY, c)
                for p in inst.plants
                for c in commodities
            )
            production_term = production_term + w * prod
            scen_cost = scen_cost + prod

            # handling cost, charged on DC inbound volume
            handling = quicksum(
                d.handling_cost * flow.sum(ANY, d.id, ANY, ANY) for d in inst.dcs
            )
            handling_term = handling_term + w * handling
            scen_cost = scen_cost + handling

            self.model.add_objective(w * (prod + handling))

            unmet_group: VarGroup | None = None
            if self.options.allow_unmet:
                unmet_group = self.model.add_vars(
                    [(z, c) for z in inst.zone_ids for c in commodities],
                    name=f"unmet[s{s}]",
                    lb=0.0,
                    obj=w * self.options.unmet_penalty,
                )
                self.unmet.append(unmet_group)
                pen = self.options.unmet_penalty * quicksum(unmet_group.values())
                unmet_term = unmet_term + w * pen
                scen_cost = scen_cost + pen

            self._scenario_variable_cost.append(scen_cost)
            self._add_flow_constraints(s, flow, demand, unmet_group)

        self.cost_terms["transport_inbound"] = transport_terms["inbound"]
        self.cost_terms["transport_primary"] = transport_terms["primary"]
        self.cost_terms["transport_outbound"] = transport_terms["outbound"]
        self.cost_terms["production"] = production_term
        self.cost_terms["handling"] = handling_term
        if self.options.allow_unmet:
            self.cost_terms["unmet_penalty"] = unmet_term

    def _add_flow_constraints(
        self,
        s: int,
        flow: VarGroup,
        demand: dict[tuple[str, str], float],
        unmet: VarGroup | None,
    ) -> None:
        inst = self.instance
        opt = self.options
        m = self.model
        commodities = inst.commodity_ids
        tag = f"s{s}"

        total_demand_c = {
            c: float(sum(v for (_, k), v in demand.items() if k == c)) for c in commodities
        }
        total_demand = float(sum(total_demand_c.values()))

        # supplier availability
        for sup in inst.suppliers:
            for c in commodities:
                cap = inst.supply.get((sup.id, c), 0.0)
                m.add(
                    flow.sum(sup.id, ANY, ANY, c) <= cap,
                    name=f"supply[{sup.id},{c},{tag}]",
                    tag=f"supply|{tag}",
                )

        # plant conservation and capacity
        for p in inst.plants:
            for c in commodities:
                m.add(
                    flow.sum(ANY, p.id, ANY, c) - flow.sum(p.id, ANY, ANY, c) == 0.0,
                    name=f"plant_balance[{p.id},{c},{tag}]",
                    tag=f"plant_balance|{tag}",
                )
            out = quicksum(flow.sum(p.id, ANY, ANY, c) for c in commodities)
            tight = throughput_big_m(p, total_demand)
            if opt.big_m_scale > 1.0:
                # With a loosened M the linking row no longer implies the
                # capacity ceiling, so it has to be stated separately. This is
                # the honest comparison: same feasible set, weaker relaxation.
                m.add(out <= p.capacity, name=f"plant_cap[{p.id},{tag}]", tag=f"plant_capacity|{tag}")
            m.link_big_m(
                out,
                self.open_plant[p.id],
                tight * opt.big_m_scale,
                name=f"plant_capacity[{p.id},{tag}]",
                tag=f"plant_capacity|{tag}",
            )

        # DC conservation, throughput ceiling and minimum volume
        for d in inst.dcs:
            for c in commodities:
                m.add(
                    flow.sum(ANY, d.id, ANY, c) - flow.sum(d.id, ANY, ANY, c) == 0.0,
                    name=f"dc_balance[{d.id},{c},{tag}]",
                    tag=f"dc_balance|{tag}",
                )
            inflow = flow.sum(ANY, d.id, ANY, ANY)
            tight = throughput_big_m(d, total_demand)
            if opt.big_m_scale > 1.0:
                m.add(inflow <= d.capacity, name=f"dc_cap[{d.id},{tag}]", tag=f"dc_throughput|{tag}")
            m.link_big_m(
                inflow,
                self.open_dc[d.id],
                tight * opt.big_m_scale,
                name=f"dc_throughput[{d.id},{tag}]",
                tag=f"dc_throughput|{tag}",
            )
            if opt.enforce_min_volume and d.min_volume > 0:
                # No big-M needed in this direction: the constraint is already
                # "0 if closed, >= min_volume if open".
                m.add(
                    inflow - d.min_volume * self.open_dc[d.id] >= 0.0,
                    name=f"dc_min_volume[{d.id},{tag}]",
                    tag=f"dc_min_volume|{tag}",
                )

        # demand satisfaction
        for z in inst.zone_ids:
            for c in commodities:
                served = flow.sum(ANY, z, ANY, c)
                if unmet is not None:
                    served = served + unmet[(z, c)]
                m.add(
                    served == demand.get((z, c), 0.0),
                    name=f"demand[{z},{c},{tag}]",
                    tag=f"demand|{tag}",
                )

        # linking outbound arcs to the open decision
        if opt.disaggregated_link:
            for ln in inst.lanes_by_echelon("dc", "zone"):
                for c in commodities:
                    dem = demand.get((ln.dest, c), 0.0)
                    if dem <= 0:
                        continue
                    m.link_big_m(
                        flow[(ln.origin, ln.dest, ln.mode, c)],
                        self.open_dc[ln.origin],
                        dem,
                        name=f"link[{ln.origin},{ln.dest},{c},{tag}]",
                        tag=f"link|{tag}",
                    )

        # single sourcing: a zone's whole volume comes from its assigned DC
        if opt.single_source and self.assign is not None:
            for d, z in self.assign:
                for c in commodities:
                    dem = demand.get((z, c), 0.0)
                    if dem <= 0:
                        continue
                    m.link_big_m(
                        flow.sum(d, z, ANY, c),
                        self.assign[(d, z)],
                        dem,
                        name=f"single_source_link[{d},{z},{c},{tag}]",
                        tag=f"single_source_link|{tag}",
                    )

        # dual sourcing: cap any one DC's share of a zone
        if opt.min_second_source_share > 0:
            cap_share = 1.0 - opt.min_second_source_share
            for z in inst.zone_ids:
                zone_total = float(sum(demand.get((z, c), 0.0) for c in commodities))
                if zone_total <= 0:
                    continue
                for d in inst.dc_ids:
                    keys = flow.select(d, z, ANY, ANY)
                    if not keys:
                        continue
                    m.add(
                        flow.sum_over(keys) <= cap_share * zone_total,
                        name=f"dual_source[{d},{z},{tag}]",
                        tag=f"dual_source|{tag}",
                    )

    # -- solving and extraction ---------------------------------------------

    def solve(self, **kwargs: Any) -> tuple[NetworkSolution, Solution]:
        raw = self.model.solve(**kwargs)
        return self.extract(raw), raw

    def lp_bound(self, **kwargs: Any) -> float:
        """Objective of the LP relaxation - the formulation-quality yardstick."""
        raw = self.model.solve(relax=True, **kwargs)
        raw.require_optimal()
        assert raw.objective is not None
        return float(raw.objective)

    def extract(self, raw: Solution, scenario: int = 0) -> NetworkSolution:
        inst = self.instance
        sol = NetworkSolution(
            status=raw.status,
            objective=raw.objective,
            runtime=raw.runtime,
            mip_gap=raw.mip_gap,
            model_stats=self.model.stats(),
        )
        if raw.x is None:
            return sol

        open_plant = raw.values(self.open_plant, integral=True)
        open_dc = raw.values(self.open_dc, integral=True)
        sol.open_plants = sorted(k for k, v in open_plant.items() if v > 0.5)
        sol.open_dcs = sorted(k for k, v in open_dc.items() if v > 0.5)

        flow = self.flow[scenario]
        records: list[FlowRecord] = []
        for (o, d, mode, c), qty in raw.values(flow, nonzero=True, tol=1e-6).items():
            lane = inst.lane(o, d, mode)
            records.append(
                FlowRecord(
                    origin=o,
                    dest=d,
                    mode=mode,
                    commodity=c,
                    units=qty,
                    distance_km=lane.distance_km,
                    cost=qty * inst.unit_cost(lane, c),
                )
            )
        records.sort(key=lambda r: -r.units)
        sol.flows = records

        sol.costs = {k: float(raw.value(v)) for k, v in self.cost_terms.items()}
        sol.scenario_costs = [float(raw.value(e)) for e in self._scenario_variable_cost]

        # utilisation, per facility that can be opened
        util: dict[str, dict[str, float]] = {}
        for fac in (*inst.plants, *inst.dcs):
            if fac.kind == "plant":
                thr = sum(r.units for r in records if r.origin == fac.id)
            else:
                thr = sum(r.units for r in records if r.dest == fac.id)
            is_open = fac.id in sol.open_plants or fac.id in sol.open_dcs
            util[fac.id] = {
                "open": float(is_open),
                "throughput": float(thr),
                "capacity": float(fac.capacity),
                "utilization": float(thr / fac.capacity) if fac.capacity > 0 else 0.0,
                "min_volume": float(fac.min_volume),
            }
        sol.utilization = util

        # zone service mix and demand-weighted distance
        zone_service: dict[str, dict[str, float]] = {}
        num = 0.0
        den = 0.0
        for r in records:
            if inst.site(r.dest).kind != "zone":
                continue
            zone_service.setdefault(r.dest, {})
            zone_service[r.dest][r.origin] = zone_service[r.dest].get(r.origin, 0.0) + r.units
            num += r.units * r.distance_km
            den += r.units
        for z, mix in zone_service.items():
            tot = sum(mix.values())
            zone_service[z] = {k: v / tot for k, v in sorted(mix.items(), key=lambda kv: -kv[1])}
        sol.zone_service = zone_service
        sol.demand_weighted_km = float(num / den) if den > 0 else 0.0

        if self.unmet:
            sol.unmet_units = float(sum(raw.values(self.unmet[scenario]).values()))
        return sol


def solve_network_design(
    instance: Instance,
    options: NetworkOptions | None = None,
    **solve_kwargs: Any,
) -> NetworkSolution:
    """Convenience wrapper: build, solve, extract."""
    builder = NetworkDesignModel(instance, options)
    sol, _ = builder.solve(**solve_kwargs)
    return sol
