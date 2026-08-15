"""netdesign - supply chain network design and flow optimization with MILP.

Layers, bottom to top:

``modeling``          variable registry, expressions, sparse assembly, HiGHS solve
``geometry``          haversine distance, center of gravity, geometric median
``instances``         data model plus a synthetic instance/scenario generator
``facility_location`` UFLP and CFLP, strong vs aggregated formulations
``network_flow``      the multi-echelon, multi-commodity design MILP
``stochastic``        two-stage deterministic equivalent, VSS and EVPI
``greenfield``        center-of-gravity heuristic, costed against the MILP
``diagnostics``       capacity ledger and elastic relaxation for infeasibility
``scenarios``         scenario runner, sensitivity sweep, network stability
``reporting``         cost breakdown, utilisation, service profile
``figures``           matplotlib network map (requires matplotlib)
"""

from __future__ import annotations

from .diagnostics import Diagnosis, capacity_ledger, diagnose
from .facility_location import (
    FacilityLocationResult,
    capacitated_facility_location,
    dc_location_subproblem,
    uncapacitated_facility_location,
)
from .geometry import center_of_gravity, haversine_km, haversine_matrix, weiszfeld
from .greenfield import (
    GreenfieldResult,
    evaluate_open_set,
    greenfield_open_set,
    greenfield_vs_milp,
    solve_with_cutoff,
)
from .instances import (
    Commodity,
    Facility,
    Instance,
    Lane,
    Mode,
    Site,
    demand_scenarios,
    generate_instance,
)
from .modeling import ANY, LinExpr, Model, Solution, VarGroup, quicksum
from .network_flow import (
    FlowRecord,
    NetworkDesignModel,
    NetworkOptions,
    NetworkSolution,
    solve_network_design,
)
from .reporting import cost_breakdown, facility_table, flow_by_echelon, format_report
from .scenarios import ScenarioRun, run_scenarios, sensitivity_sweep, stability_profile
from .stochastic import StochasticResult, solve_two_stage

__version__ = "0.1.0"

__all__ = [
    "ANY",
    "Commodity",
    "Diagnosis",
    "Facility",
    "FacilityLocationResult",
    "FlowRecord",
    "GreenfieldResult",
    "Instance",
    "Lane",
    "LinExpr",
    "Mode",
    "Model",
    "NetworkDesignModel",
    "NetworkOptions",
    "NetworkSolution",
    "ScenarioRun",
    "Site",
    "Solution",
    "StochasticResult",
    "VarGroup",
    "__version__",
    "capacitated_facility_location",
    "capacity_ledger",
    "center_of_gravity",
    "cost_breakdown",
    "dc_location_subproblem",
    "demand_scenarios",
    "diagnose",
    "evaluate_open_set",
    "facility_table",
    "flow_by_echelon",
    "format_report",
    "generate_instance",
    "greenfield_open_set",
    "greenfield_vs_milp",
    "haversine_km",
    "haversine_matrix",
    "quicksum",
    "run_scenarios",
    "sensitivity_sweep",
    "solve_network_design",
    "solve_two_stage",
    "solve_with_cutoff",
    "stability_profile",
    "uncapacitated_facility_location",
    "weiszfeld",
]
