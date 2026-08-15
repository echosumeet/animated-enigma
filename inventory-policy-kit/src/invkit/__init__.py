"""invkit - safety stock, replenishment policies and multi-echelon optimisation.

A working implementation of the inventory theory that planning systems are
supposed to run, written so that every formula can be checked against a
simulation of the policy it produces.

The five things worth looking at first:

* :func:`invkit.safety_stock.compare_service_definitions` - what a "95% service
  target" costs depending on whether it is read as cycle service level or fill
  rate.  These are not close.
* :func:`invkit.simulation.simulate_policy` - the falsification harness.  Every
  analytic result in this package is checked against it in ``tests/``.
* :func:`invkit.serial.clark_scarf` - exact echelon base-stock levels for a
  serial system, validated against a simulation of the same system.
* :func:`invkit.guaranteed_service.solve_guaranteed_service` - safety stock
  *placement* on a BOM tree, which answers a different question from "what
  service level per site".
* :func:`invkit.frontier.exchange_curve` - the cost-service curve that should
  replace the single corporate service number.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .distributions import (
    EmpiricalLTD,
    GammaLTD,
    LeadTimeDemand,
    MixtureLTD,
    NormalLTD,
    inverse_standard_normal_loss,
    standard_normal_loss,
    standard_normal_loss2,
)
from .frontier import FrontierPoint, exchange_curve, marginal_cost_of_service, service_frontier
from .guaranteed_service import (
    PlacementResult,
    Stage,
    SupplyChainTree,
    example_bom_tree,
    solve_guaranteed_service,
)
from .leadtime import (
    LeadTimeSpec,
    lead_time_variance_share,
    ltd_gamma,
    ltd_moments,
    ltd_normal,
    ltd_stochastic_exact,
)
from .lotsizing import (
    PriceBreak,
    compare_lot_sizing,
    eoq,
    eoq_all_units_discount,
    eoq_cost_sensitivity,
    eoq_total_cost,
    least_unit_cost,
    lot_for_lot,
    seasonal_demand_series,
    silver_meal,
    wagner_whitin,
)
from .newsvendor import (
    critical_fractile,
    critical_fractile_sensitivity,
    newsvendor_empirical,
    newsvendor_normal,
)
from .policies import (
    RSPolicy,
    RsSPolicy,
    SQPolicy,
    SSPolicy,
    average_inventory_sQ,
    build_RS,
    build_RsS,
    build_sQ,
    protection_interval_ltd,
)
from .pooling import pooling_with_lead_time_penalty, simulate_pooling, square_root_law
from .safety_stock import (
    SafetyStockResult,
    compare_service_definitions,
    fill_rate_of_sQ,
    ss_from_cycle_service_level,
    ss_from_empirical_quantile,
    ss_from_fill_rate,
)
from .serial import SerialStage, clark_scarf, simulate_serial_system
from .simulation import DemandProcess, simulate_policy, simulate_policy_replications

__all__ = [
    "__version__",
    # distributions
    "LeadTimeDemand",
    "NormalLTD",
    "GammaLTD",
    "EmpiricalLTD",
    "MixtureLTD",
    "standard_normal_loss",
    "standard_normal_loss2",
    "inverse_standard_normal_loss",
    # lead time
    "LeadTimeSpec",
    "ltd_moments",
    "ltd_normal",
    "ltd_gamma",
    "ltd_stochastic_exact",
    "lead_time_variance_share",
    # safety stock
    "SafetyStockResult",
    "ss_from_cycle_service_level",
    "ss_from_fill_rate",
    "ss_from_empirical_quantile",
    "fill_rate_of_sQ",
    "compare_service_definitions",
    # policies
    "SQPolicy",
    "SSPolicy",
    "RSPolicy",
    "RsSPolicy",
    "build_sQ",
    "build_RS",
    "build_RsS",
    "protection_interval_ltd",
    "average_inventory_sQ",
    # lot sizing
    "eoq",
    "eoq_total_cost",
    "eoq_cost_sensitivity",
    "PriceBreak",
    "eoq_all_units_discount",
    "wagner_whitin",
    "silver_meal",
    "least_unit_cost",
    "lot_for_lot",
    "compare_lot_sizing",
    "seasonal_demand_series",
    # newsvendor
    "critical_fractile",
    "newsvendor_normal",
    "newsvendor_empirical",
    "critical_fractile_sensitivity",
    # multi-echelon
    "SerialStage",
    "clark_scarf",
    "simulate_serial_system",
    "Stage",
    "SupplyChainTree",
    "PlacementResult",
    "solve_guaranteed_service",
    "example_bom_tree",
    # pooling and frontier
    "square_root_law",
    "pooling_with_lead_time_penalty",
    "simulate_pooling",
    "FrontierPoint",
    "service_frontier",
    "exchange_curve",
    "marginal_cost_of_service",
    # simulation
    "DemandProcess",
    "simulate_policy",
    "simulate_policy_replications",
]
