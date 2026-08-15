"""Warehouse slotting and pick-path optimization.

A small, complete library for the two decisions that set manual picking labour:
where each SKU lives, and how the picker walks between them. It models the rack
geometry properly, slots against a stated objective under real constraints, and
then measures the result by simulating the order stream through classical
routing heuristics with the optimality gap of each one reported against an
exact dynamic program.

Typical use::

    from slotting import build_instance, make_objective, velocity_slotting, evaluate_travel

    inst = build_instance(seed=7)
    obj = make_objective(inst)
    before = evaluate_travel(inst, random_assignment(inst.constraints, seed=0))
    after = evaluate_travel(inst, velocity_slotting(inst.constraints, pick_rate=inst.fit_rate))
"""

from __future__ import annotations

from .affinity import (
    AffinityGraph,
    affinity_slotting,
    cluster_quality,
    greedy_clusters,
    spectral_clusters,
)
from .assignment import (
    Assignment,
    ConstraintConfig,
    ConstraintModel,
    greedy_place,
    random_assignment,
)
from .batching import (
    Batch,
    CartCapacity,
    evaluate_batches,
    savings_batching,
    seed_batching,
    single_order_batches,
)
from .benchmark import (
    Instance,
    build_instance,
    evaluate_travel,
    make_objective,
    optimality_gaps,
    run_benchmark,
)
from .ergonomics import ErgonomicModel, golden_zone_order
from .layout import Location, Point, Warehouse, WarehouseConfig
from .localsearch import AnnealingConfig, simulated_annealing, steepest_descent
from .objective import ObjectiveWeights, SlottingObjective, build_affinity_pairs
from .orders import Order, OrderConfig, OrderStream, generate_orders
from .routing import (
    ROUTERS,
    Route,
    exact_aisle_dp,
    held_karp,
    largest_gap,
    midpoint,
    nearest_neighbour,
    return_route,
    route_picks,
    s_shape,
    two_opt,
    validate_route,
)
from .skus import SKU, Catalog, CatalogConfig, generate_catalog
from .velocity import abc_summary, class_based_slotting, velocity_slotting

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # geometry
    "Warehouse",
    "WarehouseConfig",
    "Location",
    "Point",
    # data
    "SKU",
    "Catalog",
    "CatalogConfig",
    "generate_catalog",
    "Order",
    "OrderStream",
    "OrderConfig",
    "generate_orders",
    # assignment and constraints
    "Assignment",
    "ConstraintModel",
    "ConstraintConfig",
    "random_assignment",
    "greedy_place",
    # slotting
    "velocity_slotting",
    "class_based_slotting",
    "abc_summary",
    "affinity_slotting",
    "AffinityGraph",
    "greedy_clusters",
    "spectral_clusters",
    "cluster_quality",
    "ErgonomicModel",
    "golden_zone_order",
    # objective and search
    "SlottingObjective",
    "ObjectiveWeights",
    "build_affinity_pairs",
    "simulated_annealing",
    "steepest_descent",
    "AnnealingConfig",
    # routing
    "Route",
    "ROUTERS",
    "route_picks",
    "s_shape",
    "return_route",
    "midpoint",
    "largest_gap",
    "nearest_neighbour",
    "two_opt",
    "held_karp",
    "exact_aisle_dp",
    "validate_route",
    # batching
    "Batch",
    "CartCapacity",
    "single_order_batches",
    "seed_batching",
    "savings_batching",
    "evaluate_batches",
    # harness
    "Instance",
    "build_instance",
    "make_objective",
    "evaluate_travel",
    "optimality_gaps",
    "run_benchmark",
]
