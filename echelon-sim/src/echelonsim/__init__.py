"""echelonsim -- discrete-event simulation of multi-echelon supply chains.

The package is layered strictly downward:

    engine        heapq event loop: Event, Timeout, Process, Environment
    rng           named independent streams (common random numbers)
    demand        customer demand processes
    leadtime      transit-time distributions
    forecast      the forecasters that sit inside the replenishment loop
    policies      base-stock, (s,S), (R,S), order batching
    network       nodes, echelons, inventory position, information modes
    simulation    the period structure, wired onto the engine
    metrics       warm-up truncation, batch means, paired intervals
    experiments   configs, replication, comparison
    bullwhip      amplification measurement and its decomposition
    information   decentralised vs shared POS vs vendor-managed
    disruption    outages, shocks, capacity loss, recovery measurement

Nothing above the line imports anything below it in the wrong direction, which
is what keeps the analytic cross-checks in the test suite honest: the simulator
has no knowledge of the closed-form bullwhip bounds it is checked against.
"""

from .demand import AR1, IIDNormal, SeasonalTrend, ShockOverlay
from .engine import Environment, Event, Interrupt, Process, Timeout
from .forecast import DampedTrend, ExponentialSmoothing, MovingAverage, Oracle
from .leadtime import Deterministic, DiscreteLeadTime, GammaLeadTime
from .network import Allocation, InfoMode, Node, SupplyNetwork, divergent_network, serial_chain
from .policies import Batched, BaseStock, RSPolicy, SsPolicy
from .rng import StreamBank
from .simulation import (
    CapacityLoss,
    DisruptionPlan,
    SimulationResult,
    Simulator,
    SupplyOutage,
    run_simulation,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AR1",
    "Allocation",
    "BaseStock",
    "Batched",
    "CapacityLoss",
    "DampedTrend",
    "Deterministic",
    "DiscreteLeadTime",
    "DisruptionPlan",
    "Environment",
    "Event",
    "ExponentialSmoothing",
    "GammaLeadTime",
    "IIDNormal",
    "InfoMode",
    "Interrupt",
    "MovingAverage",
    "Node",
    "Oracle",
    "Process",
    "RSPolicy",
    "SeasonalTrend",
    "ShockOverlay",
    "SimulationResult",
    "Simulator",
    "SsPolicy",
    "StreamBank",
    "SupplyNetwork",
    "SupplyOutage",
    "Timeout",
    "divergent_network",
    "run_simulation",
    "serial_chain",
]
