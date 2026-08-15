"""Information sharing: decentralised vs shared point-of-sale vs vendor-managed.

The three arrangements differ in exactly two things -- what a node forecasts
from, and what state variable it controls -- and separating those two is the
whole point of the comparison. Most "information sharing" projects deliver only
the first and are then surprised that the benefit is a fraction of what the
business case promised.

============  =========================  ==============================
mode          forecasts from             manages
============  =========================  ==============================
decentralised the orders it receives     its own installation stock
shared POS    end-customer demand        its own installation stock
VMI           end-customer demand        echelon stock (Clark & Scarf)
============  =========================  ==============================

Sharing POS data removes the *forecast* cascade. It does not remove the
*physical* cascade: each echelon still holds and replenishes its own stock on its
own review cycle, so an order still has to travel up and goods still have to
travel down. That residual is what echelon-stock control removes, and it is the
reason the literature keeps finding that the organisational change (who decides)
matters more than the data feed (who knows).

References
----------
Lee, So & Tang (2000), "The value of information sharing in a two-level supply
chain", Management Science 46(5).
Cachon & Fisher (2000), "Supply chain inventory management and the value of
shared information", Management Science 46(8).
Disney & Towill (2003), "The effect of vendor managed inventory dynamics on the
bullwhip effect in supply chains", International Journal of Production
Economics 85(2).
Chen (1998), "Echelon reorder points, installation reorder points, and the value
of centralized demand information", Management Science 44(12).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .experiments import (
    DEFAULT_CONFIG,
    ScenarioOutcome,
    compare_scenarios,
    estimate_warmup,
    merge_config,
    percent_reduction,
    run_scenario,
)
from .metrics import ConfidenceInterval
from .network import InfoMode

__all__ = ["InformationComparison", "compare_information_modes"]


@dataclass
class InformationComparison:
    """Outcomes for each information mode plus paired deltas against a reference."""

    outcomes: Dict[str, ScenarioOutcome]
    reference: str
    warmup: int
    node_order: List[str]
    calibrated_to: Optional[float] = None
    safety_factors: Dict[str, float] = field(default_factory=dict)

    def bullwhip_by_mode(self) -> Dict[str, Dict[str, ConfidenceInterval]]:
        return {
            mode: {
                node: outcome.ratio_ci(f"var_orders:{node}", "var_demand")
                for node in self.node_order
            }
            for mode, outcome in self.outcomes.items()
        }

    def chain_bullwhip(self) -> Dict[str, ConfidenceInterval]:
        return {mode: o.ci("chain_bullwhip") for mode, o in self.outcomes.items()}

    def service(self) -> Dict[str, ConfidenceInterval]:
        return {mode: o.ci("service_fill_rate") for mode, o in self.outcomes.items()}

    def inventory(self) -> Dict[str, ConfidenceInterval]:
        return {mode: o.ci("avg_inventory") for mode, o in self.outcomes.items()}

    def cost(self) -> Dict[str, ConfidenceInterval]:
        return {mode: o.ci("avg_cost") for mode, o in self.outcomes.items()}

    def paired_delta(self, mode: str, metric: str) -> ConfidenceInterval:
        """Paired-t interval for ``mode - reference`` on ``metric``."""
        return compare_scenarios(self.outcomes[mode], self.outcomes[self.reference], metric)

    def paired_percent(self, mode: str, metric: str) -> ConfidenceInterval:
        return percent_reduction(self.outcomes[mode], self.outcomes[self.reference], metric)

    def table(self) -> List[Tuple[str, float, float, float, float, float]]:
        """``(mode, chain bullwhip, half width, retailer fill %, inventory, cost)``."""
        rows = []
        for mode, outcome in self.outcomes.items():
            chain = outcome.ci("chain_bullwhip")
            rows.append(
                (
                    mode,
                    chain.mean,
                    chain.half_width,
                    100.0 * outcome.ci("service_fill_rate").mean,
                    outcome.ci("avg_inventory").mean,
                    outcome.ci("avg_cost").mean,
                )
            )
        return rows


def compare_information_modes(
    base_config: Optional[Dict[str, Any]] = None,
    modes: Sequence[str] = (
        InfoMode.DECENTRALIZED.value,
        InfoMode.POS_SHARED.value,
        InfoMode.VMI.value,
    ),
    warmup: Optional[int] = None,
    calibrate_to: Optional[float] = None,
) -> InformationComparison:
    """Run the same chain under each information mode, under common random numbers.

    Every mode shares the seed and the warm-up truncation, so replication ``i``
    of "VMI" and replication ``i`` of "decentralised" face the identical demand
    path. Without that pairing the difference between the modes is comfortably
    inside the noise at any replication count a laptop will tolerate.

    ``calibrate_to`` decides which of the two honest comparisons is run, and the
    distinction is worth being explicit about because a lot of published
    information-sharing benefits quietly rely on the first:

    * ``None`` -- hold the safety factor ``z`` fixed. Every mode runs the same
      policy parameters, so the modes end up at *different* service levels and
      different inventory. Answers "what happens if I change the information
      flow and touch nothing else", which is the honest description of a pilot.
    * a fill-rate target -- bisect ``z`` per mode until each hits the same
      simulated fill rate, then compare inventory and amplification. Answers
      "what is this worth once the planners have re-tuned", which is the honest
      description of the steady state. Comparing inventory across modes at
      different service levels is not a comparison at all.
    """
    base = merge_config(DEFAULT_CONFIG, base_config or {})
    reference = modes[0]
    if warmup is None:
        warmup = estimate_warmup(merge_config(base, {"info_mode": reference}), pilots=3)

    outcomes: Dict[str, ScenarioOutcome] = {}
    safety_factors: Dict[str, float] = {}
    for mode in modes:
        config = merge_config(base, {"info_mode": mode})
        if calibrate_to is not None:
            from .tradeoffs import calibrate_service

            calibration = calibrate_service(
                config, target_fill=calibrate_to, warmup=warmup, tolerance=1e-3
            )
            safety_factors[mode] = calibration.z
            outcomes[mode] = calibration.outcome
            outcomes[mode].name = mode
        else:
            safety_factors[mode] = float(base["policy"]["z"])
            outcomes[mode] = run_scenario(config, name=mode, warmup=warmup, keep_results=True)

    node_order = [s.name for s in outcomes[reference].results[0].stocking_series()]
    return InformationComparison(
        outcomes=outcomes,
        reference=reference,
        warmup=int(warmup),
        node_order=node_order,
        calibrated_to=calibrate_to,
        safety_factors=safety_factors,
    )
