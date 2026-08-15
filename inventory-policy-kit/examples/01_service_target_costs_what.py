"""What a "95% service target" actually costs, depending on what you meant by it.

Run:  PYTHONPATH=src python examples/01_service_target_costs_what.py

The scenario is one item: 100 units/day mean demand, 30 units/day standard
deviation, 5-day lead time, 800-unit lot, 25 per unit, 25% annual holding rate.
Nothing here is unusual - that is the point.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from invkit.leadtime import LeadTimeSpec, ltd_with_undershoot, undershoot_moments
from invkit.policies import SQPolicy
from invkit.safety_stock import (
    compare_service_definitions,
    ss_from_cycle_service_level,
    ss_from_empirical_quantile,
    ss_from_fill_rate,
)
from invkit.simulation import DemandProcess, simulate_policy

DEMAND_MEAN, DEMAND_SD, LEAD_TIME = 100.0, 30.0, 5
Q = 800.0
UNIT_COST, HOLDING_RATE_ANNUAL = 25.0, 0.25


def rule(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main() -> int:
    spec = LeadTimeSpec.deterministic(DEMAND_MEAN, DEMAND_SD, LEAD_TIME)
    eu, vu = undershoot_moments(DEMAND_MEAN, DEMAND_SD)
    ltd = ltd_with_undershoot(spec, DEMAND_MEAN, DEMAND_SD)

    rule("The demand the reorder point has to cover")
    print(f"lead-time demand              mean {DEMAND_MEAN * LEAD_TIME:>8,.1f}   "
          f"sd {DEMAND_SD * np.sqrt(LEAD_TIME):>7,.1f}")
    print(f"expected undershoot           mean {eu:>8,.1f}   sd {np.sqrt(vu):>7,.1f}")
    print(f"what s must actually cover    mean {ltd.mean:>8,.1f}   sd {ltd.sd:>7,.1f}")
    print()
    print("The undershoot term is what a continuous-review policy loses by reacting to a")
    print("transaction that crosses the reorder point rather than to the point itself.")
    print("Most systems omit it. It is worth 15 points of cycle service on this item.")

    rule("Reading the same 95% target two ways")
    c = compare_service_definitions(ltd, 0.95, Q)
    print(f"as cycle service level:  s = {ss_from_cycle_service_level(ltd, 0.95, Q).reorder_point:,.0f}"
          f"   safety stock {c['ss_csl_basis']:>7,.0f}")
    print(f"as fill rate:            s = {ss_from_fill_rate(ltd, 0.95, Q).reorder_point:,.0f}"
          f"   safety stock {c['ss_fill_basis']:>7,.0f}")
    print()
    print(f"difference                              {c['ss_delta']:>7,.0f} units "
          f"= {c['ss_delta'] * UNIT_COST:,.0f} of working capital")
    print(f"fill rate delivered by the CSL reading: {c['fill_at_csl_basis']:.4f}")
    print(f"cycle service delivered by the fill reading: {c['csl_at_fill_basis']:.3f}")
    print()
    print("Both are defensible. Only one was chosen. If the business measures fill rate -")
    print("and it almost always does - the CSL reading is 149 units of stock bought by a")
    print("definition, and the extra service it delivers is not visible to any customer.")

    rule("Neither number is worth anything until it survives a simulation")
    proc = DemandProcess(DEMAND_MEAN, DEMAND_SD, "gamma")
    for label, res in (
        ("sized on cycle service 0.95", ss_from_cycle_service_level(ltd, 0.95, Q)),
        ("sized on fill rate 0.95", ss_from_fill_rate(ltd, 0.95, Q)),
    ):
        sim = simulate_policy(
            SQPolicy(s=res.reorder_point, Q=Q), proc, {LEAD_TIME: 1.0},
            n_periods=120_000, warmup=2_000, seed=7,
            holding_cost=UNIT_COST * HOLDING_RATE_ANNUAL / 365.0,
        )
        print(f"{label:<30} simulated CSL {sim.cycle_service_level:.4f}  "
              f"fill {sim.fill_rate:.4f}  avg on hand {sim.avg_on_hand:,.0f}")

    rule("And the normal assumption is a third source of error")
    rng = np.random.default_rng(20260815)
    n = 60_000
    base = rng.lognormal(0.0, 0.35, int(0.9 * n))
    spike = rng.lognormal(1.0, 0.55, n - int(0.9 * n))
    raw = np.concatenate([base, spike])
    rng.shuffle(raw)
    errors = (raw - raw.mean()) / raw.std(ddof=1) * DEMAND_SD
    print(f"forecast-error sample: skewness "
          f"{float(((errors - errors.mean()) ** 3).mean() / errors.std() ** 3):.2f}")
    print()
    print(f"{'target':>8} {'empirical SS':>14} {'normal SS':>12} {'gap':>8}")
    for target in (0.95, 0.98, 0.99):
        emp, norm = ss_from_empirical_quantile(errors, target, LEAD_TIME)
        print(f"{target:>8.2f} {emp:>14,.0f} {norm:>12,.0f} {100 * (emp / norm - 1):>7.0f}%")
    print()
    print("Three independent modelling choices - service definition, undershoot, error")
    print("distribution - each worth tens of percent of the buffer, and none of them is")
    print("the demand forecast everybody spends the quarter arguing about.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
