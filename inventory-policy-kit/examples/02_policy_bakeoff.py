"""Four policies, one item, one simulator: what you actually give up by choosing.

Run:  PYTHONPATH=src python examples/02_policy_bakeoff.py

Every policy is sized to the same 98% fill-rate target and then run through the
same Monte Carlo evaluator, with holding, ordering and backorder costs charged.
The comparison is only meaningful because the sizing and the evaluation are
independent: nothing here tunes a policy to the simulator that scores it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invkit.leadtime import LeadTimeSpec, ltd_with_undershoot
from invkit.lotsizing import eoq
from invkit.policies import RsSPolicy, SQPolicy, SSPolicy, build_RS, build_RsS
from invkit.safety_stock import ss_from_fill_rate
from invkit.simulation import DemandProcess, simulate_policy

DEMAND_MEAN, DEMAND_SD, LEAD_TIME = 100.0, 30.0, 4
UNIT_COST, HOLDING_RATE_ANNUAL, ORDER_COST = 25.0, 0.25, 250.0
HOLDING_PER_DAY = UNIT_COST * HOLDING_RATE_ANNUAL / 365.0
BACKORDER_COST = 2.0
TARGET_FILL = 0.98
N_PERIODS = 150_000


def main() -> int:
    spec = LeadTimeSpec.deterministic(DEMAND_MEAN, DEMAND_SD, LEAD_TIME)
    proc = DemandProcess(DEMAND_MEAN, DEMAND_SD, "gamma")
    q_eoq = eoq(DEMAND_MEAN, ORDER_COST, HOLDING_PER_DAY)
    print(f"EOQ = {q_eoq:,.0f} units ({q_eoq / DEMAND_MEAN:.1f} days of demand), "
          f"lead time {LEAD_TIME} days, fill-rate target {TARGET_FILL:.0%}")
    print()

    ltd_cont = ltd_with_undershoot(spec, DEMAND_MEAN, DEMAND_SD)
    res = ss_from_fill_rate(ltd_cont, TARGET_FILL, q_eoq)
    policies: list[tuple[str, object, float]] = [
        ("(s,Q) continuous", SQPolicy(s=res.reorder_point, Q=q_eoq), res.safety_stock),
        (
            "(s,S) continuous",
            SSPolicy(s=res.reorder_point, S=res.reorder_point + q_eoq),
            res.safety_stock,
        ),
    ]
    rs, rs_res = build_RS(spec, R=7, target_fill=TARGET_FILL)
    policies.append(("(R,S) weekly review", rs, rs_res.safety_stock))
    rss, rss_res = build_RsS(spec, R=7, Q=q_eoq, target_fill=TARGET_FILL)
    policies.append(("(R,s,S) weekly review", rss, rss_res.safety_stock))

    header = (
        f"{'policy':<24}{'parameters':<26}{'SS':>8}{'fill':>7}{'CSL':>8}"
        f"{'on hand':>10}{'orders':>9}{'cost/day':>11}"
    )
    print(header)
    print("-" * len(header))
    for label, policy, safety_stock in policies:
        sim = simulate_policy(
            policy, proc, {LEAD_TIME: 1.0}, n_periods=N_PERIODS, warmup=2_000, seed=31,
            holding_cost=HOLDING_PER_DAY, order_cost=ORDER_COST,
            shortage_cost=BACKORDER_COST,
        )
        if isinstance(policy, SQPolicy):
            params = f"s={policy.s:,.0f} Q={policy.Q:,.0f}"
        elif isinstance(policy, SSPolicy):
            params = f"s={policy.s:,.0f} S={policy.S:,.0f}"
        elif isinstance(policy, RsSPolicy):
            params = f"R={policy.R} s={policy.s:,.0f} S={policy.S:,.0f}"
        else:
            params = f"R={policy.R} S={policy.S:,.0f}"
        print(
            f"{label:<24}{params:<26}{safety_stock:>8,.0f}"
            f"{sim.fill_rate:>7.4f}{sim.cycle_service_level:>8.4f}"
            f"{sim.avg_on_hand:>10,.0f}{sim.n_orders:>9,}"
            f"{sim.total_cost / sim.periods:>11,.2f}"
        )

    print()
    print("Reading it:")
    print("- (s,Q) and (s,S) are nearly identical here because demand is smooth relative to")
    print("  the lot. The gap opens up when transactions are lumpy and (s,Q) leaves the")
    print("  position sitting below s after a single large order.")
    print("- (R,S) needs far more *safety* stock than (s,Q) at the same fill rate, because")
    print("  its protection interval is R + L = 11 days rather than L = 4. Its total on-hand")
    print("  is still lower, because a weekly cycle is shorter than an EOQ cycle - but it")
    print("  places three times as many orders and ends up the most expensive option here.")
    print("  Trading cycle stock for ordering cost is the decision a review calendar makes")
    print("  on your behalf, usually without anyone pricing it.")
    print("- (R,s,S) with S = s + EOQ suppresses the small top-ups - a third of the orders -")
    print("  but the order-up-to level then covers 17 days on top of an 11-day protection")
    print("  interval, so it badly over-serves. That is the honest failure mode of the")
    print("  pragmatic construction: sizing s from a service target and S from an EOQ makes")
    print("  no joint statement about cost. A cost-based (s,S) calculation is the fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
