"""Lot sizing under a seasonal plan, and a newsvendor buy for a single season.

Run:  PYTHONPATH=src python examples/04_lot_sizing_and_newsvendor.py

Two decisions that share one idea: the cost of one unit too many versus one unit
too few. Wagner-Whitin answers it across a horizon with known demand; the
newsvendor answers it for a single period with unknown demand.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from invkit.distributions import EmpiricalLTD
from invkit.lotsizing import (
    PriceBreak,
    compare_lot_sizing,
    eoq,
    eoq_all_units_discount,
    eoq_cost_sensitivity,
    seasonal_demand_series,
)
from invkit.newsvendor import (
    critical_fractile_sensitivity,
    expected_newsvendor_cost,
    newsvendor_empirical,
    newsvendor_normal,
)


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def part_lot_sizing() -> None:
    rule("Wagner-Whitin vs the heuristics, on a seasonal 12-period plan")
    demand = seasonal_demand_series(12, base=120.0, amplitude=70.0, noise_cv=0.25, seed=5)
    setup, holding = 300.0, 1.0
    print("demand:", "  ".join(f"{d:6.1f}" for d in demand))
    print()
    plans = compare_lot_sizing(demand, setup, holding)
    opt = plans["wagner-whitin"]
    print(f"{'method':<18}{'orders':>8}{'setup':>10}{'holding':>10}{'total':>10}{'gap':>9}")
    print("-" * 65)
    for plan in plans.values():
        print(
            f"{plan.method:<18}{plan.n_setups:>8}{plan.setup_cost:>10,.0f}"
            f"{plan.holding_cost:>10,.0f}{plan.total_cost:>10,.0f}"
            f"{100 * plan.gap_vs(opt):>8.2f}%"
        )
    print()
    print("Order plan (Wagner-Whitin):", "  ".join(f"{q:6.0f}" for q in opt.orders))
    print("Order plan (Silver-Meal):  ",
          "  ".join(f"{q:6.0f}" for q in plans["silver-meal"].orders))
    print()
    print("Silver-Meal extends a block while average cost per period is still falling, so")
    print("it runs through a cheap period and then pays to carry the spike behind it. The")
    print("exact DP is O(T^2) and runs in microseconds; there has not been a computational")
    print("reason to use the heuristic since about 1975.")

    rule("EOQ, and why arguing about the ordering cost is a waste of a quarter")
    D, K, h = 100.0, 250.0, 25.0 * 0.25 / 365.0
    q = eoq(D, K, h)
    print(f"EOQ = {q:,.0f} units at K = {K:,.0f}")
    print()
    print(f"{'you order':>12}{'cost penalty':>15}")
    print("-" * 27)
    for ratio in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        print(f"{ratio * q:>11,.0f} {100 * eoq_cost_sensitivity(ratio):>13.1f}%")
    print()
    print("Being 25% off the optimal quantity costs 2.5% of the relevant cost. Round to the")
    print("pallet and move on.")

    rule("All-units quantity discounts")
    breaks = [PriceBreak(0, 10.00), PriceBreak(500, 9.60), PriceBreak(2000, 9.20)]
    r = eoq_all_units_discount(10_000.0, 400.0, 0.25, breaks)
    for b in breaks:
        print(f"  from {b.min_qty:>6,.0f} units: {b.price:>6.2f} per unit")
    print()
    print(f"optimal order quantity: {r.quantity:,.0f} at {r.unit_price:.2f} "
          f"-> total cost {r.total_cost:,.0f} per period")
    print(f"({r.breaks_evaluated} of {len(breaks)} tiers produced a feasible candidate)")


def part_newsvendor() -> None:
    rule("Newsvendor: a single seasonal buy against a skewed demand history")
    rng = np.random.default_rng(20260815)
    n = 40_000
    body = rng.lognormal(0.0, 0.35, int(0.9 * n))
    spike = rng.lognormal(1.0, 0.55, n - int(0.9 * n))
    raw = np.concatenate([body, spike])
    rng.shuffle(raw)
    sample = 500.0 + (raw - raw.mean()) * (250.0 / raw.std(ddof=1))
    sample = np.maximum(sample, 0.0)
    truth = EmpiricalLTD(sample)

    cu, co = 40.0, 4.0
    print(f"underage cost {cu:,.0f}, overage cost {co:,.0f} "
          f"-> critical fractile {cu / (cu + co):.4f}")
    print(f"demand sample: mean {sample.mean():,.0f}  sd {sample.std(ddof=1):,.0f}  "
          f"skewness {float(((sample - sample.mean()) ** 3).mean() / sample.std() ** 3):.2f}")
    print()
    emp = newsvendor_empirical(sample, cu, co)
    norm = newsvendor_normal(float(sample.mean()), float(sample.std(ddof=1)), cu, co)
    print(f"{'basis':<12}{'Q*':>10}{'cost vs real demand':>22}{'leftover':>12}{'shortage':>11}")
    print("-" * 68)
    for r in (norm, emp):
        cost, leftover, shortage = expected_newsvendor_cost(truth, r.order_quantity, cu, co)
        print(f"{r.basis:<12}{r.order_quantity:>10,.0f}{cost:>22,.0f}"
              f"{leftover:>12,.0f}{shortage:>11,.0f}")
    cost_n = expected_newsvendor_cost(truth, norm.order_quantity, cu, co)[0]
    cost_e = expected_newsvendor_cost(truth, emp.order_quantity, cu, co)[0]
    print()
    print(f"Cost penalty of assuming normal: {100 * (cost_n / cost_e - 1):+.2f}%")
    print("Both quantities are costed against the demand that actually shows up. Scoring")
    print("the normal solution under its own normal assumption is circular, and it is how")
    print("this error survives review.")

    rule("What the answer is really sensitive to")
    rows = critical_fractile_sensitivity(truth, cu, co, cu_multipliers=(0.25, 0.5, 1.0, 2.0, 4.0))
    print(f"{'Cu assumed':>12}{'fractile':>11}{'Q*':>10}{'true cost':>13}{'penalty':>10}")
    print("-" * 56)
    for r in rows:
        print(f"{r['cu_multiplier'] * cu:>12,.0f}{r['assumed_fractile']:>11.4f}"
              f"{r['order_quantity']:>10,.0f}{r['true_expected_cost']:>13,.0f}"
              f"{r['cost_penalty_pct']:>9.2f}%")
    print()
    print("The formula is trivial. The underage cost is not: get it wrong by 2x and the")
    print("implied fractile moves enough to matter. Most organisations set Cu to lost gross")
    print("margin, which ignores substitution, the customer who does not come back, and the")
    print("expedite freight that gets paid to avoid the stockout in the first place.")


def main() -> int:
    part_lot_sizing()
    part_newsvendor()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
