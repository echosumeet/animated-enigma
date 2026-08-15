"""Command line interface: ``python -m invkit <command>``.

Six subcommands, each mapping to one thing you would actually want to ask:

``safety-stock``   what a service target costs, read three different ways
``policy``         build a policy and validate it by simulation in one shot
``lotsize``        Wagner-Whitin vs the heuristics on a demand series
``newsvendor``     critical fractile, normal vs empirical
``serial``         Clark-Scarf echelon base-stock levels, checked by simulation
``place``          guaranteed-service safety stock placement on the example BOM
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .frontier import exchange_curve, marginal_cost_of_service
from .guaranteed_service import example_bom_tree, solve_guaranteed_service
from .leadtime import LeadTimeSpec, lead_time_variance_share, ltd_with_undershoot
from .lotsizing import compare_lot_sizing, eoq, seasonal_demand_series
from .newsvendor import newsvendor_empirical, newsvendor_normal
from .policies import SQPolicy, build_RS, build_sQ
from .pooling import square_root_law
from .safety_stock import compare_service_definitions, ss_from_cycle_service_level, ss_from_fill_rate
from .serial import SerialStage, clark_scarf, simulate_serial_system
from .simulation import DemandProcess, simulate_policy


def _print_rows(rows: list[dict], keys: list[str] | None = None) -> None:
    if not rows:
        print("(no rows)")
        return
    keys = keys or list(rows[0])
    widths = {k: max(len(k), *(len(_fmt(r.get(k))) for r in rows)) for k in keys}
    print("  ".join(k.ljust(widths[k]) for k in keys))
    print("  ".join("-" * widths[k] for k in keys))
    for r in rows:
        print("  ".join(_fmt(r.get(k)).ljust(widths[k]) for k in keys))


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:,.4f}" if abs(v) < 1000 else f"{v:,.1f}"
    return str(v)


def _lead_time_pmf(arg: str) -> dict[int, float]:
    """Parse ``5`` or ``4:0.6,9:0.4`` into a lead-time pmf."""
    if ":" not in arg:
        return {int(arg): 1.0}
    pmf: dict[int, float] = {}
    for part in arg.split(","):
        l, p = part.split(":")
        pmf[int(l)] = float(p)
    return pmf


def cmd_safety_stock(args: argparse.Namespace) -> int:
    spec = LeadTimeSpec(args.demand_mean, args.demand_sd, _lead_time_pmf(args.lead_time))
    ltd = ltd_with_undershoot(spec, args.demand_mean, args.demand_sd)
    print(f"lead-time demand: mean={ltd.mean:,.1f} sd={ltd.sd:,.1f} (undershoot-adjusted)")
    print(f"share of variance from lead-time variability: {lead_time_variance_share(spec):.1%}")
    print()
    rows = []
    for target in args.targets:
        c = compare_service_definitions(ltd, target, args.order_quantity)
        rows.append(
            {
                "target": c["target"],
                "SS (cycle service)": c["ss_csl_basis"],
                "SS (fill rate)": c["ss_fill_basis"],
                "difference": c["ss_delta"],
                "fill @ CSL basis": c["fill_at_csl_basis"],
                "CSL @ fill basis": c["csl_at_fill_basis"],
            }
        )
    _print_rows(rows)
    return 0


def cmd_policy(args: argparse.Namespace) -> int:
    spec = LeadTimeSpec(args.demand_mean, args.demand_sd, _lead_time_pmf(args.lead_time))
    proc = DemandProcess(args.demand_mean, args.demand_sd, "gamma")
    if args.kind == "sQ":
        ltd = ltd_with_undershoot(spec, args.demand_mean, args.demand_sd)
        if args.target_fill is not None:
            res = ss_from_fill_rate(ltd, args.target_fill, args.order_quantity)
        else:
            res = ss_from_cycle_service_level(ltd, args.target_csl, args.order_quantity)
        policy = SQPolicy(s=res.reorder_point, Q=args.order_quantity)
        print(f"(s, Q) policy: s={policy.s:,.1f}  Q={policy.Q:,.1f}")
    else:
        policy, res = build_RS(
            spec, args.review_period, target_csl=args.target_csl, target_fill=args.target_fill
        )
        print(f"(R, S) policy: R={policy.R}  S={policy.S:,.1f}")
    print(f"analytic:  CSL={res.cycle_service_level:.4f}  fill={res.fill_rate:.4f}  "
          f"SS={res.safety_stock:,.1f}")
    sim = simulate_policy(
        policy, proc, _lead_time_pmf(args.lead_time), n_periods=args.periods, warmup=1000
    )
    print(f"simulated: CSL={sim.cycle_service_level:.4f}  fill={sim.fill_rate:.4f}  "
          f"ready={sim.ready_rate:.4f}  avg on hand={sim.avg_on_hand:,.1f}")
    return 0


def cmd_lotsize(args: argparse.Namespace) -> int:
    if args.demand:
        demand = [float(x) for x in args.demand.split(",")]
    else:
        demand = seasonal_demand_series(args.periods, 120.0, 70.0, noise_cv=0.25, seed=5)
    print("demand:", ", ".join(f"{d:g}" for d in demand))
    plans = compare_lot_sizing(demand, args.setup_cost, args.holding_cost)
    opt = plans["wagner-whitin"]
    rows = [
        {
            "method": p.method,
            "setups": float(p.n_setups),
            "setup cost": p.setup_cost,
            "holding cost": p.holding_cost,
            "total": p.total_cost,
            "gap vs optimal %": 100.0 * p.gap_vs(opt),
        }
        for p in plans.values()
    ]
    _print_rows(rows)
    print(f"\nEOQ on average demand: {eoq(float(np.mean(demand)), args.setup_cost, args.holding_cost):,.1f}")
    return 0


def cmd_newsvendor(args: argparse.Namespace) -> int:
    """Both order quantities are costed against the *empirical* demand.

    Costing the normal solution under its own normal assumption is circular and
    always makes it look fine.  The honest comparison evaluates both quantities
    against the demand that actually shows up.
    """
    from .distributions import EmpiricalLTD
    from .newsvendor import expected_newsvendor_cost

    rng = np.random.default_rng(args.seed)
    sample = skewed_demand_sample(
        args.demand_mean, args.demand_sd, args.sample_size, rng
    )
    truth = EmpiricalLTD(sample)
    normal = newsvendor_normal(float(sample.mean()), float(sample.std(ddof=1)),
                               args.underage, args.overage)
    empirical = newsvendor_empirical(sample, args.underage, args.overage)

    rows = []
    for r in (normal, empirical):
        cost, leftover, shortage = expected_newsvendor_cost(
            truth, r.order_quantity, args.underage, args.overage
        )
        rows.append(
            {
                "basis": r.basis,
                "critical fractile": r.critical_fractile,
                "Q*": r.order_quantity,
                "cost vs real demand": cost,
                "E[leftover]": leftover,
                "E[shortage]": shortage,
            }
        )
    _print_rows(rows)
    gap = 100.0 * (empirical.order_quantity / normal.order_quantity - 1.0)
    penalty = 100.0 * (rows[0]["cost vs real demand"] / rows[1]["cost vs real demand"] - 1.0)
    print(f"\nsample skewness: {float(((sample - sample.mean()) ** 3).mean() / sample.std() ** 3):.2f}")
    print(f"empirical Q* is {gap:+.1f}% versus the normal assumption")
    print(f"cost penalty of assuming normal: {penalty:+.2f}%")
    return 0


def skewed_demand_sample(
    mean: float, sd: float, size: int, rng: np.random.Generator
) -> np.ndarray:
    """A right-skewed, fat-tailed demand sample with the requested moments.

    A 90/10 mixture of a tight lognormal base and a promotion-driven spike.  This
    is the shape that breaks a normal fit: the body looks fine, the upper decile
    does not, and safety stock is entirely a statement about the upper decile.
    """
    n_spike = int(0.1 * size)
    n_base = size - n_spike
    base = rng.lognormal(mean=0.0, sigma=0.35, size=n_base)
    spike = rng.lognormal(mean=1.0, sigma=0.55, size=n_spike)
    raw = np.concatenate([base, spike])
    rng.shuffle(raw)
    return mean + (raw - raw.mean()) * (sd / raw.std(ddof=1))


def cmd_serial(args: argparse.Namespace) -> int:
    stages = [
        SerialStage("store", 1, 1.2),
        SerialStage("dc", 2, 0.5),
        SerialStage("plant", 4, 0.4),
    ]
    sol = clark_scarf(stages, args.demand_mean, args.demand_sd, penalty=args.penalty)
    _print_rows(sol.summary())
    print(f"\noptimal expected cost per period: {sol.optimal_cost:,.3f}")
    sim = simulate_serial_system(
        stages, sol.echelon_base_stock, args.demand_mean, args.demand_sd,
        args.penalty, n_periods=args.periods,
    )
    print(f"simulated cost per period:        {sim['avg_cost_per_period']:,.3f} "
          f"({100 * (sim['avg_cost_per_period'] / sol.optimal_cost - 1):+.2f}%)")
    print(f"simulated fill rate: {sim['fill_rate']:.4f}")
    return 0


def cmd_place(args: argparse.Namespace) -> int:
    tree = example_bom_tree()
    res = solve_guaranteed_service(tree, args.service_level, args.holding_rate)
    _print_rows(res.table())
    print(f"\ntotal safety stock holding cost per period: {res.total_cost:,.2f}")
    print(f"decoupling points (tau > 0): {', '.join(res.decoupling_points)}")
    zero = [k for k, v in res.net_replenishment_times.items() if v == 0]
    print(f"stages holding no safety stock: {', '.join(sorted(zero))}")
    return 0


def cmd_frontier(args: argparse.Namespace) -> int:
    spec = LeadTimeSpec(args.demand_mean, args.demand_sd, _lead_time_pmf(args.lead_time))
    ltd = ltd_with_undershoot(spec, args.demand_mean, args.demand_sd)
    curves = exchange_curve(ltd, args.order_quantity, args.unit_cost, args.holding_rate, n_points=12)
    rows = [
        {
            "target": p.target,
            "basis": p.basis,
            "SS": p.safety_stock,
            "achieved fill": p.achieved_fill,
            "holding cost": p.holding_cost,
        }
        for p in curves["fill"]
    ]
    _print_rows(rows)
    print()
    _print_rows(marginal_cost_of_service(curves["fill"])[-5:])
    return 0


def cmd_pooling(args: argparse.Namespace) -> int:
    rows = []
    for rho in (0.0, 0.2, 0.5, 0.8):
        r = square_root_law(args.n_locations, args.demand_sd, args.service_level, correlation=rho)
        rows.append(
            {
                "correlation": rho,
                "decentralised SS": r.decentralised_ss,
                "centralised SS": r.centralised_ss,
                "reduction %": r.reduction_pct,
                "effective sqrt(n)": r.effective_sqrt_n,
            }
        )
    _print_rows(rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="invkit", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    def add_demand(sp, lead_default="5"):
        sp.add_argument("--demand-mean", type=float, default=100.0)
        sp.add_argument("--demand-sd", type=float, default=30.0)
        sp.add_argument("--lead-time", default=lead_default,
                        help="periods, or a pmf like '4:0.6,9:0.4'")

    sp = sub.add_parser("safety-stock", help="CSL vs fill-rate safety stock")
    add_demand(sp)
    sp.add_argument("--order-quantity", type=float, default=800.0)
    sp.add_argument("--targets", type=float, nargs="+", default=[0.90, 0.95, 0.98, 0.99])
    sp.set_defaults(func=cmd_safety_stock)

    sp = sub.add_parser("policy", help="build a policy and validate it by simulation")
    add_demand(sp)
    sp.add_argument("--kind", choices=["sQ", "RS"], default="sQ")
    sp.add_argument("--order-quantity", type=float, default=800.0)
    sp.add_argument("--review-period", type=int, default=5)
    sp.add_argument("--target-csl", type=float, default=0.95)
    sp.add_argument("--target-fill", type=float, default=None)
    sp.add_argument("--periods", type=int, default=60000)
    sp.set_defaults(func=cmd_policy)

    sp = sub.add_parser("lotsize", help="Wagner-Whitin vs heuristics")
    sp.add_argument("--demand", default=None, help="comma-separated demand series")
    sp.add_argument("--periods", type=int, default=12)
    sp.add_argument("--setup-cost", type=float, default=300.0)
    sp.add_argument("--holding-cost", type=float, default=1.0)
    sp.set_defaults(func=cmd_lotsize)

    sp = sub.add_parser("newsvendor", help="critical fractile, normal vs empirical")
    sp.add_argument("--demand-mean", type=float, default=500.0)
    sp.add_argument("--demand-sd", type=float, default=250.0)
    sp.add_argument("--underage", type=float, default=18.0)
    sp.add_argument("--overage", type=float, default=4.0)
    sp.add_argument("--sample-size", type=int, default=5000)
    sp.add_argument("--seed", type=int, default=99)
    sp.set_defaults(func=cmd_newsvendor)

    sp = sub.add_parser("serial", help="Clark-Scarf echelon base-stock levels")
    sp.add_argument("--demand-mean", type=float, default=100.0)
    sp.add_argument("--demand-sd", type=float, default=30.0)
    sp.add_argument("--penalty", type=float, default=20.0)
    sp.add_argument("--periods", type=int, default=40000)
    sp.set_defaults(func=cmd_serial)

    sp = sub.add_parser("place", help="guaranteed-service placement on the example BOM")
    sp.add_argument("--service-level", type=float, default=0.95)
    sp.add_argument("--holding-rate", type=float, default=0.25)
    sp.set_defaults(func=cmd_place)

    sp = sub.add_parser("frontier", help="cost-service efficient frontier")
    add_demand(sp)
    sp.add_argument("--order-quantity", type=float, default=800.0)
    sp.add_argument("--unit-cost", type=float, default=25.0)
    sp.add_argument("--holding-rate", type=float, default=0.25 / 365)
    sp.set_defaults(func=cmd_frontier)

    sp = sub.add_parser("pooling", help="square-root law with correlation")
    sp.add_argument("--n-locations", type=int, default=8)
    sp.add_argument("--demand-sd", type=float, default=30.0)
    sp.add_argument("--service-level", type=float, default=0.95)
    sp.set_defaults(func=cmd_pooling)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
