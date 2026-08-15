"""Command line entry point: ``python -m slotting ...``.

Three subcommands, all of which run end to end with no network and no data
files, because every input is generated from a seed.

    python -m slotting describe                  # the generated instance
    python -m slotting slot --policy velocity    # slot it and report travel
    python -m slotting route --order 3           # route one order, every policy
"""

from __future__ import annotations

import argparse

import numpy as np

from .benchmark import build_instance, evaluate_travel
from .routing import ROUTERS
from .velocity import abc_summary, class_based_slotting, velocity_slotting
from .affinity import affinity_slotting
from .assignment import random_assignment

POLICIES = ("random", "velocity", "cube", "coi", "class", "affinity")


def _slot(instance, policy: str):
    con, rate = instance.constraints, instance.fit_rate
    if policy == "random":
        return random_assignment(con, seed=0)
    if policy == "velocity":
        return velocity_slotting(con, metric="picks", pick_rate=rate)
    if policy == "cube":
        return velocity_slotting(con, metric="cube")
    if policy == "coi":
        return velocity_slotting(con, metric="coi")
    if policy == "class":
        return class_based_slotting(con, metric="picks", pick_rate=rate)
    if policy == "affinity":
        return affinity_slotting(con, instance.fit_stream, method="greedy", pick_rate=rate)[0]
    raise ValueError(f"unknown slotting policy {policy!r}; have {list(POLICIES)}")


def cmd_describe(args: argparse.Namespace) -> int:
    inst = build_instance(seed=args.seed)
    for k, v in inst.describe().items():
        print(f"{k:26s} {v:,.3f}" if isinstance(v, float) else f"{k:26s} {v:,}")
    for k, v in inst.stream.lift_summary(inst.catalog).items():
        print(f"{k:26s} {v:,.3f}")
    return 0


def cmd_slot(args: argparse.Namespace) -> int:
    inst = build_instance(seed=args.seed)
    baseline = random_assignment(inst.constraints, seed=0)
    assignment = _slot(inst, args.policy)
    violations = inst.constraints.violations(assignment)
    print(f"policy               {args.policy}")
    print(f"constraint violations {len(violations)}")

    base = evaluate_travel(inst, baseline, policies=(args.router,))[args.router]
    got = evaluate_travel(inst, assignment, policies=(args.router,))[args.router]
    n = len(inst.score_stream)
    print(f"router               {args.router}")
    print(f"baseline travel      {base / 1000:8.1f} km   ({base / n:6.1f} m/order)")
    print(f"slotted travel       {got / 1000:8.1f} km   ({got / n:6.1f} m/order)")
    print(f"reduction            {100 * (1 - got / base):8.1f}%")

    print("\nABC classes, and where each class ended up:")
    summary = abc_summary(inst.warehouse, inst.catalog, assignment, "picks")
    print(f"  {'class':6s}{'SKUs':>7s}{'pick share':>13s}{'cube share':>13s}{'mean depot m':>15s}")
    for cls, count, pick, cube, dist in summary.as_rows():
        print(f"  {cls:6s}{count:7,d}{100 * pick:12.1f}%{100 * cube:12.1f}%{dist:15.1f}")
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    inst = build_instance(seed=args.seed)
    assignment = _slot(inst, args.policy)
    order = inst.score_stream[args.order]
    picks = sorted({int(assignment.location_of[s]) for s in order.lines})
    print(f"order {order.order_id}: {order.n_lines} lines, {len(picks)} distinct pick faces")
    print(f"aisles visited: {sorted({inst.warehouse.locations[p].aisle for p in picks})}\n")

    from .routing import exact_aisle_dp

    opt = exact_aisle_dp(inst.warehouse, picks)
    print(f"  {'router':22s}{'metres':>10s}{'gap':>10s}")
    for name in ("s_shape", "return", "midpoint", "largest_gap", "nearest_neighbour", "two_opt"):
        d = ROUTERS[name](inst.warehouse, picks).distance
        gap = f"{100 * (d / opt - 1):.2f}%" if opt > 0 else "-"
        print(f"  {name:22s}{d:10.1f}{gap:>10s}")
    print(f"  {'exact (aisle DP)':22s}{opt:10.1f}{'0.00%':>10s}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="slotting", description=__doc__)
    p.add_argument("--seed", type=int, default=7)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("describe", help="print the generated instance").set_defaults(fn=cmd_describe)

    s = sub.add_parser("slot", help="slot the warehouse and report travel")
    s.add_argument("--policy", choices=POLICIES, default="velocity")
    s.add_argument("--router", choices=sorted(ROUTERS), default="two_opt")
    s.set_defaults(fn=cmd_slot)

    r = sub.add_parser("route", help="route a single order under every heuristic")
    r.add_argument("--policy", choices=POLICIES, default="velocity")
    r.add_argument("--order", type=int, default=0, help="index into the held-out stream")
    r.set_defaults(fn=cmd_route)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
