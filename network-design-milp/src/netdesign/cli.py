"""Command line interface: ``python -m netdesign <command>``.

Every command works from a generated instance by default, so the whole tool is
usable from a clean clone with no data files:

    python -m netdesign solve
    python -m netdesign solve --single-source --max-dcs 4
    python -m netdesign stochastic --scenarios 12
    python -m netdesign sweep
    python -m netdesign diagnose --demand-multiplier 1.8 --max-dcs 3
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .diagnostics import diagnose
from .greenfield import greenfield_vs_milp, snap_report, greenfield_open_set
from .instances import Instance, demand_scenarios, generate_instance
from .network_flow import NetworkDesignModel, NetworkOptions
from .reporting import format_report, markdown_table
from .scenarios import sensitivity_sweep, stability_profile, sweep_table
from .stochastic import solve_two_stage


def _load_instance(args: argparse.Namespace) -> Instance:
    if getattr(args, "instance", None):
        with open(args.instance, "r", encoding="utf-8") as fh:
            inst = Instance.from_dict(json.load(fh))
    else:
        inst = generate_instance(
            seed=args.seed,
            n_dcs=args.dcs,
            n_zones=args.zones,
            n_plants=args.plants,
            n_commodities=args.commodities,
        )
    dm = getattr(args, "demand_multiplier", 1.0)
    if dm and dm != 1.0:
        inst = inst.with_demand(inst.scaled_demand(float(dm)))
    return inst


def _options(args: argparse.Namespace) -> NetworkOptions:
    return NetworkOptions(
        single_source=getattr(args, "single_source", False),
        min_second_source_share=getattr(args, "dual_source", 0.0) or 0.0,
        max_open_dcs=getattr(args, "max_dcs", None),
        min_open_dcs=getattr(args, "min_dcs", None),
        allow_unmet=getattr(args, "allow_unmet", False),
    )


def _add_instance_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--instance", help="path to a JSON instance (default: generate one)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dcs", type=int, default=8, help="number of DC candidates")
    p.add_argument("--zones", type=int, default=25)
    p.add_argument("--plants", type=int, default=4)
    p.add_argument("--commodities", type=int, default=2)
    p.add_argument("--demand-multiplier", type=float, default=1.0)


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--single-source", action="store_true", help="one DC per zone")
    p.add_argument(
        "--dual-source",
        type=float,
        default=0.0,
        metavar="SHARE",
        help="minimum share of each zone's volume from a second DC, e.g. 0.25",
    )
    p.add_argument("--max-dcs", type=int, default=None)
    p.add_argument("--min-dcs", type=int, default=None)
    p.add_argument("--allow-unmet", action="store_true")
    p.add_argument("--time-limit", type=float, default=None)


def cmd_generate(args: argparse.Namespace) -> int:
    inst = _load_instance(args)
    payload = inst.to_json()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(payload)
        print(inst.summary())
        print(f"written to {args.out}")
    else:
        print(payload)
    return 0


def cmd_solve(args: argparse.Namespace) -> int:
    inst = _load_instance(args)
    builder = NetworkDesignModel(inst, _options(args))
    sol, _ = builder.solve(time_limit=args.time_limit)
    if args.json:
        print(
            json.dumps(
                {
                    "status": sol.status,
                    "objective": sol.objective,
                    "open_plants": sol.open_plants,
                    "open_dcs": sol.open_dcs,
                    "costs": sol.costs,
                    "demand_weighted_km": sol.demand_weighted_km,
                    "runtime_s": sol.runtime,
                },
                indent=1,
            )
        )
    else:
        print(format_report(sol, inst))
        if args.lp_bound:
            lp = builder.lp_bound()
            print(
                f"\nLP relaxation {lp:,.0f}  "
                f"(integrality gap {100 * (sol.objective - lp) / sol.objective:.2f}%)"
            )
    return 0 if sol.is_optimal else 1


def cmd_stochastic(args: argparse.Namespace) -> int:
    inst = _load_instance(args)
    scen, probs = demand_scenarios(inst, n_scenarios=args.scenarios, seed=args.scenario_seed)
    res = solve_two_stage(inst, scen, probs, _options(args), time_limit=args.time_limit)
    print(f"two-stage stochastic network design - {inst.name}")
    print(f"{args.scenarios} demand scenarios, equal probability\n")
    print(res.to_table())
    print()
    print(f"VSS  {res.vss_pct:5.2f}% of RP - what modelling the uncertainty is worth")
    print(f"EVPI {res.evpi_pct:5.2f}% of RP - the ceiling on forecast investment")
    print(f"\nRP network  {', '.join(res.rp_solution.open_dcs)}")
    print(f"EV network  {', '.join(res.ev_solution.open_dcs)}")
    print(f"wait-and-see solutions matching the RP network: {100 * res.network_stability:.0f}%")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    inst = _load_instance(args)
    runs = sensitivity_sweep(
        inst,
        demand_multipliers=args.demand,
        cost_multipliers=args.cost,
        options=_options(args),
        time_limit=args.time_limit,
    )
    headers, rows = sweep_table(runs)
    print(markdown_table(headers, rows))
    prof = stability_profile(runs)
    print()
    print(f"runs {prof['n_runs']}, distinct networks {prof['distinct_networks']}")
    print(f"core  : {', '.join(prof['core']) or '-'}")
    print(f"swing : {', '.join(prof['swing']) or '-'}")
    print(f"out   : {', '.join(prof['out']) or '-'}")
    return 0


def cmd_greenfield(args: argparse.Namespace) -> int:
    inst = _load_instance(args)
    rows = greenfield_vs_milp(inst, args.p, _options(args), time_limit=args.time_limit)
    print(snap_report(inst, greenfield_open_set(inst, args.p[0])))
    print()
    print(
        markdown_table(
            ["p", "greenfield DCs", "wtd km", "status", "cost", "gap vs MILP"],
            [
                [
                    r["p"],
                    " ".join(r["greenfield_dcs"]),
                    f"{r['weighted_km']:,.0f}",
                    r["status"],
                    f"{r['cost']:,.0f}" if r["cost"] else "-",
                    f"{r['gap_pct']:.2f}%" if r["gap_pct"] is not None else "-",
                ]
                for r in rows
            ],
        )
    )
    print(f"\nMILP optimum {rows[0]['milp_cost']:,.0f} with {' '.join(rows[0]['milp_dcs'])}")
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    inst = _load_instance(args)
    diag = diagnose(inst, _options(args), time_limit=args.time_limit)
    print(diag.summary())
    return 0 if diag.feasible else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="netdesign",
        description="supply chain network design and flow optimization with MILP",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="generate a synthetic instance as JSON")
    _add_instance_args(p)
    p.add_argument("--out", help="write JSON here instead of stdout")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("solve", help="solve the deterministic network design MILP")
    _add_instance_args(p)
    _add_model_args(p)
    p.add_argument("--json", action="store_true")
    p.add_argument("--lp-bound", action="store_true", help="also report the LP relaxation")
    p.set_defaults(func=cmd_solve)

    p = sub.add_parser("stochastic", help="two-stage stochastic design, VSS and EVPI")
    _add_instance_args(p)
    _add_model_args(p)
    p.add_argument("--scenarios", type=int, default=12)
    p.add_argument("--scenario-seed", type=int, default=11)
    p.set_defaults(func=cmd_stochastic)

    p = sub.add_parser("sweep", help="sensitivity sweep over demand and transport cost")
    _add_instance_args(p)
    _add_model_args(p)
    p.add_argument("--demand", type=float, nargs="+", default=[0.8, 0.9, 1.0, 1.1, 1.25])
    p.add_argument("--cost", type=float, nargs="+", default=[0.7, 1.0, 1.4])
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("greenfield", help="center-of-gravity heuristic vs the MILP")
    _add_instance_args(p)
    _add_model_args(p)
    p.add_argument("--p", type=int, nargs="+", default=[2, 3, 4, 5])
    p.set_defaults(func=cmd_greenfield)

    p = sub.add_parser("diagnose", help="feasibility diagnostics for an instance")
    _add_instance_args(p)
    _add_model_args(p)
    p.set_defaults(func=cmd_diagnose)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
