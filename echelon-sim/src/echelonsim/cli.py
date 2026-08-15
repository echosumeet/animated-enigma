"""Command line interface: ``python -m echelonsim <command>``.

Every command takes the same optional ``--config`` JSON file, merged over
:data:`echelonsim.experiments.DEFAULT_CONFIG`, plus a few common overrides so
the usual "run it again with more replications" loop does not need a file at
all. ``--json`` emits machine-readable output for the benchmark harness.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from .bullwhip import decompose_bullwhip, measure_by_echelon, smoothing_sweep
from .disruption import run_disruption_study
from .experiments import (
    DEFAULT_CONFIG,
    estimate_warmup,
    load_config,
    merge_config,
    run_scenario,
)
from .information import compare_information_modes
from .tradeoffs import lead_time_grid

__all__ = ["main", "build_parser"]


def _config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_config(args.config) if args.config else dict(DEFAULT_CONFIG)
    override: Dict[str, Any] = {"run": {}}
    if args.periods is not None:
        override["run"]["periods"] = args.periods
    if args.replications is not None:
        override["run"]["replications"] = args.replications
    if args.seed is not None:
        override["run"]["seed"] = args.seed
    if args.alpha is not None:
        override["forecast"] = {"kind": "exponential", "alpha": args.alpha}
    if args.levels is not None:
        override["topology"] = {"levels": args.levels}
    return merge_config(config, override)


def _emit(payload: Dict[str, Any], lines: Sequence[str], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=float))
    else:
        for line in lines:
            print(line)


def cmd_run(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    outcome = run_scenario(config, name="run", keep_results=True)
    result = outcome.results[0]
    lines = [
        f"periods {result.periods}  replications {outcome.replications}  "
        f"warm-up truncated {outcome.warmup}",
        f"{'node':<14}{'bullwhip':>10}{'fill %':>9}{'on hand':>10}{'backlog':>10}",
    ]
    payload: Dict[str, Any] = {"warmup": outcome.warmup, "nodes": {}}
    for series in result.stocking_series():
        bullwhip = outcome.ci(f"bullwhip:{series.name}").mean
        fill = outcome.ci(f"fill_rate:{series.name}").mean
        on_hand = outcome.ci(f"on_hand:{series.name}").mean
        backlog = outcome.ci(f"backlog:{series.name}").mean
        lines.append(
            f"{series.name:<14}{bullwhip:>10.3f}{100 * fill:>9.2f}{on_hand:>10.1f}{backlog:>10.2f}"
        )
        payload["nodes"][series.name] = {
            "bullwhip": bullwhip, "fill_rate": fill,
            "on_hand": on_hand, "backlog": backlog,
        }
    cost = outcome.ci("avg_cost")
    payload["avg_cost"] = cost.mean
    lines.append(f"average cost per period {cost.format(1)}")
    _emit(payload, lines, args.json)
    return 0


def cmd_bullwhip(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    amplification = measure_by_echelon(config)
    lines = [f"{'echelon':<14}{'local':>18}{'vs end demand':>22}"]
    payload = {"warmup": amplification.scenario.warmup, "echelons": {}}
    for name, local, local_hw, cumulative, cumulative_hw in amplification.table():
        lines.append(
            f"{name:<14}{local:>11.3f} +/-{local_hw:>5.3f}"
            f"{cumulative:>15.3f} +/-{cumulative_hw:>5.3f}"
        )
        payload["echelons"][name] = {
            "local": local, "local_half_width": local_hw,
            "cumulative": cumulative, "cumulative_half_width": cumulative_hw,
        }
    _emit(payload, lines, args.json)
    return 0


def cmd_decompose(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    result = decompose_bullwhip(config)
    result.check_additivity()
    lines = [
        f"amplification with all mechanisms off: {result.baseline.mean:.3f}",
        f"amplification with all mechanisms on:  {result.full.mean:.2f}",
        f"{'mechanism':<12}{'log contribution':>20}{'x factor':>12}{'share':>10}",
    ]
    payload: Dict[str, Any] = {
        "baseline": result.baseline.mean,
        "full": result.full.mean,
        "mechanisms": {},
        "cells": {"+".join(k) or "none": v for k, v in result.cell_means.items()},
    }
    for key, contribution, half_width, multiplier, share in result.table():
        lines.append(
            f"{key:<12}{contribution:>13.4f} +/-{half_width:>5.4f}"
            f"{multiplier:>12.3f}{share:>9.1f}%"
        )
        payload["mechanisms"][key] = {
            "log_contribution": contribution, "half_width": half_width,
            "multiplier": multiplier, "share_percent": share,
        }
    _emit(payload, lines, args.json)
    return 0


def cmd_information(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    comparison = compare_information_modes(config, calibrate_to=args.target_fill)
    header = f"{'mode':<16}{'chain bullwhip':>18}{'fill %':>9}{'inventory':>12}{'cost':>10}"
    lines = [header]
    payload: Dict[str, Any] = {"warmup": comparison.warmup, "modes": {}}
    for mode, chain, half_width, fill, inventory, cost in comparison.table():
        lines.append(
            f"{mode:<16}{chain:>11.2f} +/-{half_width:>5.2f}{fill:>9.2f}"
            f"{inventory:>12.1f}{cost:>10.1f}"
        )
        payload["modes"][mode] = {
            "chain_bullwhip": chain, "half_width": half_width,
            "fill_rate_percent": fill, "inventory": inventory, "cost": cost,
            "z": comparison.safety_factors.get(mode),
        }
    for mode in comparison.outcomes:
        if mode == comparison.reference:
            continue
        delta = comparison.paired_percent(mode, "chain_bullwhip")
        lines.append(f"  {mode} vs {comparison.reference}: chain bullwhip {delta.format(1)}%")
        payload["modes"][mode]["chain_bullwhip_percent_change"] = delta.mean
    _emit(payload, lines, args.json)
    return 0


def cmd_leadtime(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    cells = lead_time_grid(target_fill=args.target_fill or 0.95, base_config=config)
    lines = [f"{'mean L':>8}{'CV L':>8}{'z':>8}{'fill %':>9}{'inventory':>12}{'sigma_DL':>11}"]
    payload = {"target_fill": args.target_fill or 0.95, "cells": []}
    for cell in cells:
        mean_lead, cv_lead, z, fill, inventory, sigma = cell.row()
        lines.append(
            f"{mean_lead:>8.1f}{cv_lead:>8.2f}{z:>8.3f}{fill:>9.2f}"
            f"{inventory:>12.1f}{sigma:>11.1f}"
        )
        payload["cells"].append({
            "mean_lead": mean_lead, "cv_lead": cv_lead, "z": z,
            "fill_rate_percent": fill, "inventory": inventory, "sigma_dl": sigma,
        })
    _emit(payload, lines, args.json)
    return 0


def cmd_disrupt(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    config = merge_config(config, {"topology": {"capacity": args.capacity}})
    start = args.start
    scenarios = [
        (
            f"supplier outage {args.duration}p",
            {"disruptions": {"outages": [
                {"node": "source", "start": start, "duration": args.duration}]}},
            start,
            args.duration,
        ),
        (
            f"demand shock x{args.shock:g} {args.duration}p",
            {"demand": {"shock": {
                "start": start, "duration": args.duration, "multiplier": args.shock}}},
            start,
            args.duration,
        ),
        (
            f"factory capacity -50% {args.duration}p",
            {"disruptions": {"capacity_losses": [
                {"node": "factory", "start": start,
                 "duration": args.duration, "factor": 0.5}]}},
            start,
            args.duration,
        ),
    ]
    study = run_disruption_study(config, scenarios)
    lines = [
        f"{'scenario':<28}{'len':>5}{'trough %':>10}{'at':>5}"
        f"{'recover':>9}{'ratio':>8}{'lost units':>12}"
    ]
    payload: Dict[str, Any] = {"warmup": study.warmup, "scenarios": []}
    for profile in study.profiles:
        name, duration, trough, trough_at, recovery, ratio, lost = profile.row()
        lines.append(
            f"{name:<28}{duration:>5d}{trough:>10.1f}{trough_at:>5d}"
            f"{recovery:>9.1f}{ratio:>8.2f}{lost:>12.1f}"
        )
        payload["scenarios"].append({
            "name": name, "duration": duration, "trough_percent": trough,
            "trough_offset": trough_at, "recovery_periods": recovery,
            "recovery_ratio": ratio, "lost_units": lost,
            "censored_fraction": profile.censored_fraction,
        })
    _emit(payload, lines, args.json)
    return 0


def cmd_warmup(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    truncation = estimate_warmup(config, pilots=args.pilots)
    _emit(
        {"warmup": truncation, "pilots": args.pilots},
        [f"MSER-5 truncation point over {args.pilots} pilot replications: "
         f"{truncation} periods"],
        args.json,
    )
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    rows = smoothing_sweep(base_config=config)
    lines = [f"{'alpha':>8}{'simulated (retailer)':>24}{'analytic':>12}"]
    payload = {"rows": []}
    for alpha, interval, analytic in rows:
        lines.append(f"{alpha:>8.2f}{interval.mean:>16.3f} +/-{interval.half_width:>5.3f}"
                     f"{analytic:>12.3f}")
        payload["rows"].append({
            "alpha": alpha, "simulated": interval.mean,
            "half_width": interval.half_width, "analytic": analytic,
        })
    _emit(payload, lines, args.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m echelonsim",
        description="Discrete-event simulation of multi-echelon supply chains.",
    )
    parser.add_argument("--config", help="JSON experiment config, merged over the defaults")
    parser.add_argument("--periods", type=int, help="simulated periods per replication")
    parser.add_argument("--replications", type=int, help="number of replications")
    parser.add_argument("--seed", type=int, help="experiment seed (shared across scenarios)")
    parser.add_argument("--alpha", type=float, help="exponential smoothing constant")
    parser.add_argument("--levels", type=int, help="number of stocking echelons")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("run", help="one configuration, summary by node").set_defaults(
        handler=cmd_run
    )
    subparsers.add_parser("bullwhip", help="amplification by echelon").set_defaults(
        handler=cmd_bullwhip
    )
    subparsers.add_parser(
        "decompose", help="Shapley decomposition of amplification"
    ).set_defaults(handler=cmd_decompose)
    subparsers.add_parser("sweep", help="amplification vs smoothing constant").set_defaults(
        handler=cmd_sweep
    )

    information = subparsers.add_parser(
        "information", help="decentralised vs shared POS vs VMI"
    )
    information.add_argument(
        "--target-fill", type=float, default=None,
        help="calibrate z in every mode to this fill rate before comparing",
    )
    information.set_defaults(handler=cmd_information)

    leadtime = subparsers.add_parser(
        "leadtime", help="lead-time length vs variability at equal service"
    )
    leadtime.add_argument("--target-fill", type=float, default=0.95)
    leadtime.set_defaults(handler=cmd_leadtime)

    disrupt = subparsers.add_parser("disrupt", help="outage, shock and capacity-loss recovery")
    disrupt.add_argument("--start", type=int, default=200)
    disrupt.add_argument("--duration", type=int, default=8)
    disrupt.add_argument("--shock", type=float, default=2.0)
    disrupt.add_argument("--capacity", type=float, default=160.0)
    disrupt.set_defaults(handler=cmd_disrupt)

    warmup = subparsers.add_parser("warmup", help="MSER-5 truncation estimate")
    warmup.add_argument("--pilots", type=int, default=4)
    warmup.set_defaults(handler=cmd_warmup)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
