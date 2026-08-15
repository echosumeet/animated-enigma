"""Command line entry point: ``python -m etarisk <command>``."""

from __future__ import annotations

import argparse

import pandas as pd

from .drift import psi_table
from .figures import calibration_figure
from .generate import GeneratorConfig, describe_distribution
from .pipeline import run_pipeline


def _cfg(args) -> GeneratorConfig:
    return GeneratorConfig(n_shipments=args.n, seed=args.seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="etarisk", description=__doc__)
    parser.add_argument("command", choices=["run", "figures", "drift", "describe"])
    parser.add_argument("--n", type=int, default=40_000, help="shipments to generate")
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--outdir", default="docs")
    args = parser.parse_args(argv)

    pd.set_option("display.width", 120)

    if args.command == "describe":
        from .generate import generate_shipments

        stats = describe_distribution(generate_shipments(_cfg(args)))
        for k, v in stats.items():
            print(f"{k:>16}: {v:,.3f}")
        return 0

    res = run_pipeline(_cfg(args))
    m = res.metrics

    if args.command == "figures":
        print("wrote", calibration_figure(m, args.outdir))
        return 0

    if args.command == "drift":
        ref, cur = res.X["train_scoring"], res.X["test"]
        print(psi_table(ref, cur).to_string(index=False))
        return 0

    print(f"test MAE {m['eta_mae_h']:.2f} h vs quoted transit {m['quoted_mae_h']:.2f} h")
    print(m["coverage"].to_string(index=False))
    print(m["mae_by_horizon"].to_string(index=False))
    d = m["decision"]
    print(
        f"decision layer: {d['model_cost_per_shipment']:.2f} vs baseline "
        f"{d['baseline_cost_per_shipment']:.2f} per shipment "
        f"({d['saved_pct']:.1f}% saved)"
    )
    return 0
