"""Command line entry point: ``python -m dflab <command>``.

Commands
--------
``classify``     generate a panel and print the demand-quadrant breakdown
``backtest``     run the rolling-origin benchmark and print the tables
``reconcile``    run the hierarchical reconciliation study
``figures``      regenerate the README figures into a directory
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .classify import QUADRANTS, classify_panel
from .datagen import DGPConfig, generate_panel
from .pipeline import pinball_table, run_backtest, run_reconciliation_study
from .plots import (
    plot_accuracy_by_quadrant,
    plot_demand_quadrants,
    plot_example_series,
    plot_reconciliation_gain,
)


def _panel_from_args(args):
    cfg = DGPConfig(
        n_products=args.products,
        n_regions=args.regions,
        n_channels=args.channels,
        n_periods=args.periods,
        seed=args.seed,
    )
    return generate_panel(cfg)


def _fmt(v: float, nd: int = 4) -> str:
    return "n/a" if not np.isfinite(v) else f"{v:.{nd}f}"


def cmd_classify(args) -> int:
    panel = _panel_from_args(args)
    print(panel.describe())
    cut = panel.n_periods - args.horizon * args.windows
    profiles = classify_panel(panel.y[:, :cut])
    print(f"\nClassification computed on the first {cut} periods (training only)\n")
    print(f"{'quadrant':<14}{'series':>8}{'mean ADI':>10}{'mean CV2':>10}{'units/wk':>12}")
    for q in QUADRANTS:
        sel = [p for p in profiles if p.quadrant == q]
        if not sel:
            print(f"{q:<14}{0:>8}")
            continue
        print(
            f"{q:<14}{len(sel):>8}"
            f"{np.mean([p.adi for p in sel]):>10.2f}"
            f"{np.mean([p.cv2 for p in sel]):>10.2f}"
            f"{np.mean([p.mean_demand for p in sel]):>12.1f}"
        )
    return 0


def cmd_backtest(args) -> int:
    panel = _panel_from_args(args)
    print(panel.describe(), flush=True)
    res = run_backtest(
        panel,
        horizon=args.horizon,
        step=args.step,
        n_windows=args.windows,
        min_train=args.min_train,
        include_ml=not args.no_ml,
        verbose=True,
    )
    agg = res.overall()
    fva = res.value_add("snaive[m=52]")
    print("\nOverall (pooled over all series and windows)")
    print(f"{'method':<20}{'WAPE':>9}{'MASE':>9}{'RMSSE':>9}{'sMAPE':>9}{'bias%':>9}{'FVA vs snaive':>15}")
    for mth in sorted(agg, key=lambda k: agg[k]["wape"]):
        a = agg[mth]
        print(
            f"{mth:<20}{_fmt(a['wape'])!s:>9}{_fmt(a['mase'],3)!s:>9}"
            f"{_fmt(a['rmsse'],3)!s:>9}{_fmt(a['smape'],3)!s:>9}"
            f"{_fmt(a['pct_bias'],3)!s:>9}{_fmt(fva[mth],3)!s:>15}"
        )

    bq = res.by_quadrant()
    counts = res.quadrant_counts()
    print("\nWAPE by demand quadrant  " + str(counts))
    header = f"{'method':<20}" + "".join(f"{q:>14}" for q in QUADRANTS)
    print(header)
    for mth in sorted(agg, key=lambda k: agg[k]["wape"]):
        cells = "".join(f"{_fmt(bq[mth][q]['wape'])!s:>14}" for q in QUADRANTS)
        print(f"{mth:<20}{cells}")

    print("\nBest method per quadrant (WAPE):")
    for q, (mth, v) in res.best_by_quadrant().items():
        print(f"  {q:<14}{mth:<20}{v:.4f}")

    pt = pinball_table(res, "gbt_mean")
    if pt:
        print("\nQuantile forecasts (gradient boosting, pinball loss / coverage):")
        for k in sorted(pt):
            print(f"  {k:<18}{pt[k]:.4f}")
    return 0


def cmd_reconcile(args) -> int:
    panel = _panel_from_args(args)
    print(panel.describe())
    print(panel.hierarchy.summary(), flush=True)
    rep = run_reconciliation_study(
        panel,
        horizon=args.horizon,
        n_windows=args.windows,
        step=args.step,
        min_train=args.min_train,
        verbose=True,
    )
    print(f"\nShrinkage intensity (Schafer-Strimmer): {rep.shrinkage:.3f}\n")
    print(rep.as_markdown())
    return 0


def cmd_figures(args) -> int:
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    panel = _panel_from_args(args)
    cut = panel.n_periods - args.horizon * args.windows
    profiles = classify_panel(panel.y[:, :cut])
    paths = [plot_demand_quadrants(profiles, out / "demand_quadrants.png")]

    picks, labels = [], []
    for q in QUADRANTS:
        sel = [i for i, p in enumerate(profiles) if p.quadrant == q]
        if sel:
            best = max(sel, key=lambda i: profiles[i].mean_demand)
            picks.append(best)
            labels.append(
                f"{'/'.join(panel.keys[best])} - {q} "
                f"(ADI {profiles[best].adi:.2f}, CV2 {profiles[best].cv2:.2f})"
            )
    paths.append(
        plot_example_series(panel.y, picks, labels, out / "example_series.png", cut=cut)
    )

    res = run_backtest(
        panel,
        horizon=args.horizon,
        step=args.step,
        n_windows=args.windows,
        min_train=args.min_train,
        include_ml=not args.no_ml,
        verbose=True,
    )
    paths.append(
        plot_accuracy_by_quadrant(
            res.by_quadrant(),
            out / "accuracy_by_quadrant.png",
            methods=res.methods(),
            counts=res.quadrant_counts(),
        )
    )
    rep = run_reconciliation_study(
        panel, horizon=args.horizon, n_windows=2, step=args.step, min_train=args.min_train
    )
    paths.append(plot_reconciliation_gain(rep.table, out / "reconciliation.png"))
    for p in paths:
        print(f"wrote {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--products", type=int, default=8)
    common.add_argument("--regions", type=int, default=4)
    common.add_argument("--channels", type=int, default=3)
    common.add_argument("--periods", type=int, default=260)
    common.add_argument("--seed", type=int, default=20260815)
    common.add_argument("--horizon", type=int, default=13)
    common.add_argument("--step", type=int, default=13)
    common.add_argument("--windows", type=int, default=4)
    common.add_argument("--min-train", dest="min_train", type=int, default=156)
    common.add_argument(
        "--no-ml", action="store_true", help="skip the gradient boosting models"
    )

    p = argparse.ArgumentParser(
        prog="dflab",
        description=(
            "Benchmarking harness for intermittent and hierarchical demand forecasting"
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "classify", parents=[common], help="demand quadrant breakdown"
    ).set_defaults(func=cmd_classify)
    sub.add_parser(
        "backtest", parents=[common], help="rolling-origin benchmark"
    ).set_defaults(func=cmd_backtest)
    sub.add_parser(
        "reconcile", parents=[common], help="hierarchical reconciliation study"
    ).set_defaults(func=cmd_reconcile)
    fig = sub.add_parser(
        "figures", parents=[common], help="regenerate the README figures"
    )
    fig.add_argument("--outdir", default="docs")
    fig.set_defaults(func=cmd_figures)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
