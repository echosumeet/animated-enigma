"""Command line entry point: ``python -m scmplatform <command>``."""

from __future__ import annotations

import argparse

import pandas as pd

from .backtest import Gate, check_gate, rolling_backtest
from .contracts import demand_panel_contract, validate
from .datagen import PanelConfig, inject_quality_faults, make_panel
from .features import audit_features, default_specs, leaky_specs
from .monitoring import drift_report, slice_performance


def _panel(args) -> pd.DataFrame:
    return make_panel(PanelConfig(n_skus=args.skus, n_days=args.days, seed=args.seed))


def cmd_validate(args) -> int:
    panel = _panel(args)
    if args.faults:
        panel = inject_quality_faults(panel)
    report = validate(panel, demand_panel_contract(), now=panel["date"].max())
    print(report.summary())
    if report.violations:
        print(report.to_frame().to_string(index=False))
    return 0 if report.passed else 1


def cmd_audit(args) -> int:
    panel = _panel(args)
    specs = leaky_specs() if args.leaky else default_specs()
    findings = audit_features(panel, specs)
    if not findings:
        print(f"PASS: {len(specs)} feature specs are point-in-time correct")
        return 0
    print(f"FAIL: {len(findings)} leakage finding(s)")
    for f in findings:
        print(f"  [{f.check}] {f.feature}: {f.detail}")
    return 1


def cmd_backtest(args) -> int:
    panel = _panel(args)
    result = rolling_backtest(panel, default_specs(), folds=args.folds, horizon=args.horizon)
    gate = check_gate(result, Gate())
    print(gate.summary())
    for k, v in gate.metrics.items():
        print(f"  {k:>18}: {v:,.4f}")
    return 0 if gate.passed else 1


def cmd_monitor(args) -> int:
    panel = _panel(args)
    split = panel["date"].quantile(0.7)
    cols = ["units", "price", "on_hand"]
    print(drift_report(panel[panel["date"] <= split], panel[panel["date"] > split], cols).to_string(index=False))
    result = rolling_backtest(panel, default_specs(), folds=2, horizon=args.horizon)
    report = slice_performance(result.predictions, ["category"])
    print()
    print(report.summary())
    print(report.table.to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scmplatform", description=__doc__)
    p.add_argument("--skus", type=int, default=40)
    p.add_argument("--days", type=int, default=540)
    p.add_argument("--seed", type=int, default=7)
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="validate the generated panel against its data contract")
    v.add_argument("--faults", action="store_true", help="inject upstream data faults first")
    v.set_defaults(func=cmd_validate)

    a = sub.add_parser("audit", help="run the point-in-time leakage audit on a feature set")
    a.add_argument("--leaky", action="store_true", help="audit the deliberately buggy feature set")
    a.set_defaults(func=cmd_audit)

    b = sub.add_parser("backtest", help="rolling backtest plus the release gate")
    b.add_argument("--folds", type=int, default=3)
    b.add_argument("--horizon", type=int, default=14)
    b.set_defaults(func=cmd_backtest)

    m = sub.add_parser("monitor", help="feature drift and slice-level performance")
    m.add_argument("--horizon", type=int, default=14)
    m.set_defaults(func=cmd_monitor)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
