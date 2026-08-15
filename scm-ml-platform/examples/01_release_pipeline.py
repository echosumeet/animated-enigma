"""End-to-end release pipeline: contract -> features -> backtest gate -> registry -> monitoring.

Run from the repository root:

    PYTHONPATH=src python examples/01_release_pipeline.py

Nothing here talks to a network or a database. The registry writes JSON into a
temporary directory that is printed at the end so you can inspect the model cards.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from scmplatform.backtest import Gate, champion_challenger, check_gate, rolling_backtest
from scmplatform.contracts import demand_panel_contract, diff_contracts, validate
from scmplatform.datagen import PanelConfig, inject_quality_faults, make_panel
from scmplatform.features import audit_features, build_features, default_specs, leaky_specs
from scmplatform.monitoring import drift_report, prediction_drift, slice_performance
from scmplatform.registry import ModelCard, ModelRegistry


def rule(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def main() -> int:
    panel = make_panel(PanelConfig(n_skus=30, n_days=480, seed=7))
    contract = demand_panel_contract()
    as_of = panel["date"].max()

    rule("1. Data contract")
    print(validate(panel, contract, now=as_of).summary())
    bad = validate(inject_quality_faults(panel), contract, now=as_of)
    print(bad.summary())
    print(bad.to_frame().to_string(index=False))

    rule("2. Contract change review")
    proposed = demand_panel_contract("1.1.0")
    proposed = type(proposed)(
        name=proposed.name,
        version="1.1.0",
        columns=tuple(c for c in proposed.columns if c.name != "on_hand"),
        primary_key=proposed.primary_key,
        freshness_column=proposed.freshness_column,
        max_age_days=1.0,
    )
    diff = diff_contracts(contract, proposed)
    print(diff.summary())
    for item in diff.breaking:
        print("  BREAKING:", item)

    rule("3. Point-in-time audit")
    print("production feature set:", audit_features(panel, default_specs()) or "clean")
    for f in audit_features(panel, leaky_specs()):
        print(f"  [{f.check}] {f.feature}: {f.detail}")

    rule("4. Backtest and release gate")
    gate = Gate()
    champion = rolling_backtest(panel, default_specs(), label="champion", horizon=14, folds=3)
    challenger = rolling_backtest(
        panel, default_specs(), label="challenger", horizon=14, folds=3, bias_adjustment=0.85
    )
    champ_gate = check_gate(champion, gate)
    print(champ_gate.summary())
    print(check_gate(challenger, gate).summary())
    decision = champion_challenger(champion, challenger, gate)
    print(f"promote challenger: {decision.promote} -- {decision.rationale}")
    print(decision.as_frame().to_string(index=False))

    rule("5. Registry: promote, then roll back")
    tmp = Path(tempfile.mkdtemp(prefix="scmplatform-registry-"))
    reg = ModelRegistry(tmp)
    for version, parent, metrics in (
        ("1.0.0", None, {"wmape": champion.wmape}),
        ("1.1.0", "1.0.0", {"wmape": challenger.wmape}),
    ):
        reg.register(
            ModelCard(
                name="demand_daily",
                version=version,
                algorithm="HistGradientBoostingRegressor",
                data_contract=f"{contract.name}@{contract.version}",
                feature_specs=[s.name for s in default_specs()],
                training_window=f"{panel['date'].min():%Y-%m-%d}..{as_of:%Y-%m-%d}",
                parent_version=parent,
                metrics={k: round(v, 4) for k, v in metrics.items()},
                intended_use="Daily SKU-level unit forecast feeding the replenishment order-up-to level.",
                limitations="Fitted on synthetic data; not calibrated for new-item launches.",
            )
        )
        reg.transition("demand_daily", version, "production")
    print("in production:", reg.production("demand_daily").version)
    print("rolled back to:", reg.rollback("demand_daily", note="cost regression").version)
    print("lineage:", [c.version for c in reg.lineage("demand_daily", "1.1.0")])

    rule("6. Monitoring")
    split = panel["date"].quantile(0.7)
    feats = build_features(panel, default_specs()).dropna()
    cols = [s.name for s in default_specs()]
    print(drift_report(feats[feats["date"] <= split], feats[feats["date"] > split], cols).to_string(index=False))
    early = champion.predictions[champion.predictions["fold"] == 0]["yhat"]
    late = champion.predictions[champion.predictions["fold"] == champion.folds - 1]["yhat"]
    pd_stats = prediction_drift(early, late)
    print(f"\nprediction drift fold 0 -> fold {champion.folds - 1}: "
          f"PSI {pd_stats['psi']:.4f}, mean shift {pd_stats['mean_shift']:+.2f} units")
    slices = slice_performance(champion.predictions, ["category"])
    print(slices.summary())
    print(slices.table.to_string(index=False))

    print(f"\nregistry written to {tmp}")
    return 0


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    raise SystemExit(main())
