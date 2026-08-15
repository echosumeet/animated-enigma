#!/usr/bin/env python3
"""Make the plan add up: bottom-up, top-down, OLS and MinT compared.

Run:  python examples/03_hierarchical_reconciliation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dflab.datagen import DGPConfig, generate_panel  # noqa: E402
from dflab.hierarchy import coherency_error  # noqa: E402
from dflab.pipeline import run_reconciliation_study  # noqa: E402

HORIZON = 13


def main() -> int:
    panel = generate_panel(
        DGPConfig(n_products=4, n_regions=3, n_channels=2, n_periods=230)
    )
    hier = panel.hierarchy
    print(panel.describe())
    print(hier.summary())

    nodes = panel.node_series()
    print(
        f"\nGenerated data is coherent by construction: "
        f"max |y - S b| = {coherency_error(nodes, hier):.2e}"
    )
    print(
        "Independently produced forecasts are not. That gap is what a planning "
        "organisation spends its Monday arguing about."
    )

    rep = run_reconciliation_study(
        panel, horizon=HORIZON, n_windows=2, step=HORIZON, min_train=156, verbose=True
    )

    print(f"\nShrinkage intensity fitted for MinT: {rep.shrinkage:.3f}")
    print("(0 = trust the full sample covariance, 1 = trust only the diagonal)\n")

    width = max(len(lv) for lv in rep.levels) + 2
    print(f"{'level':<{width}}" + "".join(f"{m:>12}" for m in rep.table))
    for lv in rep.levels:
        cells = "".join(f"{rep.table[m][lv]:>12.4f}" for m in rep.table)
        print(f"{lv:<{width}}{cells}")

    print(f"\n{'coherency':<{width}}" + "".join(f"{rep.coherency[m]:>12.1e}" for m in rep.table))

    base_bottom = rep.table["base"]["bottom"]
    mint_bottom = rep.table["mint"]["bottom"]
    base_total = rep.table["base"]["total"]
    bu_total = rep.table["bottom_up"]["total"]
    best_at_bottom = min(rep.table, key=lambda m: rep.table[m]["bottom"])
    best_at_total = min(rep.table, key=lambda m: rep.table[m]["total"])
    print(
        f"\nThree numbers worth carrying into a design review:\n"
        f"  MinT changed bottom-level WAPE by "
        f"{100 * (mint_bottom / base_bottom - 1):+.1f}% versus the unreconciled "
        f"base forecast, while making the plan add up exactly.\n"
        f"  Bottom-up changed total-level WAPE by "
        f"{100 * (bu_total / base_total - 1):+.1f}% versus forecasting the total "
        f"directly. Whether summing cells helps or hurts at the top depends on how "
        f"correlated the cell errors are, which is precisely what MinT estimates "
        f"instead of assuming.\n"
        f"  Best at the bottom: {best_at_bottom}. Best at the total: "
        f"{best_at_total}. A single reconciliation method does not have to win "
        f"everywhere, but it does have to be chosen once, because you only get to "
        f"publish one plan."
    )
    print(
        "\nTop-down is the cheapest to explain and the worst at the bottom, because "
        "historical proportions carry no information about which cell is currently "
        "growing. It survives in practice because it is the only method a finance "
        "process can audit in a spreadsheet."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
