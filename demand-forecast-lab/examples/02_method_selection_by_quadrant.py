#!/usr/bin/env python3
"""Which method wins where: a rolling-origin backtest reported per quadrant.

This is the whole argument of the repository in one script. A single pooled
accuracy number picks one winner; splitting by demand pattern usually picks
four, and the operational answer is a routing rule rather than a champion.

Run:  python examples/02_method_selection_by_quadrant.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dflab.classify import QUADRANTS  # noqa: E402
from dflab.datagen import DGPConfig, generate_panel  # noqa: E402
from dflab.pipeline import run_backtest  # noqa: E402

BASELINE = "snaive[m=52]"


def main() -> int:
    # Smaller than the full benchmark so this finishes in about a minute.
    panel = generate_panel(
        DGPConfig(n_products=4, n_regions=3, n_channels=2, n_periods=240)
    )
    print(panel.describe(), flush=True)

    res = run_backtest(
        panel, horizon=13, step=13, n_windows=2, min_train=156, verbose=True
    )
    agg = res.overall()
    bq = res.by_quadrant()
    counts = res.quadrant_counts()
    fva = res.value_add(BASELINE)
    order = sorted(agg, key=lambda k: agg[k]["wape"])

    print("\nPooled over the whole panel")
    print(f"{'method':<18}{'WAPE':>9}{'MASE':>9}{'FVA vs snaive':>16}")
    for m in order:
        print(f"{m:<18}{agg[m]['wape']:>9.4f}{agg[m]['mase']:>9.3f}{fva[m]:>16.3f}")

    print("\nWAPE by demand quadrant")
    print(f"{'method':<18}" + "".join(f"{q + f' (n={counts[q]})':>22}" for q in QUADRANTS))
    for m in order:
        cells = "".join(f"{bq[m][q]['wape']:>22.4f}" for q in QUADRANTS)
        print(f"{m:<18}{cells}")

    print("\nRouting rule implied by this run")
    for q, (mth, v) in res.best_by_quadrant().items():
        base_v = bq[BASELINE][q]["wape"]
        gain = 1.0 - v / base_v if np.isfinite(base_v) and base_v > 0 else float("nan")
        print(
            f"  {q:<13} -> {mth:<16} WAPE {v:.4f}  "
            f"(seasonal naive {base_v:.4f}, FVA {gain:+.1%})"
        )

    single_champion = order[0]
    loss = {}
    for q in QUADRANTS:
        best = res.best_by_quadrant().get(q)
        if best is None:
            continue
        loss[q] = bq[single_champion][q]["wape"] - best[1]
    print(
        f"\nPicking the single pooled champion ({single_champion}) and applying it "
        f"everywhere costs, per quadrant, this much WAPE versus routing:"
    )
    for q, v in loss.items():
        print(f"  {q:<13}{v:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
