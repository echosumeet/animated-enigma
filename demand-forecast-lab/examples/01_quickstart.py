#!/usr/bin/env python3
"""Generate a demand panel, classify it, and forecast one series every way.

Run:  python examples/01_quickstart.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dflab.baselines import NaiveForecaster, SeasonalNaiveForecaster  # noqa: E402
from dflab.classify import classify_panel, classify_series  # noqa: E402
from dflab.datagen import DGPConfig, generate_panel  # noqa: E402
from dflab.ets import HoltLinear, HoltWinters, SimpleExponentialSmoothing  # noqa: E402
from dflab.intermittent import (  # noqa: E402
    CrostonForecaster,
    SBAForecaster,
    TSBForecaster,
)
from dflab.metrics import evaluate, seasonal_naive_scale  # noqa: E402

HORIZON = 13


def main() -> int:
    panel = generate_panel(DGPConfig(n_products=4, n_regions=3, n_channels=2))
    print(panel.describe())
    print(panel.hierarchy.summary())

    cut = panel.n_periods - HORIZON
    profiles = classify_panel(panel.y[:, :cut])
    print("\nDemand mix in the training window")
    for q in ("smooth", "erratic", "intermittent", "lumpy"):
        sel = [p for p in profiles if p.quadrant == q]
        if sel:
            print(
                f"  {q:<13}{len(sel):>3} series  "
                f"ADI {np.mean([p.adi for p in sel]):>5.2f}  "
                f"CV2 {np.mean([p.cv2 for p in sel]):>5.2f}"
            )

    # Take the largest intermittent series: the case where method choice matters.
    candidates = [i for i, p in enumerate(profiles) if p.quadrant == "intermittent"]
    if not candidates:
        candidates = list(range(panel.n_bottom))
    i = max(candidates, key=lambda k: profiles[k].mean_demand)
    y = panel.y[i]
    train, test = y[:cut], y[cut:]
    prof = classify_series(train)

    print(
        f"\nSeries {'/'.join(panel.keys[i])}: {prof.quadrant}, "
        f"ADI {prof.adi:.2f}, CV2 {prof.cv2:.2f}, "
        f"{prof.zero_share:.0%} zero weeks, mean {prof.mean_demand:.1f} units/week"
    )

    models = [
        NaiveForecaster(),
        SeasonalNaiveForecaster(52),
        SimpleExponentialSmoothing(),
        HoltLinear(damped=True),
        HoltWinters(52, "add", damped=True),
        CrostonForecaster(),
        SBAForecaster(),
        TSBForecaster(),
    ]

    scale = seasonal_naive_scale(train, 52)
    print(f"\nSeasonal-naive MASE denominator from the training window: {scale:.3f}")
    print(f"\n{'method':<20}{'WAPE':>9}{'MASE':>9}{'bias':>9}{'h=1 fc':>9}")
    for mdl in models:
        fc = mdl.fit(train).predict(HORIZON)
        met = evaluate(test, fc, train=train, season_length=52)
        print(
            f"{mdl.name:<20}{met['wape']:>9.4f}{met['mase']:>9.3f}"
            f"{met['bias']:>9.2f}{fc[0]:>9.2f}"
        )

    print(
        "\nRead the h=1 column next to the demand mix: on an intermittent series "
        "the classical smoothers and the Croston family all land on a similar "
        "demand *rate*. The differences that matter show up in how they react to "
        "a run of zeros, which is what example 04 isolates."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
