#!/usr/bin/env python3
"""Croston vs SBA vs TSB on a part that stops selling.

Point accuracy is not why you would choose TSB. The reason is what happens to
the forecast of an item that has gone quiet: Croston only updates at demand
epochs, so it holds its last rate forever and keeps proposing replenishment for
stock nobody will ever order. TSB decays. This script measures both the bias
correction and the decay behaviour explicitly.

Run:  python examples/04_intermittent_obsolescence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dflab.classify import classify_series  # noqa: E402
from dflab.intermittent import (  # noqa: E402
    CrostonForecaster,
    SBAForecaster,
    TSBForecaster,
)
from dflab.metrics import evaluate  # noqa: E402


def intermittent_series(n, p, mean_size, seed, cv2=0.5):
    rng = np.random.default_rng(seed)
    occ = rng.random(n) < p
    y = np.zeros(n)
    shape = 1.0 / cv2
    y[occ] = np.round(rng.gamma(shape, mean_size / shape, occ.sum()))
    return y


def main() -> int:
    ALPHA, BETA = 0.15, 0.10

    print("Part 1 -- the Syntetos-Boylan bias correction\n")
    print(
        "Croston estimates size/interval separately, then divides. Because "
        "E[z/p] != E[z]/E[p], the ratio is biased high by roughly 1/(1-alpha/2).\n"
    )
    print(
        f"{'alpha':>7}{'croston':>10}{'sba':>10}{'realised rate':>15}"
        f"{'croston err':>13}{'sba err':>10}"
    )
    for alpha in (0.05, 0.10, 0.20, 0.30):
        c_vals, s_vals, truths = [], [], []
        for seed in range(60):
            y = intermittent_series(400, 0.2, 25.0, seed=seed, cv2=0.6)
            truths.append(float(np.mean(y)))
            c_vals.append(CrostonForecaster(alpha=alpha).fit(y).predict(1)[0])
            s_vals.append(SBAForecaster(alpha=alpha).fit(y).predict(1)[0])
        c_err = float(np.mean(c_vals) - np.mean(truths))
        s_err = float(np.mean(s_vals) - np.mean(truths))
        print(
            f"{alpha:>7.2f}"
            f"{np.mean(c_vals):>10.2f}"
            f"{np.mean(s_vals):>10.2f}"
            f"{np.mean(truths):>15.2f}"
            f"{c_err:>13.3f}"
            f"{s_err:>10.3f}"
        )
    print(
        "\nThe error columns are averaged over 60 seeded replications: Croston is "
        "systematically high and the gap widens with alpha, exactly as the "
        "deflator (1 - alpha/2) predicts."
    )

    print("\n\nPart 2 -- obsolescence\n")
    live = intermittent_series(156, 0.22, 30.0, seed=7, cv2=0.8)
    dead_tail = np.zeros(52)
    full = np.concatenate([live, dead_tail])
    prof = classify_series(full)
    print(
        f"Series: 156 weeks of intermittent demand then 52 weeks of nothing "
        f"({prof.quadrant}, ADI {prof.adi:.2f}, CV2 {prof.cv2:.2f})\n"
    )

    print(f"{'weeks dead':>11}{'croston':>10}{'sba':>8}{'tsb':>8}")
    for dead in (0, 4, 13, 26, 39, 52):
        hist = np.concatenate([live, dead_tail[:dead]])
        c = CrostonForecaster(alpha=ALPHA).fit(hist).predict(1)[0]
        s = SBAForecaster(alpha=ALPHA).fit(hist).predict(1)[0]
        t = TSBForecaster(alpha=ALPHA, beta=BETA).fit(hist).predict(1)[0]
        print(f"{dead:>11}{c:>10.3f}{s:>8.3f}{t:>8.3f}")

    print(
        "\nTSB is already below the long-run rate at week 0 because the series "
        "happens to end on a run of zeros -- responsiveness to silence and a "
        "noisy level estimate are the same property, not two."
    )
    print(
        "\nCroston and SBA are frozen: they never update on a zero period, so a "
        "year of silence changes nothing. TSB's demand probability decays by "
        f"(1 - beta) every period, so after 52 dead weeks it has shed "
        f"{1 - (1 - BETA) ** 52:.1%} of its estimate."
    )

    print("\n\nPart 3 -- accuracy over the dead year\n")
    train, test = full[:156], full[156:]
    print(f"{'method':<18}{'WAPE':>10}{'MAE':>10}{'bias':>10}")
    for mdl in (
        CrostonForecaster(alpha=ALPHA),
        SBAForecaster(alpha=ALPHA),
        TSBForecaster(alpha=ALPHA, beta=BETA),
    ):
        fc = mdl.fit(train).predict(test.size)
        met = evaluate(test, fc, train=train, season_length=52)
        print(f"{mdl.name:<18}{met['wape']:>10.4f}{met['mae']:>10.3f}{met['bias']:>10.3f}")
    print(
        "\nWAPE is NaN here because the test window has no demand at all -- which "
        "is the honest answer, and the reason this repository refuses to average "
        "WAPE across series. Judge this case on MAE and on the decay table above: "
        "the whole point of TSB is that it stops proposing stock."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
