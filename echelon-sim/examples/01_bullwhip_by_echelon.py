#!/usr/bin/env python3
"""Measure amplification at every echelon, and check it against theory.

Run: ``python examples/01_bullwhip_by_echelon.py``

Two things worth noticing in the output. First, the *local* amplification is
modest at every stage -- each node roughly quadruples the variance it was handed,
which is not obviously alarming when you are looking at one node's numbers.
Second, the *cumulative* amplification is not modest at all, because those
modest factors multiply. Nobody in the chain is behaving badly and the factory
still sees an order stream with eighty times the variance of the demand that
caused it.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from echelonsim.bullwhip import exponential_smoothing_bullwhip, measure_by_echelon
from echelonsim.experiments import DEFAULT_CONFIG, estimate_warmup, merge_config

CONFIG = merge_config(DEFAULT_CONFIG, {
    "topology": {"kind": "serial", "levels": 3},
    "demand": {"kind": "iid_normal", "mean": 100.0, "std": 20.0},
    "forecast": {"kind": "exponential", "alpha": 0.3},
    "policy": {"kind": "base_stock", "z": 1.645},
    "leadtime": {"kind": "deterministic", "mean": 2.0, "order_lead_time": 1},
    "run": {"periods": 520, "replications": 12, "seed": 4242},
})


def main() -> int:
    warmup = estimate_warmup(CONFIG, pilots=3)
    print(f"MSER-5 warm-up truncation: {warmup} periods\n")

    amplification = measure_by_echelon(CONFIG, warmup=warmup)
    print(f"{'echelon':<14}{'local':>20}{'vs end demand':>22}")
    print("-" * 56)
    for name, local, local_hw, cumulative, cumulative_hw in amplification.table():
        print(f"{name:<14}{local:>13.2f} +/-{local_hw:>5.2f}"
              f"{cumulative:>15.2f} +/-{cumulative_hw:>5.2f}")

    protection = 2.0 + 1.0 + 1.0 - 1.0  # transit + order lead + review - 1
    analytic = exponential_smoothing_bullwhip(protection, 0.3)
    print(f"\nSingle-stage closed form at protection interval {protection:.0f} "
          f"and alpha 0.3: {analytic:.2f}")
    print("The retailer should land near it; everything upstream compounds "
          "beyond anything a single-stage formula predicts.")

    product = 1.0
    for name in amplification.node_order:
        product *= amplification.local[name].mean
    factory = amplification.cumulative[amplification.node_order[-1]].mean
    print(f"\nProduct of the local factors: {product:.1f}")
    print(f"Measured cumulative at the top: {factory:.1f}")
    print("Close, but not equal -- the stages are not independent, which is "
          "exactly why the decomposition in echelonsim.bullwhip uses Shapley "
          "values rather than multiplying factors together.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
