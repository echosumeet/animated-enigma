#!/usr/bin/env python3
"""Decentralised ordering vs shared point-of-sale vs vendor-managed inventory.

Run: ``python examples/02_information_sharing.py``

The comparison is calibrated: every mode's safety factor is bisected until the
retailer fill rate is the same, so what changes between the rows is inventory
and order variance, not service. Comparing information modes at different
service levels is the single easiest way to overstate the benefit, and it is
what most business cases do.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from echelonsim.experiments import DEFAULT_CONFIG, merge_config
from echelonsim.information import compare_information_modes

CONFIG = merge_config(DEFAULT_CONFIG, {
    "topology": {"kind": "serial", "levels": 3},
    "forecast": {"kind": "exponential", "alpha": 0.3},
    "run": {"periods": 520, "replications": 12, "seed": 4242},
})
TARGET_FILL = 0.975


def main() -> int:
    comparison = compare_information_modes(CONFIG, calibrate_to=TARGET_FILL)
    print(f"All modes calibrated to a {100 * TARGET_FILL:.1f}% retailer fill "
          f"rate; warm-up truncation {comparison.warmup} periods.\n")

    print(f"{'mode':<16}{'z':>7}{'factory amp':>14}{'fill %':>9}"
          f"{'inventory':>12}{'cost':>9}")
    print("-" * 67)
    for mode, chain, _hw, fill, inventory, cost in comparison.table():
        print(f"{mode:<16}{comparison.safety_factors[mode]:>7.3f}{chain:>14.2f}"
              f"{fill:>9.2f}{inventory:>12.0f}{cost:>9.0f}")

    print(f"\nAmplification by echelon (vs end-customer demand)")
    by_mode = comparison.bullwhip_by_mode()
    print(f"{'mode':<16}" + "".join(f"{name:>14}" for name in comparison.node_order))
    print("-" * (16 + 14 * len(comparison.node_order)))
    for mode in comparison.outcomes:
        print(f"{mode:<16}" + "".join(
            f"{by_mode[mode][name].mean:>14.2f}" for name in comparison.node_order
        ))

    print("\nPaired against decentralised (common random numbers, so these are "
          "paired-t intervals):")
    for mode in comparison.outcomes:
        if mode == comparison.reference:
            continue
        amplification = comparison.paired_percent(mode, "chain_bullwhip")
        inventory = comparison.paired_percent(mode, "avg_inventory")
        cost = comparison.paired_percent(mode, "avg_cost")
        print(f"  {mode:<12} factory amplification {amplification.mean:+7.1f}% "
              f"+/-{amplification.half_width:4.1f}   inventory "
              f"{inventory.mean:+7.1f}% +/-{inventory.half_width:4.1f}   cost "
              f"{cost.mean:+7.1f}% +/-{cost.half_width:4.1f}")

    print("""
Read the two shared modes carefully -- they are not ranked the same way on
every column.

Sharing POS data alone removes the forecast cascade and is the biggest single
reduction in order variance. But each node still runs its own installation
stock, and it now sizes that stock from the variability of *end demand* while
actually facing the variability of the *order stream* above it. Upstream buffers
end up too thin for what they see, which shows up as upstream backorders.

Echelon control (VMI) leaves more order variance at the factory, because the
echelon target covers the cumulative lead time down the chain and so responds
more strongly to a forecast revision. In exchange it puts the stock where the
demand actually is, and it is the cheapest of the three on total cost.

The lesson is the one worth carrying into a real programme: "share the data" and
"change who decides" are different projects with different payoffs, and only the
second one moves the physical structure of the chain.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
