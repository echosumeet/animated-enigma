#!/usr/bin/env python3
"""Is a faster supplier or a more consistent supplier worth more?

Run: ``python examples/04_leadtime_tradeoff.py``

Every cell is calibrated to the same fill rate by bisecting the safety factor,
so the inventory column is a like-for-like comparison. Without that calibration
a longer or more erratic lead time shows up partly as extra inventory and partly
as worse service, and reading either column alone understates it.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from echelonsim.tradeoffs import lead_time_demand_sigma, lead_time_grid

TARGET_FILL = 0.95
CONFIG = {"run": {"periods": 780, "replications": 10, "seed": 4242, "warmup": 60}}


def main() -> int:
    cells = lead_time_grid(means=(2.0, 4.0, 8.0), cvs=(0.0, 0.5),
                           target_fill=TARGET_FILL, base_config=CONFIG)
    print(f"Calibrated to a {100 * TARGET_FILL:.0f}% fill rate in every cell.\n")
    print(f"{'mean L':>8}{'CV L':>7}{'z':>8}{'fill %':>9}{'inventory':>12}"
          f"{'sigma_DL':>11}{'ok':>5}")
    print("-" * 60)
    for cell in cells:
        mean_lead, cv_lead, z, fill, inventory, sigma = cell.row()
        flag = "y" if cell.converged else "n"
        print(f"{mean_lead:>8.0f}{cv_lead:>7.2f}{z:>8.3f}{fill:>9.2f}"
              f"{inventory:>12.0f}{sigma:>11.0f}{flag:>5}")

    lookup = {(c.mean_lead, c.cv_lead): c for c in cells}
    fast_erratic = lookup[(2.0, 0.5)]
    slow_reliable = lookup[(4.0, 0.0)]
    slower_reliable = lookup[(8.0, 0.0)]
    print(f"\nA 2-period supplier with a 0.5 lead-time CV needs "
          f"{fast_erratic.inventory.mean:.0f} units.")
    print(f"A perfectly reliable 4-period supplier needs "
          f"{slow_reliable.inventory.mean:.0f} units -- "
          f"{100 * (1 - slow_reliable.inventory.mean / fast_erratic.inventory.mean):.0f}% "
          f"less, at twice the lead time.")
    print(f"Even a reliable 8-period supplier "
          f"({slower_reliable.inventory.mean:.0f} units) is competitive with it "
          f"at four times the lead time.")

    # Protection interval is R + L - 1: review 1, order lead 1, transit L.
    analytic_fast = lead_time_demand_sigma(3.0, 100.0, 20.0, 2.0 * 0.5)
    analytic_slow = lead_time_demand_sigma(5.0, 100.0, 20.0, 0.0)
    print(f"\nThe convolution formula says the same thing before any simulation "
          f"runs: sigma_DL is {analytic_fast:.0f} for the erratic short lead "
          f"time and {analytic_slow:.0f} for the reliable long one. The "
          f"d_bar^2 * sigma_L^2 term dominates as soon as demand is smoother "
          f"than delivery.")
    print("""
The practical version: a supplier scorecard that tracks average lead time and
not lead-time variance is optimising the smaller of the two terms. 'We cut lead
time from six weeks to four' is worth very little if the delivery window widened
to get there -- and widening the window is usually how it was achieved.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
