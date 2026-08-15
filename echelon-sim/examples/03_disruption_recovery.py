#!/usr/bin/env python3
"""How long a chain takes to recover from an outage, a shock, or a capacity loss.

Run: ``python examples/03_disruption_recovery.py``

The number that surprises people is the ratio in the last column. A supply
disruption of N periods does not produce N periods of degraded service: the
backlog has to be worked off on top of ongoing demand, through a chain that is
simultaneously re-ordering against depressed inventory positions everywhere.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from echelonsim.disruption import run_disruption_study
from echelonsim.experiments import DEFAULT_CONFIG, merge_config

START = 220
CONFIG = merge_config(DEFAULT_CONFIG, {
    "topology": {"kind": "serial", "levels": 3, "capacity": 160.0},
    "forecast": {"kind": "exponential", "alpha": 0.3},
    "run": {"periods": 420, "replications": 16, "seed": 4242},
})

SCENARIOS = [
    (f"supplier outage, {n} periods",
     {"disruptions": {"outages": [{"node": "source", "start": START, "duration": n}]}},
     START, n)
    for n in (2, 4, 8)
] + [
    ("demand shock, 2x for 4 periods",
     {"demand": {"shock": {"start": START, "duration": 4, "multiplier": 2.0}}},
     START, 4),
    ("factory capacity -50%, 8 periods",
     {"disruptions": {"capacity_losses": [
         {"node": "factory", "start": START, "duration": 8, "factor": 0.5}]}},
     START, 8),
]


def main() -> int:
    study = run_disruption_study(CONFIG, SCENARIOS, horizon=80)
    print(f"Warm-up truncation {study.warmup} periods; every scenario paired "
          f"with the same undisrupted baseline.\n")
    print(f"{'scenario':<34}{'len':>5}{'trough':>9}{'at':>5}"
          f"{'recover':>10}{'ratio':>8}{'late units':>12}")
    print("-" * 83)
    for profile in study.profiles:
        name, duration, trough, trough_at, recovery, ratio, lost = profile.row()
        print(f"{name:<34}{duration:>5d}{trough:>8.1f}%{trough_at:>5d}"
              f"{recovery:>10.0f}{ratio:>7.1f}x{lost:>12.0f}")

    absorbed = [p for p in study.profiles if p.recovery_offset.mean == 0.0]
    if absorbed:
        print("\nAbsorbed entirely (no service gap outside the measurement "
              "tolerance): " + ", ".join(p.name for p in absorbed) + ".")
        print("A short outage is soaked up by the pipeline; the chain has "
              "roughly that much slack in it by construction.")

    worst = max(study.profiles, key=lambda p: p.recovery_ratio)
    print(f"\nWorst ratio: {worst.name} -- {worst.duration} periods of "
          f"disruption, {worst.recovery_offset.mean:.0f} periods to recover "
          f"({worst.recovery_ratio:.1f}x), and the fill-rate trough arrives "
          f"{worst.trough_offset} periods *after* the disruption starts.")
    print("""
That lag is worth planning around. The service failure that gets escalated is
not concurrent with the event that caused it: by the time the shelf is empty the
supplier has usually been back for several periods, and the incident review ends
up looking at the wrong week.

Note also the asymmetry between the two 'four period' rows. A four-period supply
outage and a four-period demand doubling are both absorbed, but the demand shock
costs several times more late units, because the chain has buffers sized for
supply variability and none sized for a step change in demand.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
