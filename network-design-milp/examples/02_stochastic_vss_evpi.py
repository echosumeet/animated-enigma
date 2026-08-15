"""What uncertainty is worth: VSS and EVPI on the same instance.

    PYTHONPATH=src python examples/02_stochastic_vss_evpi.py

Two numbers come out of this and they answer different questions.

* **VSS** answers "should we model the uncertainty at all, or is designing on
  the forecast good enough?"
* **EVPI** answers "how much is a better forecast worth?" - and it is an upper
  bound, so it kills more projects than it funds.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from netdesign.instances import demand_scenarios, generate_instance
from netdesign.network_flow import NetworkOptions
from netdesign.reporting import markdown_table
from netdesign.stochastic import solve_two_stage


def main() -> None:
    inst = generate_instance(seed=7)
    scenarios, probs = demand_scenarios(inst, n_scenarios=12, seed=11)
    totals = sorted(sum(s.values()) for s in scenarios)
    print(inst.summary())
    print(
        f"\n12 equiprobable scenarios, total demand {totals[0]:,.0f} to {totals[-1]:,.0f} "
        f"units/period (nominal {inst.total_demand():,.0f})."
    )
    print(
        "The shock model is three-level - national, regional, per-zone - because only the "
        "part that does not diversify away moves the design.\n"
    )

    res = solve_two_stage(inst, scenarios, probs, NetworkOptions())
    print(res.to_table())
    print()
    print(f"VSS  = {res.vss:,.0f}  ({res.vss_pct:.2f}% of RP)")
    print(f"EVPI = {res.evpi:,.0f}  ({res.evpi_pct:.2f}% of RP)")

    print("\nWhat the two designs actually are")
    print("=" * 33)
    print(
        markdown_table(
            ["design", "DCs", "plants"],
            [
                [
                    "RP (stochastic)",
                    " ".join(res.rp_solution.open_dcs),
                    " ".join(res.rp_solution.open_plants),
                ],
                [
                    "EV (mean demand)",
                    " ".join(res.ev_solution.open_dcs),
                    " ".join(res.ev_solution.open_plants),
                ],
            ],
        )
    )

    print(
        f"\n{100 * res.network_stability:.0f}% of the wait-and-see solutions pick the same DC "
        "set as the stochastic solution."
    )
    print(
        "\nRead it this way: designing on the forecast costs "
        f"{res.vss:,.0f} per period in expectation, and no forecasting programme, however "
        f"good, can be worth more than {res.evpi:,.0f} per period. If a forecast-accuracy "
        "initiative is pitched above that number, the business case is wrong regardless "
        "of the modelling."
    )

    print("\nPer-scenario wait-and-see cost")
    print("=" * 30)
    rows = []
    for i, (obj, dcs) in enumerate(zip(res.ws_objectives, res.ws_open_sets)):
        rows.append(
            [i, f"{sum(scenarios[i].values()):,.0f}", f"{obj:,.0f}", " ".join(dcs)]
        )
    print(markdown_table(["scenario", "demand units", "cost if known in advance", "DCs"], rows))


if __name__ == "__main__":
    main()
