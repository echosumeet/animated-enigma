"""Triaging an infeasible network: from "infeasible" to "here is what to change".

    PYTHONPATH=src python examples/04_infeasibility_triage.py

The scenario is the one that actually happens: volume grows, someone caps the
number of DCs for capex reasons, and the model comes back infeasible the
morning of the review.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from netdesign.diagnostics import capacity_ledger, diagnose
from netdesign.instances import generate_instance
from netdesign.network_flow import NetworkOptions, solve_network_design


def main() -> None:
    inst = generate_instance(seed=7)
    print("Step 0 - the base plan solves")
    base = solve_network_design(inst)
    print(f"  {base.status}, {base.objective:,.0f}/period, DCs {' '.join(base.open_dcs)}\n")

    print("Step 1 - the ask: 80% more volume, and no more than 3 DCs")
    stressed = inst.with_demand(inst.scaled_demand(1.8))
    options = NetworkOptions(max_open_dcs=3)
    naive = solve_network_design(stressed, options)
    print(f"  solver says: {naive.status}")
    print("  which tells the room precisely nothing.\n")

    print("Step 2 - the capacity ledger, before touching the solver")
    ledger = capacity_ledger(stressed)
    print(f"  {'commodity':<10}{'supply':>13}{'plant cap':>13}{'dc cap':>13}{'demand':>13}")
    for c, row in ledger.items():
        print(
            f"  {c:<10}{row['supply']:>13,.0f}{row['plant_capacity']:>13,.0f}"
            f"{row['dc_capacity']:>13,.0f}{row['demand']:>13,.0f}"
        )
    print()

    print("Step 3 - elastic relaxation: minimise total violation and read it off")
    diag = diagnose(stressed, options)
    print(diag.summary())

    print("\n\nStep 4 - act on it")
    print("=" * 18)
    relaxed = NetworkOptions(max_open_dcs=5)
    with_more_dcs = solve_network_design(stressed, relaxed)
    print(f"  lift the DC cap to 5: {with_more_dcs.status}")
    if not with_more_dcs.is_optimal:
        print("  still infeasible - so the DC cap was not the binding constraint.")
    priced = solve_network_design(
        stressed, NetworkOptions(max_open_dcs=5, allow_unmet=True, unmet_penalty=45.0)
    )
    print(
        f"  price the shortfall instead: {priced.status}, {priced.objective:,.0f}/period "
        f"with {priced.unmet_units:,.0f} units unserved"
    )
    print(
        "\nThat is the useful answer. The network cannot physically carry 1.8x volume with "
        "the contracted supply, so the decision is not 'how many DCs' - it is how much "
        "demand to walk away from, or how much more supply to contract, and what that "
        "trade is worth per period."
    )


if __name__ == "__main__":
    main()
