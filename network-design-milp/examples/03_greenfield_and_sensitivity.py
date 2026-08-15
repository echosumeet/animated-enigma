"""Greenfield heuristic, then the sweep that decides what to commit to.

    PYTHONPATH=src python examples/03_greenfield_and_sensitivity.py

The greenfield answer is priced against the MILP optimum rather than compared
in kilometres, and the sweep output is reduced to the only thing a steering
committee can act on: core sites, swing sites, and sites to stop discussing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from netdesign.greenfield import (
    evaluate_open_set,
    greenfield_open_set,
    snap_report,
)
from netdesign.instances import generate_instance
from netdesign.network_flow import NetworkOptions, solve_network_design
from netdesign.reporting import markdown_table
from netdesign.scenarios import elasticity, sensitivity_sweep, stability_profile, sweep_table


def main() -> None:
    inst = generate_instance(seed=7)
    optimum = solve_network_design(inst)
    print(inst.summary())
    print(
        f"\nMILP optimum {optimum.objective:,.0f}/period with "
        f"{' '.join(optimum.open_dcs)}\n"
    )

    print("Greenfield: geography only, then priced")
    print("=" * 39)
    rows = []
    for p in (2, 3, 4, 5):
        gf = greenfield_open_set(inst, p, seed=3)
        costed = evaluate_open_set(inst, gf.snapped_dcs)
        rows.append(
            [
                p,
                " ".join(gf.snapped_dcs),
                f"{gf.weighted_km:,.0f}",
                f"{costed.objective:,.0f}" if costed.is_optimal else costed.status,
                f"{100 * (costed.objective - optimum.objective) / optimum.objective:+.2f}%"
                if costed.is_optimal
                else "-",
            ]
        )
    print(markdown_table(["p", "snapped DCs", "wtd km", "costed", "vs optimum"], rows))
    print()
    print(snap_report(inst, greenfield_open_set(inst, 3, seed=3)))
    print(
        "\nNote what the heuristic cannot see: fixed cost and capacity. It keeps adding "
        "sites because distance always falls, and the cost curve turns around long before "
        "the distance curve does."
    )

    print("\n\nSensitivity sweep: demand x transport rates")
    print("=" * 43)
    runs = sensitivity_sweep(inst, options=NetworkOptions())
    headers, table = sweep_table(runs)
    print(markdown_table(headers, table))

    prof = stability_profile(runs)
    print(
        f"\n{prof['n_runs']} runs, {prof['distinct_networks']} distinct networks, "
        f"mean {prof['mean_dcs']:.1f} DCs open"
    )
    print(f"core  (commit)          : {', '.join(prof['core']) or '-'}")
    print(f"swing (the real debate) : {', '.join(prof['swing']) or '-'}")
    print(f"out   (stop discussing) : {', '.join(prof['out']) or '-'}")
    slope = elasticity(runs, "demand_multiplier")
    print(f"\ncost elasticity to demand (log-log slope): {slope:.2f}")
    print(
        "Below 1 the network absorbs growth; above 1 it is buying lumpy capacity faster "
        "than volume arrives. Check this before signing off a savings case that assumes "
        "a volume ramp."
    )


if __name__ == "__main__":
    main()
