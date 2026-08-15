"""End-to-end network design study on a generated instance.

    PYTHONPATH=src python examples/01_network_design_walkthrough.py

Solves the base model, prints the report a network review actually needs, then
prices the two service constraints that always come up in the room: single
sourcing, and a mandated second source.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from netdesign.instances import generate_instance
from netdesign.network_flow import NetworkDesignModel, NetworkOptions, solve_network_design
from netdesign.reporting import format_report, markdown_table


def main() -> None:
    inst = generate_instance(seed=7)
    print(inst.summary())
    problems = inst.validate()
    print(f"structural problems: {problems or 'none'}\n")

    builder = NetworkDesignModel(inst, NetworkOptions())
    base, _ = builder.solve()
    print(format_report(base, inst))

    lp = builder.lp_bound()
    print()
    print(
        f"LP relaxation {lp:,.0f}  ->  integrality gap "
        f"{100 * (base.objective - lp) / base.objective:.2f}%"
    )
    print(
        "That gap is the honest measure of the formulation, not of the instance: "
        "the same feasible set with an aggregated linking constraint reports a much "
        "worse bound (see benchmarks/results.md)."
    )

    print("\n\nWhat the service constraints cost")
    print("=" * 34)
    rows = []
    for label, opts in [
        ("cost-optimal", {}),
        ("single sourcing", dict(single_source=True)),
        ("second source >= 15%", dict(min_second_source_share=0.15)),
        ("second source >= 25%", dict(min_second_source_share=0.25)),
    ]:
        sol = solve_network_design(inst, NetworkOptions(**opts))
        rows.append(
            [
                label,
                f"{sol.objective:,.0f}",
                f"{100 * (sol.objective - base.objective) / base.objective:+.2f}%",
                len(sol.open_dcs),
                f"{sum(len(m) for m in sol.zone_service.values()) / len(sol.zone_service):.2f}",
                f"{sol.demand_weighted_km:,.0f}",
            ]
        )
    print(
        markdown_table(
            ["variant", "cost/period", "vs optimum", "DCs", "DCs/zone", "wtd km"], rows
        )
    )
    print(
        "\nSingle sourcing is nearly free here because the cost-optimal answer already "
        "sends most zones to one DC. A mandated second source is not free - it forces "
        "volume onto a DC that is not the nearest, and the last-leg distance rises with it."
    )


if __name__ == "__main__":
    main()
