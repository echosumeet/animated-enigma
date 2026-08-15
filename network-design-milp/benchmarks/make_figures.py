"""Regenerate the figures in docs/ by actually solving the models.

    PYTHONPATH=src python benchmarks/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from netdesign.figures import plot_network, plot_tradeoffs  # noqa: E402
from netdesign.instances import demand_scenarios, generate_instance  # noqa: E402
from netdesign.network_flow import NetworkOptions, solve_network_design  # noqa: E402
from netdesign.scenarios import sensitivity_sweep, stability_profile  # noqa: E402
from netdesign.stochastic import solve_two_stage  # noqa: E402

DOCS = ROOT / "docs"


def main() -> int:
    DOCS.mkdir(exist_ok=True)
    inst = generate_instance(seed=7)

    base = solve_network_design(inst)
    p1 = plot_network(inst, base, str(DOCS / "network_map.png"))
    print(f"wrote {p1}")

    curve: list[tuple[int, float, float]] = []
    for k in range(2, len(inst.dcs) + 1):
        sol = solve_network_design(inst, NetworkOptions(min_open_dcs=k, max_open_dcs=k))
        if sol.objective is not None:
            curve.append((k, float(sol.objective), sol.demand_weighted_km))

    runs = sensitivity_sweep(inst)
    prof = stability_profile(runs)
    frequency = {d.id: prof["frequency"].get(d.id, 0.0) for d in inst.dcs}

    scen, probs = demand_scenarios(inst, n_scenarios=12, seed=11)
    st = solve_two_stage(inst, scen, probs)

    p2 = plot_tradeoffs(
        curve,
        frequency,
        {"ws": st.ws, "rp": st.rp, "eev": st.eev},
        str(DOCS / "tradeoffs.png"),
    )
    print(f"wrote {p2}")
    print(
        f"deterministic {base.objective:,.0f} | RP {st.rp:,.0f} | "
        f"VSS {st.vss:,.0f} | EVPI {st.evpi:,.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
