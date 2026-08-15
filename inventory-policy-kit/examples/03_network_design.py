"""Where should the inventory sit? Three views of the same network question.

Run:  PYTHONPATH=src python examples/03_network_design.py

1. Clark-Scarf: how much stock each echelon of a serial chain should hold, and
   proof by simulation that the answer is right.
2. Graves-Willems: where on a BOM tree the buffer belongs at all.
3. Risk pooling: how much consolidation is really worth once demand correlation
   and the longer outbound lead time are counted.

These are three different questions and they get confused constantly. The first
sets levels. The second sets *placement*. The third sets structure.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invkit.guaranteed_service import example_bom_tree, solve_guaranteed_service
from invkit.pooling import pooling_with_lead_time_penalty, square_root_law
from invkit.serial import SerialStage, clark_scarf, simulate_serial_system

DEMAND_MEAN, DEMAND_SD = 100.0, 30.0


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def part_one() -> None:
    rule("1. Clark-Scarf: optimal echelon base stock for a serial chain")
    stages = [
        SerialStage("store", lead_time=1, echelon_holding=1.2),
        SerialStage("dc", lead_time=2, echelon_holding=0.5),
        SerialStage("plant", lead_time=4, echelon_holding=0.4),
    ]
    penalty = 20.0
    sol = clark_scarf(stages, DEMAND_MEAN, DEMAND_SD, penalty=penalty)

    print(f"{'stage':<10}{'L':>4}{'echelon h':>12}{'installation h':>17}"
          f"{'echelon S*':>13}{'implied local stock':>22}")
    print("-" * 78)
    local = sol.installation_holding_costs()
    prev = 0.0
    for st, s, h_local in zip(stages, sol.echelon_base_stock, local):
        print(f"{st.name:<10}{st.lead_time:>4}{st.echelon_holding:>12.2f}{h_local:>17.2f}"
              f"{s:>13,.0f}{s - prev:>22,.0f}")
        prev = s
    print()
    print(f"optimal expected cost per period (exact DP): {sol.optimal_cost:,.2f}")

    sim = simulate_serial_system(
        stages, sol.echelon_base_stock, DEMAND_MEAN, DEMAND_SD, penalty,
        n_periods=120_000, seed=17,
    )
    print(f"simulated cost per period:                   {sim['avg_cost_per_period']:,.2f} "
          f"({100 * (sim['avg_cost_per_period'] / sol.optimal_cost - 1):+.2f}%)")
    print(f"simulated fill rate at the customer:         {sim['fill_rate']:.4f}")
    print()
    print("Perturbing the optimum and re-simulating:")
    for label, delta in (("store +60", (60, 0, 0)), ("dc +80", (0, 80, 0)),
                         ("plant +120", (0, 0, 120)), ("store -60", (-60, 0, 0))):
        perturbed = [s + d for s, d in zip(sol.echelon_base_stock, delta)]
        cost = simulate_serial_system(
            stages, perturbed, DEMAND_MEAN, DEMAND_SD, penalty, n_periods=120_000, seed=17
        )["avg_cost_per_period"]
        print(f"  {label:<12} {cost:>9,.2f}  ({100 * (cost / sim['avg_cost_per_period'] - 1):+.2f}%)")
    print()
    print("Note the shape of the answer: most of the echelon stock sits upstream where a")
    print("unit is cheap to hold, and the store carries only what its own 1-day exposure")
    print("needs. Setting an independent reorder point per site cannot produce this - it")
    print("charges every level for the same demand risk.")


def part_two() -> None:
    rule("2. Graves-Willems: where the buffer belongs on a BOM tree")
    tree = example_bom_tree()
    res = solve_guaranteed_service(tree, service_level=0.95, holding_rate=0.25)
    demand = tree.propagate_demand()

    print(f"{'stage':<16}{'T':>4}{'cost':>7}{'sd':>8}{'SI':>5}{'S':>5}{'tau':>6}"
          f"{'safety stock':>15}{'cost/period':>14}")
    print("-" * 80)
    for name in sorted(tree.stages, key=lambda n: -res.safety_stock_cost[n]):
        st = tree.stages[name]
        print(
            f"{name:<16}{st.processing_time:>4}{st.cost_added:>7,.0f}{demand[name][1]:>8,.1f}"
            f"{res.inbound_service_times[name]:>5}{res.service_times[name]:>5}"
            f"{res.net_replenishment_times[name]:>6}"
            f"{res.safety_stock[name]:>15,.0f}{res.safety_stock_cost[name]:>14,.0f}"
        )
    print("-" * 80)
    print(f"{'total':<16}{'':>4}{'':>7}{'':>8}{'':>5}{'':>5}{'':>6}{'':>15}"
          f"{res.total_cost:>14,.0f}")
    print()
    zero = sorted(k for k, v in res.net_replenishment_times.items() if v == 0)
    print(f"Stages holding nothing: {', '.join(zero)}")
    print(f"Decoupling points:      {', '.join(res.decoupling_points)}")
    print()
    print("`build` and `pack` are pure pass-throughs: they quote a service time equal to")
    print("their inbound time plus processing, so they never hold a buffer. `raw_A` also")
    print("holds nothing because `raw_B`, which is cheaper per unit, is the better place to")
    print("absorb the same upstream variability. That trade is the whole model - it is not")
    print("something a per-site service-level policy can express.")

    print()
    print("Sensitivity to the customer promise at dc_north:")
    from invkit.guaranteed_service import Stage, SupplyChainTree

    for promise in (0, 2, 4, 6):
        variant = SupplyChainTree(stages=dict(tree.stages), arcs=dict(tree.arcs))
        st = variant.stages["dc_north"]
        variant.stages["dc_north"] = Stage(
            st.name, st.processing_time, st.cost_added, st.demand_mean, st.demand_sd,
            max_service_time=promise,
        )
        r = solve_guaranteed_service(variant, 0.95, 0.25)
        print(f"  promise {promise} days -> total cost {r.total_cost:>9,.0f} "
              f"({len(r.decoupling_points)} decoupling points)")
    print()
    print("Every day of quoted lead time you can win back from the customer is worth real")
    print("money, and this is how you price it before the negotiation rather than after.")


def part_three() -> None:
    rule("3. Risk pooling: what consolidation is really worth")
    print(f"{'locations':>10}{'rho':>7}{'decentralised':>16}{'centralised':>14}"
          f"{'reduction':>12}")
    print("-" * 60)
    for n in (4, 8, 16):
        for rho in (0.0, 0.3, 0.6):
            r = square_root_law(n, DEMAND_SD, 0.95, correlation=rho)
            print(f"{n:>10}{rho:>7.1f}{r.decentralised_ss:>16,.0f}"
                  f"{r.centralised_ss:>14,.0f}{r.reduction_pct:>11.1f}%")
    print()
    print("And with the longer outbound lead time centralisation usually brings:")
    print()
    print(f"{'central L':>11}{'decentralised':>16}{'centralised':>14}{'net':>10}")
    print("-" * 52)
    for lt in (2.0, 4.0, 6.0, 8.0):
        out = pooling_with_lead_time_penalty(
            8, DEMAND_SD, lead_time_local=2.0, lead_time_central=lt, correlation=0.3
        )
        print(f"{lt:>11.1f}{out['decentralised_ss']:>16,.0f}"
              f"{out['centralised_ss']:>14,.0f}{out['net_reduction_pct']:>9.1f}%")
    out = pooling_with_lead_time_penalty(8, DEMAND_SD, 2.0, 2.0, correlation=0.3)
    print()
    print(f"Break-even central lead time at rho = 0.3: "
          f"{out['breakeven_central_lead_time']:.1f} days. Past that the pooling benefit is")
    print("gone and you are paying outbound freight for the privilege. The independence")
    print("assumption and the lead-time penalty are the two things network-design cases")
    print("leave out, and together they usually account for most of the promised saving.")


def main() -> int:
    part_one()
    part_two()
    part_three()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
