"""The modelling layer on its own, on a problem small enough to check by hand.

    PYTHONPATH=src python examples/05_modelling_layer_tour.py

A three-plant, four-customer transportation problem with a fixed charge, built
with the same primitives the full network model uses. The point is that the
source reads like the algebra, and that the wildcard selection is what makes it
read that way.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from netdesign.modeling import ANY, Model, quicksum

PLANTS = {"P1": 900.0, "P2": 700.0, "P3": 1200.0}          # capacity
FIXED = {"P1": 3_500.0, "P2": 2_900.0, "P3": 5_400.0}       # cost to open
DEMAND = {"C1": 400.0, "C2": 620.0, "C3": 350.0, "C4": 480.0}
SHIP = {                                                    # cost per unit
    ("P1", "C1"): 4.0, ("P1", "C2"): 6.5, ("P1", "C3"): 9.0, ("P1", "C4"): 7.5,
    ("P2", "C1"): 7.0, ("P2", "C2"): 4.5, ("P2", "C3"): 5.5, ("P2", "C4"): 8.5,
    ("P3", "C1"): 6.0, ("P3", "C2"): 7.0, ("P3", "C3"): 4.0, ("P3", "C4"): 4.5,
}


def main() -> None:
    m = Model("fixed-charge transportation", sense="min")

    # 1. variable registry: keys are meaningful, not offsets
    open_plant = m.add_vars(PLANTS, name="open", vtype="binary", obj=FIXED)
    ship = m.add_vars(SHIP, name="ship", lb=0.0, obj=SHIP)

    # 2. constraints read like the algebra
    for c, qty in DEMAND.items():
        m.add(ship.sum(ANY, c) == qty, name=f"demand[{c}]", tag="demand")

    for p, cap in PLANTS.items():
        # tight big-M: no more than the plant can make, and no more than the
        # market can absorb
        big_m = min(cap, sum(DEMAND.values()))
        m.link_big_m(ship.sum(p, ANY), open_plant[p], big_m, name=f"capacity[{p}]")

    print(m)
    print()

    lp = m.solve(relax=True)
    sol = m.solve()
    print(f"LP relaxation   {lp.objective:,.2f}")
    print(f"MILP optimum    {sol.objective:,.2f}   ({sol.status}, {sol.runtime * 1000:.0f} ms)")
    print(f"integrality gap {100 * (sol.objective - lp.objective) / sol.objective:.2f}%")
    print()

    opened = [p for p, v in sol.values(open_plant, integral=True).items() if v > 0.5]
    print(f"open plants: {', '.join(opened)}")
    print(f"{'lane':<12}{'units':>10}{'cost':>12}")
    for (p, c), qty in sol.values(ship, nonzero=True).items():
        print(f"{p + ' -> ' + c:<12}{qty:>10,.0f}{qty * SHIP[(p, c)]:>12,.0f}")

    # 3. the same model, with the classic mistake: one large constant for M
    loose = Model("fixed-charge transportation (loose M)", sense="min")
    y = loose.add_vars(PLANTS, name="open", vtype="binary", obj=FIXED)
    x = loose.add_vars(SHIP, name="ship", lb=0.0, obj=SHIP)
    for c, qty in DEMAND.items():
        loose.add(x.sum(ANY, c) == qty)
    for p, cap in PLANTS.items():
        loose.add(x.sum(p, ANY) <= cap)          # capacity, stated separately
        loose.link_big_m(x.sum(p, ANY), y[p], 1e6)  # "a big number"
    loose_lp = loose.solve(relax=True)
    loose_milp = loose.solve()
    print()
    print("Same feasible set, one large constant instead of the tight bound:")
    print(f"  LP relaxation   {loose_lp.objective:,.2f}  (was {lp.objective:,.2f})")
    print(f"  MILP optimum    {loose_milp.objective:,.2f}  (unchanged, as it must be)")
    print(
        f"  integrality gap {100 * (loose_milp.objective - loose_lp.objective) / loose_milp.objective:.2f}%"
        f"  (was {100 * (sol.objective - lp.objective) / sol.objective:.2f}%)"
    )
    print(
        "\nThe optimum is identical; the bound is not. On a problem this small it costs "
        "nothing. On a real network it is the difference between minutes and hours."
    )

    # 4. what the layer does when the model is over-constrained
    infeasible = Model("over-constrained", sense="min")
    z = infeasible.add_vars(PLANTS, name="make", lb=0.0, ub=lambda p: PLANTS[p] * 0.2)
    infeasible.add(quicksum(z.values()) >= sum(DEMAND.values()), name="cover", tag="cover")
    print(f"\nover-constrained variant: {infeasible.solve().status}")
    elastic, rows, slacks = infeasible.elastic_copy(["cover"])
    relaxed = elastic.solve()
    shortfall = sum(relaxed.values(slacks).values())
    print(
        f"minimum violation {shortfall:,.0f} units - i.e. capacity is short by that much, "
        "which is a sentence someone can act on"
    )


if __name__ == "__main__":
    main()
