"""Multi-echelon: Clark-Scarf against simulation, Graves-Willems against brute force."""

from __future__ import annotations

import math
import random
import unittest

from invkit.guaranteed_service import (
    Stage,
    SupplyChainTree,
    enumerate_optimal_cost,
    example_bom_tree,
    solve_guaranteed_service,
)
from invkit.serial import SerialStage, clark_scarf, discretised_demand_pmf, simulate_serial_system


class TestClarkScarf(unittest.TestCase):
    """The dynamic program is checked against a simulation of the same system."""

    def test_single_stage_reduces_to_the_newsvendor(self):
        """One stage: the base stock must be the (p+H)/(p+H+e) fractile of D over L+1."""
        stage = SerialStage("only", lead_time=2, echelon_holding=1.0)
        sol = clark_scarf([stage], 100.0, 30.0, penalty=20.0)
        support, pmf = discretised_demand_pmf(100.0, 30.0, stage.lead_time + 1)
        cdf = pmf.cumsum()
        ratio = (20.0 + 1.0) / (20.0 + 1.0 + 1.0)
        idx = int((cdf < ratio).sum())
        self.assertAlmostEqual(sol.echelon_base_stock[0], float(support[idx]), delta=1.0)

    def test_two_stage_cost_matches_simulation(self):
        stages = [SerialStage("retail", 2, 1.0), SerialStage("dc", 3, 0.6)]
        sol = clark_scarf(stages, 100.0, 30.0, penalty=20.0)
        sim = simulate_serial_system(
            stages, sol.echelon_base_stock, 100.0, 30.0, 20.0, n_periods=60_000, seed=17
        )
        rel = abs(sim["avg_cost_per_period"] / sol.optimal_cost - 1.0)
        self.assertLess(rel, 0.02, msg=f"dp={sol.optimal_cost} sim={sim['avg_cost_per_period']}")

    def test_three_stage_cost_matches_simulation(self):
        stages = [
            SerialStage("store", 1, 1.2),
            SerialStage("dc", 2, 0.5),
            SerialStage("plant", 4, 0.4),
        ]
        sol = clark_scarf(stages, 100.0, 30.0, penalty=20.0)
        sim = simulate_serial_system(
            stages, sol.echelon_base_stock, 100.0, 30.0, 20.0, n_periods=60_000, seed=23
        )
        rel = abs(sim["avg_cost_per_period"] / sol.optimal_cost - 1.0)
        self.assertLess(rel, 0.02, msg=f"dp={sol.optimal_cost} sim={sim['avg_cost_per_period']}")

    def test_dp_levels_beat_perturbed_levels_in_simulation(self):
        """The optimum is a real optimum, not just a fixed point of the recursion."""
        stages = [SerialStage("retail", 2, 1.0), SerialStage("dc", 3, 0.6)]
        sol = clark_scarf(stages, 100.0, 30.0, penalty=20.0)
        base = simulate_serial_system(
            stages, sol.echelon_base_stock, 100.0, 30.0, 20.0, n_periods=80_000, seed=29
        )["avg_cost_per_period"]
        for delta in ((60, 0), (-60, 0), (0, 80), (0, -80)):
            perturbed = [s + d for s, d in zip(sol.echelon_base_stock, delta)]
            cost = simulate_serial_system(
                stages, perturbed, 100.0, 30.0, 20.0, n_periods=80_000, seed=29
            )["avg_cost_per_period"]
            self.assertGreater(cost, base, msg=f"perturbation {delta}")

    def test_echelon_levels_are_monotone_upstream(self):
        stages = [
            SerialStage("store", 1, 1.2),
            SerialStage("dc", 2, 0.5),
            SerialStage("plant", 4, 0.4),
        ]
        sol = clark_scarf(stages, 100.0, 30.0, penalty=20.0)
        levels = sol.echelon_base_stock
        self.assertEqual(levels, sorted(levels))

    def test_installation_costs_recovered_from_echelon_costs(self):
        stages = [
            SerialStage("store", 1, 1.2),
            SerialStage("dc", 2, 0.5),
            SerialStage("plant", 4, 0.4),
        ]
        sol = clark_scarf(stages, 100.0, 30.0, penalty=20.0)
        self.assertEqual(sol.installation_holding_costs(), [2.1, 0.9, 0.4])

    def test_higher_penalty_raises_every_base_stock_level(self):
        stages = [SerialStage("retail", 2, 1.0), SerialStage("dc", 3, 0.6)]
        low = clark_scarf(stages, 100.0, 30.0, penalty=5.0).echelon_base_stock
        high = clark_scarf(stages, 100.0, 30.0, penalty=100.0).echelon_base_stock
        for a, b in zip(low, high):
            self.assertLess(a, b)

    def test_rejects_empty_system(self):
        with self.assertRaises(ValueError):
            clark_scarf([], 100.0, 30.0, 10.0)


class TestGuaranteedService(unittest.TestCase):
    """The tree DP is checked against exhaustive enumeration."""

    @staticmethod
    def _random_tree(rng: random.Random, n: int) -> SupplyChainTree:
        tree = SupplyChainTree(stages={})
        names = [f"n{i}" for i in range(n)]
        for name in names:
            tree.add_stage(
                Stage(name, processing_time=rng.randint(0, 4), cost_added=rng.uniform(1.0, 20.0))
            )
        for i in range(1, n):
            j = rng.randrange(i)
            if rng.random() < 0.5:
                tree.add_arc(names[i], names[j], 1.0)
            else:
                tree.add_arc(names[j], names[i], 1.0)
        for name in names:
            if not tree.customers(name):
                st = tree.stages[name]
                tree.stages[name] = Stage(
                    name,
                    st.processing_time,
                    st.cost_added,
                    demand_mean=rng.uniform(10.0, 50.0),
                    demand_sd=rng.uniform(5.0, 25.0),
                    max_service_time=rng.randint(0, 3),
                )
        return tree

    def test_dp_matches_brute_force_on_random_trees(self):
        rng = random.Random(20260815)
        for trial in range(20):
            tree = self._random_tree(rng, rng.randint(3, 5))
            dp = solve_guaranteed_service(tree, 0.95, 0.25)
            bf = enumerate_optimal_cost(tree, 0.95, 0.25)
            self.assertAlmostEqual(dp.total_cost, bf, delta=1e-6 * max(1.0, bf), msg=f"trial {trial}")

    def test_solution_is_feasible(self):
        """Every constraint of the guaranteed-service formulation must hold."""
        tree = example_bom_tree()
        res = solve_guaranteed_service(tree, 0.95, 0.25)
        for name, stage in tree.stages.items():
            si = res.inbound_service_times[name]
            s = res.service_times[name]
            self.assertGreaterEqual(si + stage.processing_time - s, 0, msg=f"{name} tau < 0")
            self.assertEqual(res.net_replenishment_times[name], si + stage.processing_time - s)
            for supplier in tree.suppliers(name):
                self.assertGreaterEqual(si, res.service_times[supplier], msg=f"{name} <- {supplier}")
            if stage.max_service_time is not None:
                self.assertLessEqual(s, stage.max_service_time, msg=f"{name} service promise")

    def test_safety_stock_follows_the_square_root_of_net_replenishment_time(self):
        tree = example_bom_tree()
        res = solve_guaranteed_service(tree, 0.95, 0.25)
        demand = tree.propagate_demand()
        for name in tree.stages:
            tau = res.net_replenishment_times[name]
            expected = res.z * demand[name][1] * math.sqrt(max(tau, 0))
            self.assertAlmostEqual(res.safety_stock[name], expected, places=9)

    def test_most_stages_hold_no_safety_stock(self):
        """The characteristic guaranteed-service answer: a few decoupling points."""
        tree = example_bom_tree()
        res = solve_guaranteed_service(tree, 0.95, 0.25)
        zero = [k for k, v in res.net_replenishment_times.items() if v == 0]
        self.assertGreaterEqual(len(zero), 3)
        self.assertLess(len(res.decoupling_points), len(tree.stages))

    def test_demand_propagates_through_the_bom_with_usage(self):
        tree = example_bom_tree()
        demand = tree.propagate_demand()
        # pack serves both DCs: 120 + 80 = 200 units per period
        self.assertAlmostEqual(demand["pack"][0], 200.0, places=9)
        self.assertAlmostEqual(demand["pack"][1], math.sqrt(40.0 ** 2 + 35.0 ** 2), places=9)
        # raw_A is consumed 2 per subassembly, which serves build (215 per period)
        self.assertAlmostEqual(demand["raw_A"][0], 2.0 * 215.0, places=9)

    def test_tighter_customer_promise_costs_more(self):
        tree = example_bom_tree()
        relaxed = SupplyChainTree(stages=dict(tree.stages), arcs=dict(tree.arcs))
        st = relaxed.stages["dc_north"]
        relaxed.stages["dc_north"] = Stage(
            st.name, st.processing_time, st.cost_added, st.demand_mean, st.demand_sd,
            max_service_time=6,
        )
        base = solve_guaranteed_service(tree, 0.95, 0.25).total_cost
        loose = solve_guaranteed_service(relaxed, 0.95, 0.25).total_cost
        self.assertLess(loose, base)

    def test_higher_service_level_scales_cost_by_z(self):
        tree = example_bom_tree()
        a = solve_guaranteed_service(tree, 0.95, 0.25)
        b = solve_guaranteed_service(tree, 0.99, 0.25)
        self.assertAlmostEqual(b.total_cost / a.total_cost, b.z / a.z, places=6)

    def test_non_tree_input_is_rejected(self):
        tree = SupplyChainTree(stages={})
        for n in ("a", "b", "c"):
            tree.add_stage(Stage(n, 1, 10.0))
        tree.add_arc("a", "b")
        tree.add_arc("b", "c")
        tree.add_arc("a", "c")  # creates a cycle in the undirected sense
        with self.assertRaises(ValueError):
            solve_guaranteed_service(tree)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
