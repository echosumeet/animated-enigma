"""Lot sizing: EOQ, discounts, and Wagner-Whitin against exhaustive enumeration."""

from __future__ import annotations

import itertools
import math
import random
import unittest

from invkit.lotsizing import (
    PriceBreak,
    _evaluate_plan,
    compare_lot_sizing,
    eoq,
    eoq_all_units_discount,
    eoq_cost_sensitivity,
    eoq_total_cost,
    least_unit_cost,
    lot_for_lot,
    seasonal_demand_series,
    silver_meal,
    wagner_whitin,
)


def brute_force_lot_sizing(demand, setup, holding):
    """Exhaustive search over every order pattern satisfying zero-inventory ordering."""
    T = len(demand)
    best = math.inf
    for mask in range(1 << (T - 1)):
        points = [0] + [i + 1 for i in range(T - 1) if mask >> i & 1]
        orders = [0.0] * T
        for a, b in zip(points, points[1:] + [T]):
            orders[a] = sum(demand[a:b])
        best = min(best, _evaluate_plan(demand, orders, setup, holding, "bf").total_cost)
    return best


class TestEOQ(unittest.TestCase):
    def test_closed_form(self):
        self.assertAlmostEqual(eoq(1000.0, 250.0, 0.02), math.sqrt(2 * 250 * 1000 / 0.02), places=9)

    def test_eoq_minimises_the_relevant_cost(self):
        D, K, h = 5000.0, 180.0, 3.0
        q_star = eoq(D, K, h)
        best = eoq_total_cost(q_star, D, K, h)
        for q in (q_star * 0.5, q_star * 0.9, q_star * 1.1, q_star * 2.0):
            self.assertGreater(eoq_total_cost(q, D, K, h), best)

    def test_ordering_and_holding_costs_are_equal_at_the_optimum(self):
        D, K, h = 5000.0, 180.0, 3.0
        q = eoq(D, K, h)
        self.assertAlmostEqual(K * D / q, h * q / 2.0, places=6)

    def test_cost_curve_is_flat(self):
        """The practical point: a 50% error in Q costs about 8% of relevant cost."""
        self.assertAlmostEqual(eoq_cost_sensitivity(1.5), 1 / 12, places=9)
        self.assertAlmostEqual(eoq_cost_sensitivity(2.0), 0.25, places=9)
        self.assertLess(eoq_cost_sensitivity(1.25), 0.03)

    def test_rejects_non_positive_inputs(self):
        with self.assertRaises(ValueError):
            eoq(0.0, 1.0, 1.0)


class TestQuantityDiscounts(unittest.TestCase):
    def setUp(self):
        self.breaks = [PriceBreak(0, 10.0), PriceBreak(500, 9.6), PriceBreak(2000, 9.2)]

    def test_matches_exhaustive_search_over_candidate_quantities(self):
        D, K, i = 10_000.0, 400.0, 0.25
        result = eoq_all_units_discount(D, K, i, self.breaks)
        # Independent check: scan a dense grid of quantities and price them.
        best_q, best_cost = None, math.inf
        for q in [x / 4.0 for x in range(4, 40_001)]:
            price = max(b.price for b in self.breaks if q >= b.min_qty)
            price = min(b.price for b in self.breaks if q >= b.min_qty)
            cost = eoq_total_cost(q, D, K, i * price, price)
            if cost < best_cost:
                best_cost, best_q = cost, q
        self.assertAlmostEqual(result.total_cost, best_cost, delta=1.0)
        self.assertAlmostEqual(result.quantity, best_q, delta=1.0)

    def test_deep_discount_pulls_the_order_up_to_the_break(self):
        cheap = [PriceBreak(0, 10.0), PriceBreak(5000, 7.0)]
        r = eoq_all_units_discount(10_000.0, 400.0, 0.25, cheap)
        self.assertEqual(r.quantity, 5000)
        self.assertEqual(r.unit_price, 7.0)

    def test_trivial_discount_leaves_the_plain_eoq(self):
        tiny = [PriceBreak(0, 10.0), PriceBreak(50_000, 9.999)]
        r = eoq_all_units_discount(10_000.0, 400.0, 0.25, tiny)
        self.assertAlmostEqual(r.quantity, eoq(10_000.0, 400.0, 0.25 * 10.0), delta=1.0)


class TestWagnerWhitin(unittest.TestCase):
    def test_matches_brute_force_on_random_instances(self):
        rng = random.Random(20260815)
        for _ in range(30):
            T = rng.randint(4, 11)
            demand = [float(rng.randint(0, 120)) for _ in range(T)]
            setup = rng.choice([50.0, 150.0, 400.0])
            holding = rng.choice([0.5, 1.0, 3.0])
            self.assertAlmostEqual(
                wagner_whitin(demand, setup, holding).total_cost,
                brute_force_lot_sizing(demand, setup, holding),
                places=6,
                msg=f"demand={demand} K={setup} h={holding}",
            )

    def test_high_setup_cost_collapses_to_a_single_order(self):
        demand = [40.0] * 8
        plan = wagner_whitin(demand, setup_cost=100_000.0, holding_cost=1.0)
        self.assertEqual(plan.n_setups, 1)
        self.assertAlmostEqual(plan.orders[0], 320.0, places=9)

    def test_zero_setup_cost_collapses_to_lot_for_lot(self):
        demand = [40.0, 10.0, 90.0, 5.0]
        plan = wagner_whitin(demand, setup_cost=0.0, holding_cost=1.0)
        self.assertAlmostEqual(plan.holding_cost, 0.0, places=9)

    def test_plan_covers_demand_exactly(self):
        demand = seasonal_demand_series(24, 120.0, 70.0, noise_cv=0.2, seed=3)
        plan = wagner_whitin(demand, 300.0, 1.0)
        self.assertAlmostEqual(sum(plan.orders), sum(demand), places=6)
        self.assertAlmostEqual(plan.ending_inventory[-1], 0.0, places=6)

    def test_empty_horizon(self):
        self.assertEqual(wagner_whitin([], 100.0, 1.0).total_cost, 0.0)


class TestHeuristics(unittest.TestCase):
    def test_no_heuristic_beats_the_exact_dp(self):
        rng = random.Random(4242)
        for _ in range(25):
            T = rng.randint(6, 18)
            demand = [float(rng.randint(0, 200)) for _ in range(T)]
            setup, holding = 300.0, 1.0
            plans = compare_lot_sizing(demand, setup, holding)
            opt = plans["wagner-whitin"].total_cost
            for name, plan in plans.items():
                self.assertGreaterEqual(plan.total_cost, opt - 1e-6, msg=name)

    def test_silver_meal_gap_is_positive_on_a_lumpy_series(self):
        """Silver-Meal over-extends through a cheap period before a spike."""
        demand = seasonal_demand_series(12, 120.0, 70.0, noise_cv=0.25, seed=5)
        plans = compare_lot_sizing(demand, 300.0, 1.0)
        gap = plans["silver-meal"].gap_vs(plans["wagner-whitin"])
        self.assertGreater(gap, 0.02)

    def test_lot_for_lot_has_no_holding_cost(self):
        demand = [10.0, 20.0, 30.0]
        plan = lot_for_lot(demand, 50.0, 2.0)
        self.assertAlmostEqual(plan.holding_cost, 0.0, places=9)
        self.assertEqual(plan.n_setups, 3)

    def test_least_unit_cost_is_a_valid_plan(self):
        demand = seasonal_demand_series(18, 90.0, 40.0, noise_cv=0.3, seed=11)
        plan = least_unit_cost(demand, 250.0, 1.0)
        self.assertAlmostEqual(sum(plan.orders), sum(demand), places=6)

    def test_silver_meal_and_wagner_whitin_agree_on_stationary_demand(self):
        demand = [100.0] * 12
        plans = compare_lot_sizing(demand, 400.0, 1.0)
        self.assertAlmostEqual(
            plans["silver-meal"].total_cost, plans["wagner-whitin"].total_cost, delta=1e-6
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
