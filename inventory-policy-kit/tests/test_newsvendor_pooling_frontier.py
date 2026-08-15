"""Newsvendor, risk pooling and the cost-service frontier."""

from __future__ import annotations

import math
import unittest

import numpy as np
from scipy import stats

from invkit.distributions import GammaLTD, NormalLTD
from invkit.frontier import exchange_curve, marginal_cost_of_service, service_frontier
from invkit.newsvendor import (
    critical_fractile,
    critical_fractile_sensitivity,
    expected_newsvendor_cost,
    newsvendor_empirical,
    newsvendor_normal,
)
from invkit.pooling import (
    pooled_sd,
    pooling_with_lead_time_penalty,
    simulate_pooling,
    square_root_law,
)


class TestNewsvendor(unittest.TestCase):
    def test_critical_fractile(self):
        self.assertAlmostEqual(critical_fractile(9.0, 1.0), 0.9, places=12)

    def test_normal_order_quantity_is_the_fractile(self):
        r = newsvendor_normal(500.0, 100.0, underage_cost=9.0, overage_cost=1.0)
        self.assertAlmostEqual(r.order_quantity, 500.0 + stats.norm.ppf(0.9) * 100.0, places=8)

    def test_order_quantity_minimises_expected_cost(self):
        dist = GammaLTD.from_moments(500.0, 150.0)
        cu, co = 12.0, 3.0
        q_star = dist.ppf(critical_fractile(cu, co))
        best = expected_newsvendor_cost(dist, q_star, cu, co)[0]
        for q in (q_star * 0.8, q_star * 0.95, q_star * 1.05, q_star * 1.3):
            self.assertGreater(expected_newsvendor_cost(dist, q, cu, co)[0], best)

    def test_leftover_shortage_identity(self):
        dist = NormalLTD(500.0, 120.0)
        _, leftover, shortage = expected_newsvendor_cost(dist, 620.0, 5.0, 2.0)
        self.assertAlmostEqual(leftover - shortage, 620.0 - dist.mean, places=9)

    def test_empirical_matches_normal_on_normal_data(self):
        rng = np.random.default_rng(1)
        sample = rng.normal(500.0, 100.0, 400_000)
        emp = newsvendor_empirical(sample, 9.0, 1.0)
        norm = newsvendor_normal(500.0, 100.0, 9.0, 1.0)
        self.assertAlmostEqual(emp.order_quantity, norm.order_quantity, delta=3.0)

    def test_empirical_departs_from_normal_on_a_fat_tail(self):
        rng = np.random.default_rng(2)
        sample = rng.lognormal(mean=6.0, sigma=0.7, size=400_000)
        emp = newsvendor_empirical(sample, 19.0, 1.0)
        norm = newsvendor_normal(
            float(sample.mean()), float(sample.std(ddof=1)), 19.0, 1.0
        )
        self.assertGreater(emp.order_quantity / norm.order_quantity, 1.05)

    def test_underestimating_underage_cost_shrinks_the_order(self):
        dist = GammaLTD.from_moments(500.0, 200.0)
        rows = critical_fractile_sensitivity(dist, 18.0, 2.0, cu_multipliers=(0.5, 1.0, 2.0))
        self.assertLess(rows[0]["order_quantity"], rows[1]["order_quantity"])
        self.assertLess(rows[1]["order_quantity"], rows[2]["order_quantity"])
        self.assertAlmostEqual(rows[1]["cost_penalty_pct"], 0.0, places=6)
        self.assertGreater(rows[0]["cost_penalty_pct"], 0.0)

    def test_rejects_non_positive_costs(self):
        with self.assertRaises(ValueError):
            critical_fractile(0.0, 1.0)


class TestRiskPooling(unittest.TestCase):
    def test_square_root_law_under_independence(self):
        r = square_root_law(9, sd_per_location=30.0, service_level=0.95)
        self.assertAlmostEqual(r.effective_sqrt_n, 3.0, places=9)
        self.assertAlmostEqual(r.reduction_pct, 100.0 * (1 - 1 / 3), places=9)

    def test_correlation_erodes_the_benefit(self):
        prev = None
        for rho in (0.0, 0.2, 0.5, 0.8):
            r = square_root_law(8, 30.0, 0.95, correlation=rho)
            if prev is not None:
                self.assertLess(r.reduction_pct, prev)
            prev = r.reduction_pct

    def test_perfect_correlation_gives_no_benefit(self):
        r = square_root_law(6, 30.0, 0.95, correlation=1.0)
        self.assertAlmostEqual(r.reduction_pct, 0.0, places=6)

    def test_pooled_sd_matches_the_equicorrelation_formula(self):
        n, sd, rho = 5, 40.0, 0.3
        expected = sd * math.sqrt(n + n * (n - 1) * rho)
        self.assertAlmostEqual(pooled_sd([sd] * n, rho), expected, places=9)

    def test_simulation_reproduces_the_analytic_ratio(self):
        for rho in (0.0, 0.4):
            out = simulate_pooling(6, 200.0, 40.0, correlation=rho, n_draws=300_000, seed=8)
            self.assertAlmostEqual(
                out["simulated_ratio"], out["analytic_ratio"], delta=0.05,
                msg=f"rho={rho}",
            )

    def test_longer_central_lead_time_can_wipe_out_the_benefit(self):
        out = pooling_with_lead_time_penalty(
            4, sd_per_location=30.0, lead_time_local=2.0, lead_time_central=8.0
        )
        self.assertAlmostEqual(out["breakeven_central_lead_time"], 2.0 * 4.0, places=9)
        self.assertAlmostEqual(out["net_reduction_pct"], 0.0, places=6)

    def test_unequal_locations_pool_less_well(self):
        equal = pooled_sd([50.0] * 4)
        skewed = pooled_sd([95.0, 20.0, 20.0, 20.0])
        self.assertGreater(skewed / (95.0 + 60.0), equal / 200.0)


class TestFrontier(unittest.TestCase):
    def setUp(self):
        self.ltd = GammaLTD.from_moments(500.0, 67.0)
        self.Q = 800.0

    def test_frontier_is_monotone_and_convex_in_service(self):
        # Evenly spaced targets, so equal steps in service can be compared.
        points = service_frontier(
            self.ltd, self.Q, list(np.linspace(0.85, 0.995, 12)),
            unit_cost=25.0, holding_rate=0.25 / 365, basis="fill",
        )
        costs = [p.holding_cost for p in points]
        self.assertEqual(costs, sorted(costs))
        increments = [b - a for a, b in zip(costs, costs[1:])]
        self.assertEqual(increments, sorted(increments))

    def test_achieved_service_equals_the_target(self):
        for basis, attr in (("fill", "achieved_fill"), ("csl", "achieved_csl")):
            for p in service_frontier(
                self.ltd, self.Q, [0.90, 0.95, 0.99], 25.0, 0.25 / 365, basis=basis
            ):
                self.assertAlmostEqual(getattr(p, attr), p.target, places=8)

    def test_fill_basis_is_cheaper_than_csl_basis_at_the_same_number(self):
        curves = exchange_curve(self.ltd, self.Q, 25.0, 0.25 / 365, n_points=8)
        for a, b in zip(curves["fill"], curves["csl"]):
            self.assertAlmostEqual(a.target, b.target, places=9)
            self.assertLess(a.holding_cost, b.holding_cost)

    def test_marginal_cost_of_service_rises(self):
        curves = exchange_curve(self.ltd, self.Q, 25.0, 0.25 / 365, n_points=12)
        rows = marginal_cost_of_service(curves["fill"])
        costs = [r["cost_per_service_point"] for r in rows]
        self.assertEqual(costs, sorted(costs))
        self.assertGreater(costs[-1] / costs[0], 2.0)

    def test_shortage_cost_produces_an_interior_optimum(self):
        points = service_frontier(
            self.ltd, self.Q, list(np.linspace(0.80, 0.999, 40)),
            unit_cost=25.0, holding_rate=0.25 / 365, basis="fill",
            shortage_cost_per_unit=1.0, demand_per_period=100.0,
        )
        totals = [p.total_cost for p in points]
        best = int(np.argmin(totals))
        self.assertGreater(best, 0)
        self.assertLess(best, len(totals) - 1)

    def test_rejects_unknown_basis(self):
        with self.assertRaises(ValueError):
            service_frontier(self.ltd, self.Q, [0.9], 25.0, 0.1, basis="nonsense")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
