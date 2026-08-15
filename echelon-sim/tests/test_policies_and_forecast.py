"""Policies, forecasters and lead-time distributions.

Unit-level checks on the pieces that sit inside the replenishment loop, done
against hand-computable values rather than against yesterday's output.
"""

import math
import unittest

import numpy as np

from echelonsim.demand import AR1, IIDNormal, SeasonalTrend, ShockOverlay, demand_path
from echelonsim.forecast import (
    DampedTrend,
    ExponentialSmoothing,
    Forecast,
    MovingAverage,
    Oracle,
)
from echelonsim.leadtime import (
    Deterministic,
    DiscreteLeadTime,
    GammaLeadTime,
    NoCrossingWrapper,
)
from echelonsim.policies import BaseStock, Batched, RSPolicy, SsPolicy


class TestForecasters(unittest.TestCase):
    def test_moving_average_is_the_arithmetic_mean_of_the_window(self):
        forecaster = MovingAverage(window=4)
        forecaster.reset(100.0, 10.0)
        for value in (10.0, 20.0, 30.0, 40.0, 50.0):
            forecaster.update(value)
        # Only the last four survive the window.
        self.assertAlmostEqual(forecaster.forecast().mean, 35.0)

    def test_moving_average_falls_back_before_it_has_data(self):
        forecaster = MovingAverage(window=6)
        forecaster.reset(120.0, 15.0)
        self.assertEqual(forecaster.forecast().mean, 120.0)
        self.assertEqual(forecaster.forecast().std, 15.0)

    def test_exponential_smoothing_follows_its_recursion(self):
        forecaster = ExponentialSmoothing(alpha=0.25)
        forecaster.reset(100.0, 0.0)
        level = 100.0
        for observation in (140.0, 90.0, 110.0):
            forecaster.update(observation)
            level = 0.25 * observation + 0.75 * level
        self.assertAlmostEqual(forecaster.forecast().mean, level, places=10)

    def test_exponential_smoothing_converges_to_a_constant_signal(self):
        forecaster = ExponentialSmoothing(alpha=0.3)
        forecaster.reset(0.0, 0.0)
        for _ in range(300):
            forecaster.update(80.0)
        self.assertAlmostEqual(forecaster.forecast().mean, 80.0, places=6)
        self.assertAlmostEqual(forecaster.forecast().std, 0.0, places=6)

    def test_mad_to_sigma_conversion_recovers_the_normal_case(self):
        rng = np.random.default_rng(3)
        forecaster = ExponentialSmoothing(alpha=0.02, beta=0.02)
        forecaster.reset(100.0, 20.0)
        for _ in range(20000):
            forecaster.update(float(rng.normal(100.0, 20.0)))
        self.assertAlmostEqual(forecaster.forecast().std, 20.0, delta=2.0)

    def test_damped_trend_extrapolates_less_than_the_raw_trend(self):
        damped = DampedTrend(alpha=0.3, beta_trend=0.2, phi=0.6)
        undamped = DampedTrend(alpha=0.3, beta_trend=0.2, phi=1.0)
        for forecaster in (damped, undamped):
            forecaster.reset(100.0, 5.0)
            for step in range(30):
                forecaster.update(100.0 + 5.0 * step)
        self.assertLess(damped.forecast().mean, undamped.forecast().mean)

    def test_oracle_ignores_observations(self):
        forecaster = Oracle()
        forecaster.reset(100.0, 20.0)
        for _ in range(50):
            forecaster.update(999.0)
        self.assertEqual(forecaster.forecast().mean, 100.0)
        self.assertEqual(forecaster.forecast().std, 20.0)


class TestPolicies(unittest.TestCase):
    def test_base_stock_target_uses_the_protection_interval(self):
        policy = BaseStock(z=2.0)
        decision = policy.decide(0.0, Forecast(100.0, 20.0, 10), protection=4.0)
        self.assertAlmostEqual(decision.target_level, 100.0 * 4.0 + 2.0 * 20.0 * 2.0)
        self.assertAlmostEqual(decision.quantity, decision.target_level)

    def test_base_stock_clamps_negative_orders_unless_returns_allowed(self):
        forecast = Forecast(100.0, 20.0, 10)
        clamped = BaseStock(z=0.0, allow_returns=False).decide(1000.0, forecast, 3.0)
        signed = BaseStock(z=0.0, allow_returns=True).decide(1000.0, forecast, 3.0)
        self.assertEqual(clamped.quantity, 0.0)
        self.assertAlmostEqual(signed.quantity, -700.0)
        self.assertAlmostEqual(clamped.raw_quantity, -700.0)

    def test_ss_policy_does_not_order_above_the_reorder_point(self):
        policy = SsPolicy(z=1.0, order_quantity=200.0)
        forecast = Forecast(100.0, 20.0, 10)
        reorder = policy.decide(0.0, forecast, 3.0).reorder_point
        self.assertEqual(policy.decide(reorder + 1.0, forecast, 3.0).quantity, 0.0)
        self.assertGreater(policy.decide(reorder - 1.0, forecast, 3.0).quantity, 0.0)

    def test_ss_policy_orders_up_to_s_after_an_undershoot(self):
        policy = SsPolicy(z=0.0, order_quantity=200.0)
        forecast = Forecast(100.0, 20.0, 10)
        decision = policy.decide(0.0, forecast, 3.0)
        self.assertAlmostEqual(decision.target_level, 300.0 + 200.0)
        self.assertAlmostEqual(decision.quantity, 500.0)

    def test_rs_policy_matches_base_stock_at_a_review_epoch(self):
        forecast = Forecast(100.0, 20.0, 10)
        base = BaseStock(z=1.5).decide(120.0, forecast, 5.0)
        periodic = RSPolicy(z=1.5, review_period=4).decide(120.0, forecast, 5.0)
        self.assertAlmostEqual(base.quantity, periodic.quantity)

    def test_batching_rounds_up_to_the_multiple(self):
        policy = Batched(inner=BaseStock(z=0.0), multiple=50.0)
        forecast = Forecast(100.0, 0.0, 10)
        decision = policy.decide(0.0, forecast, 3.05)  # raw order 305
        self.assertAlmostEqual(decision.quantity, 350.0)
        self.assertAlmostEqual(decision.raw_quantity, 305.0)

    def test_batching_is_exact_on_a_boundary(self):
        policy = Batched(inner=BaseStock(z=0.0), multiple=50.0)
        decision = policy.decide(0.0, Forecast(100.0, 0.0, 10), protection=3.0)
        self.assertAlmostEqual(decision.quantity, 300.0)

    def test_minimum_order_quantity_suppresses_small_orders(self):
        policy = Batched(inner=BaseStock(z=0.0), multiple=1.0, minimum=120.0)
        forecast = Forecast(100.0, 0.0, 10)
        self.assertEqual(policy.decide(250.0, forecast, 3.0).quantity, 0.0)  # raw 50
        self.assertGreater(policy.decide(0.0, forecast, 3.0).quantity, 0.0)

    def test_batched_requires_an_inner_policy(self):
        with self.assertRaises(ValueError):
            Batched(multiple=10.0)


class TestLeadTimes(unittest.TestCase):
    def test_gamma_lead_time_hits_its_target_moments(self):
        rng = np.random.default_rng(6)
        lead = GammaLeadTime(mean_lt=6.0, cv_lt=0.4, minimum=1.0)
        draws = np.array([lead.sample(rng) for _ in range(60000)])
        self.assertAlmostEqual(draws.mean(), 6.0, delta=0.05)
        self.assertAlmostEqual(draws.std(), 2.4, delta=0.05)
        self.assertGreaterEqual(draws.min(), 1.0)

    def test_zero_cv_gamma_degenerates_to_deterministic(self):
        rng = np.random.default_rng(1)
        lead = GammaLeadTime(mean_lt=4.0, cv_lt=0.0)
        self.assertEqual(lead.sample(rng), 4.0)
        self.assertEqual(lead.std, 0.0)

    def test_discrete_lead_time_moments_are_exact(self):
        lead = DiscreteLeadTime(values=(2.0, 10.0), probabilities=(0.75, 0.25))
        self.assertAlmostEqual(lead.mean, 4.0)
        self.assertAlmostEqual(lead.std, math.sqrt(0.75 * 4 + 0.25 * 36))

    def test_discrete_lead_time_rejects_unnormalised_probabilities(self):
        with self.assertRaises(ValueError):
            DiscreteLeadTime(values=(1.0, 2.0), probabilities=(0.5, 0.4))

    def test_no_crossing_wrapper_produces_monotone_arrivals(self):
        rng = np.random.default_rng(12)
        wrapper = NoCrossingWrapper(base=GammaLeadTime(mean_lt=4.0, cv_lt=0.8, minimum=0.5))
        wrapper.reset()
        arrivals = []
        for dispatch in range(200):
            wrapper.observe_dispatch(float(dispatch))
            arrivals.append(dispatch + wrapper.sample(rng))
        self.assertEqual(arrivals, sorted(arrivals))

    def test_deterministic_lead_time_has_no_dispersion(self):
        rng = np.random.default_rng(0)
        lead = Deterministic(3.0)
        self.assertEqual({lead.sample(rng) for _ in range(20)}, {3.0})
        self.assertEqual(lead.cv, 0.0)


class TestDemandProcesses(unittest.TestCase):
    def test_iid_normal_matches_its_parameters(self):
        rng = np.random.default_rng(2)
        path = demand_path(IIDNormal(100.0, 15.0), 40000, rng)
        self.assertAlmostEqual(path.mean(), 100.0, delta=0.5)
        self.assertAlmostEqual(path.std(), 15.0, delta=0.5)

    def test_ar1_holds_the_marginal_variance_fixed_across_rho(self):
        """Changing rho must not change how variable demand is, or the bullwhip
        comparison across correlation levels moves for the wrong reason."""
        rng = np.random.default_rng(4)
        low = demand_path(AR1(100.0, 20.0, rho=0.1), 60000, rng)
        high = demand_path(AR1(100.0, 20.0, rho=0.8), 60000, rng)
        self.assertAlmostEqual(low.std(), 20.0, delta=0.7)
        self.assertAlmostEqual(high.std(), 20.0, delta=1.2)

    def test_ar1_reproduces_its_autocorrelation(self):
        rng = np.random.default_rng(5)
        path = demand_path(AR1(100.0, 20.0, rho=0.7), 60000, rng)
        centred = path - path.mean()
        rho = float(np.dot(centred[:-1], centred[1:]) / np.dot(centred, centred))
        self.assertAlmostEqual(rho, 0.7, delta=0.02)

    def test_ar1_rejects_a_non_stationary_parameter(self):
        with self.assertRaises(ValueError):
            AR1(rho=1.0)

    def test_shock_overlay_only_affects_its_window(self):
        rng = np.random.default_rng(7)
        base = IIDNormal(100.0, 0.0)
        shocked = ShockOverlay(base=base, start=10, duration=3, multiplier=2.0)
        path = demand_path(shocked, 20, rng)
        np.testing.assert_allclose(path[10:13], 200.0)
        np.testing.assert_allclose(path[:10], 100.0)
        np.testing.assert_allclose(path[13:], 100.0)

    def test_seasonal_demand_repeats(self):
        rng = np.random.default_rng(9)
        path = demand_path(SeasonalTrend(100.0, 0.0, amplitude=25.0, period_length=12), 36, rng)
        np.testing.assert_allclose(path[:12], path[12:24], atol=1e-9)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
