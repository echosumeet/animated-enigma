"""Metric behaviour, especially the edge cases that break MAPE-style metrics."""

from __future__ import annotations

import unittest

import numpy as np

from dflab import metrics as M


class TestPointMetrics(unittest.TestCase):
    def test_perfect_forecast_is_zero_error(self):
        a = np.array([3.0, 0.0, 7.0, 0.0, 12.0])
        self.assertAlmostEqual(M.wape(a, a), 0.0)
        self.assertAlmostEqual(M.smape(a, a), 0.0)
        self.assertAlmostEqual(M.mae(a, a), 0.0)
        self.assertAlmostEqual(M.bias(a, a), 0.0)

    def test_wape_is_defined_with_zero_actuals(self):
        """MAPE is undefined here; WAPE is not. This is the whole argument."""
        a = np.array([0.0, 0.0, 10.0, 0.0])
        f = np.array([1.0, 1.0, 8.0, 1.0])
        # |e| = 1 + 1 + 2 + 1 = 5, sum|a| = 10
        self.assertAlmostEqual(M.wape(a, f), 0.5)

    def test_wape_returns_nan_on_an_all_zero_window(self):
        a = np.zeros(6)
        f = np.ones(6)
        self.assertTrue(np.isnan(M.wape(a, f)))

    def test_zero_forecast_gives_wape_of_exactly_one(self):
        """Forecasting zero always scores WAPE = 1. Anything above 1 is worse
        than giving up, which makes WAPE readable without a baseline column."""
        rng = np.random.default_rng(3)
        a = rng.gamma(2.0, 5.0, size=40)
        self.assertAlmostEqual(M.wape(a, np.zeros(40)), 1.0)

    def test_smape_is_bounded_and_symmetric_in_magnitude(self):
        a = np.array([10.0, 10.0])
        over = M.smape(a, np.array([20.0, 20.0]))
        under = M.smape(a, np.array([0.0, 0.0]))
        self.assertLessEqual(over, 1.0)
        self.assertLessEqual(under, 1.0)
        self.assertLess(over, under)  # sMAPE is not actually symmetric

    def test_bias_and_tracking_signal_detect_systematic_over_forecast(self):
        a = np.full(20, 10.0)
        f = np.full(20, 13.0)
        self.assertAlmostEqual(M.bias(a, f), 3.0)
        self.assertAlmostEqual(M.percent_bias(a, f), 0.3)
        # every error has the same sign, so |TS| equals the number of periods
        self.assertAlmostEqual(M.tracking_signal(a, f), 20.0)

    def test_tracking_signal_stays_inside_the_limit_for_unbiased_noise(self):
        """A limit scaled to the review window separates clean from biased.

        Note what this test also demonstrates: the folklore +/-4 limit applied
        to a 13-period window has a false-alarm rate well above a third,
        because the statistic's spread is about 1.25*sqrt(n). Scaling the limit
        to the window length is not a refinement, it is the difference between
        an exception report people read and one they mute."""
        rng = np.random.default_rng(11)
        n = 13
        limit = 2.5 * np.sqrt(n)
        clean = biased = folklore_false_alarms = 0
        for _ in range(400):
            a = np.full(n, 50.0)
            f_clean = a + rng.normal(0.0, 5.0, size=n)
            f_bias = a + 5.0 + rng.normal(0.0, 5.0, size=n)
            ts_clean = abs(M.tracking_signal(a, f_clean))
            clean += ts_clean > limit
            biased += abs(M.tracking_signal(a, f_bias)) > limit
            folklore_false_alarms += ts_clean > 4.0
        self.assertLess(clean / 400.0, 0.12)
        self.assertGreater(biased / 400.0, 0.70)
        self.assertGreater(folklore_false_alarms / 400.0, 0.25)

    def test_tracking_signal_grows_with_window_length_under_no_bias(self):
        rng = np.random.default_rng(17)
        def spread(n):
            vals = [
                abs(M.tracking_signal(np.full(n, 50.0), 50.0 + rng.normal(0, 5, n)))
                for _ in range(300)
            ]
            return float(np.mean(vals))
        self.assertGreater(spread(200), 2.0 * spread(13))


class TestScaledMetrics(unittest.TestCase):
    def test_seasonal_scale_uses_the_seasonal_lag(self):
        m = 4
        train = np.tile([10.0, 20.0, 30.0, 40.0], 6)
        # perfectly seasonal: naive-1 sees big swings, seasonal naive sees none
        self.assertTrue(np.isnan(M.seasonal_naive_scale(train, m)))
        self.assertGreater(M.seasonal_naive_scale(train, 1), 0.0)

    def test_mase_of_seasonal_naive_is_about_one_on_its_own_denominator(self):
        rng = np.random.default_rng(7)
        m = 12
        season = np.array([5, 6, 9, 14, 20, 25, 27, 24, 18, 12, 8, 6], dtype=float)
        y = np.tile(season, 10) + rng.normal(0, 1.5, 120)
        train, test = y[:108], y[108:]
        f = train[-m:]
        scale = M.seasonal_naive_scale(train, m)
        val = M.mase(test, f, scale)
        self.assertGreater(val, 0.4)
        self.assertLess(val, 1.8)

    def test_mase_is_nan_when_scale_is_degenerate(self):
        self.assertTrue(np.isnan(M.mase([1.0, 2.0], [1.0, 2.0], float("nan"))))
        self.assertTrue(np.isnan(M.rmsse([1.0, 2.0], [1.0, 2.0], 0.0)))

    def test_scale_requires_enough_history(self):
        with self.assertRaises(ValueError):
            M.seasonal_naive_scale(np.arange(5.0), season_length=12)


class TestProbabilisticMetrics(unittest.TestCase):
    def test_pinball_loss_is_minimised_at_the_true_quantile(self):
        rng = np.random.default_rng(42)
        sample = rng.gamma(3.0, 4.0, size=200_000)
        tau = 0.9
        truth = float(np.quantile(sample, tau))
        actual = rng.gamma(3.0, 4.0, size=200_000)
        best = M.pinball_loss(actual, np.full(actual.size, truth), tau)
        for delta in (-2.0, -1.0, 1.0, 2.0):
            worse = M.pinball_loss(actual, np.full(actual.size, truth + delta), tau)
            self.assertGreater(worse, best)

    def test_pinball_penalises_asymmetrically(self):
        a = np.array([10.0])
        tau = 0.9
        under = M.pinball_loss(a, np.array([5.0]), tau)   # forecast too low
        over = M.pinball_loss(a, np.array([15.0]), tau)   # forecast too high
        self.assertAlmostEqual(under, 0.9 * 5.0)
        self.assertAlmostEqual(over, 0.1 * 5.0)
        self.assertGreater(under, over)

    def test_pinball_rejects_invalid_quantiles(self):
        with self.assertRaises(ValueError):
            M.pinball_loss([1.0], [1.0], 0.0)
        with self.assertRaises(ValueError):
            M.pinball_loss([1.0], [1.0], 1.0)

    def test_coverage_matches_the_nominal_level_for_a_correct_quantile(self):
        rng = np.random.default_rng(5)
        a = rng.normal(100.0, 10.0, size=50_000)
        q95 = float(np.quantile(a, 0.95))
        self.assertAlmostEqual(M.coverage(a, np.full(a.size, q95), 0.95), 0.95, places=2)


class TestValueAddAndBundle(unittest.TestCase):
    def test_forecast_value_add_signs(self):
        self.assertAlmostEqual(M.forecast_value_add(0.4, 0.5), 0.2)
        self.assertAlmostEqual(M.forecast_value_add(0.6, 0.5), -0.2)
        self.assertTrue(np.isnan(M.forecast_value_add(0.4, 0.0)))

    def test_evaluate_returns_the_full_bundle_with_pooling_components(self):
        rng = np.random.default_rng(1)
        train = rng.gamma(2.0, 5.0, size=60)
        actual = rng.gamma(2.0, 5.0, size=12)
        fc = np.full(12, float(np.mean(train)))
        out = M.evaluate(
            actual,
            fc,
            train=train,
            season_length=12,
            quantile_forecasts={0.5: fc, 0.9: fc * 1.6},
        )
        for key in ("wape", "mase", "rmsse", "smape", "abs_err", "abs_actual", "n"):
            self.assertIn(key, out)
        self.assertAlmostEqual(out["abs_actual"], float(np.sum(actual)))
        self.assertIn("pinball_0.5", out)
        self.assertIn("coverage_0.9", out)
        self.assertIn("pinball_mean", out)

    def test_shape_mismatch_is_an_error(self):
        with self.assertRaises(ValueError):
            M.wape([1.0, 2.0], [1.0])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
