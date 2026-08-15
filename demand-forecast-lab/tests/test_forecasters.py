"""Baselines, exponential smoothing and intermittent-demand methods."""

from __future__ import annotations

import unittest

import numpy as np

from dflab.baselines import (
    DriftForecaster,
    MeanForecaster,
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    ZeroForecaster,
)
from dflab.ets import HoltLinear, HoltWinters, SimpleExponentialSmoothing
from dflab.intermittent import CrostonForecaster, SBAForecaster, TSBForecaster

ALL_LOCAL = [
    NaiveForecaster(),
    SeasonalNaiveForecaster(12),
    MovingAverageForecaster(4),
    DriftForecaster(),
    MeanForecaster(),
    ZeroForecaster(),
    SimpleExponentialSmoothing(),
    HoltLinear(damped=True),
    HoltWinters(12, "add"),
    HoltWinters(12, "mul"),
    CrostonForecaster(),
    SBAForecaster(),
    TSBForecaster(),
]


def seasonal_series(n=120, m=12, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    level = 100.0 + 0.4 * t
    season = 1.0 + 0.35 * np.sin(2 * np.pi * t / m)
    return np.clip(level * season + rng.normal(0, 4.0, n), 0.0, None)


class TestContract(unittest.TestCase):
    def test_every_forecaster_returns_the_requested_horizon(self):
        y = seasonal_series()
        for mdl in ALL_LOCAL:
            with self.subTest(model=mdl.name):
                out = mdl.fit(y).predict(7)
                self.assertEqual(out.shape, (7,))
                self.assertTrue(np.all(np.isfinite(out)))

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            NaiveForecaster().predict(3)

    def test_invalid_horizon_raises(self):
        mdl = NaiveForecaster().fit(np.arange(10.0))
        with self.assertRaises(ValueError):
            mdl.predict(0)

    def test_non_finite_input_is_rejected(self):
        with self.assertRaises(ValueError):
            NaiveForecaster().fit(np.array([1.0, np.nan, 3.0]))


class TestBaselines(unittest.TestCase):
    def test_naive_repeats_the_last_value(self):
        y = np.array([4.0, 9.0, 2.0, 11.0])
        np.testing.assert_allclose(NaiveForecaster().fit(y).predict(3), [11.0] * 3)

    def test_seasonal_naive_repeats_the_last_cycle(self):
        m = 4
        y = np.tile([1.0, 2.0, 3.0, 4.0], 5)
        out = SeasonalNaiveForecaster(m).fit(y).predict(6)
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0, 4.0, 1.0, 2.0])

    def test_seasonal_naive_degrades_gracefully_on_short_history(self):
        out = SeasonalNaiveForecaster(52).fit(np.arange(1.0, 6.0)).predict(3)
        self.assertEqual(out.shape, (3,))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_drift_extrapolates_the_average_slope(self):
        y = np.arange(0.0, 10.0)  # slope exactly 1
        out = DriftForecaster().fit(y).predict(3)
        np.testing.assert_allclose(out, [10.0, 11.0, 12.0])

    def test_moving_average_is_flat_at_the_window_mean(self):
        y = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        out = MovingAverageForecaster(3).fit(y).predict(2)
        np.testing.assert_allclose(out, [20.0, 20.0])

    def test_zero_forecaster_is_exactly_zero(self):
        np.testing.assert_allclose(
            ZeroForecaster().fit(seasonal_series()).predict(5), np.zeros(5)
        )


class TestExponentialSmoothing(unittest.TestCase):
    def test_ses_reproduces_a_constant_series(self):
        y = np.full(50, 42.0)
        out = SimpleExponentialSmoothing().fit(y).predict(4)
        np.testing.assert_allclose(out, np.full(4, 42.0), atol=1e-6)

    def test_ses_forecast_is_flat(self):
        out = SimpleExponentialSmoothing().fit(seasonal_series()).predict(6)
        self.assertAlmostEqual(float(np.std(out)), 0.0, places=9)

    def test_ses_alpha_stays_inside_its_bounds(self):
        mdl = SimpleExponentialSmoothing().fit(seasonal_series())
        self.assertGreaterEqual(mdl.params_["alpha"], 0.01)
        self.assertLessEqual(mdl.params_["alpha"], 0.99)

    def test_holt_tracks_a_linear_trend(self):
        y = 10.0 + 2.0 * np.arange(60.0)
        out = HoltLinear(damped=False).fit(y).predict(5)
        expected = 10.0 + 2.0 * np.arange(60.0, 65.0)
        np.testing.assert_allclose(out, expected, rtol=0.06)

    def test_damping_shortens_the_trend_extrapolation(self):
        y = 10.0 + 2.0 * np.arange(60.0)
        undamped = HoltLinear(damped=False).fit(y).predict(24)
        damped = HoltLinear(damped=True).fit(y).predict(24)
        self.assertLess(damped[-1], undamped[-1])

    def test_holt_winters_recovers_a_known_seasonal_pattern(self):
        m = 12
        y = seasonal_series(n=180, m=m, seed=2)
        train, test = y[:-m], y[-m:]
        add = HoltWinters(m, "add").fit(train).predict(m)
        flat = SimpleExponentialSmoothing().fit(train).predict(m)
        err_add = float(np.mean(np.abs(test - add)))
        err_flat = float(np.mean(np.abs(test - flat)))
        self.assertLess(err_add, err_flat)

    def test_holt_winters_forecast_is_seasonally_periodic(self):
        m = 12
        mdl = HoltWinters(m, "mul", damped=True).fit(seasonal_series(180, m, seed=4))
        out = mdl.predict(2 * m)
        ratio = out[m:] / np.maximum(out[:m], 1e-9)
        # with damping the level drifts, but the seasonal shape must repeat
        self.assertLess(float(np.std(ratio)) / float(np.mean(ratio)), 0.05)

    def test_holt_winters_falls_back_to_additive_on_zero_heavy_data(self):
        """Multiplicative seasonality divides by the seasonal index, so a zero
        week is not a small problem, it is an undefined model. The fallback
        must be automatic and flagged."""
        y = np.zeros(60)
        y[::5] = 3.0
        mdl = HoltWinters(12, "mul").fit(y)
        self.assertTrue(mdl.degenerate_)
        self.assertEqual(mdl.seasonal, "add")
        out = mdl.predict(6)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertLess(float(np.max(out)), 100.0 * float(np.mean(y[y > 0])))

    def test_holt_winters_keeps_multiplicative_on_strictly_positive_data(self):
        mdl = HoltWinters(12, "mul").fit(seasonal_series(n=120, m=12, seed=13) + 1.0)
        self.assertFalse(mdl.degenerate_)
        self.assertEqual(mdl.seasonal, "mul")

    def test_forecasts_are_never_negative(self):
        rng = np.random.default_rng(9)
        y = np.clip(50.0 - 0.6 * np.arange(120.0) + rng.normal(0, 3, 120), 0.0, None)
        for mdl in (HoltLinear(damped=False), HoltWinters(12, "add")):
            with self.subTest(model=mdl.name):
                self.assertTrue(np.all(mdl.fit(y).predict(40) >= 0.0))

    def test_fitted_values_are_one_step_ahead_not_in_sample_cheats(self):
        y = seasonal_series(n=100, m=12, seed=6)
        mdl = SimpleExponentialSmoothing(alpha=0.3).fit(y)
        fitted = mdl.fitted_values
        # SES recursion: fitted[t+1] = a*y[t] + (1-a)*fitted[t]
        for t in range(1, 40):
            self.assertAlmostEqual(
                fitted[t], 0.3 * y[t - 1] + 0.7 * fitted[t - 1], places=8
            )


class TestIntermittent(unittest.TestCase):
    @staticmethod
    def intermittent_series(n=200, p=0.25, size=20.0, seed=1):
        rng = np.random.default_rng(seed)
        occ = rng.random(n) < p
        y = np.zeros(n)
        y[occ] = rng.gamma(4.0, size / 4.0, occ.sum())
        return y

    def test_croston_estimates_the_demand_rate(self):
        y = self.intermittent_series(n=400, p=0.25, size=20.0, seed=3)
        rate = float(np.mean(y))
        fc = CrostonForecaster().fit(y).predict(1)[0]
        self.assertGreater(fc, 0.5 * rate)
        self.assertLess(fc, 2.5 * rate)

    def test_sba_is_the_croston_estimate_deflated(self):
        y = self.intermittent_series(seed=8)
        alpha = 0.15
        c = CrostonForecaster(alpha=alpha).fit(y).predict(1)[0]
        s = SBAForecaster(alpha=alpha).fit(y).predict(1)[0]
        self.assertAlmostEqual(s, c * (1.0 - alpha / 2.0), places=9)
        self.assertLess(s, c)

    def test_croston_and_sba_are_flat_forecasts(self):
        y = self.intermittent_series(seed=12)
        for mdl in (CrostonForecaster(), SBAForecaster(), TSBForecaster()):
            with self.subTest(model=mdl.name):
                out = mdl.fit(y).predict(10)
                self.assertAlmostEqual(float(np.std(out)), 0.0, places=12)

    def test_tsb_decays_on_a_discontinued_item_but_croston_does_not(self):
        """The obsolescence property TSB was written for."""
        y = np.concatenate([self.intermittent_series(n=150, seed=4), np.zeros(60)])
        alive = y[:150]
        c_alive = CrostonForecaster(alpha=0.15).fit(alive).predict(1)[0]
        c_dead = CrostonForecaster(alpha=0.15).fit(y).predict(1)[0]
        t_alive = TSBForecaster(alpha=0.15, beta=0.1).fit(alive).predict(1)[0]
        t_dead = TSBForecaster(alpha=0.15, beta=0.1).fit(y).predict(1)[0]
        self.assertAlmostEqual(c_dead, c_alive, places=9)   # frozen
        self.assertLess(t_dead, 0.05 * t_alive)             # decayed away

    def test_all_zero_history_forecasts_zero(self):
        y = np.zeros(80)
        for mdl in (CrostonForecaster(), SBAForecaster(), TSBForecaster()):
            with self.subTest(model=mdl.name):
                np.testing.assert_allclose(mdl.fit(y).predict(4), np.zeros(4))

    def test_fitted_values_do_not_use_the_current_observation(self):
        """A single large spike must not appear in that period's own fit."""
        y = np.zeros(60)
        y[10] = 5.0
        y[30] = 5.0
        y[50] = 900.0
        mdl = CrostonForecaster(alpha=0.2).fit(y)
        f = mdl.fitted_values
        self.assertLess(f[50], 10.0)      # spike not yet absorbed
        self.assertGreater(f[51], f[50])  # absorbed from the next period on

    def test_intermittent_methods_beat_naive_on_intermittent_demand(self):
        y = self.intermittent_series(n=300, p=0.2, size=30.0, seed=21)
        train, test = y[:260], y[260:]
        err = {}
        for mdl in (CrostonForecaster(), SBAForecaster(), TSBForecaster(),
                    NaiveForecaster()):
            fc = mdl.fit(train).predict(test.size)
            err[mdl.name] = float(np.mean((test - fc) ** 2))
        for m in ("croston", "sba", "tsb"):
            self.assertLess(err[m], err["naive"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
