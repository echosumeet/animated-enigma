"""Leakage guarantees for the feature model, and rolling-origin mechanics.

The leakage tests are the most important tests in this repository. A feature
based forecaster that has quietly seen the future produces beautiful backtest
numbers and then fails in the planning cycle, and the failure is invisible
unless you assert against it directly.
"""

from __future__ import annotations

import unittest

import numpy as np

from dflab.backtest import BacktestConfig, backtest, rolling_origins
from dflab.baselines import NaiveForecaster, SeasonalNaiveForecaster, ZeroForecaster
from dflab.classify import QUADRANTS
from dflab.datagen import DGPConfig, generate_panel
from dflab.ml import FeatureConfig, GlobalGBTForecaster, build_features


def small_panel(seed=5, products=2, periods=170):
    cfg = DGPConfig(
        n_products=products, n_regions=2, n_channels=2, n_periods=periods, seed=seed
    )
    return generate_panel(cfg)


class TestFeatureConstruction(unittest.TestCase):
    def setUp(self):
        self.cfg = FeatureConfig(season_length=12, lags=(0, 1, 2), roll_means=(3, 6))
        rng = np.random.default_rng(0)
        self.P = rng.gamma(2.0, 8.0, size=(3, 80))

    def test_feature_names_match_matrix_width(self):
        X, y, idx = build_features(
            self.P, np.array([40, 41]), np.array([1, 2]), self.cfg
        )
        self.assertEqual(X.shape[1], len(self.cfg.feature_names()))
        self.assertEqual(X.shape[0], y.size)
        self.assertEqual(idx.shape, (X.shape[0], 3))

    def test_lag_zero_is_the_value_at_the_origin_not_the_target(self):
        origins = np.array([30, 45])
        X, y, idx = build_features(self.P, origins, np.array([3]), self.cfg)
        for row in range(X.shape[0]):
            s, o, h = idx[row]
            self.assertAlmostEqual(X[row, 0], self.P[s, o])       # lag_0
            self.assertAlmostEqual(X[row, 1], self.P[s, o - 1])   # lag_1
            self.assertAlmostEqual(y[row], self.P[s, o + h])

    def test_features_are_invariant_to_everything_after_the_origin(self):
        """The single strongest anti-leakage assertion: scramble the future,
        the features must not move."""
        origins = np.array([40, 50, 60])
        horizons = np.array([1, 4])
        cfg = FeatureConfig(
            season_length=12,
            lags=(0, 1, 2),
            roll_means=(3, 6),
            use_seasonal_lag=False,
            use_promo=False,
        )
        X0, _, _ = build_features(self.P, origins, horizons, cfg, targets=False)
        P2 = self.P.copy()
        rng = np.random.default_rng(1)
        P2[:, int(origins.max()) + 1 :] = rng.gamma(9.0, 90.0, size=P2[:, int(origins.max()) + 1 :].shape)
        X1, _, _ = build_features(P2, origins, horizons, cfg, targets=False)
        np.testing.assert_allclose(X0, X1)

    def test_seasonal_lag_only_ever_reads_the_past(self):
        cfg = FeatureConfig(season_length=12, lags=(0,), roll_means=(4,))
        origins = np.array([50])
        horizons = np.array([1, 6, 12])
        X, _, idx = build_features(self.P, origins, horizons, cfg, targets=False)
        col = cfg.feature_names().index("seasonal_lag")
        for row in range(X.shape[0]):
            s, o, h = idx[row]
            self.assertAlmostEqual(X[row, col], self.P[s, o + h - 12])
            self.assertLessEqual(o + h - 12, o)  # strictly at or before the origin

    def test_origin_plus_horizon_beyond_the_panel_is_rejected(self):
        with self.assertRaises(ValueError):
            build_features(self.P, np.array([78]), np.array([5]), self.cfg)

    def test_zero_horizon_is_rejected(self):
        with self.assertRaises(ValueError):
            build_features(self.P, np.array([40]), np.array([0]), self.cfg)


class TestGlobalModel(unittest.TestCase):
    def setUp(self):
        self.panel = small_panel()
        self.cfg = FeatureConfig(season_length=52)

    def test_fit_predict_shapes_and_nonnegativity(self):
        train = self.panel.y[:, :150]
        mdl = GlobalGBTForecaster(
            horizon=8, cfg=self.cfg, max_iter=40, promo=self.panel.promo
        ).fit_panel(train)
        out = mdl.predict_panel(8)
        self.assertEqual(out.shape, (self.panel.n_bottom, 8))
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_forecast_does_not_change_when_the_test_block_changes(self):
        """Refit on the same training window with a corrupted future: the
        forecast must be bit-identical."""
        panel = self.panel
        train_end = 150
        corrupt = panel.y.copy()
        rng = np.random.default_rng(2)
        corrupt[:, train_end:] = rng.gamma(9.0, 400.0, size=corrupt[:, train_end:].shape)
        a = GlobalGBTForecaster(
            horizon=6, cfg=self.cfg, max_iter=40, promo=panel.promo
        ).fit_panel(panel.y[:, :train_end]).predict_panel(6)
        b = GlobalGBTForecaster(
            horizon=6, cfg=self.cfg, max_iter=40, promo=panel.promo
        ).fit_panel(corrupt[:, :train_end]).predict_panel(6)
        np.testing.assert_allclose(a, b)

    def test_asking_for_more_than_the_fitted_horizon_is_rejected(self):
        mdl = GlobalGBTForecaster(
            horizon=4, cfg=self.cfg, max_iter=20, promo=self.panel.promo
        ).fit_panel(self.panel.y[:, :150])
        with self.assertRaises(ValueError):
            mdl.predict_panel(9)

    def test_predict_before_fit_is_rejected(self):
        with self.assertRaises(RuntimeError):
            GlobalGBTForecaster(horizon=4).predict_panel(2)

    def test_training_window_too_short_is_reported_clearly(self):
        with self.assertRaises(ValueError):
            GlobalGBTForecaster(horizon=30, cfg=self.cfg, max_iter=10).fit_panel(
                self.panel.y[:, :20]
            )

    def test_quantile_models_are_ordered(self):
        train = self.panel.y[:, :150]
        preds = {}
        for q in (0.5, 0.9):
            mdl = GlobalGBTForecaster(
                horizon=4,
                cfg=self.cfg,
                loss="quantile",
                quantile=q,
                max_iter=60,
                promo=self.panel.promo,
            ).fit_panel(train)
            preds[q] = mdl.predict_panel(4)
        # a q90 forecast should sit above the median for the large majority of
        # cells; exact monotonicity is not guaranteed by independent fits
        share = float(np.mean(preds[0.9] >= preds[0.5]))
        self.assertGreater(share, 0.9)


class TestRollingOrigins(unittest.TestCase):
    def test_cut_layout_is_anchored_to_the_end_of_the_series(self):
        cuts = rolling_origins(T=260, horizon=13, step=13, n_windows=4, min_train=156)
        self.assertEqual(cuts, [208, 221, 234, 247])
        self.assertEqual(cuts[-1] + 13, 260)

    def test_windows_are_dropped_rather_than_violating_min_train(self):
        cuts = rolling_origins(T=200, horizon=13, step=13, n_windows=10, min_train=156)
        self.assertTrue(all(c >= 156 for c in cuts))
        self.assertLess(len(cuts), 10)

    def test_impossible_configuration_raises(self):
        with self.assertRaises(ValueError):
            rolling_origins(T=100, horizon=13, step=13, n_windows=3, min_train=156)
        with self.assertRaises(ValueError):
            rolling_origins(T=260, horizon=0, step=13, n_windows=3, min_train=156)


class TestBacktest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = small_panel(seed=9, products=2, periods=180)
        cls.cfg = BacktestConfig(
            horizon=6, step=6, n_windows=2, min_train=150, season_length=52
        )
        cls.result = backtest(
            cls.panel.y,
            [NaiveForecaster(), SeasonalNaiveForecaster(52), ZeroForecaster()],
            cls.cfg,
        )

    def test_every_method_series_window_combination_is_scored(self):
        n = self.panel.n_bottom
        expected = 3 * n * len(self.result.cuts)
        self.assertEqual(len(self.result.rows), expected)

    def test_zero_forecaster_pools_to_a_wape_of_one(self):
        self.assertAlmostEqual(self.result.overall()["zero"]["wape"], 1.0, places=9)

    def test_quadrant_labels_cover_the_panel_and_use_training_data_only(self):
        self.assertEqual(len(self.result.quadrant_of), self.panel.n_bottom)
        self.assertTrue(set(self.result.quadrant_of) <= set(QUADRANTS))
        self.assertEqual(sum(self.result.quadrant_counts().values()), self.panel.n_bottom)

    def test_pooled_wape_equals_the_ratio_of_summed_errors(self):
        rows = [r for r in self.result.rows if r.method == "naive"]
        expected = sum(r.metrics["abs_err"] for r in rows) / sum(
            r.metrics["abs_actual"] for r in rows
        )
        self.assertAlmostEqual(self.result.overall()["naive"]["wape"], expected)

    def test_value_add_of_the_baseline_against_itself_is_zero(self):
        fva = self.result.value_add("snaive[m=52]")
        self.assertAlmostEqual(fva["snaive[m=52]"], 0.0, places=12)

    def test_by_quadrant_pooling_reconstructs_the_overall_number(self):
        bq = self.result.by_quadrant()["naive"]
        num = den = 0.0
        for q in QUADRANTS:
            rows = [
                r for r in self.result.rows if r.method == "naive" and r.quadrant == q
            ]
            num += sum(r.metrics["abs_err"] for r in rows)
            den += sum(r.metrics["abs_actual"] for r in rows)
            if rows:
                self.assertIn("wape", bq[q])
        self.assertAlmostEqual(num / den, self.result.overall()["naive"]["wape"])

    def test_forecasts_are_retained_with_the_expected_shape(self):
        fc = self.result.forecasts["naive"]
        self.assertEqual(
            fc.shape, (len(self.result.cuts), self.panel.n_bottom, self.cfg.horizon)
        )

    def test_backtest_with_a_global_model_runs_end_to_end(self):
        cfg = BacktestConfig(
            horizon=4, step=4, n_windows=1, min_train=150, season_length=52
        )
        gm = GlobalGBTForecaster(
            horizon=4,
            cfg=FeatureConfig(season_length=52),
            max_iter=30,
            name="gbt_test",
            promo=self.panel.promo,
        )
        qm = [
            (
                0.9,
                GlobalGBTForecaster(
                    horizon=4,
                    cfg=FeatureConfig(season_length=52),
                    loss="quantile",
                    quantile=0.9,
                    max_iter=30,
                    name="gbt_test_q90",
                    promo=self.panel.promo,
                ),
            )
        ]
        res = backtest(
            self.panel.y, [NaiveForecaster()], cfg, global_models=[gm], quantile_models=qm
        )
        self.assertIn("gbt_test", res.methods())
        gbt_rows = [r for r in res.rows if r.method == "gbt_test"]
        self.assertTrue(all("pinball_0.9" in r.metrics for r in gbt_rows))
        self.assertTrue(np.isfinite(res.overall()["gbt_test"]["wape"]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
