"""Data generator, demand classification, pipeline wiring and the CLI."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from dflab.classify import ADI_CUT, CV2_CUT, QUADRANTS, adi, classify_panel, classify_series, cv_squared
from dflab.cli import main
from dflab.datagen import DGPConfig, generate_panel
from dflab.hierarchy import coherency_error
from dflab.pipeline import run_reconciliation_study, select_base_model, static_codes


class TestClassification(unittest.TestCase):
    def test_adi_counts_periods_per_demand(self):
        y = np.zeros(52)
        y[::4] = 5.0  # 13 demands in 52 periods
        self.assertAlmostEqual(adi(y), 4.0)

    def test_adi_of_a_dead_series_is_infinite(self):
        self.assertEqual(adi(np.zeros(20)), float("inf"))

    def test_cv_squared_ignores_the_zeros(self):
        y = np.array([0.0, 10.0, 0.0, 10.0, 0.0, 10.0])
        self.assertAlmostEqual(cv_squared(y), 0.0)
        z = np.array([10.0, 10.0, 10.0])
        self.assertAlmostEqual(cv_squared(y[y > 0]), cv_squared(z))

    def test_cv_squared_matches_the_closed_form(self):
        nz = np.array([4.0, 9.0, 16.0, 25.0])
        expected = (np.std(nz, ddof=1) / np.mean(nz)) ** 2
        self.assertAlmostEqual(cv_squared(nz), float(expected))

    def test_the_four_quadrants_are_assigned_at_the_cut_points(self):
        rng = np.random.default_rng(0)
        n = 400
        cases = {
            "smooth": np.full(n, 100.0) + rng.normal(0, 3, n),
            "erratic": np.abs(rng.lognormal(3.0, 1.1, n)) + 1.0,
        }
        occ = rng.random(n) < 0.25
        inter = np.zeros(n)
        inter[occ] = 20.0 + rng.normal(0, 1.0, occ.sum())
        cases["intermittent"] = inter
        lumpy = np.zeros(n)
        lumpy[occ] = rng.lognormal(2.0, 1.2, occ.sum())
        cases["lumpy"] = lumpy
        for expected, y in cases.items():
            with self.subTest(quadrant=expected):
                self.assertEqual(classify_series(y).quadrant, expected)

    def test_profile_fields_are_consistent(self):
        y = np.zeros(100)
        y[::5] = 7.0
        p = classify_series(y)
        self.assertEqual(p.n_periods, 100)
        self.assertEqual(p.n_nonzero, 20)
        self.assertAlmostEqual(p.zero_share, 0.8)
        self.assertAlmostEqual(p.mean_demand, 1.4)
        self.assertIn(p.quadrant, QUADRANTS)
        self.assertIn("adi", p.as_dict())

    def test_cut_points_are_configurable(self):
        y = np.zeros(100)
        y[::2] = 10.0  # ADI 2.0
        self.assertEqual(classify_series(y).quadrant, "intermittent")
        self.assertEqual(classify_series(y, adi_cut=5.0).quadrant, "smooth")
        self.assertGreater(ADI_CUT, 1.0)
        self.assertGreater(CV2_CUT, 0.0)

    def test_empty_series_is_rejected(self):
        with self.assertRaises(ValueError):
            classify_series(np.array([]))


class TestDataGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = DGPConfig(n_products=4, n_regions=3, n_channels=2, n_periods=160)
        cls.panel = generate_panel(cls.cfg)

    def test_shape_and_key_layout(self):
        self.assertEqual(self.panel.y.shape, (4 * 3 * 2, 160))
        self.assertEqual(len(self.panel.keys), 24)
        self.assertEqual(len(set(self.panel.keys)), 24)

    def test_demand_is_nonnegative_and_integral(self):
        self.assertGreaterEqual(float(self.panel.y.min()), 0.0)
        np.testing.assert_allclose(self.panel.y, np.round(self.panel.y))

    def test_generation_is_reproducible_and_seed_sensitive(self):
        again = generate_panel(self.cfg)
        np.testing.assert_allclose(self.panel.y, again.y)
        other = generate_panel(DGPConfig(**{**self.cfg.__dict__, "seed": 999}))
        self.assertFalse(np.allclose(self.panel.y, other.y))

    def test_aggregated_nodes_are_coherent_by_construction(self):
        nodes = self.panel.node_series()
        self.assertEqual(nodes.shape[0], self.panel.hierarchy.n_nodes)
        self.assertLess(coherency_error(nodes, self.panel.hierarchy), 1e-9)

    def test_the_panel_spans_multiple_demand_quadrants(self):
        profiles = classify_panel(self.panel.y)
        found = {p.quadrant for p in profiles}
        self.assertGreaterEqual(len(found), 3)

    def test_lifecycle_effects_are_present_and_labelled(self):
        launched = np.flatnonzero(self.panel.launch > 0)
        dead = np.flatnonzero(self.panel.discontinued < self.panel.n_periods)
        self.assertGreater(launched.size, 0)
        self.assertGreater(dead.size, 0)
        for i in launched:
            start = int(self.panel.launch[i])
            self.assertEqual(float(self.panel.y[i, : max(start, 1)].sum()), 0.0)

    def test_promotions_lift_average_demand(self):
        promo = self.panel.promo > 0
        has_promo = promo.any(axis=1)
        y = self.panel.y[has_promo]
        p = promo[has_promo]
        on = float(y[p].mean())
        off = float(y[~p].mean())
        self.assertGreater(on, off)

    def test_seasonality_is_visible_at_the_top_level(self):
        total = self.panel.node_series()[0]
        m = self.cfg.season_length
        cycles = total[: (total.size // m) * m].reshape(-1, m).mean(axis=0)
        self.assertGreater(float(np.std(cycles) / np.mean(cycles)), 0.05)

    def test_node_promo_intensity_is_a_share(self):
        np_share = self.panel.node_promo()
        self.assertGreaterEqual(float(np_share.min()), 0.0)
        self.assertLessEqual(float(np_share.max()), 1.0)


class TestPipelineHelpers(unittest.TestCase):
    def test_static_codes_are_dense_integers_per_dimension(self):
        keys = [("P1", "R1", "A"), ("P2", "R1", "B"), ("P1", "R2", "A")]
        codes = static_codes(keys)
        self.assertEqual(codes.shape, (3, 3))
        for d in range(3):
            col = codes[:, d]
            self.assertEqual(set(col.tolist()), set(range(len(set(col.tolist())))))

    def test_base_model_selection_rules(self):
        rng = np.random.default_rng(1)
        dense = np.abs(rng.normal(100, 10, 120))
        self.assertEqual(select_base_model(dense, 52).name, "hw_add[m=52]")
        sparse = np.zeros(120)
        sparse[::6] = 5.0
        self.assertEqual(select_base_model(sparse, 52).name, "sba")
        short = np.abs(rng.normal(100, 10, 60))
        self.assertEqual(select_base_model(short, 52).name, "ses")

    def test_reconciliation_study_produces_coherent_results(self):
        panel = generate_panel(
            DGPConfig(n_products=2, n_regions=2, n_channels=2, n_periods=140, seed=3)
        )
        rep = run_reconciliation_study(
            panel, horizon=6, n_windows=1, step=6, min_train=120, residual_window=60
        )
        self.assertGreater(rep.coherency["base"], 0.0)
        for mth in ("bottom_up", "top_down", "ols", "mint"):
            self.assertLess(rep.coherency[mth], 1e-6)
        self.assertGreaterEqual(rep.shrinkage, 0.0)
        self.assertLessEqual(rep.shrinkage, 1.0)
        self.assertIn("| method |", rep.as_markdown())
        for lv in rep.levels:
            self.assertTrue(np.isfinite(rep.table["mint"][lv]))


class TestCLI(unittest.TestCase):
    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()

    def test_classify_command(self):
        code, out = self._run(
            ["classify", "--products", "2", "--regions", "2", "--channels", "2",
             "--periods", "140", "--horizon", "6", "--windows", "2"]
        )
        self.assertEqual(code, 0)
        self.assertIn("DemandPanel", out)
        for q in QUADRANTS:
            self.assertIn(q, out)

    def test_backtest_command_without_ml(self):
        code, out = self._run(
            ["backtest", "--products", "2", "--regions", "2", "--channels", "2",
             "--periods", "150", "--horizon", "6", "--step", "6", "--windows", "1",
             "--min-train", "130", "--no-ml"]
        )
        self.assertEqual(code, 0)
        self.assertIn("WAPE by demand quadrant", out)
        self.assertIn("Best method per quadrant", out)

    def test_figures_command_writes_png_files(self):
        with tempfile.TemporaryDirectory() as d:
            code, out = self._run(
                ["figures", "--outdir", d, "--products", "2", "--regions", "2",
                 "--channels", "2", "--periods", "150", "--horizon", "6",
                 "--step", "6", "--windows", "1", "--min-train", "130", "--no-ml"]
            )
            self.assertEqual(code, 0)
            pngs = sorted(p.name for p in Path(d).glob("*.png"))
            self.assertIn("accuracy_by_quadrant.png", pngs)
            self.assertIn("demand_quadrants.png", pngs)
            self.assertGreaterEqual(len(pngs), 4)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
