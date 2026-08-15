import unittest

import numpy as np

from etarisk.decision import ACTIONS, CostMatrix, compare_to_baseline, optimal_actions
from etarisk.drift import psi, psi_table
from etarisk.generate import GeneratorConfig
from etarisk.pipeline import run_pipeline
from etarisk.risk import expected_calibration_error


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = run_pipeline(
            GeneratorConfig(n_shipments=12_000, days=540, seed=17),
            alphas=(0.2, 0.1),
            max_iter=120,
        )
        cls.m = cls.res.metrics

    def test_model_beats_the_quoted_transit_time(self):
        self.assertLess(self.m["eta_mae_h"], self.m["quoted_mae_h"])

    def test_conformal_coverage_is_close_to_nominal(self):
        for _, row in self.m["coverage"].iterrows():
            self.assertGreater(row["empirical"], row["nominal"] - 0.06)
            self.assertLess(row["empirical"], row["nominal"] + 0.08)

    def test_tighter_alpha_gives_wider_intervals(self):
        cov = self.m["coverage"].sort_values("alpha")
        widths = cov["mean_width_h"].to_numpy()
        self.assertTrue(np.all(np.diff(widths) < 0))

    def test_intervals_are_ordered_and_non_negative(self):
        cp = self.res.conformal[0.1]
        X = self.res.X["test"]
        scale = self.res.splits["test"]["planned_transit_h"].to_numpy()
        point, lo, hi = cp.predict_interval(X, scale)
        self.assertTrue(np.all(lo <= point))
        self.assertTrue(np.all(point <= hi))
        self.assertGreaterEqual(float(lo.min()), 0.0)

    def test_mae_by_horizon_covers_every_test_shipment(self):
        table = self.m["mae_by_horizon"]
        self.assertEqual(int(table["n"].sum()), self.m["n_test"])
        self.assertTrue((table["mae_h"] > 0).all())

    def test_isotonic_calibration_improves_reliability(self):
        self.assertLess(self.m["ece_calibrated"], self.m["ece_raw"])
        # Isotonic buys calibration, not sharpness: Brier is allowed to move a
        # little either way, but it must not blow up.
        self.assertLessEqual(self.m["brier_calibrated"], self.m["brier_raw"] + 0.005)

    def test_decision_layer_beats_the_fixed_rule(self):
        d = self.m["decision"]
        self.assertLess(d["model_cost_per_shipment"], d["baseline_cost_per_shipment"])
        self.assertGreater(d["saved_per_shipment"], 0.0)
        self.assertLess(d["oracle_cost_per_shipment"], d["model_cost_per_shipment"])


class TestReliability(unittest.TestCase):
    def test_a_perfectly_calibrated_forecast_has_near_zero_ece(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.02, 0.98, 40_000)
        y = (rng.random(40_000) < p).astype(int)
        self.assertLess(expected_calibration_error(y, p), 0.02)


class TestDecision(unittest.TestCase):
    def test_actions_follow_the_cost_thresholds(self):
        costs = CostMatrix()
        thr = costs.thresholds()
        p = np.array([0.0, thr["nothing_to_notify"] + 0.01, thr["notify_to_expedite"] + 0.01, 1.0])
        acts = [ACTIONS[i] for i in optimal_actions(p, costs)]
        self.assertEqual(acts, ["nothing", "notify", "expedite", "expedite"])

    def test_a_perfect_forecast_reaches_the_oracle(self):
        rng = np.random.default_rng(3)
        y = (rng.random(2000) < 0.3).astype(int)
        baseline = np.zeros(2000, dtype=int)
        out = compare_to_baseline(y.astype(float), y, baseline)
        self.assertAlmostEqual(out["model_cost_per_shipment"], out["oracle_cost_per_shipment"], places=6)


class TestDrift(unittest.TestCase):
    def test_psi_is_zero_for_the_same_distribution(self):
        rng = np.random.default_rng(5)
        a = rng.normal(size=50_000)
        b = rng.normal(size=50_000)
        self.assertLess(psi(a, b), 0.01)

    def test_psi_flags_a_shifted_distribution(self):
        rng = np.random.default_rng(6)
        a = rng.normal(size=20_000)
        b = rng.normal(1.0, 1.0, size=20_000)
        self.assertGreater(psi(a, b), 0.25)

    def test_psi_table_ranks_the_worst_feature_first(self):
        import pandas as pd

        rng = np.random.default_rng(7)
        ref = pd.DataFrame({"stable": rng.normal(size=10_000), "moved": rng.normal(size=10_000)})
        cur = pd.DataFrame({"stable": rng.normal(size=10_000), "moved": rng.normal(2.0, 1.0, 10_000)})
        table = psi_table(ref, cur)
        self.assertEqual(table.iloc[0]["feature"], "moved")
        self.assertEqual(table.iloc[0]["verdict"], "broken")
        self.assertEqual(table.iloc[-1]["verdict"], "stable")


if __name__ == "__main__":
    unittest.main()
