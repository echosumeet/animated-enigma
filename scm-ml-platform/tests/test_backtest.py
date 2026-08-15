import unittest

import numpy as np

from scmplatform.backtest import (
    Gate,
    bias,
    champion_challenger,
    check_gate,
    rolling_backtest,
    wmape,
)
from scmplatform.datagen import PanelConfig, make_panel
from scmplatform.decisions import InventoryEconomics, evaluate_decisions, period_cost
from scmplatform.features import default_specs

PANEL = make_panel(PanelConfig(n_skus=8, n_days=300, seed=2))


class TestMetrics(unittest.TestCase):
    def test_wmape_and_bias_on_a_hand_worked_example(self):
        y = np.array([10.0, 20.0, 30.0])
        yhat = np.array([12.0, 18.0, 30.0])
        self.assertAlmostEqual(wmape(y, yhat), 4.0 / 60.0)
        self.assertAlmostEqual(bias(y, yhat), 0.0)


class TestBacktestAndGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.champion = rolling_backtest(
            PANEL, default_specs(), label="champion", horizon=10, folds=2, min_train_days=200
        )
        cls.challenger = rolling_backtest(
            PANEL,
            default_specs(),
            label="challenger",
            horizon=10,
            folds=2,
            min_train_days=200,
            bias_adjustment=0.80,
        )

    def test_backtest_produces_clean_out_of_sample_predictions(self):
        p = self.champion.predictions
        self.assertEqual(self.champion.folds, 2)
        self.assertFalse(p[["y", "yhat"]].isna().any().any())
        self.assertTrue((p["yhat"] >= 0).all())
        self.assertGreater(len(p), 100)

    def test_a_reasonable_model_clears_the_default_gate(self):
        result = check_gate(self.champion, Gate())
        self.assertTrue(result.passed, result.summary())

    def test_the_gate_fails_and_explains_itself(self):
        result = check_gate(self.champion, Gate(max_wmape=0.01, max_abs_bias=0.0001, max_cost_per_unit=0.0))
        self.assertFalse(result.passed)
        self.assertEqual(len(result.reasons), 3)
        self.assertIn("wMAPE", result.summary())

    def test_deliberate_under_forecasting_is_rejected_on_cost(self):
        decision = champion_challenger(self.champion, self.challenger, Gate())
        self.assertFalse(decision.promote)
        self.assertLess(self.challenger.bias, self.champion.bias)
        self.assertGreater(
            self.challenger.metrics()["cost_per_unit"], self.champion.metrics()["cost_per_unit"]
        )


class TestDecisionQuality(unittest.TestCase):
    def test_period_cost_charges_holding_and_shortage_separately(self):
        econ = InventoryEconomics(unit_cost=10.0, holding_rate_annual=0.365, shortage_multiple=2.0)
        over = period_cost(np.array([5.0]), np.array([15.0]), econ)[0]
        short = period_cost(np.array([15.0]), np.array([5.0]), econ)[0]
        self.assertAlmostEqual(over, 10 * econ.holding_cost_per_unit)
        self.assertAlmostEqual(short, 10 * 20.0)
        self.assertGreater(short, over)

    def test_a_worse_forecast_costs_more_and_regret_is_non_negative(self):
        p = self.frame()
        good = evaluate_decisions(p.assign(yhat=p["y"] * 1.02))
        bad = evaluate_decisions(p.assign(yhat=p["y"] * 0.60))
        self.assertGreater(bad.realised_cost, good.realised_cost)
        self.assertGreaterEqual(good.regret, 0.0)
        self.assertLess(bad.fill_rate, good.fill_rate)

    @staticmethod
    def frame():
        import pandas as pd

        rng = np.random.default_rng(0)
        y = rng.gamma(6.0, 5.0, 400)
        return pd.DataFrame({"y": y, "yhat": y, "sigma": np.full(400, 8.0)})



if __name__ == "__main__":
    unittest.main()
