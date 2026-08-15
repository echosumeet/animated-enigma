"""Validation of the simulator against closed-form bullwhip results.

These are the tests that make the rest of the package believable. They run the
*whole* stack -- event loop, network, policies, forecasters -- and compare the
measured order variance against an expression derived independently of the
simulator. Nothing in ``echelonsim.simulation`` knows these formulas exist.
"""

import unittest

import numpy as np

from echelonsim.bullwhip import (
    MECHANISMS,
    chen_moving_average_bullwhip,
    decompose_bullwhip,
    exponential_smoothing_bullwhip,
    measure_by_echelon,
)
from echelonsim.demand import IIDNormal
from echelonsim.forecast import ExponentialSmoothing, MovingAverage, Oracle
from echelonsim.leadtime import Deterministic
from echelonsim.metrics import bullwhip_ratio
from echelonsim.network import serial_chain
from echelonsim.policies import BaseStock
from echelonsim.simulation import run_simulation

PERIODS = 30_200
WARMUP = 200


def single_stage_ratio(forecaster_factory, transit=2.0, order_lead=1, seed=7,
                       demand_std=10.0, periods=PERIODS):
    """Amplification at a single order-up-to stage fed by an infinite source."""
    network = serial_chain(
        levels=1,
        policy_factory=lambda level: BaseStock(z=0.0, allow_returns=True),
        forecaster_factory=forecaster_factory,
        transit_factory=lambda level: Deterministic(transit),
        order_lead_time=order_lead,
    )
    result = run_simulation(
        network, IIDNormal(100.0, demand_std), periods=periods, seed=seed
    ).trim(WARMUP)
    series = result.nodes["retailer"]
    return bullwhip_ratio(series.orders_placed, series.demand_received)


class TestAnalyticFormulas(unittest.TestCase):
    def test_moving_average_matches_chen_et_al_theorem_1(self):
        for window in (5, 10, 20):
            with self.subTest(window=window):
                simulated = single_stage_ratio(lambda level, p=window: MovingAverage(p))
                analytic = chen_moving_average_bullwhip(protection=3.0, window=window)
                self.assertAlmostEqual(simulated / analytic, 1.0, delta=0.02)

    def test_exponential_smoothing_matches_the_derived_expression(self):
        for alpha in (0.1, 0.3, 0.5):
            with self.subTest(alpha=alpha):
                simulated = single_stage_ratio(
                    lambda level, a=alpha: ExponentialSmoothing(a)
                )
                analytic = exponential_smoothing_bullwhip(protection=3.0, alpha=alpha)
                self.assertAlmostEqual(simulated / analytic, 1.0, delta=0.02)

    def test_amplification_grows_with_the_protection_interval(self):
        short = single_stage_ratio(lambda level: ExponentialSmoothing(0.3), transit=1.0)
        long = single_stage_ratio(lambda level: ExponentialSmoothing(0.3), transit=5.0)
        self.assertGreater(long, 2.0 * short)

    def test_more_smoothing_means_less_amplification(self):
        ratios = [
            single_stage_ratio(lambda level, a=alpha: ExponentialSmoothing(a))
            for alpha in (0.1, 0.2, 0.4, 0.6)
        ]
        self.assertEqual(ratios, sorted(ratios))

    def test_a_perfect_forecast_produces_no_amplification(self):
        ratio = single_stage_ratio(
            lambda level: Oracle(mean=100.0, std=10.0), periods=5200
        )
        self.assertAlmostEqual(ratio, 1.0, delta=1e-6)

    def test_analytic_helpers_reject_bad_parameters(self):
        with self.assertRaises(ValueError):
            chen_moving_average_bullwhip(3.0, 0)
        with self.assertRaises(ValueError):
            exponential_smoothing_bullwhip(3.0, 0.0)


class TestEchelonGrowth(unittest.TestCase):
    def test_amplification_grows_strictly_with_echelon_depth(self):
        config = {
            "forecast": {"kind": "exponential", "alpha": 0.3},
            "topology": {"kind": "serial", "levels": 4},
            "run": {"periods": 520, "replications": 8, "warmup": 60},
        }
        amplification = measure_by_echelon(config, warmup=60)
        cumulative = [amplification.cumulative[name].mean for name in amplification.node_order]
        self.assertEqual(cumulative, sorted(cumulative))
        self.assertGreater(cumulative[-1], 5.0 * cumulative[0])

    def test_echelon_base_stock_amplification_tracks_the_cumulative_interval(self):
        """Under echelon control the factory's order responds to the forecast
        change scaled by the *cumulative* protection interval, so the analytic
        single-stage expression evaluated at that interval should be close."""
        config = {
            "info_mode": "vmi",
            "forecast": {"kind": "exponential", "alpha": 0.3},
            "topology": {"kind": "serial", "levels": 3},
            "run": {"periods": 1040, "replications": 8, "warmup": 100},
        }
        amplification = measure_by_echelon(config, warmup=100)
        measured = amplification.cumulative["factory"].mean
        predicted = exponential_smoothing_bullwhip(protection=9.0, alpha=0.3)
        self.assertAlmostEqual(measured / predicted, 1.0, delta=0.15)


class TestDecomposition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = decompose_bullwhip(
            {"run": {"periods": 400, "replications": 6}}, warmup=60
        )

    def test_shapley_contributions_reconstruct_the_total(self):
        self.result.check_additivity()

    def test_every_mechanism_is_present(self):
        self.assertEqual(set(self.result.contributions), set(MECHANISMS))

    def test_all_mechanisms_off_gives_no_amplification(self):
        self.assertAlmostEqual(self.result.baseline.mean, 1.0, delta=0.02)

    def test_every_mechanism_contributes_non_negatively(self):
        for name, interval in self.result.contributions.items():
            with self.subTest(mechanism=name):
                self.assertGreater(interval.mean, -1e-6)

    def test_lead_time_alone_does_nothing_but_interacts_strongly(self):
        """A long lead time with a perfect forecast and no batching produces no
        amplification at all -- the order still equals the demand. Its Shapley
        value is therefore entirely interaction, which is exactly why an
        ablate-one-at-a-time attribution would score it at zero and lead a
        reader to the wrong investment."""
        alone = self.result.cell_means[("leadtime",)]
        self.assertAlmostEqual(alone, 1.0, delta=0.05)
        self.assertGreater(self.result.contributions["leadtime"].mean, 0.05)

    def test_cells_cover_the_full_factorial(self):
        self.assertEqual(len(self.result.cell_means), 2 ** len(MECHANISMS))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
