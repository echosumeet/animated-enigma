"""Two-stage stochastic design: the bounds WS <= RP <= EEV are the test.

They are not decoration. If the scenario block is wired up wrong - shared flow
variables across scenarios, a probability weight applied twice, first-stage
binaries accidentally replicated - one of these inequalities flips, and it is
essentially the only way to notice.
"""

from __future__ import annotations

import unittest

import numpy as np

from netdesign.instances import demand_scenarios, generate_instance
from netdesign.network_flow import NetworkOptions
from netdesign.stochastic import solve_two_stage, wait_and_see


def instance():
    return generate_instance(seed=7, n_dcs=5, n_zones=10, n_commodities=1, n_plants=3)


class TestStochasticBounds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = instance()
        cls.scen, cls.probs = demand_scenarios(cls.inst, n_scenarios=6, seed=5)
        cls.res = solve_two_stage(cls.inst, cls.scen, cls.probs)

    def test_ws_le_rp_le_eev(self):
        self.assertLessEqual(self.res.ws, self.res.rp + 1e-6)
        self.assertLessEqual(self.res.rp, self.res.eev + 1e-6)

    def test_vss_and_evpi_are_non_negative(self):
        self.assertGreaterEqual(self.res.vss, -1e-6)
        self.assertGreaterEqual(self.res.evpi, -1e-6)

    def test_check_bounds_accepts_a_valid_result(self):
        self.res.check_bounds()

    def test_check_bounds_rejects_an_impossible_result(self):
        self.res.ws = self.res.rp * 2.0
        with self.assertRaises(AssertionError):
            self.res.check_bounds()
        self.res.ws = self.res.rp - abs(self.res.rp) * 0.01  # restore a valid ordering

    def test_a_deterministic_scenario_set_collapses_the_value_of_information(self):
        # every scenario identical -> perfect information is worth nothing and
        # the mean-value design is optimal
        same = [dict(self.inst.demand) for _ in range(4)]
        res = solve_two_stage(self.inst, same, np.full(4, 0.25))
        self.assertAlmostEqual(res.evpi, 0.0, delta=1e-3 * res.rp)
        self.assertAlmostEqual(res.vss, 0.0, delta=1e-3 * res.rp)

    def test_wait_and_see_expectation_matches_its_scenario_objectives(self):
        ws, objs, sets = wait_and_see(
            self.inst, self.scen, self.probs, NetworkOptions(allow_unmet=True)
        )
        self.assertAlmostEqual(ws, float(np.dot(self.probs, objs)), delta=1e-6 * abs(ws))
        self.assertEqual(len(sets), len(self.scen))

    def test_recourse_is_always_enabled_for_the_stochastic_study(self):
        res = solve_two_stage(
            self.inst, self.scen, self.probs, NetworkOptions(allow_unmet=False)
        )
        res.check_bounds()
        self.assertGreaterEqual(res.eev, res.rp - 1e-6)


class TestScenarioGeneration(unittest.TestCase):
    def test_scenarios_are_reproducible_from_the_seed(self):
        inst = instance()
        a, _ = demand_scenarios(inst, n_scenarios=4, seed=99)
        b, _ = demand_scenarios(inst, n_scenarios=4, seed=99)
        self.assertEqual(a, b)

    def test_regional_correlation_produces_more_aggregate_spread_than_zone_noise(self):
        inst = generate_instance(seed=7, n_zones=30, n_dcs=5, n_commodities=1, n_plants=3)
        corr, _ = demand_scenarios(
            inst, n_scenarios=60, seed=1, national_sigma=0.25, regional_sigma=0.2, zone_sigma=0.05
        )
        indep, _ = demand_scenarios(
            inst, n_scenarios=60, seed=1, national_sigma=0.0, regional_sigma=0.0, zone_sigma=0.25
        )
        cv_corr = np.std([sum(s.values()) for s in corr]) / np.mean([sum(s.values()) for s in corr])
        cv_indep = np.std([sum(s.values()) for s in indep]) / np.mean(
            [sum(s.values()) for s in indep]
        )
        self.assertGreater(
            cv_corr,
            3 * cv_indep,
            "independent zone noise diversifies away; that is why it yields VSS = 0",
        )

    def test_probabilities_are_a_valid_distribution(self):
        inst = instance()
        _, probs = demand_scenarios(inst, n_scenarios=7, seed=2)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=12)
        self.assertTrue(np.all(probs > 0))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
