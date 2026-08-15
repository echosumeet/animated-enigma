"""Experiment plumbing, information/disruption studies, and the CLI.

Fast configurations throughout -- these test that the machinery is wired
correctly, not that the numbers have converged.
"""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

from echelonsim.cli import main
from echelonsim.disruption import (
    measure_recovery,
    retailer_fill_path,
    run_disruption_study,
)
from echelonsim.experiments import (
    DEFAULT_CONFIG,
    build_demand,
    build_forecaster,
    build_leadtime,
    build_network,
    build_policy,
    compare_scenarios,
    estimate_warmup,
    load_config,
    merge_config,
    run_scenario,
    save_config,
)
from echelonsim.forecast import ExponentialSmoothing, MovingAverage, Oracle
from echelonsim.information import compare_information_modes
from echelonsim.leadtime import Deterministic, GammaLeadTime
from echelonsim.network import InfoMode
from echelonsim.policies import Batched, SsPolicy
from echelonsim.tradeoffs import calibrate_service, lead_time_demand_sigma, lead_time_grid

FAST = {"run": {"periods": 300, "replications": 5, "warmup": 60}}


class TestConfig(unittest.TestCase):
    def test_merge_is_recursive_and_non_destructive(self):
        merged = merge_config(DEFAULT_CONFIG, {"run": {"periods": 42}})
        self.assertEqual(merged["run"]["periods"], 42)
        self.assertEqual(merged["run"]["seed"], DEFAULT_CONFIG["run"]["seed"])
        self.assertEqual(DEFAULT_CONFIG["run"]["periods"], 520)

    def test_lists_are_replaced_not_merged(self):
        base = {"a": [1, 2, 3]}
        self.assertEqual(merge_config(base, {"a": [9]})["a"], [9])

    def test_config_round_trips_through_json(self):
        config = merge_config(DEFAULT_CONFIG, {"review_period": 3})
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "config.json")
            save_config(config, path)
            self.assertEqual(load_config(path)["review_period"], 3)

    def test_builders_dispatch_on_kind(self):
        config = merge_config(DEFAULT_CONFIG, {
            "forecast": {"kind": "oracle"},
            "policy": {"kind": "ss", "batch_multiple": 25.0},
            "leadtime": {"kind": "gamma", "mean": 5.0, "cv": 0.3},
        })
        self.assertIsInstance(build_forecaster(config, 0), Oracle)
        policy = build_policy(config, 0)
        self.assertIsInstance(policy, Batched)
        self.assertIsInstance(policy.inner, SsPolicy)
        self.assertIsInstance(build_leadtime(config, 0), GammaLeadTime)

    def test_per_level_parameters_are_indexed_by_level(self):
        config = merge_config(DEFAULT_CONFIG, {"leadtime": {"mean": [1.0, 4.0, 9.0]}})
        self.assertEqual(build_leadtime(config, 0).mean, 1.0)
        self.assertEqual(build_leadtime(config, 2).mean, 9.0)
        # Beyond the list, the last entry repeats.
        self.assertEqual(build_leadtime(config, 5).mean, 9.0)

    def test_unknown_kinds_are_rejected(self):
        for section, kind in (("forecast", "kind"), ("policy", "kind"), ("leadtime", "kind")):
            config = merge_config(DEFAULT_CONFIG, {section: {kind: "nonsense"}})
            with self.assertRaises(ValueError):
                build_forecaster(config, 0) if section == "forecast" else (
                    build_policy(config, 0) if section == "policy" else build_leadtime(config, 0)
                )

    def test_vmi_forces_continuous_review_at_the_retailer(self):
        config = merge_config(DEFAULT_CONFIG, {"info_mode": "vmi", "review_period": 6})
        network = build_network(config)
        self.assertEqual(network.retailers[0].review_period, 1)
        self.assertEqual(network.nodes["factory"].review_period, 6)

    def test_demand_shock_is_built_from_config(self):
        config = merge_config(DEFAULT_CONFIG, {
            "demand": {"shock": {"start": 5, "duration": 3, "multiplier": 2.0}}
        })
        process = build_demand(config)
        self.assertTrue(process.active(6))
        self.assertFalse(process.active(9))


class TestScenarioRunning(unittest.TestCase):
    def test_scenario_records_one_value_per_replication(self):
        outcome = run_scenario(merge_config(DEFAULT_CONFIG, FAST), name="x")
        self.assertEqual(outcome.replications, 5)
        self.assertEqual(len(outcome.values("chain_bullwhip")), 5)
        self.assertEqual(outcome.warmup, 60)

    def test_unknown_metric_raises_with_a_useful_message(self):
        outcome = run_scenario(merge_config(DEFAULT_CONFIG, FAST))
        with self.assertRaises(KeyError):
            outcome.values("no_such_metric")

    def test_auto_warmup_estimates_a_truncation_point(self):
        config = merge_config(DEFAULT_CONFIG, {"run": {"periods": 400, "replications": 2}})
        truncation = estimate_warmup(config, pilots=2)
        self.assertGreater(truncation, 0)
        self.assertLess(truncation, 200)

    def test_paired_comparison_requires_a_shared_seed(self):
        first = run_scenario(merge_config(DEFAULT_CONFIG, FAST), name="a")
        second_config = merge_config(DEFAULT_CONFIG, FAST)
        second_config["run"]["seed"] += 1
        second = run_scenario(second_config, name="b")
        with self.assertRaises(ValueError):
            compare_scenarios(first, second, "chain_bullwhip")

    def test_paired_comparison_requires_a_shared_warmup(self):
        first = run_scenario(merge_config(DEFAULT_CONFIG, FAST), name="a", warmup=60)
        second = run_scenario(merge_config(DEFAULT_CONFIG, FAST), name="b", warmup=80)
        with self.assertRaises(ValueError):
            compare_scenarios(first, second, "chain_bullwhip")

    def test_scenarios_under_crn_see_identical_demand(self):
        base = merge_config(DEFAULT_CONFIG, FAST)
        smoothing = run_scenario(
            merge_config(base, {"forecast": {"kind": "exponential", "alpha": 0.5}}),
            keep_results=True,
        )
        average = run_scenario(
            merge_config(base, {"forecast": {"kind": "moving_average", "window": 12}}),
            keep_results=True,
        )
        for left, right in zip(smoothing.results, average.results):
            np.testing.assert_array_equal(left.customer_demand, right.customer_demand)


class TestInformationStudy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comparison = compare_information_modes(
            merge_config(FAST, {"forecast": {"kind": "exponential", "alpha": 0.3}}),
            warmup=60,
        )

    def test_all_three_modes_are_run(self):
        self.assertEqual(set(self.comparison.outcomes), set(m.value for m in InfoMode))

    def test_sharing_reduces_upstream_amplification(self):
        chain = self.comparison.chain_bullwhip()
        self.assertLess(chain["pos_shared"].mean, chain["decentralized"].mean)
        self.assertLess(chain["vmi"].mean, chain["decentralized"].mean)

    def test_the_retailer_is_unaffected_by_what_upstream_knows(self):
        """Sanity check on the mechanism: information sharing changes what the
        *upstream* nodes see, so the retailer's own amplification must be
        essentially unchanged. If it moves, the modes differ in something other
        than information."""
        bullwhip = self.comparison.bullwhip_by_mode()
        values = [bullwhip[mode]["retailer"].mean for mode in bullwhip]
        self.assertLess(max(values) - min(values), 0.2)

    def test_paired_percent_change_is_negative_for_sharing(self):
        change = self.comparison.paired_percent("pos_shared", "chain_bullwhip")
        self.assertLess(change.high, 0.0)


class TestDisruptionStudy(unittest.TestCase):
    def test_recovery_rule_needs_a_sustained_return(self):
        """A single good period inside the outage must not count as recovery."""
        periods = 120
        good = np.ones(periods)
        bad = good.copy()
        bad[40:60] = 0.5
        bad[50] = 1.0  # one deceptive spike mid-outage

        class Stub:
            def __init__(self, fill):
                self.periods = periods
                self._fill = fill

            def stocking_series(self):
                return [self]

            level = 0
            name = "retailer"

            @property
            def fill_numerator(self):
                return self._fill * 100.0

            @property
            def demand_received(self):
                return np.full(periods, 100.0)

            @property
            def backlog(self):
                return np.zeros(periods)

            nodes = {}

        disrupted = Stub(bad)
        baseline = Stub(good)
        disrupted.nodes = {"retailer": disrupted}
        baseline.nodes = {"retailer": baseline}
        profile = measure_recovery([disrupted], [baseline], start=35, duration=20,
                                   horizon=60, bootstrap=20)
        self.assertGreater(profile.recovery_offset.mean, 20.0)
        self.assertLess(profile.trough, 0.6)

    def test_study_reports_a_profile_per_scenario(self):
        study = run_disruption_study(
            merge_config(FAST, {"topology": {"capacity": 160.0}}),
            scenarios=[
                ("outage", {"disruptions": {"outages": [
                    {"node": "source", "start": 150, "duration": 6}]}}, 150, 6),
                ("shock", {"demand": {"shock": {
                    "start": 150, "duration": 4, "multiplier": 2.0}}}, 150, 4),
            ],
            warmup=60,
            horizon=60,
        )
        self.assertEqual(len(study.profiles), 2)
        for profile in study.profiles:
            self.assertLess(profile.trough, 1.0)
            self.assertGreaterEqual(profile.recovery_offset.mean, 0.0)

    def test_a_disruption_inside_the_warmup_is_rejected(self):
        with self.assertRaises(ValueError):
            run_disruption_study(
                FAST,
                scenarios=[("bad", {}, 30, 4)],
                warmup=60,
            )

    def test_fill_path_is_unit_weighted_across_retailers(self):
        outcome = run_scenario(
            merge_config(FAST, {"topology": {"kind": "divergent", "n_retailers": 3}}),
            keep_results=True,
        )
        path = retailer_fill_path(outcome.results[0])
        self.assertEqual(path.shape, (240,))
        self.assertLessEqual(path.max(), 1.0)
        self.assertGreaterEqual(path.min(), 0.0)


class TestLeadTimeTradeoffs(unittest.TestCase):
    def test_convolution_formula_matches_hand_calculation(self):
        sigma = lead_time_demand_sigma(mean_lead=4.0, demand_mean=100.0,
                                       demand_std=20.0, lead_std=1.0)
        self.assertAlmostEqual(sigma, (4 * 400 + 10000 * 1) ** 0.5)

    def test_variability_dominates_length_in_the_analytic_formula(self):
        longer = lead_time_demand_sigma(8.0, 100.0, 20.0, 0.0)
        erratic = lead_time_demand_sigma(4.0, 100.0, 20.0, 2.0)
        self.assertGreater(erratic, longer)

    def test_calibration_hits_the_service_target(self):
        config = merge_config(DEFAULT_CONFIG, {
            "topology": {"levels": 1},
            "leadtime": {"kind": "gamma", "mean": 3.0, "cv": 0.2},
            "run": {"periods": 400, "replications": 6, "warmup": 60},
        })
        calibration = calibrate_service(config, target_fill=0.95, tolerance=2e-3)
        self.assertTrue(calibration.converged)
        self.assertAlmostEqual(calibration.achieved.mean, 0.95, delta=3e-3)

    def test_grid_holds_service_constant_while_inventory_moves(self):
        cells = lead_time_grid(
            means=(3.0, 6.0),
            cvs=(0.0, 0.4),
            target_fill=0.95,
            base_config={"run": {"periods": 400, "replications": 6, "warmup": 60}},
        )
        self.assertEqual(len(cells), 4)
        for cell in cells:
            self.assertAlmostEqual(cell.fill_rate.mean, 0.95, delta=0.01)
        by_key = {(c.mean_lead, c.cv_lead): c for c in cells}
        self.assertGreater(
            by_key[(3.0, 0.4)].inventory.mean, by_key[(3.0, 0.0)].inventory.mean
        )


class TestCli(unittest.TestCase):
    def _run(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(argv)
        self.assertEqual(code, 0)
        return buffer.getvalue()

    def test_run_command_prints_a_node_table(self):
        output = self._run(["--periods", "300", "--replications", "3", "run"])
        self.assertIn("retailer", output)
        self.assertIn("average cost", output)

    def test_bullwhip_command_emits_json(self):
        output = self._run(
            ["--periods", "300", "--replications", "3", "--json", "bullwhip"]
        )
        payload = json.loads(output)
        self.assertIn("factory", payload["echelons"])
        self.assertGreater(payload["echelons"]["factory"]["cumulative"], 1.0)

    def test_warmup_command_reports_a_truncation_point(self):
        output = self._run(["--periods", "400", "warmup", "--pilots", "2"])
        self.assertIn("MSER-5", output)

    def test_information_command_covers_every_mode(self):
        output = self._run(
            ["--periods", "300", "--replications", "4", "--json", "information"]
        )
        payload = json.loads(output)
        self.assertEqual(set(payload["modes"]), {"decentralized", "pos_shared", "vmi"})

    def test_config_file_is_honoured(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "cfg.json")
            save_config(merge_config(DEFAULT_CONFIG, {
                "topology": {"levels": 2},
                "run": {"periods": 300, "replications": 3, "warmup": 60},
            }), path)
            output = self._run(["--config", path, "--json", "run"])
            payload = json.loads(output)
            self.assertEqual(set(payload["nodes"]), {"retailer", "distributor"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
