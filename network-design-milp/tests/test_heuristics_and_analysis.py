"""Greenfield heuristic, sensitivity tooling, diagnostics, reporting and CLI."""

from __future__ import annotations

import io
import json
import contextlib
import tempfile
import unittest
from pathlib import Path

from netdesign.cli import main
from netdesign.diagnostics import capacity_ledger, diagnose
from netdesign.geometry import haversine_matrix
from netdesign.greenfield import (
    evaluate_open_set,
    greenfield_centers,
    greenfield_open_set,
    solve_with_cutoff,
)
from netdesign.instances import Instance, generate_instance
from netdesign.network_flow import NetworkOptions, solve_network_design
from netdesign.reporting import cost_breakdown, flow_by_echelon, format_report, markdown_table
from netdesign.scenarios import elasticity, sensitivity_sweep, stability_profile


def instance():
    return generate_instance(seed=7, n_dcs=5, n_zones=12, n_commodities=1, n_plants=3)


class TestGreenfield(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = instance()

    def test_more_centers_never_increase_weighted_distance(self):
        pts = [z.coords for z in self.inst.zones]
        w = [self.inst.zone_demand(z.id) for z in self.inst.zones]
        costs = [greenfield_centers(pts, w, p, seed=1, restarts=4)[2] for p in (1, 2, 3, 4)]
        for a, b in zip(costs, costs[1:]):
            self.assertLessEqual(b, a + 1e-6)

    def test_the_one_median_is_at_least_as_good_as_the_best_demand_point(self):
        pts = [z.coords for z in self.inst.zones]
        w = [self.inst.zone_demand(z.id) for z in self.inst.zones]
        centers, _, cost, _ = greenfield_centers(pts, w, 1, seed=1, restarts=3)
        dm = haversine_matrix(pts, pts)
        best_point = min(float((dm[:, j] * w).sum()) for j in range(len(pts)))
        self.assertLessEqual(cost, best_point + 1e-6)

    def test_snapped_set_has_the_requested_size_and_real_candidate_ids(self):
        res = greenfield_open_set(self.inst, 3, seed=1, restarts=3)
        self.assertEqual(len(res.snapped_dcs), 3)
        self.assertEqual(len(set(res.snapped_dcs)), 3)
        self.assertTrue(set(res.snapped_dcs) <= set(self.inst.dc_ids))

    def test_the_heuristic_network_is_never_cheaper_than_the_milp_optimum(self):
        optimum = solve_network_design(self.inst)
        for p in (2, 3):
            res = greenfield_open_set(self.inst, p, seed=1, restarts=3)
            costed = evaluate_open_set(self.inst, res.snapped_dcs)
            if costed.is_optimal:
                self.assertGreaterEqual(costed.objective, optimum.objective - 1e-6)

    def test_evaluate_open_set_opens_exactly_the_requested_dcs(self):
        chosen = self.inst.dc_ids[:3]
        sol = evaluate_open_set(self.inst, chosen)
        self.assertEqual(sorted(sol.open_dcs), sorted(chosen))

    def test_a_valid_cutoff_does_not_remove_the_optimum(self):
        optimum = solve_network_design(self.inst)
        heuristic = evaluate_open_set(
            self.inst, greenfield_open_set(self.inst, 3, seed=1, restarts=3).snapped_dcs
        )
        sol, _ = solve_with_cutoff(self.inst, upper_bound=heuristic.objective)
        self.assertTrue(sol.is_optimal)
        self.assertAlmostEqual(sol.objective, optimum.objective, delta=1e-4 * optimum.objective)


class TestSensitivity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = instance()
        cls.runs = sensitivity_sweep(
            cls.inst, demand_multipliers=(0.9, 1.0, 1.15), cost_multipliers=(0.8, 1.3)
        )

    def test_every_sweep_cell_solved(self):
        self.assertEqual(len(self.runs), 6)
        for r in self.runs:
            self.assertEqual(r.status, "optimal", r.label)

    def test_cost_rises_with_demand_at_fixed_transport_rates(self):
        by_cost = {}
        for r in self.runs:
            by_cost.setdefault(r.params["cost_multiplier"], []).append(
                (r.params["demand_multiplier"], r.objective)
            )
        for cm, pairs in by_cost.items():
            pairs.sort()
            values = [v for _, v in pairs]
            self.assertEqual(values, sorted(values), msg=f"cost multiplier {cm}")

    def test_stability_profile_partitions_every_opened_dc(self):
        prof = stability_profile(self.runs)
        buckets = set(prof["core"]) | set(prof["swing"]) | set(prof["out"])
        self.assertEqual(buckets, set(prof["frequency"]))
        self.assertGreaterEqual(prof["n_runs"], 1)

    def test_cost_elasticity_to_demand_is_positive_and_near_unity(self):
        # Below 1 means fixed costs are being spread over more volume; above 1
        # means lumpy capacity is being added faster than volume grows. Both
        # happen on real networks, so the test brackets rather than assumes.
        slope = elasticity(self.runs, "demand_multiplier")
        self.assertIsNotNone(slope)
        self.assertGreater(slope, 0.3)
        self.assertLess(slope, 1.7)


class TestDiagnostics(unittest.TestCase):
    def test_a_feasible_instance_is_reported_feasible(self):
        diag = diagnose(instance())
        self.assertTrue(diag.feasible)
        self.assertEqual(diag.violations, [])

    def test_an_over_constrained_instance_names_the_binding_group(self):
        inst = instance()
        stressed = inst.with_demand(inst.scaled_demand(2.2))
        diag = diagnose(stressed, NetworkOptions(max_open_dcs=2))
        self.assertFalse(diag.feasible)
        self.assertGreater(diag.total_violation, 0.0)
        self.assertIn("demand", diag.by_group)
        self.assertTrue(any("supply" in p or "throughput" in p for p in diag.structural_problems))
        self.assertIn("model is INFEASIBLE", diag.summary())

    def test_capacity_ledger_totals_are_consistent(self):
        inst = instance()
        ledger = capacity_ledger(inst)
        self.assertAlmostEqual(ledger["TOTAL"]["demand"], inst.total_demand(), places=6)
        self.assertAlmostEqual(
            ledger["TOTAL"]["supply"], sum(inst.supply.values()), places=6
        )


class TestReporting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = instance()
        cls.sol = solve_network_design(cls.inst)

    def test_cost_breakdown_shares_sum_to_one_hundred(self):
        rows = cost_breakdown(self.sol)
        self.assertAlmostEqual(sum(r[2] for r in rows), 100.0, places=6)

    def test_flow_by_echelon_covers_every_shipped_unit(self):
        rows = flow_by_echelon(self.sol, self.inst)
        total = sum(float(r["units"]) for r in rows)
        self.assertAlmostEqual(total, sum(r.units for r in self.sol.flows), places=4)

    def test_report_names_every_open_facility(self):
        text = format_report(self.sol, self.inst)
        for dc in self.sol.open_dcs:
            self.assertIn(dc, text)
        self.assertIn("cost breakdown", text)

    def test_markdown_table_has_a_header_rule(self):
        md = markdown_table(["a", "b"], [[1, "x"], [2, "y"]])
        self.assertEqual(len(md.splitlines()), 4)
        self.assertIn("|", md.splitlines()[1])


class TestInstanceRoundTrip(unittest.TestCase):
    def test_json_round_trip_preserves_the_instance(self):
        inst = instance()
        clone = Instance.from_dict(json.loads(inst.to_json()))
        self.assertEqual(clone.demand, inst.demand)
        self.assertEqual(len(clone.lanes), len(inst.lanes))
        self.assertAlmostEqual(clone.total_demand(), inst.total_demand(), places=6)
        self.assertEqual(
            solve_network_design(clone).open_dcs, solve_network_design(inst).open_dcs
        )

    def test_generated_instance_is_structurally_valid(self):
        self.assertEqual(instance().validate(), [])


class TestCLI(unittest.TestCase):
    def test_solve_json_output_is_parseable(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["solve", "--zones", "10", "--dcs", "4", "--commodities", "1", "--json"])
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["status"], "optimal")
        self.assertGreater(payload["objective"], 0.0)
        self.assertTrue(payload["open_dcs"])

    def test_generate_writes_a_loadable_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inst.json"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = main(["generate", "--zones", "8", "--dcs", "4", "--out", str(path)])
            self.assertEqual(rc, 0)
            inst = Instance.from_dict(json.loads(path.read_text()))
            self.assertEqual(len(inst.zones), 8)

    def test_diagnose_returns_a_non_zero_code_for_an_infeasible_instance(self):
        with contextlib.redirect_stdout(io.StringIO()):
            rc = main(
                [
                    "diagnose",
                    "--zones",
                    "10",
                    "--dcs",
                    "4",
                    "--commodities",
                    "1",
                    "--demand-multiplier",
                    "2.5",
                    "--max-dcs",
                    "2",
                ]
            )
        self.assertEqual(rc, 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
