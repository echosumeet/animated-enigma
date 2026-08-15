"""The network design model, checked against its own physics.

Every assertion here is a property the *solution* must satisfy - conservation,
capacity, coverage - rather than a recorded objective value. A regression test
on the objective tells you something changed; these tell you what broke.
"""

from __future__ import annotations

import unittest
from collections import defaultdict

from netdesign.instances import generate_instance
from netdesign.network_flow import NetworkDesignModel, NetworkOptions, solve_network_design

TOL = 1e-5


def small_instance(**kwargs):
    params = dict(seed=7, n_dcs=5, n_zones=12, n_commodities=2, n_plants=3)
    params.update(kwargs)
    return generate_instance(**params)


class TestFeasibilityInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = small_instance()
        cls.sol = solve_network_design(cls.inst)

    def test_solves_to_optimality(self):
        self.assertTrue(self.sol.is_optimal, self.sol.status)

    def test_all_demand_is_served(self):
        served = defaultdict(float)
        for r in self.sol.flows:
            if self.inst.site(r.dest).kind == "zone":
                served[(r.dest, r.commodity)] += r.units
        for key, qty in self.inst.demand.items():
            self.assertAlmostEqual(served[key], qty, delta=max(TOL, 1e-6 * qty), msg=str(key))

    def test_flow_is_conserved_at_every_open_facility(self):
        inflow = defaultdict(float)
        outflow = defaultdict(float)
        for r in self.sol.flows:
            outflow[(r.origin, r.commodity)] += r.units
            inflow[(r.dest, r.commodity)] += r.units
        for fac in (*self.inst.plants, *self.inst.dcs):
            for c in self.inst.commodity_ids:
                self.assertAlmostEqual(
                    inflow[(fac.id, c)],
                    outflow[(fac.id, c)],
                    delta=max(TOL, 1e-6 * max(1.0, inflow[(fac.id, c)])),
                    msg=f"{fac.id}/{c}",
                )

    def test_closed_facilities_carry_no_flow(self):
        for fac in (*self.inst.plants, *self.inst.dcs):
            if fac.id in self.sol.open_plants or fac.id in self.sol.open_dcs:
                continue
            moved = sum(r.units for r in self.sol.flows if fac.id in (r.origin, r.dest))
            self.assertLess(moved, TOL, msg=f"{fac.id} moved volume while closed")

    def test_capacity_and_minimum_volume_hold_at_open_dcs(self):
        for dc in self.inst.dcs:
            u = self.sol.utilization[dc.id]
            self.assertLessEqual(u["throughput"], dc.capacity + TOL, msg=dc.id)
            if dc.id in self.sol.open_dcs:
                self.assertGreaterEqual(u["throughput"], dc.min_volume - TOL, msg=dc.id)

    def test_supplier_availability_is_respected(self):
        shipped = defaultdict(float)
        for r in self.sol.flows:
            if self.inst.site(r.origin).kind == "supplier":
                shipped[(r.origin, r.commodity)] += r.units
        for key, cap in self.inst.supply.items():
            self.assertLessEqual(shipped[key], cap + TOL, msg=str(key))

    def test_cost_buckets_add_up_to_the_objective(self):
        self.assertAlmostEqual(
            sum(self.sol.costs.values()),
            self.sol.objective,
            delta=1e-4 * abs(self.sol.objective),
        )


class TestFormulationStrength(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = small_instance()

    def _bound_and_optimum(self, **opts):
        builder = NetworkDesignModel(self.inst, NetworkOptions(**opts))
        sol, _ = builder.solve()
        return builder.lp_bound(), sol.objective

    def test_lp_bound_never_exceeds_the_milp_optimum(self):
        lp, milp = self._bound_and_optimum()
        self.assertLessEqual(lp, milp + 1e-6)

    def test_disaggregated_linking_tightens_the_bound_without_changing_the_optimum(self):
        strong_lp, strong_obj = self._bound_and_optimum(disaggregated_link=True)
        weak_lp, weak_obj = self._bound_and_optimum(disaggregated_link=False)
        self.assertAlmostEqual(strong_obj, weak_obj, delta=1e-4 * strong_obj)
        self.assertGreater(strong_lp, weak_lp)

    def test_loose_big_m_weakens_the_bound_but_keeps_the_same_optimum(self):
        tight_lp, tight_obj = self._bound_and_optimum(disaggregated_link=False)
        loose_lp, loose_obj = self._bound_and_optimum(disaggregated_link=False, big_m_scale=50.0)
        self.assertAlmostEqual(tight_obj, loose_obj, delta=1e-4 * tight_obj)
        self.assertLess(loose_lp, tight_lp)

    def test_a_big_m_below_the_tight_value_is_rejected(self):
        with self.assertRaises(ValueError):
            NetworkOptions(big_m_scale=0.5).validate()


class TestServiceConstraints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = small_instance()
        cls.base = solve_network_design(cls.inst)

    def test_single_sourcing_leaves_one_dc_per_zone_and_is_never_cheaper(self):
        sol = solve_network_design(self.inst, NetworkOptions(single_source=True))
        self.assertTrue(sol.is_optimal)
        self.assertGreaterEqual(sol.objective, self.base.objective - 1e-6)
        for zone, mix in sol.zone_service.items():
            self.assertEqual(len(mix), 1, msg=f"{zone} served by {sorted(mix)}")

    def test_dual_sourcing_caps_any_single_dc_share_and_is_never_cheaper(self):
        share = 0.25
        sol = solve_network_design(self.inst, NetworkOptions(min_second_source_share=share))
        self.assertTrue(sol.is_optimal)
        self.assertGreaterEqual(sol.objective, self.base.objective - 1e-6)
        for zone, mix in sol.zone_service.items():
            self.assertGreaterEqual(len(mix), 2, msg=zone)
            self.assertLessEqual(max(mix.values()), 1.0 - share + 1e-6, msg=zone)

    def test_single_and_dual_sourcing_together_are_rejected(self):
        with self.assertRaises(ValueError):
            NetworkOptions(single_source=True, min_second_source_share=0.2).validate()

    def test_max_open_dcs_is_binding(self):
        cap = max(1, len(self.base.open_dcs))
        sol = solve_network_design(self.inst, NetworkOptions(max_open_dcs=cap))
        self.assertTrue(sol.is_optimal)
        self.assertLessEqual(len(sol.open_dcs), cap)
        self.assertGreaterEqual(sol.objective, self.base.objective - 1e-6)

    def test_a_dc_cap_below_total_capacity_need_is_infeasible_not_silently_wrong(self):
        # each candidate is sized at a fraction of total demand, so too few
        # sites cannot cover it; the model must say so rather than drop demand
        sol = solve_network_design(self.inst, NetworkOptions(max_open_dcs=1))
        self.assertEqual(sol.status, "infeasible")

    def test_fixing_the_open_set_reproduces_that_network(self):
        chosen = {d.id: int(d.id in self.base.open_dcs) for d in self.inst.dcs}
        sol = solve_network_design(self.inst, NetworkOptions(fixed_open_dcs=chosen))
        self.assertEqual(sorted(sol.open_dcs), sorted(self.base.open_dcs))
        self.assertAlmostEqual(sol.objective, self.base.objective, delta=1e-4 * self.base.objective)

    def test_unmet_demand_is_only_used_when_it_has_to_be(self):
        sol = solve_network_design(self.inst, NetworkOptions(allow_unmet=True, unmet_penalty=1e4))
        self.assertLess(sol.unmet_units, TOL)


class TestScenarioBlock(unittest.TestCase):
    def test_one_scenario_reproduces_the_deterministic_model(self):
        inst = small_instance()
        det = solve_network_design(inst)
        builder = NetworkDesignModel(inst, scenarios=[dict(inst.demand)], probabilities=[1.0])
        sol, _ = builder.solve()
        self.assertAlmostEqual(sol.objective, det.objective, delta=1e-4 * det.objective)
        self.assertEqual(sorted(sol.open_dcs), sorted(det.open_dcs))

    def test_probabilities_must_sum_to_one(self):
        inst = small_instance()
        with self.assertRaises(ValueError):
            NetworkDesignModel(
                inst, scenarios=[dict(inst.demand), dict(inst.demand)], probabilities=[0.5, 0.9]
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
