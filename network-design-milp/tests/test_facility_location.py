"""Facility location: checked against exhaustive enumeration, not against itself."""

from __future__ import annotations

import unittest

from netdesign.facility_location import (
    brute_force_uflp,
    capacitated_facility_location,
    dc_location_subproblem,
    uncapacitated_facility_location,
)
from netdesign.instances import generate_instance


class TestUFLP(unittest.TestCase):
    def test_milp_matches_brute_force_on_several_seeds(self):
        for seed in (1, 5, 9):
            inst = generate_instance(seed=seed, n_dcs=6, n_zones=10, n_commodities=1, n_plants=2)
            fixed, unit_cost, demand, _ = dc_location_subproblem(inst)
            res = uncapacitated_facility_location(fixed, unit_cost, demand)
            best, best_set = brute_force_uflp(fixed, unit_cost, demand)
            self.assertTrue(res.status == "optimal")
            self.assertAlmostEqual(res.objective, best, delta=1e-4 * best)
            self.assertEqual(sorted(res.open_facilities), sorted(best_set))

    def test_strong_formulation_dominates_the_aggregated_one(self):
        inst = generate_instance(seed=3, n_dcs=6, n_zones=12, n_commodities=1, n_plants=2)
        fixed, unit_cost, demand, _ = dc_location_subproblem(inst)
        strong = uncapacitated_facility_location(
            fixed, unit_cost, demand, formulation="strong", with_lp_bound=True
        )
        weak = uncapacitated_facility_location(
            fixed, unit_cost, demand, formulation="aggregated", with_lp_bound=True
        )
        self.assertAlmostEqual(strong.objective, weak.objective, delta=1e-4 * strong.objective)
        self.assertGreater(strong.lp_bound, weak.lp_bound)
        self.assertLessEqual(strong.integrality_gap, weak.integrality_gap)

    def test_every_customer_is_fully_served(self):
        inst = generate_instance(seed=2, n_dcs=5, n_zones=9, n_commodities=1, n_plants=2)
        fixed, unit_cost, demand, _ = dc_location_subproblem(inst)
        res = uncapacitated_facility_location(fixed, unit_cost, demand)
        for zone, mix in res.assignment.items():
            self.assertAlmostEqual(sum(mix.values()), 1.0, places=6, msg=zone)
            self.assertTrue(set(mix) <= set(res.open_facilities))

    def test_p_median_constraint_is_honoured(self):
        inst = generate_instance(seed=4, n_dcs=6, n_zones=10, n_commodities=1, n_plants=2)
        fixed, unit_cost, demand, _ = dc_location_subproblem(inst)
        res = uncapacitated_facility_location(fixed, unit_cost, demand, p_facilities=3)
        self.assertEqual(len(res.open_facilities), 3)


class TestCFLP(unittest.TestCase):
    def setUp(self):
        self.inst = generate_instance(seed=6, n_dcs=5, n_zones=12, n_commodities=1, n_plants=2)
        self.fixed, self.unit_cost, self.demand, self.capacity = dc_location_subproblem(self.inst)

    def test_capacity_is_respected(self):
        res = capacitated_facility_location(self.fixed, self.unit_cost, self.demand, self.capacity)
        load = {f: 0.0 for f in self.fixed}
        for zone, mix in res.assignment.items():
            for f, share in mix.items():
                load[f] += share * self.demand[zone]
        for f, v in load.items():
            self.assertLessEqual(v, self.capacity[f] + 1e-6, msg=f)

    def test_capacitated_optimum_is_never_cheaper_than_uncapacitated(self):
        cap = capacitated_facility_location(
            self.fixed, self.unit_cost, self.demand, self.capacity
        )
        unc = uncapacitated_facility_location(self.fixed, self.unit_cost, self.demand)
        self.assertGreaterEqual(cap.objective, unc.objective - 1e-6)

    def test_single_sourcing_gives_integral_assignments_and_costs_money(self):
        multi = capacitated_facility_location(
            self.fixed, self.unit_cost, self.demand, self.capacity, single_source=False
        )
        single = capacitated_facility_location(
            self.fixed, self.unit_cost, self.demand, self.capacity, single_source=True
        )
        self.assertGreaterEqual(single.objective, multi.objective - 1e-6)
        for zone, mix in single.assignment.items():
            self.assertEqual(len(mix), 1, msg=f"{zone} was split under single sourcing")
            self.assertAlmostEqual(next(iter(mix.values())), 1.0, places=6)

    def test_minimum_volume_forces_load_onto_open_sites(self):
        min_volume = {f: 0.2 * self.capacity[f] for f in self.fixed}
        res = capacitated_facility_location(
            self.fixed, self.unit_cost, self.demand, self.capacity, min_volume=min_volume
        )
        self.assertEqual(res.status, "optimal")
        load = {f: 0.0 for f in self.fixed}
        for zone, mix in res.assignment.items():
            for f, share in mix.items():
                load[f] += share * self.demand[zone]
        for f in res.open_facilities:
            self.assertGreaterEqual(load[f], min_volume[f] - 1e-6, msg=f)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
