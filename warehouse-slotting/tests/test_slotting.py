"""Data generation, constraints, slotting policies and the surrogate objective."""

from __future__ import annotations

import unittest

import numpy as np

from slotting import (
    CatalogConfig,
    ConstraintConfig,
    ConstraintModel,
    OrderConfig,
    WarehouseConfig,
    affinity_slotting,
    build_instance,
    class_based_slotting,
    evaluate_travel,
    generate_catalog,
    make_objective,
    random_assignment,
    steepest_descent,
    velocity_slotting,
)
from slotting.objective import ObjectiveWeights

SMALL = dict(
    warehouse_config=WarehouseConfig(n_aisles=6, n_bays=12, n_levels=3),
    catalog_config=CatalogConfig(n_skus=200, n_families=8),
    order_config=OrderConfig(n_orders=600),
)


def small_instance(seed: int = 5):
    return build_instance(seed=seed, **SMALL)


class TestGeneratedData(unittest.TestCase):
    def test_catalog_is_reproducible_and_skewed(self):
        a = generate_catalog(CatalogConfig(n_skus=300), seed=3)
        b = generate_catalog(CatalogConfig(n_skus=300), seed=3)
        np.testing.assert_allclose(a.picks, b.picks)
        # Zipf demand: the fast fifth of the range carries most of the picks.
        self.assertGreater(a.concentration(0.20), 0.5)
        self.assertLess(a.concentration(0.05), a.concentration(0.20))

    def test_order_stream_split_is_disjoint_and_ordered(self):
        inst = small_instance()
        self.assertEqual(len(inst.fit_stream) + len(inst.score_stream), len(inst.stream))
        fit_ids = {o.order_id for o in inst.fit_stream}
        score_ids = {o.order_id for o in inst.score_stream}
        self.assertFalse(fit_ids & score_ids)
        self.assertLess(max(fit_ids), min(score_ids))

    def test_orders_are_themed_so_affinity_exists_to_find(self):
        inst = small_instance()
        lift = inst.stream.lift_summary(inst.catalog)
        self.assertGreater(lift["within_family_lift"], 1.0)
        self.assertGreater(inst.stream.mean_lines(), 1.0)


class TestConstraints(unittest.TestCase):
    def test_random_fill_is_complete_and_feasible(self):
        inst = small_instance()
        a = random_assignment(inst.constraints, seed=0)
        self.assertTrue(a.is_complete())
        self.assertEqual(inst.constraints.violations(a), [])
        self.assertEqual(len(set(a.location_of.tolist())), inst.catalog.n_skus)

    def test_hazmat_is_kept_on_the_floor(self):
        inst = small_instance()
        a = random_assignment(inst.constraints, seed=1)
        hazmat = np.flatnonzero(inst.catalog.hazmat != "none")
        self.assertGreater(len(hazmat), 0)
        cfg = inst.constraints.config
        for s in hazmat:
            level = inst.warehouse.locations[int(a.location_of[s])].level
            self.assertLessEqual(level, cfg.hazmat_max_level)

    def test_heavy_skus_are_not_slotted_high(self):
        inst = small_instance()
        a = velocity_slotting(inst.constraints, metric="picks", pick_rate=inst.fit_rate)
        w = inst.warehouse
        slot_weight = inst.catalog.slot_weight(
            w.config.slot_cube_m3, w.config.max_cases_per_slot
        )
        for s in range(inst.catalog.n_skus):
            level = w.locations[int(a.location_of[s])].level
            self.assertLessEqual(float(slot_weight[s]), w.weight_capacity(level) + 1e-6)

    def test_undersized_forward_area_is_refused(self):
        cfg = WarehouseConfig(n_aisles=2, n_bays=2, n_levels=1)
        inst_cat = generate_catalog(CatalogConfig(n_skus=200), seed=2)
        from slotting import Warehouse

        con = ConstraintModel(Warehouse(cfg), inst_cat, ConstraintConfig())
        with self.assertRaises(ValueError):
            random_assignment(con, seed=0)


class TestSlottingPolicies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = small_instance()
        cls.baseline = random_assignment(cls.inst.constraints, seed=0)
        cls.velocity = velocity_slotting(
            cls.inst.constraints, metric="picks", pick_rate=cls.inst.fit_rate
        )

    def test_velocity_slotting_cuts_out_of_sample_travel(self):
        base = evaluate_travel(self.inst, self.baseline, policies=("two_opt",))["two_opt"]
        slotted = evaluate_travel(self.inst, self.velocity, policies=("two_opt",))["two_opt"]
        self.assertLess(slotted, 0.85 * base)

    def test_every_policy_returns_a_feasible_complete_assignment(self):
        con = self.inst.constraints
        rate = self.inst.fit_rate
        policies = {
            "velocity": self.velocity,
            "coi": velocity_slotting(con, metric="coi"),
            "class": class_based_slotting(con, metric="picks", pick_rate=rate),
            "affinity": affinity_slotting(con, self.inst.fit_stream, method="greedy", pick_rate=rate)[0],
        }
        for name, a in policies.items():
            with self.subTest(policy=name):
                self.assertTrue(a.is_complete())
                self.assertEqual(con.violations(a), [])


class TestObjectiveAndSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = small_instance(seed=6)
        cls.obj = make_objective(cls.inst, weights=ObjectiveWeights(velocity=1.0, affinity=0.5))

    def test_swap_delta_matches_a_full_recompute(self):
        a = random_assignment(self.inst.constraints, seed=2)
        rng = np.random.default_rng(0)
        for _ in range(40):
            i, j = rng.integers(0, self.inst.catalog.n_skus, size=2)
            if i == j:
                continue
            before = self.obj.total(a)
            delta = self.obj.swap_delta(a, int(i), int(j))
            a.swap(int(i), int(j))
            self.assertAlmostEqual(self.obj.total(a) - before, delta, places=5)
            a.swap(int(i), int(j))

    def test_steepest_descent_only_ever_improves_and_stays_feasible(self):
        start = velocity_slotting(
            self.inst.constraints, metric="picks", pick_rate=self.inst.fit_rate
        )
        before = self.obj.total(start)
        result = steepest_descent(self.obj, start, self.inst.constraints)
        self.assertLessEqual(self.obj.total(result.assignment), before + 1e-6)
        self.assertEqual(self.inst.constraints.violations(result.assignment), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
