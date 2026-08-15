"""Order batching: capacity feasibility and the travel it actually saves."""

from __future__ import annotations

import unittest

from slotting import (
    CartCapacity,
    CatalogConfig,
    OrderConfig,
    WarehouseConfig,
    build_instance,
    evaluate_batches,
    savings_batching,
    seed_batching,
    single_order_batches,
    velocity_slotting,
)


class TestBatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = build_instance(
            seed=5,
            warehouse_config=WarehouseConfig(n_aisles=6, n_bays=12, n_levels=3),
            catalog_config=CatalogConfig(n_skus=200, n_families=8),
            order_config=OrderConfig(n_orders=600),
        )
        cls.assignment = velocity_slotting(
            cls.inst.constraints, metric="picks", pick_rate=cls.inst.fit_rate
        )
        cls.cap = CartCapacity()
        cls.stream = cls.inst.score_stream

    def _all_batches(self):
        return {
            "seed": seed_batching(
                self.stream, self.assignment, self.inst.catalog, self.inst.warehouse, self.cap
            ),
            "savings": savings_batching(
                self.stream, self.assignment, self.inst.catalog, self.inst.warehouse, self.cap
            ),
        }

    def test_every_order_is_picked_exactly_once(self):
        expected = sorted(o.order_id for o in self.stream)
        for name, batches in self._all_batches().items():
            with self.subTest(policy=name):
                got = sorted(oid for b in batches for oid in b.order_ids)
                self.assertEqual(got, expected)

    def test_no_batch_exceeds_cart_capacity(self):
        """Capacity holds for every batch the policies actually built.

        The one class of exception is an order that does not fit a cart on its
        own: nothing can be merged into it, and neither policy splits an order
        across trips. Those singletons are carried through unchanged, so the
        invariant is stated as "feasible, or an oversize order picked alone".
        """
        oversize = {
            b.order_ids[0]
            for b in single_order_batches(self.stream, self.assignment, self.inst.catalog)
            if not self.cap.fits(1, b.lines, b.cube)
        }
        self.assertGreater(len(oversize), 0, "no oversize orders in this stream to check")
        for name, batches in self._all_batches().items():
            for b in batches:
                with self.subTest(policy=name):
                    if self.cap.fits(len(b.order_ids), b.lines, b.cube):
                        continue
                    self.assertEqual(len(b.order_ids), 1)
                    self.assertIn(b.order_ids[0], oversize)

    def test_batching_beats_strict_single_order_picking(self):
        single = evaluate_batches(
            self.inst.warehouse,
            single_order_batches(self.stream, self.assignment, self.inst.catalog),
            "s_shape",
        )
        for name, batches in self._all_batches().items():
            with self.subTest(policy=name):
                result = evaluate_batches(self.inst.warehouse, batches, "s_shape")
                self.assertLess(result.total_distance_m, single.total_distance_m)
                self.assertLess(result.distance_per_order_m, single.distance_per_order_m)
                self.assertLess(result.n_batches, single.n_batches)

    def test_savings_batching_is_at_least_as_good_as_seed_batching(self):
        batches = self._all_batches()
        seeded = evaluate_batches(self.inst.warehouse, batches["seed"], "s_shape")
        saved = evaluate_batches(self.inst.warehouse, batches["savings"], "s_shape")
        self.assertLessEqual(saved.total_distance_m, seeded.total_distance_m)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
