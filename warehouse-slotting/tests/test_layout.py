"""Geometry and the travel metric."""

from __future__ import annotations

import unittest

import networkx as nx
import numpy as np

from slotting import Point, Warehouse, WarehouseConfig
from slotting.ergonomics import golden_zone_order


class TestWarehouseConfig(unittest.TestCase):
    def test_rejects_degenerate_geometry(self):
        for kwargs in (
            {"n_aisles": 0},
            {"n_bays": 0},
            {"n_levels": 0},
            {"n_blocks": 0},
            {"depot_aisle": 99},
            {"weight_capacity_decay": 0.0},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    WarehouseConfig(**kwargs)


class TestTravelMetric(unittest.TestCase):
    """The closed form must equal a shortest path on the explicit corridor graph."""

    def _check_against_dijkstra(self, config: WarehouseConfig) -> None:
        w = Warehouse(config)
        g = w.to_networkx()
        lengths = dict(nx.all_pairs_dijkstra_path_length(g, weight="weight"))
        for a in range(config.n_aisles):
            for b in range(config.n_bays):
                for a2 in range(config.n_aisles):
                    for b2 in range(config.n_bays):
                        closed = w.travel_distance(Point(a, w.bay_y(b)), Point(a2, w.bay_y(b2)))
                        graph = lengths[(a, w.bay_y(b))][(a2, w.bay_y(b2))]
                        self.assertAlmostEqual(closed, graph, places=6)

    def test_matches_dijkstra_single_block(self):
        self._check_against_dijkstra(WarehouseConfig(n_aisles=4, n_bays=5, n_levels=2))

    def test_matches_dijkstra_two_blocks(self):
        self._check_against_dijkstra(
            WarehouseConfig(n_aisles=4, n_bays=6, n_levels=2, n_blocks=2)
        )

    def test_symmetric_and_zero_on_diagonal(self):
        w = Warehouse(WarehouseConfig(n_aisles=3, n_bays=4, n_levels=2))
        m = w.access_distance_matrix()
        np.testing.assert_allclose(m, m.T, atol=1e-9)
        np.testing.assert_allclose(np.diag(m), 0.0, atol=1e-9)

    def test_triangle_inequality(self):
        w = Warehouse(WarehouseConfig(n_aisles=4, n_bays=5, n_levels=2))
        m = w.access_distance_matrix()
        n = m.shape[0]
        rng = np.random.default_rng(0)
        for _ in range(300):
            i, j, k = rng.integers(0, n, size=3)
            self.assertLessEqual(m[i, j], m[i, k] + m[k, j] + 1e-9)

    def test_facing_locations_are_far_apart_not_near(self):
        """Two faces across a rack are metres apart in the plane, a walk apart in fact."""
        w = Warehouse(WarehouseConfig(n_aisles=6, n_bays=20, n_levels=2))
        mid = 10
        same_aisle = w.travel_distance(Point(2, w.bay_y(mid)), Point(2, w.bay_y(mid + 1)))
        across = w.travel_distance(Point(2, w.bay_y(mid)), Point(3, w.bay_y(mid)))
        self.assertAlmostEqual(same_aisle, w.config.bay_depth_m, places=6)
        self.assertGreater(across, 10 * same_aisle)

    def test_mid_cross_aisle_shortens_mid_warehouse_moves(self):
        base = dict(n_aisles=6, n_bays=24, n_levels=2)
        one = Warehouse(WarehouseConfig(n_blocks=1, **base))
        two = Warehouse(WarehouseConfig(n_blocks=2, **base))
        p, q = Point(0, one.bay_y(11)), Point(5, one.bay_y(12))
        self.assertLess(two.travel_distance(p, q), one.travel_distance(p, q))

    def test_levels_share_one_access_point(self):
        w = Warehouse(WarehouseConfig(n_aisles=3, n_bays=4, n_levels=3))
        floor = w.index[w.locations[0]]
        top = [i for i, l in enumerate(w.locations) if l.aisle == 0 and l.bay == 0 and l.level == 2][0]
        self.assertEqual(w.distance(floor, top), 0.0)
        self.assertAlmostEqual(w.distance_from_depot(floor), w.distance_from_depot(top))


class TestCapabilityAndErgonomics(unittest.TestCase):
    def test_weight_capacity_falls_with_level(self):
        w = Warehouse(WarehouseConfig(n_levels=4))
        caps = [w.weight_capacity(l) for l in range(4)]
        self.assertEqual(caps, sorted(caps, reverse=True))
        self.assertAlmostEqual(caps[1] / caps[0], w.config.weight_capacity_decay)

    def test_golden_zone_order_is_a_permutation_and_favours_waist_height(self):
        order = golden_zone_order(4, 1.0)
        self.assertEqual(sorted(order), [0, 1, 2, 3])
        self.assertIn(order[0], (0, 1))
        self.assertEqual(order[-1], 3)

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
