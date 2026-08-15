"""Routing heuristics, their legality, and their gap to the exact reference."""

from __future__ import annotations

import unittest

import numpy as np

from slotting import (
    ROUTERS,
    Warehouse,
    WarehouseConfig,
    exact_aisle_dp,
    held_karp,
    s_shape,
    two_opt,
    validate_route,
)

HEURISTICS = ("s_shape", "return", "midpoint", "largest_gap", "nearest_neighbour", "two_opt")


def small_warehouse() -> Warehouse:
    return Warehouse(WarehouseConfig(n_aisles=5, n_bays=10, n_levels=3))


def random_pick_sets(w: Warehouse, n_sets: int, max_picks: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    for _ in range(n_sets):
        k = int(rng.integers(1, max_picks + 1))
        yield sorted(rng.choice(w.n_locations, size=k, replace=False).tolist())


class TestRouteLegality(unittest.TestCase):
    def test_every_heuristic_produces_a_legal_closed_tour(self):
        w = small_warehouse()
        for picks in random_pick_sets(w, 25, 8, seed=1):
            for name in HEURISTICS:
                with self.subTest(policy=name, n=len(picks)):
                    route = ROUTERS[name](w, picks)
                    validate_route(w, route, picks)

    def test_route_starts_and_ends_at_the_depot(self):
        w = small_warehouse()
        route = s_shape(w, [3, 40, 120])
        self.assertEqual(route.waypoints[0], w.depot)
        self.assertEqual(route.waypoints[-1], w.depot)

    def test_reported_distance_equals_the_walked_path(self):
        w = small_warehouse()
        for picks in random_pick_sets(w, 12, 7, seed=2):
            for name in HEURISTICS:
                route = ROUTERS[name](w, picks)
                walked = sum(
                    w.travel_distance(p, q)
                    for p, q in zip(route.waypoints, route.waypoints[1:])
                )
                self.assertAlmostEqual(route.distance, walked, places=6)

    def test_empty_pick_list_is_a_zero_length_tour(self):
        w = small_warehouse()
        for name in HEURISTICS:
            self.assertEqual(ROUTERS[name](w, []).distance, 0.0)

    def test_single_pick_costs_the_round_trip(self):
        w = small_warehouse()
        loc = 200
        expected = 2 * w.distance_from_depot(loc)
        for name in HEURISTICS:
            with self.subTest(policy=name):
                self.assertAlmostEqual(ROUTERS[name](w, [loc]).distance, expected, places=6)


class TestExactReferences(unittest.TestCase):
    def test_aisle_dp_agrees_with_held_karp(self):
        """Two independent exact methods, on every instance small enough for both."""
        w = Warehouse(WarehouseConfig(n_aisles=4, n_bays=8, n_levels=2))
        checked = 0
        for picks in random_pick_sets(w, 60, 8, seed=3):
            positions = {(w.locations[p].aisle, w.locations[p].bay) for p in picks}
            if len(positions) > 8:
                continue
            dp = exact_aisle_dp(w, picks)
            hk = held_karp(w, picks, max_picks=10).distance
            self.assertAlmostEqual(dp, hk, places=6)
            checked += 1
        self.assertGreater(checked, 20)

    def test_no_heuristic_ever_beats_the_exact_route(self):
        w = small_warehouse()
        for picks in random_pick_sets(w, 40, 9, seed=4):
            opt = exact_aisle_dp(w, picks)
            for name in HEURISTICS:
                with self.subTest(policy=name):
                    self.assertGreaterEqual(ROUTERS[name](w, picks).distance, opt - 1e-6)

    def test_two_opt_never_worse_than_its_nearest_neighbour_start(self):
        w = small_warehouse()
        for picks in random_pick_sets(w, 25, 10, seed=5):
            nn = ROUTERS["nearest_neighbour"](w, picks).distance
            self.assertLessEqual(two_opt(w, picks).distance, nn + 1e-9)

    def test_two_opt_is_near_optimal_on_sparse_orders(self):
        """The claim the benchmark leans on: 2-opt is a usable stand-in for exact."""
        w = small_warehouse()
        gaps = []
        for picks in random_pick_sets(w, 60, 6, seed=6):
            opt = exact_aisle_dp(w, picks)
            if opt <= 0:
                continue
            gaps.append(two_opt(w, picks).distance / opt - 1.0)
        self.assertLess(float(np.mean(gaps)), 0.02)


class TestHeuristicStructure(unittest.TestCase):
    def test_return_route_stays_out_of_the_back_cross_aisle_for_front_picks(self):
        w = Warehouse(WarehouseConfig(n_aisles=5, n_bays=10, n_levels=1))
        picks = [w.index[l] for l in w.locations if l.aisle in (1, 3) and l.bay == 0]
        route = ROUTERS["return"](w, picks)
        self.assertLess(max(p.y for p in route.waypoints), w.aisle_length_m)
        self.assertLess(route.distance, s_shape(w, picks).distance)

    def test_multi_block_layout_rejects_the_single_block_heuristics(self):
        w = Warehouse(WarehouseConfig(n_aisles=4, n_bays=8, n_levels=2, n_blocks=2))
        picks = [5, 60, 119]
        for name in ("s_shape", "midpoint", "largest_gap"):
            with self.subTest(policy=name):
                with self.assertRaises(ValueError):
                    ROUTERS[name](w, picks)
        # The tour-based routers are geometry-agnostic and must still work.
        route = two_opt(w, picks)
        validate_route(w, route, picks)
        self.assertGreater(route.distance, 0.0)

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
