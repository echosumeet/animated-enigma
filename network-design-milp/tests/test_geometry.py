"""Distance and continuous-location primitives, checked against closed forms."""

from __future__ import annotations

import math
import unittest

import numpy as np

from netdesign.geometry import (
    EARTH_RADIUS_KM,
    center_of_gravity,
    haversine_km,
    haversine_matrix,
    road_distance_km,
    weiszfeld,
)


class TestHaversine(unittest.TestCase):
    def test_one_degree_of_latitude_is_a_known_distance(self):
        d = haversine_km(0.0, 0.0, 1.0, 0.0)
        self.assertAlmostEqual(d, math.pi * EARTH_RADIUS_KM / 180.0, places=6)

    def test_quarter_circumference_between_pole_and_equator(self):
        d = haversine_km(0.0, 0.0, 90.0, 0.0)
        self.assertAlmostEqual(d, 0.5 * math.pi * EARTH_RADIUS_KM, places=4)

    def test_symmetry_and_identity(self):
        self.assertAlmostEqual(haversine_km(41.0, -87.0, 34.0, -118.0), haversine_km(34.0, -118.0, 41.0, -87.0))
        self.assertAlmostEqual(haversine_km(41.0, -87.0, 41.0, -87.0), 0.0)

    def test_matrix_agrees_with_the_scalar_form(self):
        o = [(41.0, -87.0), (34.0, -118.0)]
        d = [(29.8, -95.4), (47.6, -122.3), (25.8, -80.2)]
        m = haversine_matrix(o, d)
        for i, (a, b) in enumerate(o):
            for j, (c, e) in enumerate(d):
                self.assertAlmostEqual(m[i, j], haversine_km(a, b, c, e), places=6)

    def test_circuity_inflates_road_distance(self):
        gc = haversine_km(41.0, -87.0, 34.0, -118.0)
        self.assertGreater(road_distance_km(41.0, -87.0, 34.0, -118.0), gc)


class TestContinuousLocation(unittest.TestCase):
    def _weighted_distance(self, center, pts, w):
        return float(sum(wi * haversine_km(center[0], center[1], p[0], p[1]) for p, wi in zip(pts, w)))

    def test_geometric_median_beats_the_centroid_on_weighted_distance(self):
        rng = np.random.default_rng(0)
        pts = [(float(a), float(b)) for a, b in rng.uniform([30, -120], [46, -75], size=(40, 2))]
        w = list(rng.lognormal(0.0, 0.9, size=40))
        cog = center_of_gravity(pts, w)
        med = weiszfeld(pts, w)
        self.assertLess(
            self._weighted_distance(med, pts, w),
            self._weighted_distance(cog, pts, w),
            "the centroid minimises squared distance, not distance",
        )

    def test_geometric_median_of_a_single_point_is_that_point(self):
        lat, lon = weiszfeld([(35.0, -90.0)], [3.0])
        self.assertAlmostEqual(lat, 35.0, places=4)
        self.assertAlmostEqual(lon, -90.0, places=4)

    def test_a_dominant_weight_pulls_the_median_onto_it(self):
        pts = [(30.0, -100.0), (45.0, -80.0), (40.0, -90.0)]
        med = weiszfeld(pts, [1.0, 1.0, 500.0])
        self.assertLess(haversine_km(med[0], med[1], 40.0, -90.0), 40.0)

    def test_center_of_gravity_is_the_weighted_mean(self):
        lat, lon = center_of_gravity([(0.0, 0.0), (10.0, 20.0)], [3.0, 1.0])
        self.assertAlmostEqual(lat, 2.5)
        self.assertAlmostEqual(lon, 5.0)

    def test_degenerate_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            weiszfeld([(1.0, 1.0), (2.0, 2.0)], [0.0, 0.0])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
