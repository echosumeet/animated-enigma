import unittest

import numpy as np

from etarisk.generate import (
    GeneratorConfig,
    default_network,
    describe_distribution,
    generate_shipments,
)
from etarisk.rng import StreamBank, stream_key


class TestNetwork(unittest.TestCase):
    def test_lane_shares_are_a_distribution(self):
        spec = default_network()
        self.assertAlmostEqual(sum(ln.share for ln in spec.lanes), 1.0, places=9)

    def test_every_lane_mode_has_a_carrier(self):
        spec = default_network()
        covered = {m for c in spec.carriers for m in c.modes}
        self.assertTrue({ln.mode for ln in spec.lanes}.issubset(covered))


class TestStreams(unittest.TestCase):
    def test_stream_keys_are_stable_and_distinct(self):
        self.assertEqual(stream_key("customs"), stream_key("customs"))
        self.assertNotEqual(stream_key("customs"), stream_key("weather"))

    def test_streams_are_independent(self):
        a = StreamBank(5)
        b = StreamBank(5)
        b.get("weather").normal(size=1000)  # consume an unrelated stream
        self.assertTrue(np.allclose(a.get("customs").normal(size=10), b.get("customs").normal(size=10)))


class TestGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_shipments(GeneratorConfig(n_shipments=6000, days=400, seed=3))

    def test_reproducible(self):
        other = generate_shipments(GeneratorConfig(n_shipments=6000, days=400, seed=3))
        self.assertTrue(np.allclose(self.df["actual_transit_h"], other["actual_transit_h"]))

    def test_sorted_by_ship_time(self):
        ts = self.df["ship_ts"].to_numpy()
        self.assertTrue(np.all(ts[1:] >= ts[:-1]))

    def test_transit_is_right_skewed_and_heavy_tailed(self):
        stats = describe_distribution(self.df)
        self.assertGreater(stats["skew"], 1.0)
        self.assertGreater(stats["excess_kurtosis"], 1.0)
        self.assertGreater(stats["mean"], stats["median"])
        self.assertGreater(stats["p99_over_p50"], 2.0)

    def test_latent_terms_are_non_negative_and_sum_to_the_target(self):
        latent = [
            "latent_core_h",
            "latent_dwell_h",
            "latent_congestion_h",
            "latent_weather_h",
            "latent_customs_h",
            "latent_disruption_h",
        ]
        for col in latent:
            self.assertGreaterEqual(float(self.df[col].min()), 0.0, col)
        total = self.df[latent].sum(axis=1).to_numpy()
        self.assertTrue(np.allclose(total, self.df["actual_transit_h"], atol=0.02))

    def test_customs_holds_only_on_cross_border_lanes(self):
        domestic = self.df.loc[~self.df["cross_border"], "latent_customs_h"]
        self.assertEqual(float(domestic.max()), 0.0)

    def test_late_rate_is_operationally_plausible(self):
        rate = float(self.df["is_late"].mean())
        self.assertGreater(rate, 0.05)
        self.assertLess(rate, 0.60)

    def test_regime_shift_degrades_reliability(self):
        base = generate_shipments(GeneratorConfig(n_shipments=6000, days=400, seed=3))
        shocked = generate_shipments(
            GeneratorConfig(
                n_shipments=6000,
                days=400,
                seed=3,
                regime_shift_day=200,
                regime_congestion_shock=0.35,
                regime_carrier_degradation=2.0,
            )
        )
        late_base = base.loc[base["day_index"] >= 200, "is_late"].mean()
        late_shock = shocked.loc[shocked["day_index"] >= 200, "is_late"].mean()
        self.assertGreater(late_shock, late_base)


if __name__ == "__main__":
    unittest.main()
