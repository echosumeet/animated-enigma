import unittest

import numpy as np
import pandas as pd

from scmplatform.monitoring import (
    drift_over_time,
    drift_report,
    prediction_drift,
    psi,
    psi_band,
    slice_performance,
)

RNG = np.random.default_rng(42)


class TestDrift(unittest.TestCase):
    def test_psi_is_near_zero_for_two_draws_from_one_distribution(self):
        a, b = RNG.normal(size=5000), RNG.normal(size=5000)
        self.assertLess(psi(a, b), 0.02)

    def test_psi_increases_monotonically_with_the_shift(self):
        ref = RNG.normal(size=5000)
        values = [psi(ref, RNG.normal(loc=s, size=5000)) for s in (0.25, 0.75, 1.5)]
        self.assertEqual(values, sorted(values))
        self.assertEqual(psi_band(values[-1]), "major")
        self.assertEqual(psi_band(0.02), "stable")

    def test_drift_report_ranks_the_shifted_feature_first(self):
        n = 3000
        ref = pd.DataFrame({"stable": RNG.normal(size=n), "shifted": RNG.normal(size=n)})
        cur = pd.DataFrame({"stable": RNG.normal(size=n), "shifted": RNG.normal(loc=1.2, size=n)})
        table = drift_report(ref, cur, ["stable", "shifted"])
        self.assertEqual(table.iloc[0]["feature"], "shifted")
        self.assertEqual(table.iloc[0]["band"], "major")
        self.assertLess(table.iloc[0]["ks_pvalue"], 0.01)

    def test_prediction_drift_reports_the_direction_of_the_shift(self):
        out = prediction_drift(RNG.normal(100, 10, 4000), RNG.normal(115, 10, 4000))
        self.assertGreater(out["mean_shift"], 10.0)
        self.assertGreater(out["relative_mean_shift"], 0.08)
        self.assertGreater(out["psi"], 0.25)

    def test_drift_over_time_produces_one_row_per_window_and_feature(self):
        dates = pd.date_range("2025-01-01", periods=180, freq="D").repeat(10)
        frame = pd.DataFrame({"date": dates, "x": RNG.normal(size=len(dates))})
        split = pd.Timestamp("2025-03-01")
        out = drift_over_time(frame, ["x"], split, window_days=30)
        self.assertGreaterEqual(len(out), 3)
        self.assertEqual(set(out["feature"]), {"x"})
        self.assertTrue((out["psi"] < 0.1).all())


class TestSlicePerformance(unittest.TestCase):
    def setUp(self):
        n = 600
        y = RNG.gamma(5.0, 4.0, n)
        region = np.where(np.arange(n) % 3 == 0, "north", "south")
        noise = np.where(region == "north", 0.45, 0.05)
        self.pred = pd.DataFrame(
            {"y": y, "yhat": y * (1 + RNG.normal(0, 1, n) * noise), "region": region}
        )

    def test_the_bad_slice_is_flagged(self):
        report = slice_performance(self.pred, ["region"])
        self.assertEqual(list(report.degraded["region"]), ["north"])
        self.assertGreater(report.degraded.iloc[0]["ratio_to_overall"], 1.25)


if __name__ == "__main__":
    unittest.main()
