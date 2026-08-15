"""Random-stream isolation and output analysis.

These two subjects live in one file because they are the two halves of the same
claim: that a difference between two reported numbers is a real difference and
not an artefact of how the randomness or the averaging was arranged.
"""

import unittest

import numpy as np

from echelonsim.metrics import (
    batch_means_ci,
    bullwhip_ratio,
    lag1_autocorrelation,
    mean_ci,
    mser5_truncation,
    paired_difference_ci,
    variance_ratio_ci,
    welch_moving_average,
)
from echelonsim.rng import StreamBank, stream_key


class TestStreamBank(unittest.TestCase):
    def test_same_seed_and_replication_reproduce_a_stream(self):
        first = StreamBank(seed=7, replication=3).stream("demand").normal(size=50)
        second = StreamBank(seed=7, replication=3).stream("demand").normal(size=50)
        np.testing.assert_array_equal(first, second)

    def test_different_replications_are_independent(self):
        first = StreamBank(seed=7, replication=1).stream("demand").normal(size=200)
        second = StreamBank(seed=7, replication=2).stream("demand").normal(size=200)
        self.assertGreater(np.abs(first - second).mean(), 0.5)

    def test_named_streams_do_not_interfere(self):
        """The core CRN property: drawing from one stream cannot move another.

        This is what lets a scenario add a lead-time draw without shifting the
        demand path, which is the entire reason the comparison is paired.
        """
        bank = StreamBank(seed=11, replication=0)
        alone = bank.stream("demand").normal(size=30)

        other = StreamBank(seed=11, replication=0)
        other.stream("transit").normal(size=5000)  # burn a lot of draws elsewhere
        interleaved = other.stream("demand").normal(size=30)
        np.testing.assert_array_equal(alone, interleaved)

    def test_stream_key_is_stable_across_processes(self):
        self.assertEqual(stream_key("customer-demand"), stream_key("customer-demand"))
        self.assertNotEqual(stream_key("customer-demand"), stream_key("transit:retailer"))

    def test_child_bank_changes_only_the_replication(self):
        bank = StreamBank(seed=5, replication=0)
        child = bank.child(4)
        self.assertEqual(child.seed, 5)
        self.assertEqual(child.replication, 4)


class TestWarmupDetection(unittest.TestCase):
    def _transient_series(self, transient=120, steady=900, seed=3):
        rng = np.random.default_rng(seed)
        decay = 60.0 * np.exp(-np.arange(transient) / 25.0)
        head = 100.0 + decay + rng.normal(0, 4, transient)
        tail = 100.0 + rng.normal(0, 4, steady)
        return np.concatenate([head, tail])

    def test_mser5_finds_a_truncation_inside_the_transient_region(self):
        series = self._transient_series()
        truncation = mser5_truncation(series)
        self.assertGreater(truncation, 20)
        self.assertLess(truncation, 250)

    def test_truncation_removes_most_of_the_bias(self):
        series = self._transient_series()
        truncation = mser5_truncation(series)
        biased = abs(series.mean() - 100.0)
        corrected = abs(series[truncation:].mean() - 100.0)
        self.assertLess(corrected, biased)

    def test_stationary_series_needs_almost_no_truncation(self):
        rng = np.random.default_rng(17)
        series = 100.0 + rng.normal(0, 5, 1000)
        self.assertLess(mser5_truncation(series), 200)

    def test_short_series_returns_zero(self):
        self.assertEqual(mser5_truncation(np.arange(10.0)), 0)

    def test_welch_average_smooths_across_replications(self):
        rng = np.random.default_rng(1)
        reps = [100.0 + rng.normal(0, 10, 200) for _ in range(8)]
        smoothed = welch_moving_average(reps, window=10)
        self.assertEqual(smoothed.shape, (200,))
        self.assertLess(smoothed.std(), np.mean([r.std() for r in reps]))


class TestIntervals(unittest.TestCase):
    def test_mean_ci_covers_the_truth_about_95_percent_of_the_time(self):
        rng = np.random.default_rng(42)
        covered = 0
        trials = 500
        for _ in range(trials):
            sample = rng.normal(10.0, 2.0, 25)
            interval = mean_ci(sample, 0.95)
            covered += int(interval.low <= 10.0 <= interval.high)
        self.assertGreater(covered / trials, 0.90)
        self.assertLess(covered / trials, 0.99)

    def test_batch_means_widens_the_interval_on_correlated_data(self):
        """The whole point of batching: an autocorrelated series is not n
        independent observations, and pretending otherwise gives an interval
        that is far too narrow."""
        rng = np.random.default_rng(9)
        n = 4000
        noise = rng.normal(0, 1, n)
        series = np.empty(n)
        series[0] = noise[0]
        for t in range(1, n):
            series[t] = 0.9 * series[t - 1] + noise[t]
        naive = mean_ci(series).half_width
        batched = batch_means_ci(series, n_batches=20).half_width
        self.assertGreater(batched, 2.0 * naive)

    def test_batch_means_flags_batches_that_are_too_short(self):
        rng = np.random.default_rng(4)
        n = 600
        noise = rng.normal(0, 1, n)
        series = np.empty(n)
        series[0] = noise[0]
        for t in range(1, n):
            series[t] = 0.97 * series[t - 1] + noise[t]
        interval = batch_means_ci(series, n_batches=30)
        self.assertIn("correlation", interval.note)

    def test_batch_means_rejects_impossible_batch_counts(self):
        with self.assertRaises(ValueError):
            batch_means_ci(np.arange(10.0), n_batches=20)

    def test_paired_interval_is_tighter_than_unpaired_under_crn(self):
        rng = np.random.default_rng(21)
        common = rng.normal(0, 10, 40)  # the shared "demand path" effect
        control = 100.0 + common + rng.normal(0, 1, 40)
        treatment = control - 5.0 + rng.normal(0, 1, 40)
        paired = paired_difference_ci(treatment, control).half_width
        unpaired = np.hypot(mean_ci(treatment).half_width, mean_ci(control).half_width)
        self.assertLess(paired, 0.3 * unpaired)

    def test_variance_ratio_interval_stays_positive(self):
        rng = np.random.default_rng(5)
        numerator = rng.gamma(2.0, 30.0, 30)
        denominator = rng.gamma(2.0, 10.0, 30)
        interval = variance_ratio_ci(numerator, denominator)
        self.assertGreater(interval.low, 0.0)
        self.assertLess(interval.low, interval.mean)
        self.assertLess(interval.mean, interval.high)

    def test_lag1_autocorrelation_recovers_a_known_ar1(self):
        rng = np.random.default_rng(8)
        n = 20000
        noise = rng.normal(0, 1, n)
        series = np.empty(n)
        series[0] = noise[0]
        for t in range(1, n):
            series[t] = 0.6 * series[t - 1] + noise[t]
        self.assertAlmostEqual(lag1_autocorrelation(series), 0.6, delta=0.03)

    def test_bullwhip_ratio_is_one_when_orders_replicate_demand(self):
        rng = np.random.default_rng(2)
        demand = rng.normal(100, 20, 500)
        self.assertAlmostEqual(bullwhip_ratio(demand, demand), 1.0, places=12)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
