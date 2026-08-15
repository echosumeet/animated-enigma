"""Distribution layer: loss functions, quantiles, convolution, mixtures."""

from __future__ import annotations

import math
import unittest

import numpy as np
from scipy import stats

from invkit.distributions import (
    EmpiricalLTD,
    GammaLTD,
    MixtureLTD,
    NormalLTD,
    inverse_standard_normal_loss,
    standard_normal_loss,
    standard_normal_loss2,
)


class TestStandardNormalLoss(unittest.TestCase):
    def test_matches_numeric_integration(self):
        """G(z) = E[(Z - z)^+] against direct numerical integration."""
        for z in (-2.0, -0.5, 0.0, 0.7, 1.645, 3.0):
            grid = np.linspace(z, z + 30.0, 400_001)
            integral = np.trapezoid((grid - z) * stats.norm.pdf(grid), grid)
            self.assertAlmostEqual(float(standard_normal_loss(z)), float(integral), places=6)

    def test_known_value_at_zero(self):
        """G(0) = phi(0) = 1/sqrt(2*pi)."""
        self.assertAlmostEqual(
            float(standard_normal_loss(0.0)), 1.0 / math.sqrt(2 * math.pi), places=12
        )

    def test_symmetry_identity(self):
        """G(-z) - G(z) = z, which falls out of the definition."""
        for z in (0.3, 1.0, 2.5):
            lhs = float(standard_normal_loss(-z) - standard_normal_loss(z))
            self.assertAlmostEqual(lhs, z, places=10)

    def test_strictly_decreasing(self):
        z = np.linspace(-5, 5, 501)
        g = standard_normal_loss(z)
        self.assertTrue(np.all(np.diff(g) < 0))

    def test_second_order_loss_is_integral_of_first(self):
        """G2(z) = integral_z^inf G(u) du."""
        for z in (-1.0, 0.0, 1.5):
            grid = np.linspace(z, z + 25.0, 200_001)
            integral = np.trapezoid(standard_normal_loss(grid), grid)
            self.assertAlmostEqual(float(standard_normal_loss2(z)), float(integral), places=6)

    def test_inversion_round_trips(self):
        for z in (-3.0, -0.4, 0.0, 1.2, 3.5):
            target = float(standard_normal_loss(z))
            self.assertAlmostEqual(inverse_standard_normal_loss(target), z, places=8)

    def test_inversion_rejects_non_positive_target(self):
        with self.assertRaises(ValueError):
            inverse_standard_normal_loss(0.0)


class TestNormalLTD(unittest.TestCase):
    def test_loss_matches_standard_form(self):
        d = NormalLTD(500.0, 67.0)
        x = 615.0
        z = (x - d.mu) / d.sigma
        self.assertAlmostEqual(d.loss(x), d.sigma * float(standard_normal_loss(z)), places=10)

    def test_loss_identity_against_expectation(self):
        """E[(D - x)^+] - E[(x - D)^+] = E[D] - x for any x."""
        d = NormalLTD(120.0, 40.0)
        rng = np.random.default_rng(0)
        sample = rng.normal(120.0, 40.0, 2_000_000)
        for x in (60.0, 120.0, 200.0):
            emp = float(np.maximum(sample - x, 0.0).mean())
            self.assertAlmostEqual(d.loss(x), emp, delta=0.15)


class TestGammaLTD(unittest.TestCase):
    def test_from_moments_round_trips(self):
        d = GammaLTD.from_moments(500.0, 67.0)
        self.assertAlmostEqual(d.mean, 500.0, places=8)
        self.assertAlmostEqual(d.sd, 67.0, places=8)

    def test_convolution_is_exact(self):
        """Sum of L iid gammas at fixed scale is Gamma(L*k, theta), exactly."""
        per_period = GammaLTD.from_moments(100.0, 30.0)
        over_five = per_period.convolve(5)
        self.assertAlmostEqual(over_five.mean, 500.0, places=9)
        self.assertAlmostEqual(over_five.sd, 30.0 * math.sqrt(5.0), places=9)
        # And it agrees with the empirical sum of five draws.
        rng = np.random.default_rng(3)
        draws = per_period.rvs(400_000, rng).reshape(80_000, 5).sum(axis=1)
        self.assertAlmostEqual(float(draws.mean()), over_five.mean, delta=1.5)
        self.assertAlmostEqual(float(draws.std(ddof=1)), over_five.sd, delta=1.5)

    def test_loss_matches_monte_carlo(self):
        d = GammaLTD.from_moments(500.0, 67.0)
        rng = np.random.default_rng(11)
        sample = d.rvs(2_000_000, rng)
        for x in (450.0, 500.0, 620.0, 700.0):
            emp = float(np.maximum(sample - x, 0.0).mean())
            self.assertAlmostEqual(d.loss(x), emp, delta=0.3)

    def test_loss_below_zero_is_mean_minus_x(self):
        d = GammaLTD.from_moments(80.0, 25.0)
        self.assertAlmostEqual(d.loss(-10.0), d.mean + 10.0, places=9)

    def test_ppf_inverts_cdf(self):
        d = GammaLTD.from_moments(500.0, 67.0)
        for p in (0.05, 0.5, 0.95, 0.999):
            self.assertAlmostEqual(d.cdf(d.ppf(p)), p, places=9)

    def test_gamma_upper_tail_exceeds_normal(self):
        """The practical reason to prefer gamma: a fatter right tail."""
        mean, sd = 500.0, 200.0
        g = GammaLTD.from_moments(mean, sd)
        n = NormalLTD(mean, sd)
        self.assertGreater(g.ppf(0.99), n.ppf(0.99))
        self.assertGreaterEqual(g.ppf(1e-9), 0.0)


class TestEmpiricalLTD(unittest.TestCase):
    def test_loss_is_exact_sample_mean(self):
        sample = np.array([1.0, 4.0, 4.0, 10.0])
        d = EmpiricalLTD(sample)
        self.assertAlmostEqual(d.loss(4.0), (0 + 0 + 0 + 6) / 4.0, places=12)
        self.assertAlmostEqual(d.loss(0.0), sample.mean(), places=12)

    def test_quantiles_track_the_sample(self):
        rng = np.random.default_rng(7)
        sample = rng.gamma(2.0, 50.0, 200_000)
        d = EmpiricalLTD(sample)
        self.assertAlmostEqual(d.ppf(0.9), float(np.quantile(sample, 0.9)), places=8)

    def test_rejects_tiny_sample(self):
        with self.assertRaises(ValueError):
            EmpiricalLTD([1.0])


class TestMixtureLTD(unittest.TestCase):
    def test_moments_match_law_of_total_variance(self):
        a = NormalLTD(400.0, 40.0)
        b = NormalLTD(900.0, 60.0)
        m = MixtureLTD([a, b], [0.7, 0.3])
        expected_mean = 0.7 * 400 + 0.3 * 900
        second = 0.7 * (40 ** 2 + 400 ** 2) + 0.3 * (60 ** 2 + 900 ** 2)
        self.assertAlmostEqual(m.mean, expected_mean, places=9)
        self.assertAlmostEqual(m.var, second - expected_mean ** 2, places=6)

    def test_ppf_inverts_mixture_cdf(self):
        m = MixtureLTD(
            [GammaLTD.from_moments(400.0, 40.0), GammaLTD.from_moments(900.0, 60.0)],
            [0.6, 0.4],
        )
        for p in (0.1, 0.5, 0.9, 0.99):
            self.assertAlmostEqual(m.cdf(m.ppf(p)), p, places=7)

    def test_weights_must_be_a_distribution(self):
        a = NormalLTD(1.0, 1.0)
        with self.assertRaises(ValueError):
            MixtureLTD([a, a], [0.5, 0.6])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
