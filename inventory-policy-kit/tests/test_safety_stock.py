"""Safety stock: cycle service vs fill rate, lead-time variance, empirical tails."""

from __future__ import annotations

import unittest

import numpy as np
from scipy import stats

from invkit.distributions import GammaLTD, NormalLTD
from invkit.leadtime import (
    LeadTimeSpec,
    lead_time_variance_share,
    ltd_moments,
    ltd_stochastic_exact,
    undershoot_moments,
)
from invkit.safety_stock import (
    compare_service_definitions,
    fill_rate_of_sQ,
    ss_from_cycle_service_level,
    ss_from_empirical_quantile,
    ss_from_fill_rate,
)


class TestCycleServiceLevel(unittest.TestCase):
    def test_reorder_point_is_the_quantile(self):
        ltd = NormalLTD(500.0, 67.0)
        res = ss_from_cycle_service_level(ltd, 0.95, Q=800.0)
        self.assertAlmostEqual(res.reorder_point, 500.0 + stats.norm.ppf(0.95) * 67.0, places=8)
        self.assertAlmostEqual(res.safety_stock, stats.norm.ppf(0.95) * 67.0, places=8)

    def test_target_must_be_a_probability(self):
        with self.assertRaises(ValueError):
            ss_from_cycle_service_level(NormalLTD(1.0, 1.0), 1.0)


class TestFillRate(unittest.TestCase):
    def test_inversion_hits_the_target(self):
        for ltd in (NormalLTD(500.0, 67.0), GammaLTD.from_moments(500.0, 67.0)):
            for target in (0.90, 0.95, 0.99, 0.999):
                res = ss_from_fill_rate(ltd, target, Q=800.0)
                self.assertAlmostEqual(
                    fill_rate_of_sQ(ltd, res.reorder_point, 800.0), target, places=9
                )

    def test_first_order_form_matches_closed_form_inversion(self):
        """For a normal, the approximate inversion is G(z) = Q(1-beta)/sigma."""
        ltd = NormalLTD(500.0, 67.0)
        Q, beta = 800.0, 0.98
        res = ss_from_fill_rate(ltd, beta, Q, exact=False)
        z = (res.reorder_point - ltd.mu) / ltd.sigma
        from invkit.distributions import standard_normal_loss

        self.assertAlmostEqual(
            float(standard_normal_loss(z)), Q * (1 - beta) / ltd.sigma, places=9
        )

    def test_exact_form_needs_less_stock_than_approximation(self):
        """Dropping the loss at s+Q understates fill, so it over-buffers."""
        ltd = NormalLTD(500.0, 200.0)
        Q = 300.0
        exact = ss_from_fill_rate(ltd, 0.98, Q, exact=True)
        approx = ss_from_fill_rate(ltd, 0.98, Q, exact=False)
        self.assertLess(exact.reorder_point, approx.reorder_point)

    def test_fill_rate_increases_with_order_quantity(self):
        """At a fixed reorder point, a bigger lot serves a larger share of demand."""
        ltd = GammaLTD.from_moments(500.0, 67.0)
        fills = [fill_rate_of_sQ(ltd, 500.0, q) for q in (200.0, 400.0, 800.0, 1600.0)]
        self.assertEqual(fills, sorted(fills))

    def test_large_Q_permits_negative_safety_stock(self):
        """The headline: a 95% fill target with a big lot does not need a buffer."""
        ltd = GammaLTD.from_moments(500.0, 67.0)
        res = ss_from_fill_rate(ltd, 0.95, Q=800.0)
        self.assertLess(res.safety_stock, 0.0)
        self.assertAlmostEqual(fill_rate_of_sQ(ltd, res.reorder_point, 800.0), 0.95, places=9)


class TestServiceDefinitionGap(unittest.TestCase):
    def test_csl_basis_always_holds_more_than_fill_basis(self):
        ltd = GammaLTD.from_moments(500.0, 67.0)
        for target in (0.90, 0.95, 0.98, 0.99):
            for Q in (200.0, 400.0, 800.0, 1600.0):
                c = compare_service_definitions(ltd, target, Q)
                self.assertGreater(
                    c["ss_csl_basis"], c["ss_fill_basis"],
                    msg=f"target={target} Q={Q}",
                )

    def test_gap_widens_with_order_quantity(self):
        """Fill rate benefits from a large lot; cycle service is blind to it."""
        ltd = GammaLTD.from_moments(500.0, 67.0)
        deltas = [compare_service_definitions(ltd, 0.95, q)["ss_delta"] for q in (200.0, 800.0, 3200.0)]
        self.assertEqual(deltas, sorted(deltas))

    def test_csl_basis_overshoots_the_fill_target(self):
        ltd = GammaLTD.from_moments(500.0, 67.0)
        c = compare_service_definitions(ltd, 0.95, 800.0)
        self.assertGreater(c["fill_at_csl_basis"], 0.99)


class TestLeadTimeVariability(unittest.TestCase):
    def test_variance_convolution(self):
        spec = LeadTimeSpec(100.0, 30.0, {4: 0.5, 8: 0.5})
        mean, sd = ltd_moments(spec)
        self.assertAlmostEqual(mean, 600.0, places=9)
        # E[L]Var(d) + E[d]^2 Var(L) = 6*900 + 10000*4
        self.assertAlmostEqual(sd ** 2, 6 * 900 + 10000 * 4, places=6)

    def test_stochastic_lead_time_dominates_when_supplier_is_unreliable(self):
        spec = LeadTimeSpec(100.0, 30.0, {4: 0.5, 8: 0.5})
        self.assertGreater(lead_time_variance_share(spec), 0.85)
        det = LeadTimeSpec.deterministic(100.0, 30.0, 6)
        self.assertAlmostEqual(lead_time_variance_share(det), 0.0, places=12)

    def test_moment_matching_is_wrong_in_both_directions_on_a_bimodal_lead_time(self):
        """Matching two moments to a unimodal shape cannot represent two modes.

        With a 70/30 split between a 2-period and a 14-period lead time, the
        moment-matched gamma has exactly the right mean and variance and still
        gets the reorder point badly wrong: it *understates* through the whole
        useful service range (the second mode has not been reached yet) and
        *overstates* far out in the tail.  Both errors are expensive, and the
        first one is the dangerous one because it is invisible until the long
        lead time lands.
        """
        spec = LeadTimeSpec(100.0, 30.0, {2: 0.7, 14: 0.3})
        exact = ltd_stochastic_exact(spec, family="gamma")
        mean, sd = ltd_moments(spec)
        matched = GammaLTD.from_moments(mean, sd)
        self.assertAlmostEqual(exact.mean, matched.mean, delta=1.0)
        self.assertAlmostEqual(exact.sd, matched.sd, delta=1.0)
        self.assertGreater(exact.ppf(0.90), matched.ppf(0.90) * 1.10)
        self.assertLess(exact.ppf(0.99), matched.ppf(0.99) * 0.75)

    def test_exact_mixture_reduces_to_the_single_component(self):
        spec = LeadTimeSpec.deterministic(100.0, 30.0, 5)
        d = ltd_stochastic_exact(spec)
        self.assertAlmostEqual(d.mean, 500.0, places=6)


class TestUndershoot(unittest.TestCase):
    def test_matches_the_renewal_theory_formula(self):
        """E[U] = mu/2 + sigma^2/(2 mu) for the equilibrium excess distribution."""
        mu, sd = 100.0, 30.0
        eu, _ = undershoot_moments(mu, sd)
        self.assertAlmostEqual(eu, mu / 2.0 + sd ** 2 / (2.0 * mu), places=9)

    def test_matches_simulation_of_a_crossing_process(self):
        rng = np.random.default_rng(21)
        mu, sd = 100.0, 30.0
        theta = sd ** 2 / mu
        k = mu / theta
        draws = rng.gamma(k, theta, 4_000_000)
        position = 400_000.0
        threshold = 200_000.0
        # Walk down and record the overshoot past the threshold.
        cum = position - np.cumsum(draws)
        idx = int(np.argmax(cum <= threshold))
        undershoots = []
        level = threshold
        for i in range(idx, len(cum)):
            if cum[i] <= level:
                undershoots.append(level - cum[i])
                level -= 500.0
                if len(undershoots) > 40_000:
                    break
        eu, vu = undershoot_moments(mu, sd)
        self.assertAlmostEqual(float(np.mean(undershoots)), eu, delta=1.5)
        self.assertAlmostEqual(float(np.var(undershoots)), vu, delta=0.12 * vu)

    def test_does_not_vanish_as_transactions_shrink(self):
        """The sigma^2/(2 mu) term is invariant to slicing the period finer."""
        floors = []
        for m in (1, 10, 100, 1000):
            eu, _ = undershoot_moments(100.0 / m, 30.0 / np.sqrt(m))
            floors.append(eu)
        self.assertAlmostEqual(floors[-1], 30.0 ** 2 / (2 * 100.0), delta=0.06)
        self.assertTrue(all(f > 4.4 for f in floors))


class TestEmpiricalSafetyStock(unittest.TestCase):
    @staticmethod
    def _skewed_errors(n=60_000, seed=5):
        rng = np.random.default_rng(seed)
        base = rng.lognormal(0.0, 0.35, int(0.9 * n))
        spike = rng.lognormal(1.0, 0.55, n - int(0.9 * n))
        raw = np.concatenate([base, spike])
        rng.shuffle(raw)
        return (raw - raw.mean()) / raw.std(ddof=1) * 30.0

    def test_empirical_exceeds_normal_on_skewed_errors(self):
        errors = self._skewed_errors()
        emp, norm = ss_from_empirical_quantile(errors, 0.98, lead_time_periods=5)
        self.assertGreater(emp / norm - 1.0, 0.25)

    def test_gap_shrinks_as_the_lead_time_aggregates(self):
        """Central limit theorem: longer lead times pull the sum toward normal."""
        errors = self._skewed_errors()
        gaps = []
        for L in (1, 5, 20):
            emp, norm = ss_from_empirical_quantile(errors, 0.99, lead_time_periods=L)
            gaps.append(emp / norm - 1.0)
        self.assertGreater(gaps[0], gaps[1])
        self.assertGreater(gaps[1], gaps[2])

    def test_symmetric_errors_reproduce_the_normal_answer(self):
        rng = np.random.default_rng(9)
        errors = rng.normal(0.0, 30.0, 200_000)
        emp, norm = ss_from_empirical_quantile(errors, 0.95, lead_time_periods=5)
        self.assertAlmostEqual(emp / norm, 1.0, delta=0.02)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
