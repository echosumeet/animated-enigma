"""Summing matrix construction and reconciliation properties."""

from __future__ import annotations

import itertools
import unittest

import numpy as np

from dflab.hierarchy import (
    build_hierarchy,
    coherency_error,
    reconcile,
    shrink_covariance,
)

METHODS = ("bottom_up", "top_down", "ols", "mint")


def small_hierarchy(n_p=3, n_r=2, n_c=2):
    keys = [
        (f"P{p}", f"R{r}", c)
        for p, r, c in itertools.product(
            range(1, n_p + 1), range(1, n_r + 1), ["RETAIL", "ECOM"][:n_c]
        )
    ]
    return build_hierarchy(keys)


class TestSummingMatrix(unittest.TestCase):
    def setUp(self):
        self.h = small_hierarchy()

    def test_shape_and_level_counts(self):
        # 1 total + 3 products + 2 regions + 2 channels
        # + 6 product_x_region + 6 product_x_channel + 4 region_x_channel
        # + 12 bottom
        self.assertEqual(self.h.n_bottom, 12)
        self.assertEqual(self.h.n_nodes, 1 + 3 + 2 + 2 + 6 + 6 + 4 + 12)
        self.assertEqual(len(self.h.level_index("total")), 1)
        self.assertEqual(len(self.h.level_index("product")), 3)
        self.assertEqual(len(self.h.level_index("bottom")), 12)

    def test_summing_matrix_is_binary_with_an_identity_tail(self):
        S = self.h.S
        self.assertTrue(np.all((S == 0) | (S == 1)))
        np.testing.assert_allclose(S[-self.h.n_bottom :], np.eye(self.h.n_bottom))
        np.testing.assert_allclose(S[0], np.ones(self.h.n_bottom))

    def test_aggregation_reproduces_group_sums(self):
        rng = np.random.default_rng(0)
        b = rng.gamma(2.0, 10.0, size=(self.h.n_bottom, 5))
        agg = self.h.aggregate(b)
        np.testing.assert_allclose(agg[0], b.sum(axis=0))
        for lv in ("product", "region", "channel"):
            idx = self.h.level_index(lv)
            np.testing.assert_allclose(agg[idx].sum(axis=0), b.sum(axis=0), rtol=1e-10)

    def test_mismatched_key_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            build_hierarchy([("A", "B", "C"), ("A", "B")])
        with self.assertRaises(ValueError):
            build_hierarchy([])


class TestShrinkage(unittest.TestCase):
    def test_intensity_is_in_the_unit_interval_and_matrix_is_psd(self):
        rng = np.random.default_rng(1)
        R = rng.normal(size=(40, 25))
        W, lam = shrink_covariance(R)
        self.assertGreaterEqual(lam, 0.0)
        self.assertLessEqual(lam, 1.0)
        self.assertTrue(np.allclose(W, W.T))
        self.assertGreater(float(np.min(np.linalg.eigvalsh(W))), 0.0)

    def test_shrinkage_is_stronger_when_there_is_less_data(self):
        """With a genuine common factor the intensity is interior, and it
        rises as the sample shrinks -- exactly the adaptive behaviour that
        makes MinT usable on a wide hierarchy with short history."""
        rng = np.random.default_rng(2)
        n = 20
        factor = rng.normal(size=(500, 1))
        base = factor @ rng.normal(size=(1, n)) + rng.normal(size=(500, n))
        _, lam_long = shrink_covariance(base)
        _, lam_short = shrink_covariance(base[:25])
        self.assertLess(lam_long, 1.0)
        self.assertGreater(lam_short, lam_long)

    def test_full_shrinkage_when_nodes_are_genuinely_independent(self):
        rng = np.random.default_rng(4)
        _, lam = shrink_covariance(rng.normal(size=(60, 20)))
        self.assertGreater(lam, 0.8)

    def test_shrunk_covariance_is_invertible_when_the_sample_one_is_not(self):
        rng = np.random.default_rng(3)
        R = rng.normal(size=(8, 30))  # rank deficient by construction
        sample = np.cov(R, rowvar=False)
        self.assertLess(np.linalg.matrix_rank(sample), 30)
        W, _ = shrink_covariance(R)
        self.assertEqual(np.linalg.matrix_rank(W), 30)

    def test_too_few_observations_is_an_error(self):
        with self.assertRaises(ValueError):
            shrink_covariance(np.zeros((2, 5)))


class TestReconciliation(unittest.TestCase):
    def setUp(self):
        self.h = small_hierarchy()
        rng = np.random.default_rng(11)
        self.rng = rng
        n = self.h.n_nodes
        self.residuals = rng.normal(0.0, 3.0, size=(60, n))
        self.props = rng.gamma(2.0, 5.0, size=self.h.n_bottom)
        # deliberately incoherent base forecasts
        bottom = rng.gamma(3.0, 20.0, size=(self.h.n_bottom, 4))
        self.base = self.h.aggregate(bottom) + rng.normal(0, 6.0, size=(n, 4))

    def test_base_forecasts_are_incoherent_and_every_method_fixes_it(self):
        self.assertGreater(coherency_error(self.base, self.h), 1.0)
        for mth in METHODS:
            with self.subTest(method=mth):
                rec = reconcile(
                    self.base,
                    self.h,
                    method=mth,
                    residuals=self.residuals,
                    proportions=self.props,
                )
                self.assertLess(coherency_error(rec, self.h), 1e-8)

    def test_reconciliation_preserves_already_coherent_forecasts(self):
        """S G S = S, so a coherent input must come back unchanged."""
        bottom = self.rng.gamma(3.0, 20.0, size=(self.h.n_bottom, 3))
        coherent = self.h.aggregate(bottom)
        for mth in ("bottom_up", "ols", "mint"):
            with self.subTest(method=mth):
                rec = reconcile(
                    coherent,
                    self.h,
                    method=mth,
                    residuals=self.residuals,
                    nonnegative=False,
                )
                np.testing.assert_allclose(rec, coherent, atol=1e-6)

    def test_bottom_up_keeps_the_bottom_level_untouched(self):
        rec = reconcile(self.base, self.h, method="bottom_up", nonnegative=False)
        nb = self.h.n_bottom
        np.testing.assert_allclose(rec[-nb:], self.base[-nb:], atol=1e-9)

    def test_top_down_splits_the_total_by_the_given_proportions(self):
        rec = reconcile(
            self.base, self.h, method="top_down", proportions=self.props,
            nonnegative=False,
        )
        nb = self.h.n_bottom
        share = self.props / self.props.sum()
        expected = np.outer(share, self.base[0])
        np.testing.assert_allclose(rec[-nb:], expected, rtol=1e-9)
        np.testing.assert_allclose(rec[0], self.base[0], rtol=1e-9)

    def test_mint_beats_base_and_bottom_up_when_errors_are_correlated(self):
        """The case MinT exists for: node errors are not independent."""
        rng = np.random.default_rng(5)
        h = small_hierarchy(3, 2, 2)
        S, nb, n = h.S, h.n_bottom, h.n_nodes
        T = 300
        truth_bottom = rng.gamma(6.0, 15.0, size=(nb, T))
        truth = S @ truth_bottom
        # shared shock across all nodes plus node-specific noise
        shared = rng.normal(0.0, 12.0, size=(1, T))
        noise = shared + rng.normal(0.0, 6.0, size=(n, T))
        base = truth + noise
        resid = noise[:, :200].T
        errs = {}
        for mth in ("base", "bottom_up", "ols", "mint"):
            if mth == "base":
                rec = base[:, 200:]
            else:
                rec = reconcile(
                    base[:, 200:], h, method=mth, residuals=resid, nonnegative=False
                )
            errs[mth] = float(np.mean((truth[:, 200:] - rec) ** 2))
        self.assertLess(errs["mint"], errs["base"])
        self.assertLess(errs["mint"], errs["bottom_up"])

    def test_nonnegative_option_clips_the_bottom_level(self):
        base = self.base.copy()
        base[-1] = -500.0
        rec = reconcile(base, self.h, method="ols", nonnegative=True)
        self.assertGreaterEqual(float(np.min(rec[-self.h.n_bottom :])), 0.0)
        self.assertLess(coherency_error(rec, self.h), 1e-8)

    def test_missing_inputs_and_bad_shapes_are_errors(self):
        with self.assertRaises(ValueError):
            reconcile(self.base, self.h, method="mint")
        with self.assertRaises(ValueError):
            reconcile(self.base, self.h, method="top_down")
        with self.assertRaises(ValueError):
            reconcile(self.base, self.h, method="nope")
        with self.assertRaises(ValueError):
            reconcile(np.zeros((3, 2)), self.h, method="ols")

    def test_one_dimensional_input_returns_one_dimensional_output(self):
        rec = reconcile(self.base[:, 0], self.h, method="ols")
        self.assertEqual(rec.ndim, 1)
        self.assertEqual(rec.size, self.h.n_nodes)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
