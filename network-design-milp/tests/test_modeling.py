"""The modelling layer is the foundation; if it is wrong every model is wrong.

These tests check it against things that can be verified independently: hand
computed LP optima, brute-force key selection, and the assembled matrix itself.
"""

from __future__ import annotations

import unittest

import numpy as np

from netdesign.modeling import ANY, LinExpr, Model, quicksum


class TestLinExpr(unittest.TestCase):
    def test_arithmetic_folds_constants_and_coefficients(self):
        m = Model()
        x = m.add_var("x")
        y = m.add_var("y")
        e = 3 * x + 2 * y - x + 5 - 2
        self.assertEqual(e.constant, 3.0)
        self.assertEqual(sorted(e.coeffs.values()), [2.0, 2.0])

    def test_cancelling_terms_are_dropped_from_the_sparsity_pattern(self):
        m = Model()
        x = m.add_var("x")
        e = x - x
        self.assertEqual(len(e.coeffs), 0, "a zero coefficient must not occupy a matrix entry")

    def test_variable_products_are_rejected(self):
        m = Model()
        x = m.add_var("x")
        with self.assertRaises(TypeError):
            _ = x * x

    def test_comparison_moves_constants_to_the_bound(self):
        m = Model()
        x = m.add_var("x")
        row = (2 * x + 5) <= 11
        self.assertEqual(row.ub, 6.0)
        self.assertEqual(row.lb, -float("inf"))
        self.assertEqual(list(row.coeffs.values()), [2.0])

    def test_quicksum_matches_python_sum(self):
        m = Model()
        xs = [m.add_var(f"x{i}") for i in range(20)]
        a = quicksum(3 * x for x in xs)
        b = sum((3 * x for x in xs), LinExpr())
        self.assertEqual(a.coeffs, b.coeffs)


class TestVarGroup(unittest.TestCase):
    def setUp(self):
        self.m = Model()
        self.keys = [(o, d, k) for o in "AB" for d in "XYZ" for k in ("s1", "s2")]
        self.g = self.m.add_vars(self.keys, name="f")

    def test_wildcard_selection_matches_brute_force(self):
        for pattern in [("A", ANY, ANY), (ANY, "Y", ANY), ("B", "Z", "s1"), (ANY, ANY, "s2")]:
            expected = sorted(
                k
                for k in self.keys
                if all(p is ANY or p == v for p, v in zip(pattern, k))
            )
            self.assertEqual(sorted(self.g.select(*pattern)), expected, pattern)

    def test_selection_preserves_insertion_order(self):
        self.assertEqual(self.g.select(ANY, ANY, ANY), self.keys)

    def test_sum_over_empty_selection_is_zero(self):
        e = self.g.sum("C", ANY, ANY)
        self.assertEqual(len(e.coeffs), 0)
        self.assertEqual(e.constant, 0.0)

    def test_wrong_arity_pattern_is_an_error_not_a_silent_empty_sum(self):
        with self.assertRaises(ValueError):
            self.g.select("A", ANY)

    def test_duplicate_group_name_is_rejected(self):
        with self.assertRaises(KeyError):
            self.m.add_vars([1, 2], name="f")


class TestSolve(unittest.TestCase):
    def test_tiny_lp_matches_the_hand_computed_optimum(self):
        # max 3x + 2y  s.t.  x + y <= 4, x + 3y <= 6, x,y >= 0  ->  x=4, y=0, obj 12
        m = Model(sense="max")
        x = m.add_var("x")
        y = m.add_var("y")
        m.set_objective(3 * x + 2 * y)
        m.add(x + y <= 4)
        m.add(x + 3 * y <= 6)
        sol = m.solve()
        self.assertTrue(sol.is_optimal)
        self.assertAlmostEqual(sol.objective, 12.0, places=6)
        self.assertAlmostEqual(sol.value(x), 4.0, places=6)

    def test_integrality_changes_the_answer(self):
        # max x  s.t. 2x <= 3  ->  1.5 relaxed, 1 integral
        m = Model(sense="max")
        x = m.add_var("x", ub=10, vtype="integer")
        m.set_objective(x)
        m.add(2 * x <= 3)
        self.assertAlmostEqual(m.solve().objective, 1.0, places=6)
        self.assertAlmostEqual(m.solve(relax=True).objective, 1.5, places=6)

    def test_assembled_matrix_matches_the_declared_rows(self):
        m = Model()
        g = m.add_vars(["a", "b", "c"], name="v")
        m.add(2 * g["a"] + 3 * g["c"] >= 6)
        m.add(g["b"] <= 4)
        parts = m.assemble()
        dense = parts["A"].toarray()
        np.testing.assert_allclose(dense, [[2.0, 0.0, 3.0], [0.0, 1.0, 0.0]])
        np.testing.assert_allclose(parts["constraint_lb"], [6.0, -np.inf])
        np.testing.assert_allclose(parts["constraint_ub"], [np.inf, 4.0])

    def test_infeasible_model_reports_infeasible(self):
        m = Model()
        x = m.add_var("x", lb=0.0, ub=1.0)
        m.add(x >= 2.0)
        self.assertEqual(m.solve().status, "infeasible")

    def test_constant_row_that_cannot_hold_is_rejected_at_build_time(self):
        m = Model()
        x = m.add_var("x")
        with self.assertRaises(ValueError):
            m.add((x - x) >= 5.0, name="nonsense")

    def test_big_m_helper_rejects_a_non_finite_m(self):
        m = Model()
        x = m.add_var("x")
        y = m.add_var("y", vtype="binary")
        with self.assertRaises(ValueError):
            m.link_big_m(x, y, float("inf"))


class TestElasticRelaxation(unittest.TestCase):
    def test_elastic_copy_finds_the_minimum_violation(self):
        # x <= 1 and x >= 4 : minimum total violation is 3
        m = Model()
        x = m.add_var("x", lb=0.0, ub=10.0)
        m.add(x <= 1.0, name="upper", tag="cap")
        m.add(x >= 4.0, name="lower", tag="need")
        self.assertEqual(m.solve().status, "infeasible")

        elastic, rows, slacks = m.elastic_copy(["cap", "need"])
        sol = elastic.solve()
        self.assertTrue(sol.is_optimal)
        self.assertAlmostEqual(sol.objective, 3.0, places=6)
        self.assertGreater(len(rows), 0)
        self.assertTrue(any(v > 1e-9 for v in sol.values(slacks).values()))

    def test_elastic_copy_leaves_the_original_model_untouched(self):
        m = Model()
        x = m.add_var("x", lb=0.0, ub=1.0)
        m.add(x >= 4.0, tag="need")
        before = m.stats()
        m.elastic_copy(["need"])
        self.assertEqual(m.stats(), before)
        self.assertEqual(m.solve().status, "infeasible")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
