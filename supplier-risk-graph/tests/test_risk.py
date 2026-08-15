import unittest

from riskgraph.concentration import (
    effective_sources,
    geographic_hhi,
    hhi,
    hhi_by_tier,
    hidden_dependencies,
    supplier_hhi,
)
from riskgraph.generate import generate_network
from riskgraph.spof import articulation_points, degree_ranking, rank_spofs, sole_source_parts
from toy import toy_network


class TestHHI(unittest.TestCase):
    def test_single_source_is_one(self):
        self.assertAlmostEqual(hhi({"a": 100.0}), 1.0)

    def test_four_equal_suppliers(self):
        self.assertAlmostEqual(hhi({k: 25.0 for k in "abcd"}), 0.25)
        self.assertAlmostEqual(effective_sources({k: 25.0 for k in "abcd"}), 4.0)

class TestSpof(unittest.TestCase):
    def setUp(self):
        self.net = toy_network()

    def test_sole_source_parts_found(self):
        sole = sole_source_parts(self.net)
        self.assertEqual(sole["MSHARED"], "SITE-X")
        self.assertNotIn("C1", sole)

    def test_ranking_is_by_revenue_not_degree(self):
        ranked = rank_spofs(self.net)
        self.assertEqual(ranked[0].revenue_share, 1.0)
        self.assertGreaterEqual(ranked[0].revenue_at_risk, ranked[-1].revenue_at_risk)

    def test_site_x_is_an_articulation_point(self):
        self.assertIn("site:SITE-X", articulation_points(self.net))

    def test_degree_ranking_is_a_different_ordering(self):
        net = generate_network(seed=7)
        by_revenue = [r.node_id for r in rank_spofs(net, top_n=10)]
        by_degree = degree_ranking(net, top_n=10)
        self.assertNotEqual(by_revenue, by_degree)

    def test_single_site_suppliers_are_not_double_counted(self):
        ranked = rank_spofs(self.net, top_n=50)
        self.assertNotIn("supplier:SUP-X", [r.node_id for r in ranked])
        self.assertIn("site:SITE-X", [r.node_id for r in ranked])


class TestHiddenDependencies(unittest.TestCase):
    def test_toy_diamond_is_detected(self):
        found = hidden_dependencies(toy_network(), "FG")
        top = found[0]
        self.assertEqual(top.supplier_id, "SUP-X")
        self.assertEqual(top.tier1_suppliers, 2)
        self.assertAlmostEqual(top.revenue_share, 1.0)

    def test_planted_diamond_in_generated_network_is_top_ranked(self):
        net = generate_network(seed=7)
        fg = net.finished_goods()[0].part_id
        crit_site = net.sources("MAT-CRIT")[0].site_id
        crit_supplier = net.site_supplier(crit_site)
        found = hidden_dependencies(net, fg)
        self.assertEqual(found[0].supplier_id, crit_supplier)
        self.assertGreaterEqual(found[0].tier1_suppliers, 3)

    def test_tier1_hhi_looks_healthy_while_the_diamond_is_total(self):
        net = generate_network(seed=7)
        fg = net.finished_goods()[0].part_id
        tier1 = hhi_by_tier(net, fg)[1]
        self.assertLess(tier1, 0.25)  # >= 4 effective tier-1 suppliers
        self.assertAlmostEqual(hidden_dependencies(net, fg)[0].revenue_share, 1.0)

    def test_geographic_and_supplier_hhi_are_bounded(self):
        net = generate_network(seed=7)
        fg = net.finished_goods()[0].part_id
        geo, by_country = geographic_hhi(net, fg)
        self.assertTrue(0.0 < geo <= 1.0)
        self.assertAlmostEqual(sum(by_country.values()), 1.0, places=6)
        self.assertTrue(0.0 < supplier_hhi(net, fg) <= 1.0)


if __name__ == "__main__":
    unittest.main()
