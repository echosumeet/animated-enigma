import unittest

from riskgraph.flow import expand, output_fraction, site_flow
from toy import toy_network


class TestExpansion(unittest.TestCase):
    def setUp(self):
        self.net = toy_network()
        self.rows = {r.part_id: r for r in expand(self.net, "FG")}

    def test_every_reachable_part_is_expanded(self):
        self.assertEqual(set(self.rows), {"FG", "SA1", "SA2", "C1", "C2", "M1", "MSHARED"})

    def test_quantity_per_finished_good_multiplies_down_the_bom(self):
        self.assertAlmostEqual(self.rows["C1"].qty_per_fg, 3.0)  # 1 x 3
        self.assertAlmostEqual(self.rows["M1"].qty_per_fg, 6.0)  # 1 x 3 x 2

    def test_multi_path_part_accumulates_quantity_from_both_branches(self):
        # MSHARED = C1 path (1x3x1) + C2 path (2x1x4) = 11 per finished good
        self.assertAlmostEqual(self.rows["MSHARED"].qty_per_fg, 11.0)

    def test_sole_source_flag(self):
        self.assertTrue(self.rows["MSHARED"].sole_source)
        self.assertFalse(self.rows["C1"].sole_source)

class TestFlowAndKernel(unittest.TestCase):
    def setUp(self):
        self.net = toy_network()

    def test_site_flow_conserves_spend(self):
        total_flow = sum(site_flow(self.net, "FG").values())
        sourced_spend = sum(r.annual_spend for r in expand(self.net, "FG") if self.net.sources(r.part_id))
        self.assertAlmostEqual(total_flow, sourced_spend, places=6)

    def test_no_disruption_means_full_output(self):
        self.assertAlmostEqual(output_fraction(self.net, "FG"), 1.0)

    def test_sole_source_outage_stops_the_finished_good(self):
        self.assertAlmostEqual(output_fraction(self.net, "FG", ["SITE-X"]), 0.0)

    def test_availability_propagates_as_a_minimum_not_a_product(self):
        net = self.net
        # dropping a 0.6 share of C1 alone leaves 0.4; the FG cannot exceed that
        frac = output_fraction(net, "SA1", ["SITE-B1"])  # C1 keeps 0.4, its children are elsewhere
        self.assertAlmostEqual(frac, 0.4)

    def test_flex_lets_survivors_ramp_but_caps_at_one(self):
        self.assertAlmostEqual(output_fraction(self.net, "SA1", ["SITE-B1"], flex=1.5), 0.6)
        self.assertAlmostEqual(output_fraction(self.net, "SA1", ["SITE-B1"], flex=10.0), 1.0)

if __name__ == "__main__":
    unittest.main()
