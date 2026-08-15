import unittest

from riskgraph.generate import generate_network
from riskgraph.mitigation import score_actions, with_dual_source, with_prequalification
from riskgraph.simulate import simulate
from toy import toy_network


def _no_disruption(net):
    data = net.model_dump()
    for s in data["sites"]:
        s["disruption_rate"] = 0.0
    return type(net).model_validate(data)


class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.net = toy_network()

    def test_zero_disruption_rate_gives_zero_loss(self):
        res = simulate(_no_disruption(self.net), trials=100, seed=1)
        self.assertEqual(res.expected_loss, 0.0)
        self.assertEqual(res.p_any_impact, 0.0)
        self.assertAlmostEqual(res.mean_service_level, 1.0)

    def test_same_seed_reproduces(self):
        a = simulate(self.net, trials=200, seed=4)
        b = simulate(self.net, trials=200, seed=4)
        self.assertEqual(a.as_dict(), b.as_dict())

    def test_tail_is_above_the_mean(self):
        res = simulate(self.net, trials=500, seed=4)
        self.assertGreater(res.p95_loss, res.expected_loss)
        self.assertGreaterEqual(res.max_loss, res.p95_loss)

    def test_ttr_is_bounded_by_the_horizon(self):
        res = simulate(self.net, trials=500, seed=4)
        self.assertGreater(res.mean_ttr_days, 0.0)
        self.assertLessEqual(res.p95_ttr_days, res.horizon_days)

    def test_buffer_reduces_expected_loss(self):
        base = simulate(self.net, trials=400, seed=6)
        buffered = simulate(self.net, trials=400, seed=6, buffers={"MSHARED": 60.0})
        self.assertLess(buffered.expected_loss, base.expected_loss)


class TestMitigation(unittest.TestCase):
    def test_dual_source_produces_a_valid_network_with_a_second_site(self):
        net = with_dual_source(toy_network(), "MSHARED")
        srcs = net.sources("MSHARED")
        self.assertEqual(len(srcs), 2)
        self.assertAlmostEqual(sum(e.share for e in srcs), 1.0, places=4)
        self.assertNotEqual(net.site(srcs[0].site_id).country, net.site(srcs[1].site_id).country)

    def test_prequalification_shortens_recovery_only_for_the_target(self):
        base = toy_network()
        net = with_prequalification(base, "MSHARED")
        self.assertLess(net.site("SITE-X").mean_recovery_days, base.site("SITE-X").mean_recovery_days)
        self.assertEqual(net.site("SITE-B1").mean_recovery_days, base.site("SITE-B1").mean_recovery_days)

    def test_mitigating_the_diamond_reduces_expected_loss(self):
        net = generate_network(seed=7)
        base, rows = score_actions(net, ["MAT-CRIT"], trials=400, seed=6)
        best = rows[0]
        self.assertGreater(best.risk_reduction, 0.0)
        self.assertLess(best.expected_loss_after, base.expected_loss)


if __name__ == "__main__":
    unittest.main()
