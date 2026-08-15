import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from riskgraph.generate import generate_network
from riskgraph.model import BomEdge, SupplyEdge, SupplyNetwork, load_network
from toy import toy_network


class TestModelValidation(unittest.TestCase):
    def test_shares_must_sum_to_one(self):
        data = toy_network().model_dump()
        data["supply"].append(SupplyEdge(part_id="C2", site_id="SITE-A1", share=0.3).model_dump())
        with self.assertRaises(ValidationError) as ctx:
            SupplyNetwork.model_validate(data)
        self.assertIn("allocation shares", str(ctx.exception))

    def test_unknown_site_reference_rejected(self):
        data = toy_network().model_dump()
        data["supply"][0]["site_id"] = "SITE-NOPE"
        with self.assertRaises(ValidationError):
            SupplyNetwork.model_validate(data)

    def test_bom_cycle_rejected(self):
        data = toy_network().model_dump()
        data["bom"].append(BomEdge(parent="MSHARED", child="SA1", qty_per=1.0).model_dump())
        with self.assertRaises(ValidationError) as ctx:
            SupplyNetwork.model_validate(data)
        self.assertIn("cycle", str(ctx.exception))

    def test_graph_has_one_node_per_entity(self):
        net = toy_network()
        g = net.to_graph()
        self.assertEqual(len(g.nodes), len(net.parts) + len(net.sites) + len(net.suppliers))
        self.assertEqual(len(g.edges), len(net.bom) + len(net.supply) + len(net.sites))

    def test_json_round_trip(self):
        net = generate_network(seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "net.json"
            net.to_json(path)
            reloaded = load_network(path)
        self.assertEqual(len(reloaded.parts), len(net.parts))
        self.assertEqual(len(reloaded.supply), len(net.supply))
        self.assertEqual(reloaded.model_dump(), net.model_dump())


class TestGenerator(unittest.TestCase):
    def test_generator_is_deterministic(self):
        a, b = generate_network(seed=5), generate_network(seed=5)
        self.assertEqual(a.model_dump(), b.model_dump())

    def test_generator_plants_a_sole_sourced_shared_material(self):
        net = generate_network(seed=7)
        srcs = net.sources("MAT-CRIT")
        self.assertEqual(len(srcs), 1)
        parents = [e.parent for e in net.bom if e.child == "MAT-CRIT"]
        self.assertGreaterEqual(len(parents), 3)


if __name__ == "__main__":
    unittest.main()
