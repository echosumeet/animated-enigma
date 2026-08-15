import tempfile
import unittest
from pathlib import Path

from scmplatform.registry import ModelCard, ModelRegistry, RegistryError


def card(version: str, parent: str | None = None) -> ModelCard:
    return ModelCard(
        name="demand_daily",
        version=version,
        algorithm="HistGradientBoostingRegressor",
        data_contract="demand_panel@1.0.0",
        parent_version=parent,
        metrics={"wmape": 0.2},
    )


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.reg = ModelRegistry(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_register_and_read_back(self):
        self.reg.register(card("1.0.0"))
        got = self.reg.get("demand_daily", "1.0.0")
        self.assertEqual(got.stage, "dev")
        self.assertEqual(got.data_contract, "demand_panel@1.0.0")

    def test_duplicate_registration_is_refused(self):
        self.reg.register(card("1.0.0"))
        with self.assertRaises(RegistryError):
            self.reg.register(card("1.0.0"))
        self.reg.register(card("1.0.0"), overwrite=True)

    def test_promotion_archives_the_incumbent(self):
        self.reg.register(card("1.0.0"))
        self.reg.register(card("1.1.0", parent="1.0.0"))
        self.reg.transition("demand_daily", "1.0.0", "production")
        self.reg.transition("demand_daily", "1.1.0", "production")
        self.assertEqual(self.reg.production("demand_daily").version, "1.1.0")
        self.assertEqual(self.reg.get("demand_daily", "1.0.0").stage, "archived")

    def test_rollback_restores_the_previous_production_version(self):
        for v in ("1.0.0", "1.1.0"):
            self.reg.register(card(v))
            self.reg.transition("demand_daily", v, "production")
        restored = self.reg.rollback("demand_daily", note="drift incident")
        self.assertEqual(restored.version, "1.0.0")
        self.assertEqual(self.reg.production("demand_daily").version, "1.0.0")
        self.assertEqual(self.reg.get("demand_daily", "1.1.0").stage, "archived")

    def test_rollback_without_history_raises(self):
        self.reg.register(card("1.0.0"))
        self.reg.transition("demand_daily", "1.0.0", "production")
        with self.assertRaises(RegistryError):
            self.reg.rollback("demand_daily")

    def test_lineage_walks_parent_pointers_to_the_root(self):
        self.reg.register(card("1.0.0"))
        self.reg.register(card("1.1.0", parent="1.0.0"))
        self.reg.register(card("2.0.0", parent="1.1.0"))
        chain = [c.version for c in self.reg.lineage("demand_daily", "2.0.0")]
        self.assertEqual(chain, ["2.0.0", "1.1.0", "1.0.0"])


if __name__ == "__main__":
    unittest.main()
