import unittest

from sccopilot.tools import ToolError, default_registry
from sccopilot.warehouse import build_warehouse, row_counts


class TestWarehouse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = build_warehouse()
        cls.reg = default_registry(cls.conn)

    def test_star_schema_populated(self):
        counts = row_counts(self.conn)
        for table in ("fact_orders", "fact_shipments", "fact_inventory_snapshot",
                      "dim_sku", "dim_location", "dim_supplier", "dim_calendar"):
            self.assertGreater(counts[table], 0, table)
        self.assertGreater(counts["fact_orders"] + counts["fact_shipments"]
                           + counts["fact_inventory_snapshot"], 3000)

    def test_generation_is_seeded(self):
        other = build_warehouse()
        a = self.conn.execute("SELECT SUM(qty_ordered) FROM fact_orders").fetchone()[0]
        b = other.execute("SELECT SUM(qty_ordered) FROM fact_orders").fetchone()[0]
        self.assertEqual(a, b)

    def test_referential_integrity(self):
        orphans = self.conn.execute(
            "SELECT COUNT(*) FROM fact_shipments f LEFT JOIN dim_supplier d "
            "ON d.supplier_id=f.supplier_id WHERE d.supplier_id IS NULL").fetchone()[0]
        self.assertEqual(orphans, 0)

    def test_tool_specs_are_json_schema(self):
        specs = self.reg.specs()
        self.assertEqual(len(specs), 5)
        for spec in specs:
            self.assertIn("properties", spec["input_schema"])
            self.assertTrue(spec["description"])

    def test_inventory_position(self):
        out = self.reg.call("inventory_position", {"sku_id": "SKU-0007", "location_id": "DC-EAST"})
        self.assertEqual(out["sku_id"], "SKU-0007")
        self.assertGreaterEqual(out["on_hand"], 0)
        truth = self.conn.execute(
            "SELECT SUM(on_hand) FROM fact_inventory_snapshot WHERE sku_id='SKU-0007' "
            "AND location_id='DC-EAST' AND date_id=(SELECT MAX(date_id) FROM fact_inventory_snapshot)"
        ).fetchone()[0]
        self.assertEqual(out["on_hand"], truth)

    def test_unknown_sku_raises_tool_error(self):
        with self.assertRaises(ToolError):
            self.reg.call("inventory_position", {"sku_id": "SKU-9999"})

    def test_invalid_arguments_raise_tool_error(self):
        with self.assertRaises(ToolError):
            self.reg.call("stockout_risk", {"top_n": 500})
        with self.assertRaises(ToolError):
            self.reg.call("inventory_position", {})
        with self.assertRaises(ToolError):
            self.reg.call("delete_everything", {})

    def test_stockout_risk_sorted_and_bounded(self):
        out = self.reg.call("stockout_risk", {"location_id": "DC-NORTH", "top_n": 5})
        scores = [r["risk_score"] for r in out["at_risk"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))
        self.assertLessEqual(len(out["at_risk"]), 5)

    def test_supplier_lead_time_stats(self):
        out = self.reg.call("supplier_lead_time_stats", {})
        self.assertEqual(len(out["suppliers"]), 8)
        worst = min(out["suppliers"], key=lambda r: r["on_time_rate"])
        self.assertEqual(out["worst_on_time_supplier"], worst["supplier_id"])

    def test_what_if_demand_uplift_reduces_cover(self):
        base = self.reg.call("run_what_if", {"sku_id": "SKU-0004", "location_id": "DC-EAST"})
        up = self.reg.call("run_what_if", {"sku_id": "SKU-0004", "location_id": "DC-EAST",
                                           "demand_uplift_pct": 50.0})
        self.assertLess(up["scenario_days_of_cover"], base["scenario_days_of_cover"])
        self.assertGreaterEqual(up["projected_shortfall_units"], base["projected_shortfall_units"])

    def test_what_if_lead_time_slip_increases_shortfall(self):
        base = self.reg.call("run_what_if", {"sku_id": "SKU-0012"})
        slip = self.reg.call("run_what_if", {"sku_id": "SKU-0012", "lead_time_delta_days": 30.0})
        self.assertGreater(slip["scenario_lead_time_days"], base["scenario_lead_time_days"])
        self.assertGreaterEqual(slip["projected_shortfall_units"], base["projected_shortfall_units"])

    def test_sql_query_tool_goes_through_guard(self):
        with self.assertRaises(ToolError):
            self.reg.call("sql_query", {"sql": "DROP TABLE fact_orders"})
        ok = self.reg.call("sql_query", {"sql": "SELECT COUNT(*) AS n FROM dim_sku"})
        self.assertEqual(ok["rows"][0][0], 40)


if __name__ == "__main__":
    unittest.main()
