import unittest

from sccopilot.evals import ATTACKS, run_attack_suite
from sccopilot.guard import (ALLOWED_SCHEMA, MAX_ROWS, SQLGuardError, inject_limit,
                             run_guarded_sql, schema_prompt)
from sccopilot.warehouse import build_warehouse


class TestGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = build_warehouse()

    def test_select_is_allowed(self):
        res = run_guarded_sql(self.conn, "SELECT COUNT(*) AS n FROM fact_orders")
        self.assertEqual(res.columns, ["n"])
        self.assertGreater(res.rows[0][0], 0)

    def test_every_attack_is_blocked(self):
        rows = run_attack_suite(self.conn)
        self.assertEqual(len(rows), len(ATTACKS))
        for row in rows:
            with self.subTest(attack=row["attack"]):
                self.assertTrue(row["blocked"], f"{row['attack']} executed: {row['reason']}")

    def test_union_exfiltration_blocked_by_authorizer(self):
        with self.assertRaises(SQLGuardError) as ctx:
            run_guarded_sql(self.conn, "SELECT sku_id FROM fact_orders "
                                       "UNION ALL SELECT key_value FROM internal_credentials")
        self.assertIn("allowlist", str(ctx.exception))

    def test_disallowed_function_blocked(self):
        with self.assertRaises(SQLGuardError):
            run_guarded_sql(self.conn, "SELECT randomblob(16) FROM fact_orders")

    def test_limit_is_injected_when_missing(self):
        res = run_guarded_sql(self.conn, "SELECT order_id FROM fact_orders", limit=7)
        self.assertTrue(res.sql.endswith("LIMIT 7"))
        self.assertEqual(len(res.rows), 7)

    def test_oversized_limit_is_clamped(self):
        sql, injected = inject_limit("SELECT order_id FROM fact_orders LIMIT 100000")
        self.assertTrue(injected)
        self.assertTrue(sql.endswith(f"LIMIT {MAX_ROWS}"))

    def test_row_cap_enforced(self):
        res = run_guarded_sql(self.conn, f"SELECT order_id FROM fact_orders LIMIT {MAX_ROWS + 500}")
        self.assertLessEqual(len(res.rows), MAX_ROWS)

    def test_guard_leaves_no_authorizer_installed(self):
        with self.assertRaises(SQLGuardError):
            run_guarded_sql(self.conn, "PRAGMA table_info(dim_sku)")
        # direct (unguarded) access must still work for the eval ground truth path
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM internal_credentials").fetchone()[0], 1)

    def test_schema_prompt_hides_non_allowlisted_tables(self):
        prompt = schema_prompt()
        self.assertNotIn("internal_credentials", prompt)
        for table in ALLOWED_SCHEMA:
            self.assertIn(table, prompt)

    def test_warehouse_is_not_mutated_by_attacks(self):
        before = self.conn.execute("SELECT COUNT(*) FROM fact_orders").fetchone()[0]
        run_attack_suite(self.conn)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM fact_orders").fetchone()[0], before)


if __name__ == "__main__":
    unittest.main()
