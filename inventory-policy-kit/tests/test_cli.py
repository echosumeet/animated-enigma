"""The CLI has to work from a clean clone with no arguments beyond the subcommand."""

from __future__ import annotations

import contextlib
import io
import unittest

from invkit.cli import _lead_time_pmf, main


def run(argv: list[str]) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(argv)
    assert code == 0, f"{argv} exited {code}"
    return buf.getvalue()


class TestCLI(unittest.TestCase):
    def test_lead_time_parsing(self):
        self.assertEqual(_lead_time_pmf("5"), {5: 1.0})
        self.assertEqual(_lead_time_pmf("4:0.6,9:0.4"), {4: 0.6, 9: 0.4})

    def test_safety_stock_command(self):
        out = run(["safety-stock", "--targets", "0.95", "0.98"])
        self.assertIn("SS (cycle service)", out)
        self.assertIn("SS (fill rate)", out)

    def test_safety_stock_accepts_a_stochastic_lead_time(self):
        out = run(["safety-stock", "--lead-time", "3:0.6,9:0.4", "--targets", "0.95"])
        self.assertIn("share of variance from lead-time variability", out)

    def test_policy_command_reports_analytic_and_simulated(self):
        out = run(["policy", "--periods", "20000"])
        self.assertIn("analytic:", out)
        self.assertIn("simulated:", out)

    def test_policy_command_periodic_review(self):
        out = run(["policy", "--kind", "RS", "--periods", "20000"])
        self.assertIn("(R, S) policy", out)

    def test_lotsize_command(self):
        out = run(["lotsize", "--demand", "10,60,12,130,150,129"])
        self.assertIn("wagner-whitin", out)
        self.assertIn("silver-meal", out)

    def test_newsvendor_command(self):
        out = run(["newsvendor", "--sample-size", "5000"])
        self.assertIn("critical fractile", out)
        self.assertIn("cost penalty of assuming normal", out)

    def test_serial_command(self):
        out = run(["serial", "--periods", "20000"])
        self.assertIn("optimal expected cost per period", out)
        self.assertIn("simulated cost per period", out)

    def test_place_command(self):
        out = run(["place"])
        self.assertIn("decoupling points", out)
        self.assertIn("dc_north", out)

    def test_frontier_command(self):
        out = run(["frontier"])
        self.assertIn("achieved fill", out)

    def test_pooling_command(self):
        out = run(["pooling", "--n-locations", "6"])
        self.assertIn("effective sqrt(n)", out)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
