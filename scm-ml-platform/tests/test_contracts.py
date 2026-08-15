import unittest

import pandas as pd

from scmplatform.contracts import (
    ColumnSpec,
    DataContract,
    demand_panel_contract,
    diff_contracts,
    validate,
)
from scmplatform.datagen import PanelConfig, inject_quality_faults, make_panel


class TestValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = make_panel(PanelConfig(n_skus=6, n_days=120, seed=3))
        cls.contract = demand_panel_contract()
        cls.now = cls.panel["date"].max()

    def test_clean_panel_passes(self):
        report = validate(self.panel, self.contract, now=self.now)
        self.assertTrue(report.passed, report.to_frame().to_string())
        self.assertIn("PASS", report.summary())

    def test_faulty_panel_fails_with_specific_rules(self):
        report = validate(inject_quality_faults(self.panel), self.contract, now=self.now)
        self.assertFalse(report.passed)
        rules = set(report.to_frame()["rule"])
        self.assertIn("not_null", rules)
        self.assertIn("min_value", rules)
        self.assertIn("unique_key", rules)

    def test_freshness_violation(self):
        stale_now = self.panel["date"].max() + pd.Timedelta(days=9)
        report = validate(self.panel, self.contract, now=stale_now)
        self.assertIn("freshness", set(report.to_frame()["rule"]))

    def test_warn_severity_does_not_fail_the_report(self):
        panel = self.panel.copy()
        panel.loc[panel.index[:3], "on_hand"] = -5.0
        report = validate(panel, self.contract, now=self.now)
        self.assertTrue(report.passed)
        self.assertEqual(len(report.violations), 1)
        self.assertEqual(report.violations[0].severity, "warn")

    def test_allowed_values(self):
        panel = self.panel.copy()
        panel.loc[panel.index[:2], "region"] = "offworld"
        report = validate(panel, self.contract, now=self.now)
        self.assertIn("allowed_values", set(report.to_frame()["rule"]))


def _contract(**kw) -> DataContract:
    cols = kw.pop("columns")
    return DataContract(name="t", version="1", columns=cols, **kw)


class TestContractDiff(unittest.TestCase):
    def setUp(self):
        self.base = _contract(
            columns=(
                ColumnSpec("a", "float", nullable=True, min_value=0.0, max_value=100.0),
                ColumnSpec("b", "str", nullable=False, allowed=("x", "y", "z")),
            ),
            primary_key=("a",),
            max_age_days=3.0,
        )

    def test_dropping_a_column_is_breaking(self):
        new = _contract(columns=self.base.columns[:1], primary_key=("a",), max_age_days=3.0)
        d = diff_contracts(self.base, new)
        self.assertTrue(d.is_breaking)
        self.assertTrue(any("column removed: b" in r for r in d.breaking))

    def test_adding_a_nullable_column_is_non_breaking(self):
        new = _contract(
            columns=self.base.columns + (ColumnSpec("c", "float", nullable=True),),
            primary_key=("a",),
            max_age_days=3.0,
        )
        d = diff_contracts(self.base, new)
        self.assertFalse(d.is_breaking)
        self.assertTrue(any("column added: c" in r for r in d.non_breaking))

    def test_tightening_a_bound_is_breaking_and_widening_is_not(self):
        tighter = _contract(
            columns=(ColumnSpec("a", "float", True, 0.0, 50.0), self.base.columns[1]),
            primary_key=("a",),
            max_age_days=3.0,
        )
        wider = _contract(
            columns=(ColumnSpec("a", "float", True, 0.0, 500.0), self.base.columns[1]),
            primary_key=("a",),
            max_age_days=3.0,
        )
        self.assertTrue(diff_contracts(self.base, tighter).is_breaking)
        self.assertFalse(diff_contracts(self.base, wider).is_breaking)

    def test_dtype_change_and_narrowed_enum_are_breaking(self):
        new = _contract(
            columns=(
                ColumnSpec("a", "int", nullable=True, min_value=0.0, max_value=100.0),
                ColumnSpec("b", "str", nullable=False, allowed=("x", "y")),
            ),
            primary_key=("a",),
            max_age_days=3.0,
        )
        d = diff_contracts(self.base, new)
        self.assertEqual(len(d.breaking), 2)


if __name__ == "__main__":
    unittest.main()
