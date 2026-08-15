import unittest

import numpy as np

from scmplatform.datagen import PanelConfig, make_panel
from scmplatform.features import (
    FeatureSpec,
    audit_features,
    build_features,
    default_specs,
    detect_skew,
    knowledge_time_check,
    leaky_specs,
    truncation_check,
)


class TestFeatureConstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = make_panel(PanelConfig(n_skus=5, n_days=150, seed=5))

    def test_lag_feature_matches_a_manual_shift(self):
        feats = build_features(self.panel, [FeatureSpec("u1", "units", "lag", lag=1)])
        one = self.panel[self.panel["sku"] == "SKU-000"].sort_values("date")
        got = feats[feats["sku"] == "SKU-000"].sort_values("date")["u1"].to_numpy()
        np.testing.assert_allclose(got[1:], one["units"].to_numpy()[:-1])
        self.assertTrue(np.isnan(got[0]))


class TestLeakageDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = make_panel(PanelConfig(n_skus=5, n_days=150, seed=5))

    def test_production_feature_set_is_clean(self):
        self.assertEqual(audit_features(self.panel, default_specs()), [])

    def test_knowledge_delay_violation_is_caught(self):
        findings = knowledge_time_check(leaky_specs())
        names = {f.feature for f in findings}
        self.assertIn("returns_lag_1", names)
        self.assertNotIn("units_lag_1", names)

    def test_full_sample_statistic_is_caught_by_truncation(self):
        findings = truncation_check(self.panel, leaky_specs())
        self.assertEqual({f.feature for f in findings}, {"sku_mean_units"})

    def test_audit_reports_both_bugs_with_distinct_checks(self):
        findings = audit_features(self.panel, leaky_specs())
        self.assertEqual(len(findings), 2)
        self.assertEqual({f.check for f in findings}, {"knowledge_time", "truncation"})


class TestSkew(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = make_panel(PanelConfig(n_skus=5, n_days=150, seed=5))
        cls.specs = default_specs()
        cls.offline = build_features(cls.panel, cls.specs)
        cls.names = [s.name for s in cls.specs]

    def test_identical_pipelines_show_no_skew(self):
        report = detect_skew(self.offline, self.offline.copy(), ["sku", "date"], self.names)
        self.assertEqual(report.failing(), [])
        self.assertAlmostEqual(report.worst_mismatch_rate, 0.0)

    def test_a_serving_side_fillna_difference_is_detected(self):
        online = self.offline.copy()
        online["units_ma_7"] = online["units_ma_7"].fillna(0.0) * 1.05
        report = detect_skew(self.offline, online, ["sku", "date"], self.names)
        self.assertEqual(report.failing(threshold=0.01), ["units_ma_7"])
        row = report.per_feature.set_index("feature").loc["units_ma_7"]
        self.assertGreater(row["mismatch_rate"], 0.5)
        self.assertGreater(row["max_abs_gap"], 0.0)


if __name__ == "__main__":
    unittest.main()
