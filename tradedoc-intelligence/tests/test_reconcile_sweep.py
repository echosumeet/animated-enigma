import random
import unittest

from tradedoc.evaluate import run
from tradedoc.generate import NoiseConfig, generate_documents, make_shipment
from tradedoc.pipeline import extract_document
from tradedoc.reconcile import evaluate_detection, reconcile
from tradedoc.sweep import field_accuracy_table, operating_point, sweep_thresholds

CLEAN = NoiseConfig(0.0, 0.0, 0.0)


def shipment_results(seed, noise=CLEAN):
    rng = random.Random(seed)
    sh = make_shipment(rng, 1)
    docs, seeded = generate_documents(sh, rng, noise)
    return {d.doc_type: extract_document(d) for d in docs}, seeded


class TestReconcile(unittest.TestCase):
    def test_consistent_shipment_raises_nothing(self):
        results, seeded = shipment_results(4)
        self.assertEqual(seeded, [])
        self.assertEqual(reconcile(results), [])

    def test_quantity_mismatch_is_detected_and_graded(self):
        results, _ = shipment_results(4)
        row = results["packing_list"].line_items[0]
        row["quantity"] = row["quantity"] * 2
        found = reconcile(results)
        kinds = {d.kind: d for d in found}
        self.assertIn("quantity_mismatch", kinds)
        self.assertEqual(kinds["quantity_mismatch"].severity, "critical")
        self.assertEqual(kinds["quantity_mismatch"].hs_code, row["hs_code"])

    def test_severity_grading_uses_relative_deviation(self):
        results, _ = shipment_results(4)
        gross = results["packing_list"].fields["gross_weight_kg"].value
        results["bill_of_lading"].fields["gross_weight_kg"].value = gross * 1.03
        found = [d for d in reconcile(results) if d.kind == "gross_weight_mismatch"]
        self.assertEqual(found[0].severity, "major")

    def test_rounding_noise_is_not_an_alert(self):
        results, _ = shipment_results(4)
        gross = results["packing_list"].fields["gross_weight_kg"].value
        results["bill_of_lading"].fields["gross_weight_kg"].value = gross * 1.001
        self.assertEqual([d for d in reconcile(results)
                          if d.kind == "gross_weight_mismatch"], [])

    def test_unreadable_table_is_not_reconciled(self):
        results, _ = shipment_results(4)
        results["packing_list"].line_items[0]["hs_code"] = None
        self.assertEqual([d for d in reconcile(results)
                          if d.kind in ("quantity_mismatch", "line_item_missing")], [])

    def test_detection_scoring_counts_misses_and_false_alerts(self):
        seeded = {"A": [{"kind": "quantity_mismatch"}], "B": [{"kind": "gross_weight_mismatch"}]}
        results, _ = shipment_results(4)
        found = reconcile(results)
        self.assertEqual(found, [])
        score = evaluate_detection(seeded, {"A": [], "B": []})
        self.assertEqual(score["recall"], 0.0)
        self.assertEqual(score["false_negatives"], 2)


class TestSweep(unittest.TestCase):
    def setUp(self):
        self.conf = [0.95, 0.90, 0.80, 0.60, 0.40]
        self.ok = [True, True, False, True, False]
        self.points = sweep_thresholds(self.conf, self.ok)

    def test_stp_is_monotone_decreasing_in_threshold(self):
        rates = [p.stp_rate for p in self.points]
        self.assertEqual(rates, sorted(rates, reverse=True))

    def test_precision_and_recall_definitions(self):
        p = next(x for x in self.points if abs(x.threshold - 0.90) < 1e-9)
        self.assertEqual(p.n_auto, 2)
        self.assertEqual(p.precision, 1.0)
        self.assertAlmostEqual(p.recall, 2 / 3, places=4)

    def test_operating_point_respects_the_error_budget(self):
        op = operating_point(self.points, error_budget=0.0)
        self.assertIsNotNone(op)
        self.assertEqual(op.n_auto_wrong, 0)
        self.assertIsNone(operating_point(self.points, error_budget=-1.0))

    def test_field_accuracy_table_shapes(self):
        table = field_accuracy_table([
            ("commercial_invoice", {"currency": True, "incoterm": False}),
            ("commercial_invoice", {"currency": True, "incoterm": True}),
        ])
        self.assertEqual(table["commercial_invoice"]["fields"]["currency"], 1.0)
        self.assertEqual(table["commercial_invoice"]["fields"]["incoterm"], 0.5)
        self.assertEqual(table["commercial_invoice"]["n_docs"], 2)


class TestEndToEnd(unittest.TestCase):
    def test_run_is_reproducible_and_internally_consistent(self):
        a = run(15, seed=2)
        b = run(15, seed=2)
        self.assertEqual(a.confidences, b.confidences)
        self.assertEqual(len(a.results), 45)
        self.assertEqual(len(a.confidences), len(a.correct))
        self.assertTrue(0.0 <= a.detection["recall"] <= 1.0)
        self.assertEqual(set(a.accuracy), {"commercial_invoice", "packing_list",
                                           "bill_of_lading"})

    def test_noise_free_corpus_is_fully_straight_through(self):
        out = run(10, seed=2, noise=CLEAN)
        self.assertTrue(all(out.correct))
        self.assertEqual(out.points[0].stp_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
