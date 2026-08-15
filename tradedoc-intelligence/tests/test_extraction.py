import random
import unittest

from tradedoc.extract import count_table_lines, extract_line_items, extract_scalars
from tradedoc.generate import DOC_TYPES, NoiseConfig, build_corpus, generate_documents, make_shipment
from tradedoc.llm import StubProvider
from tradedoc.pipeline import document_correct, extract_document, field_correctness

CLEAN = NoiseConfig(char_corruption_rate=0.0, field_dropout_rate=0.0,
                    inconsistency_rate=0.0)


def clean_docs(seed=3, n=1):
    rng = random.Random(seed)
    sh = make_shipment(rng, 1)
    docs, _ = generate_documents(sh, rng, CLEAN)
    return sh, {d.doc_type: d for d in docs}


class TestGenerator(unittest.TestCase):
    def test_three_document_types_per_shipment(self):
        _, by_type = clean_docs()
        self.assertEqual(set(by_type), set(DOC_TYPES))

    def test_corpus_is_deterministic_for_a_seed(self):
        a = build_corpus(5, seed=11)
        b = build_corpus(5, seed=11)
        self.assertEqual([d.text for d in a.docs], [d.text for d in b.docs])

    def test_dropout_sets_ground_truth_to_none(self):
        corpus = build_corpus(40, seed=5,
                              noise=NoiseConfig(0.0, 0.5, 0.0))
        dropped = sum(1 for d in corpus.docs if d.truth.get("doc_number") is None)
        self.assertGreater(dropped, 0)
        for doc in corpus.docs:
            if doc.truth.get("doc_number") is None:
                self.assertNotIn("INV-", doc.text.split("\n")[3])

    def test_seeded_inconsistencies_are_recorded(self):
        corpus = build_corpus(40, seed=5, noise=NoiseConfig(0.0, 0.0, 1.0))
        self.assertTrue(all(len(v) == 1 for v in corpus.seeded.values()))


class TestRules(unittest.TestCase):
    def test_scalars_on_a_clean_invoice(self):
        sh, by_type = clean_docs()
        got = extract_scalars(by_type["commercial_invoice"].text, "commercial_invoice")
        self.assertEqual(got["currency"], sh.currency)
        self.assertEqual(got["incoterm"], sh.incoterm)
        self.assertEqual(got["origin_country"], sh.origin_country)

    def test_blank_label_does_not_capture_the_next_line(self):
        text = "Vessel: \nPort of Loading: Hamburg\n"
        got = extract_scalars(text, "bill_of_lading")
        self.assertIsNone(got["vessel"])
        self.assertEqual(got["port_of_loading"], "Hamburg")

    def test_line_item_rows_and_expected_count_agree(self):
        _, by_type = clean_docs()
        for doc_type in DOC_TYPES:
            rows, expected = extract_line_items(by_type[doc_type].text, doc_type)
            self.assertEqual(len(rows), expected, doc_type)

    def test_count_table_lines_needs_two_rules(self):
        self.assertEqual(count_table_lines("no table here"), 0)


class TestHybridPipeline(unittest.TestCase):
    def test_clean_documents_extract_exactly(self):
        _, by_type = clean_docs()
        for doc_type, doc in by_type.items():
            res = extract_document(doc)
            self.assertTrue(document_correct(res, doc.truth), doc_type)

    def test_omitted_field_is_never_confidently_invented(self):
        """A field that is not on the page may still be guessed by the LLM pass -- the
        invariant that matters is that such a guess can never carry rule-level
        confidence, so it can never be auto-approved."""
        corpus = build_corpus(20, seed=5, noise=NoiseConfig(0.0, 0.5, 0.0))
        absent = invented = 0
        for doc in corpus.docs:
            res = extract_document(doc)
            for name, fr in res.fields.items():
                if doc.truth.get(name) is None:
                    absent += 1
                    if fr.value is not None:
                        invented += 1
                        self.assertLess(fr.confidence, 0.5, f"{doc.doc_type}.{name}")
        self.assertGreater(absent, 0)
        self.assertLess(invented / absent, 0.1)

    def test_corruption_lowers_confidence_on_the_affected_field(self):
        _, by_type = clean_docs()
        doc = by_type["bill_of_lading"]
        clean_conf = extract_document(doc).fields["container_no"].confidence
        doc.text = doc.text.replace("Container No: ", "Container No: X")
        dirty = extract_document(doc).fields["container_no"]
        self.assertIsNone(dirty.value)
        self.assertLess(dirty.confidence, clean_conf)

    def test_stub_provider_is_deterministic_and_offline(self):
        _, by_type = clean_docs()
        text = by_type["commercial_invoice"].text
        a, b = StubProvider(), StubProvider()
        self.assertEqual(a.extract_field(text, "currency", "commercial_invoice"),
                         b.extract_field(text, "currency", "commercial_invoice"))

    def test_confirm_mode_changes_call_volume_not_values(self):
        _, by_type = clean_docs()
        doc = by_type["commercial_invoice"]
        none_r = extract_document(doc, confirm="none")
        all_r = extract_document(doc, confirm="all")
        self.assertLess(none_r.llm_calls, all_r.llm_calls)
        self.assertEqual(none_r.values(), all_r.values())
        self.assertGreater(all_r.min_confidence(), none_r.min_confidence())

    def test_field_correctness_counts_absence_as_a_value(self):
        _, by_type = clean_docs()
        doc = by_type["packing_list"]
        res = extract_document(doc)
        flags = field_correctness(res, doc.truth)
        self.assertTrue(all(flags.values()))


if __name__ == "__main__":
    unittest.main()
