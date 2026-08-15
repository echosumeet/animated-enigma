import unittest
from datetime import date

from tradedoc.schemas import (
    INCOTERMS_2020,
    LineItem,
    TradeDocument,
    ValidationIssue,
    normalize_hs_code,
    normalize_uom,
    parse_amount,
    parse_date,
)


class TestNormalisers(unittest.TestCase):
    def test_hs_code_strips_separators(self):
        self.assertEqual(normalize_hs_code("8471.30.01"), "84713001")
        self.assertEqual(normalize_hs_code("847130"), "847130")

    def test_hs_code_rejects_bad_length_and_chapter(self):
        for bad in ("8471", "84713", "98123456", "001234", "84713001234"):
            with self.assertRaises(ValidationIssue):
                normalize_hs_code(bad)

    def test_hs_code_rejects_ocr_substitution(self):
        # 'O' for '0' leaves five digits, which is not a valid subheading length.
        with self.assertRaises(ValidationIssue):
            normalize_hs_code("39269O")

    def test_uom_aliases_map_to_unece_codes(self):
        self.assertEqual(normalize_uom("pcs"), "PCE")
        self.assertEqual(normalize_uom(" Cartons "), "CTN")
        self.assertEqual(normalize_uom("KGS"), "KGM")
        self.assertEqual(normalize_uom("MTR"), "MTR")

    def test_uom_rejects_unknown(self):
        with self.assertRaises(ValidationIssue):
            normalize_uom("widgets")

    def test_date_multi_format(self):
        self.assertEqual(parse_date("2026-03-05"), date(2026, 3, 5))
        self.assertEqual(parse_date("05-Mar-2026"), date(2026, 3, 5))
        self.assertEqual(parse_date("05.03.2026"), date(2026, 3, 5))

    def test_date_ambiguity_resolved_by_hint(self):
        self.assertEqual(parse_date("03/05/2026", dayfirst=True), date(2026, 5, 3))
        self.assertEqual(parse_date("03/05/2026", dayfirst=False), date(2026, 3, 5))

    def test_date_rejects_garbage(self):
        with self.assertRaises(ValidationIssue):
            parse_date("20Z6-03-05")

    def test_amount_both_thousands_conventions(self):
        self.assertEqual(parse_amount("4,316.20"), 4316.20)
        self.assertEqual(parse_amount("4.316,20"), 4316.20)
        self.assertEqual(parse_amount("1234"), 1234.0)

    def test_amount_refuses_letters_rather_than_stripping(self):
        # "1OO" must not silently become 1.
        with self.assertRaises(ValidationIssue):
            parse_amount("1OO")


class TestModels(unittest.TestCase):
    def test_line_item_validates_and_totals(self):
        li = LineItem(description="Gasket", hs_code="4016.93", quantity=10,
                      uom="pcs", unit_price=2.5)
        self.assertEqual(li.hs_code, "401693")
        self.assertEqual(li.uom, "PCE")
        self.assertEqual(li.line_total(), 25.0)

    def test_document_enum_validators(self):
        doc = TradeDocument(doc_type="commercial_invoice", doc_number="INV-1",
                            doc_date="05-Mar-2026", currency="usd", incoterm="cif",
                            origin_country="de")
        self.assertEqual((doc.currency, doc.incoterm, doc.origin_country),
                         ("USD", "CIF", "DE"))
        self.assertEqual(len(INCOTERMS_2020), 11)

    def test_document_rejects_retired_incoterm(self):
        with self.assertRaises(Exception):
            TradeDocument(doc_type="commercial_invoice", doc_number="INV-1",
                          doc_date="2026-01-01", incoterm="DAT")

    def test_totals_and_weight_consistency_checks(self):
        items = [LineItem(description="a", hs_code="401693", quantity=10, uom="PCE",
                          unit_price=2.0)]
        good = TradeDocument(doc_type="commercial_invoice", doc_number="INV-1",
                             doc_date="2026-01-01", total_value=20.0, line_items=items)
        bad = TradeDocument(doc_type="commercial_invoice", doc_number="INV-1",
                            doc_date="2026-01-01", total_value=30.0, line_items=items)
        self.assertTrue(good.totals_consistent())
        self.assertFalse(bad.totals_consistent())
        heavy = TradeDocument(doc_type="packing_list", doc_number="PL-1",
                              doc_date="2026-01-01", net_weight_kg=100,
                              gross_weight_kg=90)
        self.assertFalse(heavy.net_le_gross())


if __name__ == "__main__":
    unittest.main()
