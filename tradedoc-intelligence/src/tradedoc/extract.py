"""Deterministic rule/layout extraction pass.

Rules go first because they are cheap, auditable and correct on the ~80% of fields that
sit next to a predictable label. What they cannot do is recover from an unseen label
phrasing or a mangled table row -- that residual is what the LLM pass in ``llm.py``
exists for.
"""

from __future__ import annotations

import re
from typing import Optional

# Label patterns per field. Multiple alternatives per field cover the layout variants.
# Free-text fields require the colon to be present: with the separator optional, a
# blank field lets the capture group start on the colon itself.
# ``H`` is horizontal whitespace only. Using \s* after a label is the single most common
# defect in hand-written document rules: on a form where the labelled field is blank it
# happily walks to the next line and returns the following label's value with full
# confidence. Every pattern here is confined to one line.
H = r"[^\S\n]"
LABEL_PATTERNS: dict[str, list[str]] = {
    "doc_number": [
        rf"Invoice{H}*No{H}*[:.]*{H}*([A-Za-z0-9\-]{{4,}}){H}*$",
        rf"Document{H}*Number[ \t.]*[:.]*{H}*([A-Za-z0-9\-]{{4,}}){H}*$",
        # No separator tolerated after INV#: the number is printed flush against it, and
        # allowing whitespace lets a blank field capture the next label on the line.
        rf"INV#([A-Za-z0-9\-]{{4,}})",
        rf"Packing{H}*List{H}*No{H}*[:.]*{H}*([A-Za-z0-9\-]{{4,}}){H}*$",
        rf"B/L{H}*No{H}*[:.]*{H}*([A-Za-z0-9\-]{{4,}}){H}*$",
    ],
    "doc_date": [
        rf"(?:Invoice{H}*Date|Date{H}*of{H}*Issue|Issued|Dated|Date)[ \t.]*[:.]*{H}*"
        r"([0-9A-Za-z]{1,4}[-/. ][0-9A-Za-z]{1,4}[-/. ][0-9]{2,4})",
    ],
    "buyer": [rf"(?:Sold{H}*To|Consignee|Bill{H}*To){H}*:{H}*(\S.*)$"],
    "currency": [rf"Currency{H}*[:.]*{H}*([A-Za-z]{{3}}){H}*$"],
    "incoterm": [rf"(?:Terms{H}*of{H}*Delivery|Incoterm[s]?){H}*[:.]*{H}*([A-Za-z]{{3}})\b"],
    "origin_country": [rf"Country{H}*of{H}*Origin{H}*[:.]*{H}*([A-Za-z]{{2}}){H}*$"],
    "total_value": [rf"TOTAL{H}+(?:[A-Za-z]{{3}}{H}+)?([\d.,]+){H}*$"],
    "package_count": [
        rf"(?:Total{H}*Packages|Number{H}*of{H}*Packages){H}*[:.]*{H}*([\d,]+){H}*$"],
    "net_weight_kg": [rf"Net{H}*Weight{H}*[:.]*{H}*([\d.,]+)"],
    "gross_weight_kg": [rf"Gross{H}*Weight{H}*[:.]*{H}*([\d.,]+)"],
    "vessel": [rf"Vessel{H}*:{H}*(\S.*)$"],
    "port_of_loading": [rf"Port{H}*of{H}*Loading{H}*:{H}*(\S.*)$"],
    "port_of_discharge": [rf"Port{H}*of{H}*Discharge{H}*:{H}*(\S.*)$"],
    "container_no": [rf"Container{H}*No{H}*[:.]*{H}*([A-Za-z0-9]{{8,}}){H}*$"],
}

FIELDS_BY_DOC = {
    "commercial_invoice": ["doc_number", "doc_date", "buyer", "currency", "incoterm",
                           "origin_country", "total_value"],
    "packing_list": ["doc_number", "doc_date", "buyer", "package_count",
                     "net_weight_kg", "gross_weight_kg"],
    "bill_of_lading": ["doc_number", "doc_date", "buyer", "vessel", "port_of_loading",
                       "port_of_discharge", "container_no", "package_count",
                       "gross_weight_kg"],
}

# HS codes and quantities are the fields OCR wrecks most often, so the row patterns
# deliberately accept the corrupted alphabet and let validation reject it downstream.
_HS = r"[0-9OlSBZG]{6,10}"
_NUM = r"[\d.,OlSBZG]+"
INVOICE_ROW = re.compile(
    rf"^({_HS})\s+(.+?)\s{{2,}}({_NUM})\s+([A-Za-z]{{1,5}})\s+({_NUM})\s+({_NUM})\s*$"
)
PACKING_ROW = re.compile(rf"^({_HS})\s+(.+?)\s{{2,}}({_NUM})\s+([A-Za-z]{{1,5}})\s*$")
BOL_ROW = re.compile(rf"^\s{{2}}(.+?)\s{{2,}}HS\s+({_HS})\s*$")


def _first_match(text: str, patterns: list[str]) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            val = m.group(1).strip()
            if val:
                return val
    return None


def extract_scalars(text: str, doc_type: str) -> dict[str, Optional[str]]:
    """Label-anchored extraction of the scalar fields relevant to ``doc_type``."""
    out: dict[str, Optional[str]] = {}
    for fieldname in FIELDS_BY_DOC[doc_type]:
        out[fieldname] = _first_match(text, LABEL_PATTERNS[fieldname])
    return out


SEPARATOR = re.compile(r"^-{20,}\s*$")


def count_table_lines(text: str) -> int:
    """Count body lines between the first and second horizontal rule.

    Row-level regexes fail silently: a corrupted row simply does not match and the table
    comes back one line short with nothing to indicate it. Counting the lines the layout
    says are there gives the confidence model something to compare against.
    """
    lines = text.splitlines()
    bounds = [i for i, ln in enumerate(lines) if SEPARATOR.match(ln)]
    if len(bounds) < 2:
        return 0
    return sum(1 for ln in lines[bounds[0] + 1:bounds[1]] if ln.strip())


def extract_line_items(text: str, doc_type: str) -> tuple[list[dict[str, str]], int]:
    """Return the parsed rows and the number of body lines the layout implies."""
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if doc_type == "commercial_invoice":
            m = INVOICE_ROW.match(line)
            if m:
                rows.append({"hs_code": m.group(1), "description": m.group(2).strip(),
                             "quantity": m.group(3), "uom": m.group(4),
                             "unit_price": m.group(5), "amount": m.group(6)})
        elif doc_type == "packing_list":
            m = PACKING_ROW.match(line)
            if m:
                rows.append({"hs_code": m.group(1), "description": m.group(2).strip(),
                             "quantity": m.group(3), "uom": m.group(4)})
        else:
            m = BOL_ROW.match(line)
            if m:
                rows.append({"description": m.group(1).strip(), "hs_code": m.group(2)})
    return rows, count_table_lines(text)
