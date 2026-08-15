"""Hybrid extraction and per-field confidence.

Order of operations per field: rules, then the LLM pass only on the residual, then a
validator, then a confidence score built from three observable signals -- whether each
extractor produced anything, whether they agree after normalisation, and whether the
value passes its validator. No model is fitted here; the score is a transparent
additive rule so that a customs broker can be told *why* a document was held.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Optional

from .extract import FIELDS_BY_DOC, extract_line_items, extract_scalars
from .generate import GeneratedDoc
from .llm import LLMProvider, StubProvider
from .schemas import (
    INCOTERMS_2020,
    ISO_3166,
    ISO_4217,
    ValidationIssue,
    normalize_hs_code,
    normalize_uom,
    parse_amount,
    parse_date,
)

CONTAINER_RE = re.compile(r"^[A-Z]{4}\d{7}$")


def _norm_docnum(raw: str, ctx: dict) -> str:
    val = raw.strip().upper()
    if not re.fullmatch(r"[A-Z0-9\-]{4,20}", val):
        raise ValidationIssue(f"implausible document number {raw!r}")
    return val


def _norm_date(raw: str, ctx: dict) -> str:
    return parse_date(raw, dayfirst=ctx.get("dayfirst")).isoformat()


def _norm_text(raw: str, ctx: dict) -> str:
    val = " ".join(raw.split())
    if len(val) < 3 or not re.search(r"[A-Za-z]", val):
        raise ValidationIssue(f"implausible text value {raw!r}")
    return val


def _norm_enum(allowed: tuple[str, ...], label: str) -> Callable[[str, dict], str]:
    def _fn(raw: str, ctx: dict) -> str:
        val = raw.strip().upper()
        if val not in allowed:
            raise ValidationIssue(f"{val!r} is not a valid {label}")
        return val
    return _fn


def _norm_money(raw: str, ctx: dict) -> float:
    val = parse_amount(raw)
    if val < 0:
        raise ValidationIssue("negative amount")
    return round(val, 2)


def _norm_count(raw: str, ctx: dict) -> int:
    val = parse_amount(raw)
    if val <= 0 or abs(val - round(val)) > 1e-9:
        raise ValidationIssue(f"package count {raw!r} is not a positive integer")
    return int(round(val))


def _norm_container(raw: str, ctx: dict) -> str:
    val = re.sub(r"\s", "", raw).upper()
    if not CONTAINER_RE.fullmatch(val):
        raise ValidationIssue(f"{val!r} is not an ISO 6346 container number")
    return val


NORMALIZERS: dict[str, Callable[[str, dict], Any]] = {
    "doc_number": _norm_docnum,
    "doc_date": _norm_date,
    "buyer": _norm_text,
    "currency": _norm_enum(ISO_4217, "ISO 4217 currency"),
    "incoterm": _norm_enum(INCOTERMS_2020, "Incoterms 2020 rule"),
    "origin_country": _norm_enum(ISO_3166, "ISO 3166-1 country"),
    "total_value": _norm_money,
    "net_weight_kg": _norm_money,
    "gross_weight_kg": _norm_money,
    "package_count": _norm_count,
    "vessel": _norm_text,
    "port_of_loading": _norm_text,
    "port_of_discharge": _norm_text,
    "container_no": _norm_container,
}

# Additive confidence weights. Tuned once against the calibration corpus, then frozen;
# they are reported in the README so the score is reproducible rather than magic.
W_BASE = 0.35
W_RULE_OK = 0.53
W_LLM_PRESENT = 0.05
W_AGREE = 0.06
W_DISAGREE = -0.33
W_RULE_INVALID = -0.30
# Both extractors silent. Two independent readers finding nothing is itself a form of
# agreement, and field omission is far more common in this corpus than a label mangled
# past recognition -- so absence is asserted with confidence, not hedged.
C_ABSENT = 0.80


@dataclass
class FieldResult:
    name: str
    value: Any
    confidence: float
    source: str
    rule_raw: Optional[str] = None
    llm_raw: Optional[str] = None
    validator_ok: bool = True
    agreement: Optional[bool] = None


@dataclass
class DocResult:
    doc_type: str
    shipment_id: str
    fields: dict[str, FieldResult]
    line_items: list[dict[str, Any]]
    line_confidence: float
    llm_calls: int

    def min_confidence(self) -> float:
        vals = [f.confidence for f in self.fields.values()] + [self.line_confidence]
        return min(vals) if vals else 0.0

    def values(self) -> dict[str, Any]:
        return {k: v.value for k, v in self.fields.items()}


def _resolve(name: str, rule_raw: Optional[str], llm_raw: Optional[str],
             ctx: dict) -> FieldResult:
    norm = NORMALIZERS[name]
    rule_val = llm_val = None
    rule_ok = llm_ok = False
    if rule_raw is not None:
        try:
            rule_val, rule_ok = norm(rule_raw, ctx), True
        except (ValidationIssue, ValueError):
            rule_ok = False
    if llm_raw is not None:
        try:
            llm_val, llm_ok = norm(llm_raw, ctx), True
        except (ValidationIssue, ValueError):
            llm_ok = False

    if rule_raw is None and llm_raw is None:
        return FieldResult(name, None, C_ABSENT, "absent")

    agreement = None
    if rule_ok and llm_ok:
        agreement = rule_val == llm_val

    score = W_BASE
    if rule_ok:
        score += W_RULE_OK
    elif rule_raw is not None:
        score += W_RULE_INVALID
    if llm_raw is not None:
        score += W_LLM_PRESENT
    if agreement is True:
        score += W_AGREE
    elif agreement is False:
        score += W_DISAGREE
    score = min(0.99, max(0.02, score))

    if rule_ok:
        value, source = rule_val, "rule"
    elif llm_ok:
        value, source = llm_val, "llm"
    else:
        value, source = None, "failed"
        score = min(score, 0.15)
    return FieldResult(name, value, round(score, 4), source, rule_raw, llm_raw,
                       rule_ok or llm_ok, agreement)


def _score_line_items(rows: list[dict[str, str]], expected_rows: int
                      ) -> tuple[list[dict[str, Any]], float]:
    """Normalise table rows and return a single confidence for the table as a whole."""
    out: list[dict[str, Any]] = []
    penalties = 0
    for row in rows:
        item: dict[str, Any] = {"description": " ".join(row["description"].split())}
        try:
            item["hs_code"] = normalize_hs_code(row["hs_code"])
        except ValidationIssue:
            item["hs_code"] = None
            penalties += 1
        if "quantity" in row:
            try:
                item["quantity"] = round(parse_amount(row["quantity"]), 2)
            except ValidationIssue:
                item["quantity"] = None
                penalties += 1
            try:
                item["uom"] = normalize_uom(row["uom"])
            except ValidationIssue:
                item["uom"] = None
                penalties += 1
        for money in ("unit_price", "amount"):
            if money in row:
                try:
                    item[money] = round(parse_amount(row[money]), 2)
                except ValidationIssue:
                    item[money] = None
                    penalties += 1
        out.append(item)
    # Rows the layout implies but the row regex could not parse are counted as failures,
    # not as an empty table.
    penalties += max(0, expected_rows - len(out))
    if not out:
        return out, 0.10  # a document with no readable table is never straight-through
    # One unreadable cell contaminates the whole table: a customs line is filed as a
    # unit, so partial credit here would be dishonest.
    conf = max(0.05, 0.97 - 0.45 * penalties)
    return out, round(conf, 4)


# Fields worth spending a second-opinion call on when the rules already succeeded:
# the ones that carry money, weight or a filing date.
CONFIRM_HIGH_VALUE = ("total_value", "gross_weight_kg", "doc_date", "incoterm")


def extract_document(doc: GeneratedDoc, provider: Optional[LLMProvider] = None,
                     confirm: str = "high_value") -> DocResult:
    """Run the hybrid pipeline over one rendered document.

    ``confirm`` controls how much the second pass is used once the rules have already
    produced a valid value. ``"none"`` is cheapest, ``"all"`` buys the most confidence
    resolution, ``"high_value"`` is the default compromise. The residual (rules found
    nothing) always goes to the provider regardless.
    """
    provider = provider or StubProvider()
    ctx = {"dayfirst": doc.dayfirst}
    rules = extract_scalars(doc.text, doc.doc_type)
    fields: dict[str, FieldResult] = {}
    calls = 0
    for name in FIELDS_BY_DOC[doc.doc_type]:
        rule_raw = rules.get(name)
        llm_raw = None
        ask = rule_raw is None or confirm == "all" or (
            confirm == "high_value" and name in CONFIRM_HIGH_VALUE)
        if ask:
            llm_raw = provider.extract_field(doc.text, name, doc.doc_type)
            calls += 1
        fields[name] = _resolve(name, rule_raw, llm_raw, ctx)
    rows, expected = extract_line_items(doc.text, doc.doc_type)
    items, line_conf = _score_line_items(rows, expected)
    return DocResult(doc.doc_type, doc.shipment_id, fields, items, line_conf, calls)


def _eq(a: Any, b: Any, tol: float = 0.02) -> bool:
    if a is None or b is None:
        return a is b or (a is None and b is None)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    return str(a).strip().upper() == str(b).strip().upper()


def field_correctness(result: DocResult, truth: dict[str, Any]) -> dict[str, bool]:
    """Per-field exact match against generator ground truth (None counts as a value)."""
    out = {}
    for name, fr in result.fields.items():
        out[name] = _eq(fr.value, truth.get(name))
    return out


def line_items_correct(result: DocResult, truth: dict[str, Any]) -> bool:
    true_rows = truth.get("line_items", [])
    if len(true_rows) != len(result.line_items):
        return False
    for got, want in zip(result.line_items, true_rows):
        if got.get("hs_code") != want["hs_code"]:
            return False
        if "quantity" in got and not _eq(got.get("quantity"), want["quantity"]):
            return False
        if "uom" in got and got.get("uom") != want["uom"]:
            return False
    return True


def document_correct(result: DocResult, truth: dict[str, Any]) -> bool:
    return all(field_correctness(result, truth).values()) and line_items_correct(result, truth)
