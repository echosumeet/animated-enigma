"""Validated trade-document schemas.

The point of this module is that every field a downstream system cares about has a
*decidable* validity test: an HS code either is or is not a well-formed subheading, a
three-letter code either is or is not in ISO 4217, an Incoterm either is or is not one
of the eleven rules published in Incoterms 2020. Extraction confidence in this repo is
built partly out of these outcomes, so the validators have to be real, not decorative.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Incoterms 2020 (ICC publication 723E). Eleven rules; DAT was renamed DPU in 2020.
INCOTERMS_2020 = (
    "EXW", "FCA", "FAS", "FOB", "CFR", "CIF", "CPT", "CIP", "DAP", "DPU", "DDP",
)
# Sea/inland-waterway-only rules -- using these for airfreight is a classic filing error.
SEA_ONLY_INCOTERMS = ("FAS", "FOB", "CFR", "CIF")

# Working subset of ISO 4217 / ISO 3166-1 alpha-2. Kept explicit so validation failures
# are honest rather than "any three uppercase letters pass".
ISO_4217 = (
    "USD", "EUR", "GBP", "JPY", "CNY", "INR", "SGD", "AUD", "CAD", "CHF", "MXN", "VND",
)
ISO_3166 = (
    "US", "DE", "GB", "JP", "CN", "IN", "SG", "AU", "CA", "CH", "MX", "VN", "NL", "TH",
)

# UOM aliases -> UN/ECE Recommendation 20 codes.
UOM_ALIASES = {
    "pc": "PCE", "pcs": "PCE", "piece": "PCE", "pieces": "PCE", "ea": "PCE", "each": "PCE",
    "unit": "PCE", "units": "PCE", "pce": "PCE",
    "kg": "KGM", "kgs": "KGM", "kilo": "KGM", "kilos": "KGM", "kilogram": "KGM",
    "kilograms": "KGM", "kgm": "KGM",
    "ctn": "CTN", "ctns": "CTN", "carton": "CTN", "cartons": "CTN",
    "box": "BOX", "boxes": "BOX", "bx": "BOX",
    "m": "MTR", "mtr": "MTR", "metre": "MTR", "metres": "MTR", "meter": "MTR",
    "l": "LTR", "ltr": "LTR", "litre": "LTR", "liter": "LTR", "liters": "LTR",
    "set": "SET", "sets": "SET",
    "pr": "PR", "pair": "PR", "pairs": "PR",
}

# Ordered most- to least-specific. Ambiguous numeric formats come last on purpose: a
# bare 03/04/2026 cannot be resolved without knowing the issuing convention, and
# pretending otherwise is how silent one-month date errors get into customs filings.
DATE_FORMATS = (
    "%Y-%m-%d", "%d-%b-%Y", "%d %b %Y", "%b %d, %Y", "%d.%m.%Y", "%Y/%m/%d",
    "%d/%m/%Y", "%m/%d/%Y",
)
AMBIGUOUS_DATE_FORMATS = ("%d/%m/%Y", "%m/%d/%Y")


class ValidationIssue(Exception):
    """Raised by the normalisers below; carried as a field-level validator outcome."""


def normalize_hs_code(raw: str) -> str:
    """Return the digits of an HS code, or raise.

    Harmonized System subheadings are six digits; national tariff lines extend to eight
    or ten. Separators vary by country and by whoever typed the invoice, so we strip
    them and check length only.
    """
    digits = re.sub(r"\D", "", str(raw))
    if len(digits) < 6 or len(digits) > 10 or len(digits) % 2 == 1:
        raise ValidationIssue(f"HS code {raw!r} is not 6/8/10 digits")
    if digits[:2] == "00" or int(digits[:2]) > 97:
        raise ValidationIssue(f"HS chapter {digits[:2]} does not exist")
    return digits


def normalize_uom(raw: str) -> str:
    key = re.sub(r"[^a-z]", "", str(raw).strip().lower())
    if key in UOM_ALIASES:
        return UOM_ALIASES[key]
    upper = str(raw).strip().upper()
    if upper in set(UOM_ALIASES.values()):
        return upper
    raise ValidationIssue(f"unknown unit of measure {raw!r}")


def parse_date(raw: str, dayfirst: Optional[bool] = None) -> date:
    """Parse a date string across the formats trade documents actually use.

    ``dayfirst`` disambiguates the numeric-only formats. Left as None, the parser
    prefers day-first (the majority convention outside North America) and accepts the
    reading that produces a valid calendar date.
    """
    text = str(raw).strip()
    order = list(DATE_FORMATS)
    if dayfirst is False:
        order.remove("%m/%d/%Y")
        order.insert(order.index("%d/%m/%Y"), "%m/%d/%Y")
    for fmt in order:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValidationIssue(f"unparseable date {raw!r}")


def parse_amount(raw: str) -> float:
    """Parse a money/quantity string with either thousands convention."""
    if re.search(r"[A-Za-z]", str(raw)):
        # A digit field carrying a letter is an OCR substitution, not a number. Silently
        # stripping it turns "1OO" into 1 and puts a wrong quantity into a customs
        # filing with full confidence; refusing to parse turns it into a validator hit.
        raise ValidationIssue(f"alphabetic character in numeric field {raw!r}")
    text = re.sub(r"[^\d.,\-]", "", str(raw))
    if not text:
        raise ValidationIssue(f"unparseable amount {raw!r}")
    if "," in text and "." in text:
        # Whichever separator appears last is the decimal point.
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        frac = text.rsplit(",", 1)[1]
        text = text.replace(",", "." if len(frac) in (1, 2) else "")
    try:
        return float(text)
    except ValueError as exc:  # pragma: no cover - guarded by the regex above
        raise ValidationIssue(f"unparseable amount {raw!r}") from exc


class LineItem(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    description: str
    hs_code: str
    quantity: float = Field(gt=0)
    uom: str
    unit_price: Optional[float] = Field(default=None, ge=0)
    amount: Optional[float] = Field(default=None, ge=0)

    @field_validator("hs_code", mode="before")
    @classmethod
    def _hs(cls, v: str) -> str:
        return normalize_hs_code(v)

    @field_validator("uom", mode="before")
    @classmethod
    def _uom(cls, v: str) -> str:
        return normalize_uom(v)

    def line_total(self) -> Optional[float]:
        if self.unit_price is None:
            return self.amount
        return round(self.quantity * self.unit_price, 2)


class TradeDocument(BaseModel):
    """Normalised view of any of the three document types."""

    model_config = ConfigDict(str_strip_whitespace=True)

    doc_type: str
    doc_number: str
    doc_date: date
    seller: Optional[str] = None
    buyer: Optional[str] = None
    origin_country: Optional[str] = None
    currency: Optional[str] = None
    incoterm: Optional[str] = None
    incoterm_place: Optional[str] = None
    total_value: Optional[float] = Field(default=None, ge=0)
    gross_weight_kg: Optional[float] = Field(default=None, ge=0)
    net_weight_kg: Optional[float] = Field(default=None, ge=0)
    package_count: Optional[int] = Field(default=None, ge=0)
    vessel: Optional[str] = None
    port_of_loading: Optional[str] = None
    port_of_discharge: Optional[str] = None
    container_no: Optional[str] = None
    line_items: list[LineItem] = Field(default_factory=list)

    @field_validator("doc_type")
    @classmethod
    def _doc_type(cls, v: str) -> str:
        allowed = {"commercial_invoice", "packing_list", "bill_of_lading"}
        if v not in allowed:
            raise ValidationIssue(f"unsupported document type {v!r}")
        return v

    @field_validator("currency", mode="before")
    @classmethod
    def _ccy(cls, v):
        if v is None:
            return v
        code = str(v).strip().upper()
        if code not in ISO_4217:
            raise ValidationIssue(f"{code!r} is not an ISO 4217 code in scope")
        return code

    @field_validator("origin_country", mode="before")
    @classmethod
    def _country(cls, v):
        if v is None:
            return v
        code = str(v).strip().upper()
        if code not in ISO_3166:
            raise ValidationIssue(f"{code!r} is not an ISO 3166-1 alpha-2 code in scope")
        return code

    @field_validator("incoterm", mode="before")
    @classmethod
    def _incoterm(cls, v):
        if v is None:
            return v
        code = str(v).strip().upper()
        if code not in INCOTERMS_2020:
            raise ValidationIssue(f"{code!r} is not an Incoterms 2020 rule")
        return code

    @field_validator("doc_date", mode="before")
    @classmethod
    def _date(cls, v):
        if isinstance(v, (date, datetime)):
            return v
        return parse_date(v)

    def net_le_gross(self) -> bool:
        if self.net_weight_kg is None or self.gross_weight_kg is None:
            return True
        return self.net_weight_kg <= self.gross_weight_kg + 1e-6

    def totals_consistent(self, tol: float = 0.01) -> bool:
        """Do the line amounts add up to the stated invoice total?"""
        if self.total_value is None or not self.line_items:
            return True
        summed = sum(li.line_total() or 0.0 for li in self.line_items)
        if summed <= 0:
            return True
        return abs(summed - self.total_value) <= tol * max(summed, self.total_value)
