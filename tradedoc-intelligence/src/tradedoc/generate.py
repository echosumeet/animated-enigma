"""Synthetic trade-document generator with ground truth.

Every document this module emits comes with the field values that produced it, so
extraction accuracy is measurable rather than eyeballed. Three noise mechanisms are
modelled separately because they fail differently in operations:

1. *Cross-document inconsistency* -- the shipment really does have a quantity on the
   packing list that disagrees with the invoice. Ground truth for that document
   reflects the wrong value; reconciliation is supposed to catch it.
2. *Field omission* -- the field was never printed. Ground truth becomes ``None``;
   the correct extraction is "absent", not a guess.
3. *Character corruption* -- OCR-style substitutions applied to the rendered text only.
   Ground truth is untouched, so a corrupted field is an extraction error.

Mixing these three into one "noise level" knob is what makes vendor accuracy claims
unreproducible, so they are separately parameterised here.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Optional

from .schemas import INCOTERMS_2020, ISO_3166, ISO_4217

DOC_TYPES = ("commercial_invoice", "packing_list", "bill_of_lading")

VENDORS = [
    ("Rhein Kunststoff GmbH", "DE"), ("Kanto Seiki Co Ltd", "JP"),
    ("Nanhai Precision Works", "CN"), ("Deccan Forge Industries", "IN"),
    ("Straits Component Pte Ltd", "SG"), ("Bajio Metalworks SA de CV", "MX"),
    ("Mekong Textile JSC", "VN"), ("Thames Instrumentation Ltd", "GB"),
]
BUYERS = [
    "Northline Distribution Inc", "Vantage Retail Group", "Harbour Point Trading BV",
    "Cascade Industrial Supply", "Meridian Wholesale Ltd",
]
PORTS = [
    ("Hamburg", "DE"), ("Yokohama", "JP"), ("Shenzhen", "CN"), ("Nhava Sheva", "IN"),
    ("Singapore", "SG"), ("Rotterdam", "NL"), ("Los Angeles", "US"), ("Laem Chabang", "TH"),
]
VESSELS = ["MV Corallia", "MV Northern Kestrel", "MV Anseong Bay", "MV Ligurian Star"]
GOODS = [
    ("Injection moulded housing, ABS", "392690", "PCE", 2.4, 0.18),
    ("Ball bearing, deep groove 6204", "848210", "PCE", 1.9, 0.11),
    ("Aluminium extrusion profile", "760810", "MTR", 7.8, 0.95),
    ("Woven polyester fabric, 180gsm", "540752", "MTR", 3.1, 0.19),
    ("Stainless fastener assortment", "731815", "KGM", 6.5, 1.0),
    ("Printed circuit assembly", "853400", "PCE", 18.4, 0.05),
    ("Rubber gasket set", "401693", "SET", 4.2, 0.07),
    ("Cold rolled steel sheet", "720917", "KGM", 1.15, 1.0),
]

OCR_SUBS = {
    "0": "O", "O": "0", "1": "l", "l": "1", "5": "S", "S": "5", "8": "B", "B": "8",
    "2": "Z", "Z": "2", "6": "G", "G": "6",
}


@dataclass
class NoiseConfig:
    """Independent knobs for the three failure modes described in the module docstring."""

    char_corruption_rate: float = 0.006   # per-character substitution probability
    field_dropout_rate: float = 0.07      # probability a printed field is omitted
    inconsistency_rate: float = 0.18      # probability a shipment carries a real discrepancy


@dataclass
class Shipment:
    shipment_id: str
    seller: str
    origin_country: str
    buyer: str
    currency: str
    incoterm: str
    incoterm_place: str
    ship_date: date
    vessel: str
    port_of_loading: str
    port_of_discharge: str
    container_no: str
    lines: list[dict[str, Any]] = field(default_factory=list)

    def total_value(self) -> float:
        return round(sum(li["amount"] for li in self.lines), 2)

    def net_weight(self) -> float:
        return round(sum(li["quantity"] * li["kg_per_unit"] for li in self.lines), 1)

    def gross_weight(self) -> float:
        return round(self.net_weight() * 1.08 + 12.0, 1)

    def package_count(self) -> int:
        return max(1, int(sum(li["quantity"] for li in self.lines) // 40) + len(self.lines))


@dataclass
class GeneratedDoc:
    doc_type: str
    shipment_id: str
    text: str
    truth: dict[str, Any]
    layout: int
    dayfirst: bool


def make_shipment(rng: random.Random, idx: int) -> Shipment:
    seller, origin = rng.choice(VENDORS)
    pol = rng.choice([p for p in PORTS if p[1] == origin] or PORTS)
    pod = rng.choice([p for p in PORTS if p[1] != origin])
    n_lines = rng.randint(2, 4)
    lines = []
    for desc, hs, uom, price, kg in rng.sample(GOODS, n_lines):
        qty = float(rng.choice([25, 40, 60, 100, 120, 240, 500, 750]))
        unit_price = round(price * rng.uniform(0.85, 1.2), 2)
        lines.append({
            "description": desc, "hs_code": hs, "uom": uom, "quantity": qty,
            "unit_price": unit_price, "amount": round(qty * unit_price, 2),
            "kg_per_unit": kg,
        })
    return Shipment(
        shipment_id=f"SHP-{idx:05d}",
        seller=seller,
        origin_country=origin,
        buyer=rng.choice(BUYERS),
        currency=rng.choice(ISO_4217[:6]),
        incoterm=rng.choice(INCOTERMS_2020),
        incoterm_place=pol[0],
        ship_date=date(2026, 1, 1) + timedelta(days=rng.randint(0, 300)),
        vessel=rng.choice(VESSELS),
        port_of_loading=pol[0],
        port_of_discharge=pod[0],
        container_no=f"{rng.choice(['MSKU', 'TGHU', 'CMAU', 'HLXU'])}{rng.randint(1000000, 9999999)}",
        lines=lines,
    )


def _fmt_date(d: date, layout: int) -> tuple[str, bool]:
    """Return the rendered date and whether the format is day-first."""
    if layout == 0:
        return d.strftime("%Y-%m-%d"), True
    if layout == 1:
        return d.strftime("%d-%b-%Y"), True
    return d.strftime("%m/%d/%Y"), False


def _corrupt(text: str, rate: float, rng: random.Random) -> str:
    if rate <= 0:
        return text
    out = []
    for ch in text:
        if ch in OCR_SUBS and rng.random() < rate:
            out.append(OCR_SUBS[ch])
        else:
            out.append(ch)
    return "".join(out)


def _maybe_drop(truth: dict, keys: list[str], rate: float, rng: random.Random) -> None:
    for key in keys:
        if rng.random() < rate:
            truth[key] = None


def _money(v: float, layout: int) -> str:
    return f"{v:,.2f}" if layout != 2 else f"{v:.2f}"


def _render_invoice(sh: Shipment, truth: dict, layout: int) -> str:
    ds = truth["_date_str"]
    head = [
        f"{sh.seller}", f"Country of Origin: {truth.get('origin_country') or ''}", "",
    ]
    if layout == 0:
        head += ["COMMERCIAL INVOICE", f"Invoice No: {truth['doc_number'] or ''}",
                 f"Invoice Date: {ds if truth['doc_date'] else ''}"]
    elif layout == 1:
        head += ["*** COMMERCIAL INVOICE ***",
                 f"Document Number . . . {truth['doc_number'] or ''}",
                 f"Issued . . . . . . . . {ds if truth['doc_date'] else ''}"]
    else:
        head += ["Commercial Invoice",
                 f"INV#{truth['doc_number'] or ''}   Dated {ds if truth['doc_date'] else ''}"]
    body = [
        "", f"Sold To: {truth.get('buyer') or ''}",
        f"Currency: {truth.get('currency') or ''}",
        f"Terms of Delivery: {(truth.get('incoterm') or '')} {sh.incoterm_place}".rstrip(),
        "", "HS Code       Description                          Qty      UOM    Unit Price      Amount",
        "-" * 92,
    ]
    for li in truth["line_items"]:
        body.append(
            f"{li['hs_code']:<13} {li['description'][:34]:<34} {li['quantity']:>8.0f}"
            f"   {li['uom']:<5} {_money(li['unit_price'], layout):>11} {_money(li['amount'], layout):>13}"
        )
    body.append("-" * 92)
    tv = truth.get("total_value")
    body.append(f"TOTAL {truth.get('currency') or ''} {_money(tv, layout) if tv is not None else ''}")
    body.append("")
    body.append("We certify the above information is true and correct.")
    return "\n".join(head + body)


def _render_packing_list(sh: Shipment, truth: dict, layout: int) -> str:
    ds = truth["_date_str"]
    title = ["PACKING LIST", "P A C K I N G   L I S T", "Packing List / Weight Note"][layout]
    lines = [
        sh.seller, "", title,
        f"Packing List No: {truth['doc_number'] or ''}",
        f"Date: {ds if truth['doc_date'] else ''}",
        f"Consignee: {truth.get('buyer') or ''}", "",
        "HS Code       Description                          Qty      UOM",
        "-" * 70,
    ]
    for li in truth["line_items"]:
        lines.append(
            f"{li['hs_code']:<13} {li['description'][:34]:<34} {li['quantity']:>8.0f}   {li['uom']:<5}"
        )
    lines += [
        "-" * 70,
        f"Total Packages: {truth['package_count'] if truth['package_count'] is not None else ''}",
        f"Net Weight: {truth['net_weight_kg'] if truth['net_weight_kg'] is not None else ''} KGS",
        f"Gross Weight: {truth['gross_weight_kg'] if truth['gross_weight_kg'] is not None else ''} KGS",
    ]
    return "\n".join(lines)


def _render_bol(sh: Shipment, truth: dict, layout: int) -> str:
    ds = truth["_date_str"]
    head = ["BILL OF LADING", "ORIGINAL BILL OF LADING", "Sea Waybill / Bill of Lading"][layout]
    lines = [
        head,
        f"B/L No: {truth['doc_number'] or ''}",
        f"Date of Issue: {ds if truth['doc_date'] else ''}", "",
        f"Shipper: {sh.seller}",
        f"Consignee: {truth.get('buyer') or ''}", "",
        f"Vessel: {truth.get('vessel') or ''}",
        f"Port of Loading: {truth.get('port_of_loading') or ''}",
        f"Port of Discharge: {truth.get('port_of_discharge') or ''}",
        f"Container No: {truth.get('container_no') or ''}", "",
        "Marks and Numbers / Description of Goods",
        "-" * 70,
    ]
    for li in truth["line_items"]:
        lines.append(f"  {li['description'][:44]:<44} HS {li['hs_code']}")
    lines += [
        "-" * 70,
        f"Number of Packages: {truth['package_count'] if truth['package_count'] is not None else ''}",
        f"Gross Weight: {truth['gross_weight_kg'] if truth['gross_weight_kg'] is not None else ''} KGS",
        "SHIPPED on board in apparent good order and condition.",
    ]
    return "\n".join(lines)


def _base_truth(sh: Shipment, doc_type: str, prefix: str, layout: int) -> dict:
    ds, dayfirst = _fmt_date(sh.ship_date, layout)
    truth: dict[str, Any] = {
        "doc_type": doc_type,
        "doc_number": f"{prefix}-{sh.shipment_id[4:]}",
        "doc_date": sh.ship_date.isoformat(),
        "_date_str": ds,
        "_dayfirst": dayfirst,
        "buyer": sh.buyer,
        "line_items": [
            {k: li[k] for k in ("description", "hs_code", "uom", "quantity", "unit_price", "amount")}
            for li in sh.lines
        ],
    }
    return truth


def generate_documents(
    sh: Shipment, rng: random.Random, noise: NoiseConfig
) -> tuple[list[GeneratedDoc], list[dict]]:
    """Render all three documents for one shipment.

    Returns the documents and the list of *seeded* discrepancies, which is the ground
    truth the reconciliation evaluation is scored against.
    """
    layout = rng.randint(0, 2)
    seeded: list[dict] = []

    inv = _base_truth(sh, "commercial_invoice", "INV", layout)
    inv.update({
        "seller": sh.seller, "origin_country": sh.origin_country, "currency": sh.currency,
        "incoterm": sh.incoterm, "total_value": sh.total_value(),
    })
    pkl = _base_truth(sh, "packing_list", "PL", layout)
    pkl.update({
        "package_count": sh.package_count(), "net_weight_kg": sh.net_weight(),
        "gross_weight_kg": sh.gross_weight(),
    })
    bol = _base_truth(sh, "bill_of_lading", "BOL", layout)
    bol.update({
        "vessel": sh.vessel, "port_of_loading": sh.port_of_loading,
        "port_of_discharge": sh.port_of_discharge, "container_no": sh.container_no,
        "package_count": sh.package_count(), "gross_weight_kg": sh.gross_weight(),
    })

    # (1) cross-document inconsistencies: real data defects, not reading errors.
    if rng.random() < noise.inconsistency_rate:
        kind = rng.choice(["qty", "weight", "packages", "value"])
        if kind == "qty":
            i = rng.randrange(len(pkl["line_items"]))
            pkl["line_items"][i]["quantity"] = round(
                pkl["line_items"][i]["quantity"] * rng.choice([0.5, 0.9, 1.1, 2.0]), 0)
            seeded.append({"kind": "quantity_mismatch", "hs_code": pkl["line_items"][i]["hs_code"]})
        elif kind == "weight":
            bol["gross_weight_kg"] = round(bol["gross_weight_kg"] * rng.choice([0.82, 1.15]), 1)
            seeded.append({"kind": "gross_weight_mismatch", "hs_code": None})
        elif kind == "packages":
            bol["package_count"] = max(1, bol["package_count"] + rng.choice([-3, -1, 2, 5]))
            seeded.append({"kind": "package_count_mismatch", "hs_code": None})
        else:
            inv["total_value"] = round(inv["total_value"] * rng.choice([0.93, 1.07]), 2)
            seeded.append({"kind": "invoice_total_mismatch", "hs_code": None})

    docs = []
    renderers = {
        "commercial_invoice": (_render_invoice, inv,
                               ["doc_number", "doc_date", "buyer", "currency", "incoterm",
                                "origin_country", "total_value"]),
        "packing_list": (_render_packing_list, pkl,
                         ["doc_number", "doc_date", "buyer", "package_count",
                          "net_weight_kg", "gross_weight_kg"]),
        "bill_of_lading": (_render_bol, bol,
                           ["doc_number", "doc_date", "buyer", "vessel", "port_of_loading",
                            "port_of_discharge", "container_no", "package_count",
                            "gross_weight_kg"]),
    }
    for doc_type, (fn, truth, droppable) in renderers.items():
        # (2) omission: the field is not on the page, so "absent" is the right answer.
        _maybe_drop(truth, droppable, noise.field_dropout_rate, rng)
        text = fn(sh, truth, layout)
        # (3) OCR corruption: rendered text only.
        text = _corrupt(text, noise.char_corruption_rate, rng)
        clean = {k: v for k, v in truth.items() if not k.startswith("_")}
        docs.append(GeneratedDoc(doc_type, sh.shipment_id, text, clean, layout,
                                 truth["_dayfirst"]))
    return docs, seeded


@dataclass
class Corpus:
    shipments: list[Shipment]
    docs: list[GeneratedDoc]
    seeded: dict[str, list[dict]]

    def by_shipment(self, shipment_id: str) -> dict[str, GeneratedDoc]:
        return {d.doc_type: d for d in self.docs if d.shipment_id == shipment_id}


def build_corpus(n_shipments: int = 120, seed: int = 7,
                 noise: Optional[NoiseConfig] = None) -> Corpus:
    rng = random.Random(seed)
    noise = noise or NoiseConfig()
    shipments, docs, seeded = [], [], {}
    for i in range(n_shipments):
        sh = make_shipment(rng, i)
        d, s = generate_documents(sh, rng, noise)
        shipments.append(sh)
        docs.extend(d)
        seeded[sh.shipment_id] = s
    return Corpus(shipments, docs, seeded)


def write_corpus(corpus: Corpus, outdir: str, pdf: bool = False) -> int:
    """Write text renderings plus ground-truth JSON (and optionally PDFs) to disk."""
    import os

    os.makedirs(outdir, exist_ok=True)
    written = 0
    for doc in corpus.docs:
        stem = os.path.join(outdir, f"{doc.shipment_id}_{doc.doc_type}")
        with open(stem + ".txt", "w", encoding="utf-8") as fh:
            fh.write(doc.text)
        with open(stem + ".truth.json", "w", encoding="utf-8") as fh:
            json.dump(doc.truth, fh, indent=2, sort_keys=True)
        written += 1
        if pdf:
            from .pdfout import write_pdf

            write_pdf(doc, stem + ".pdf")
    return written
