"""Three-way reconciliation across invoice, packing list and bill of lading.

The operational failure this addresses is that each document is usually validated in
isolation, by a different party, at a different time. A packing list that is internally
perfect and disagrees with the invoice on one line quantity will pass every
single-document check and then stop the container at the border. Reconciliation has to
be a first-class step, not a report someone runs afterwards.

Severity is graded on relative deviation because the cost of a discrepancy scales with
how far off it is, with an absolute floor so that rounding noise on a 750-piece line
does not generate a queue item.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .pipeline import DocResult

REL_TOL = 0.005          # below this, treat as rounding
MAJOR_THRESHOLD = 0.02   # above this, a filing-relevant deviation
CRITICAL_THRESHOLD = 0.10


@dataclass
class Discrepancy:
    kind: str
    severity: str
    field: str
    doc_a: str
    doc_b: str
    value_a: Any
    value_b: Any
    rel_delta: Optional[float] = None
    hs_code: Optional[str] = None

    def as_row(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "severity": self.severity, "field": self.field,
            "docs": f"{self.doc_a}/{self.doc_b}", "a": self.value_a, "b": self.value_b,
            "rel_delta": None if self.rel_delta is None else round(self.rel_delta, 4),
            "hs_code": self.hs_code,
        }


def _severity(rel: float) -> str:
    if rel >= CRITICAL_THRESHOLD:
        return "critical"
    if rel >= MAJOR_THRESHOLD:
        return "major"
    return "minor"


def _compare(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Relative deviation, or None when either side is missing or both are zero."""
    if a is None or b is None:
        return None
    denom = max(abs(float(a)), abs(float(b)))
    if denom == 0:
        return None
    return abs(float(a) - float(b)) / denom


def _table_readable(res: Optional[DocResult], min_conf: float) -> bool:
    """Only reconcile tables we actually read.

    Comparing a table that contains a cell the extractor rejected produces an alert
    about the OCR, dressed up as an alert about the shipment. Those false alerts are
    what kill trust in an exception queue, so a low-confidence table is routed to review
    instead of being reconciled.
    """
    if res is None or not res.line_items or res.line_confidence < min_conf:
        return False
    return all(row.get("hs_code") is not None for row in res.line_items)


def reconcile(results: dict[str, DocResult], min_table_conf: float = 0.9
              ) -> list[Discrepancy]:
    """``results`` maps doc_type -> DocResult for a single shipment."""
    inv = results.get("commercial_invoice")
    pkl = results.get("packing_list")
    bol = results.get("bill_of_lading")
    found: list[Discrepancy] = []

    # 1. Line quantities, invoice vs packing list, matched on HS code.
    if _table_readable(inv, min_table_conf) and _table_readable(pkl, min_table_conf):
        pkl_by_hs: dict[str, float] = {}
        for row in pkl.line_items:
            if row.get("hs_code") and row.get("quantity") is not None:
                pkl_by_hs[row["hs_code"]] = pkl_by_hs.get(row["hs_code"], 0.0) + row["quantity"]
        inv_by_hs: dict[str, float] = {}
        for row in inv.line_items:
            if row.get("hs_code") and row.get("quantity") is not None:
                inv_by_hs[row["hs_code"]] = inv_by_hs.get(row["hs_code"], 0.0) + row["quantity"]
        for hs in sorted(set(inv_by_hs) | set(pkl_by_hs)):
            a, b = inv_by_hs.get(hs), pkl_by_hs.get(hs)
            if a is None or b is None:
                found.append(Discrepancy(
                    "line_item_missing", "critical", "line_items", "commercial_invoice",
                    "packing_list", a, b, None, hs))
                continue
            rel = _compare(a, b)
            if rel is not None and rel > REL_TOL:
                found.append(Discrepancy(
                    "quantity_mismatch", _severity(rel), "quantity", "commercial_invoice",
                    "packing_list", a, b, rel, hs))

    # 2. Gross weight, packing list vs bill of lading.
    if pkl and bol:
        a = pkl.fields["gross_weight_kg"].value
        b = bol.fields["gross_weight_kg"].value
        rel = _compare(a, b)
        if rel is not None and rel > REL_TOL:
            found.append(Discrepancy("gross_weight_mismatch", _severity(rel),
                                     "gross_weight_kg", "packing_list", "bill_of_lading",
                                     a, b, rel))
        # 3. Package count.
        pa = pkl.fields["package_count"].value
        pb = bol.fields["package_count"].value
        if pa is not None and pb is not None and pa != pb:
            rel = _compare(pa, pb) or 0.0
            found.append(Discrepancy("package_count_mismatch", _severity(rel),
                                     "package_count", "packing_list", "bill_of_lading",
                                     pa, pb, rel))

    # 4. Invoice total vs the sum of its own line amounts. Internal, but it is the check
    #    that most often exposes a mis-keyed unit price.
    if inv and inv.line_items and all(r.get("amount") is not None for r in inv.line_items):
        total = inv.fields["total_value"].value
        summed = sum(r["amount"] for r in inv.line_items)
        rel = _compare(total, summed) if summed > 0 else None
        if rel is not None and rel > REL_TOL:
            found.append(Discrepancy("invoice_total_mismatch", _severity(rel),
                                     "total_value", "commercial_invoice",
                                     "commercial_invoice", total, round(summed, 2), rel))

    # 5. Net weight must not exceed gross.
    if pkl:
        net = pkl.fields["net_weight_kg"].value
        gross = pkl.fields["gross_weight_kg"].value
        if net is not None and gross is not None and net > gross:
            found.append(Discrepancy("net_exceeds_gross", "critical", "net_weight_kg",
                                     "packing_list", "packing_list", net, gross,
                                     _compare(net, gross)))
    return found


def evaluate_detection(seeded: dict[str, list[dict]],
                       detected: dict[str, list[Discrepancy]],
                       min_severity: str = "major") -> dict[str, Any]:
    """Score reconciliation against the discrepancies the generator actually seeded.

    Only ``major`` and above count as an alert, because a system that fires on every
    rounding difference is one nobody looks at after week two.
    """
    order = {"minor": 0, "major": 1, "critical": 2}
    floor = order[min_severity]
    tp = fn = fp = 0
    by_kind: dict[str, list[int]] = {}
    for shipment_id, seeds in seeded.items():
        alerts = [d for d in detected.get(shipment_id, []) if order[d.severity] >= floor]
        kinds = {d.kind for d in alerts}
        seeded_kinds = {s["kind"] for s in seeds}
        for kind in seeded_kinds:
            hit = int(kind in kinds)
            by_kind.setdefault(kind, [0, 0])
            by_kind[kind][0] += hit
            by_kind[kind][1] += 1
            tp += hit
            fn += 1 - hit
        fp += len(kinds - seeded_kinds)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "true_positives": tp, "false_negatives": fn, "false_positives": fp,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "by_kind": {k: {"detected": v[0], "seeded": v[1],
                        "rate": round(v[0] / v[1], 4) if v[1] else 0.0}
                    for k, v in sorted(by_kind.items())},
    }
