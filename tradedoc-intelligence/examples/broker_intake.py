"""End-to-end intake for one shipment: generate three documents, extract, reconcile,
and apply an auto-approve/review decision at a chosen confidence threshold.

    PYTHONPATH=src python examples/broker_intake.py

Writes a small corpus (text + ground-truth JSON + PDFs) to ``example_out/`` so the
rendered documents can be inspected alongside the extraction they produced.
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tradedoc.generate import Corpus, NoiseConfig, generate_documents, make_shipment, write_corpus  # noqa: E402
from tradedoc.pipeline import document_correct, extract_document  # noqa: E402
from tradedoc.reconcile import reconcile  # noqa: E402

THRESHOLD = 0.82
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "example_out")


def main() -> int:
    # Seed chosen so the shipment carries a real cross-document discrepancy and a
    # mix of auto-approve and review outcomes.
    rng = random.Random(6)
    shipment = make_shipment(rng, 1)
    docs, seeded = generate_documents(rng=rng, sh=shipment,
                                      noise=NoiseConfig(inconsistency_rate=1.0))

    print(f"Shipment {shipment.shipment_id}: {shipment.seller} ({shipment.origin_country})"
          f" -> {shipment.buyer}")
    print(f"  {shipment.incoterm} {shipment.incoterm_place}, {shipment.currency} "
          f"{shipment.total_value():,.2f}, {len(shipment.lines)} line items")
    print(f"  seeded defect: {seeded[0]['kind'] if seeded else 'none'}\n")

    results = {}
    for doc in docs:
        res = extract_document(doc)
        results[doc.doc_type] = res
        conf = res.min_confidence()
        decision = "AUTO-APPROVE" if conf >= THRESHOLD else "REVIEW"
        print(f"{doc.doc_type:<20} confidence {conf:.2f}  {decision:<12} "
              f"llm calls {res.llm_calls}  correct={document_correct(res, doc.truth)}")
        weakest = min(res.fields.values(), key=lambda f: f.confidence)
        print(f"{'':<20} weakest field: {weakest.name}={weakest.value!r} "
              f"({weakest.confidence:.2f}, source={weakest.source})")

    print("\nThree-way reconciliation:")
    found = reconcile(results)
    if not found:
        print("  no discrepancies above tolerance")
    for d in found:
        delta = "" if d.rel_delta is None else f", {100 * d.rel_delta:.1f}% apart"
        print(f"  [{d.severity:<8}] {d.kind} on {d.field} "
              f"({d.doc_a} {d.value_a} vs {d.doc_b} {d.value_b}{delta})")

    written = write_corpus(Corpus([shipment], docs, {shipment.shipment_id: seeded}),
                           OUTDIR, pdf=True)
    print(f"\nwrote {written} documents (text, PDF, ground-truth JSON) to "
          f"{os.path.normpath(OUTDIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
