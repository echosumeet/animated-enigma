# Design notes

## Why the repository generates its own documents

Extraction accuracy claims are unfalsifiable without ground truth, and real trade
documents cannot be published. So the generator is the reference: one shipment record
produces a commercial invoice, a packing list and a bill of lading, and each rendered
document ships with the exact field values that produced it. Every number in the README
is recomputed from that record, not annotated by hand.

The generator separates three noise mechanisms that vendor benchmarks usually collapse
into one "difficulty" knob, because they fail differently and are fixed differently:

- **Character corruption** (OCR substitution) — the value is on the page and read wrong.
  Ground truth is unchanged, so this is an extraction error.
- **Field omission** — the value was never printed. Ground truth becomes `None` and the
  correct answer is "absent". A system that guesses here is worse than one that stops.
- **Cross-document inconsistency** — the shipment genuinely has a packing list that
  disagrees with the invoice. Ground truth for that document reflects the wrong value;
  this is reconciliation's job, not extraction's.

Reporting a single accuracy figure over all three hides which of the three you actually
have.

## Rules first, model on the residual

The deterministic pass is label-anchored regex plus table-row layout matching. It is
cheap, auditable and correct on most fields. The LLM pass runs behind a thin
`LLMProvider` protocol, and the default `StubProvider` is offline and deterministic — it
is a keyword-proximity reader that does *no* validation, deliberately different in kind
from the rules. That difference is what makes agreement between the two passes carry
information. If the second pass were a copy of the first, the confidence score built on
their agreement would be a constant.

Confidence is an additive rule over three observable signals: did each pass return
anything, do they agree after normalisation, does the value pass its validator. No
model is fitted. An operator asking why a document was held gets an answer in those
terms.

## Validation as a confidence signal, not a gate

The validators (HS chapter and subheading length, Incoterms 2020, ISO 4217, ISO 3166,
UN/ECE units, multi-format dates, ISO 6346 container numbers) exist mainly so that a
corruption becomes *visible*. `parse_amount` refusing a numeric field that contains a
letter is the clearest case: stripping the letter turns `1OO` into `1` and files a wrong
quantity with full confidence, whereas refusing turns it into a low-confidence field
that lands in review. Converting silent errors into loud ones is most of the value here.

## Reconciliation only reads what it could read

Comparing a table containing a cell the extractor rejected produces an alert about the
OCR, dressed up as an alert about the shipment. Tables below the confidence floor are
routed to review instead of reconciled. On the benchmark corpus that gate moves
discrepancy precision from 0.37 to 0.76 and costs roughly ten points of recall — the
right trade, because an exception queue nobody trusts is not an exception queue.
