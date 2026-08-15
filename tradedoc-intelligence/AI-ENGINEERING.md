# AI Engineering skills map — `tradedoc-intelligence`

This repository extracts fields from commercial invoices, packing lists and bills of
lading, reconciles the three documents of a shipment against each other, and reports a
straight-through-processing (STP) curve: at an agreed escaped-error budget, what share
of documents clear with no human touching them. It generates its own corpus, so ground
truth is exact at field level and every accuracy number is recomputed rather than
annotated.

Mapped against Andrew Ng's AI Engineering Skills Map (14 Aug 2026), this repository is
strongest on **skill 1 (building AI applications)** — a model in the running product
behind a provider protocol, with an eval harness and a governance threshold on top — and
on **skill 2 (software engineering fundamentals)**, where the cost and reliability
tradeoffs around that model are the design. **Skill 3** appears in the build process, not
the product. **Skill 4 (shaping the build)** shows up as one scope cut and one choice of
headline metric, and that section is short rather than inflated.

## Disclosure

The code in this repository was written by Claude subagents on 2026-08-15, in a single
session, under my direction. I set the goal, the constraints and the metric; the agents
wrote the design notes, the generator, the extractor, the reconciler, the tests, the
benchmark harness and the figure. This document maps both the product and that build
process against the skills map. The repository has **40 tests, all passing**
(`PYTHONPATH=src python -m unittest discover -s tests`). Nothing here has run in
production, at scale, or at any company.

## The four skills, as they actually appeared

### 1. Building and deploying AI applications

`src/tradedoc/llm.py` defines an `LLMProvider` protocol with one method,
`extract_field(text, field, doc_type)`. Three implementations sit behind it:
`AnthropicProvider` and `OpenAIProvider`, which import their SDKs lazily inside the call,
and `StubProvider`, a deterministic offline keyword-proximity reader that every test and
benchmark here runs against. The stub hashes content rather than drawing from an RNG, so
repeated runs are byte-identical.

The architecture is rules first, model on the residual. `src/tradedoc/pipeline.py` runs
label-anchored regex and table-layout matching, then asks the provider only for fields
the rules did not resolve. Per-field confidence is an additive rule over three observable
signals — did each pass return a value, do they agree after normalisation, does the value
pass its validator (`W_BASE = 0.35`, `W_RULE_OK = 0.53`, `W_AGREE = 0.06`,
`W_DISAGREE = -0.33`, `W_RULE_INVALID = -0.30`). No model is fitted, so an operator
asking why a document was held gets an answer in those terms.

The eval discipline is the point of the repository. On 200 shipments = 600 documents at
seed 7, macro field accuracy is 97.9% / 98.6% / 98.0% for invoice, packing list and bill
of lading, and **document-level accuracy — every field and every line item correct — is
78.7%** (`benchmarks/results.md` §1). That gap is why the headline is the sweep in
`src/tradedoc/sweep.py`, which reports STP rate, precision, recall and escaped error at
each confidence threshold. At a 5% escaped-error budget the operating point is t=0.82,
clearing **46.2% of documents untouched**. A 2% or 3% budget is **infeasible at every
threshold** on this corpus, because the residual errors are OCR substitutions that leave
a syntactically valid value behind (`INV-O0004` for `INV-00004`), and no signal inside a
single document separates them. That negative result was left in the README rather than
tuned away.

Two model-behaviour findings the harness produced:

| second-pass policy | LLM calls/doc | STP @ 5% budget |
|---|---|---|
| `none` (residual only) | 0.53 | 46.7% @ t=0.82 |
| `high_value` | 2.68 | 46.2% @ t=0.82 |
| `all` (confirm every field) | 7.33 | 45.8% @ t=0.82 |

Fourteen times the calls buys marginally less. The second pass reads the same corrupted
glyphs the rules read, so it agrees with them for the same wrong reason; confirmation
only pays where the two passes have independent views. Separately, the stub hallucinates
on blank fields, and
`tests/test_extraction.py::test_omitted_field_is_never_confidently_invented` asserts the
invariant that matters: such a value can never carry rule-level confidence, so it can
never be auto-approved.

### 2. Software engineering fundamentals

The tradeoffs are explicit and measured, not asserted.

- **Cost/reliability**: the table above is a cost decision made with numbers.
- **Precision/recall in the exception queue**: reconciliation is suppressed on tables the
  extractor could not read cleanly (`_table_readable`, `min_table_conf=0.9` in
  `src/tradedoc/reconcile.py`). That gate moves discrepancy precision from 0.37 to 0.76
  for roughly ten points of recall. Final detection is precision 0.756, recall 0.838 —
  31 caught, 6 missed, 10 false alerts.
- **Failing loud over failing silent**: `parse_amount` in `src/tradedoc/schemas.py`
  refuses a numeric field containing a letter instead of stripping it. Stripping turns
  `1OO` into `1` and files a wrong quantity at full confidence.
- **Dependency discipline**: the container had no PyPI access, so validators are pydantic
  models written against the published standards (WCO HS structure, Incoterms 2020,
  ISO 4217, ISO 3166-1 alpha-2, ISO 6346, UN/ECE Rec. 20). Tests are stdlib `unittest`.

The 40 tests cover validators including OCR substitution in HS codes
(`tests/test_schemas.py`), determinism and absence handling
(`tests/test_extraction.py`), and severity grading, the unreadable-table gate and
monotonicity of STP in the threshold (`tests/test_reconcile_sweep.py`).

### 3. Using coding agents

This is a build-process skill here, not a product feature. What made it work was a
shared `CONVENTIONS.md` handed to every subagent as the context contract, and verifiers
the agent had to run rather than claim: the unittest suite, `benchmarks/run_benchmarks.py`,
and `examples/broker_intake.py`. The README had to be written from real output;
fabricating a number was prohibited. That closed the loop — the agent could not declare
done without producing a `benchmarks/results.md` that a later reader could regenerate. I
re-ran the harness while writing this document and every table reproduced exactly; only
the wall-clock line moved (results.md records 1.2s, my rerun 1.9s on this container).

A separate audit agent, which wrote none of the code, re-ran all ten repositories and
checked headline numbers against benchmark output. It caught one honesty defect elsewhere
in the portfolio — a README claiming an 11-second pipeline whose benchmark said 4.2s.
That is the argument for a verifier that is not the author.

### 4. Shaping the build

Two real decisions. First, choosing the metric: field accuracy is what document-AI
vendors sell, and at 97.9% per field the invoice is fully correct 78.7% of the time, so
the spec named STP-at-an-error-budget as the headline instead. Second, a scope cut. The
first run of the portfolio overshot its line budget badly and projected to about three
hours on two effective cores; the run was stopped and the remainder relaunched under a
hard budget. This repository lost two document types (certificate of origin, customs
declaration), duty and landed-cost estimation, and the HTTP serving layer. What was
protected was the generator-truth loop and the sweep, because those carry the argument.

## AI during development vs AI in the product

| AI during development | AI in the running product |
|---|---|
| Claude subagents wrote all code, tests, benchmarks and figure on 2026-08-15 | `LLMProvider` second pass on fields the rules cannot resolve (`src/tradedoc/llm.py`) |
| An audit agent re-ran the suite and checked README numbers against benchmark output | Per-field confidence from rule/LLM agreement plus validator outcome (`pipeline.py`) |
| Human direction set constraints, metric and the scope cut | Confidence-threshold governance: the operating point at a fixed error budget (`sweep.py`) |

Both columns are populated here, which is not true of the classical repositories in this
portfolio.

## What I would do differently

1. **Add cross-document field voting.** The irreducible errors are single-document-
   invisible. The same buyer, vessel and port appear on all three documents; voting
   across them should recover most of what the 2% budget currently cannot reach. This is
   the highest-value missing feature and it is not in the code.
2. **Measure on OCR of the PDFs, not the text rendering.** The per-character corruption
   model at 0.006 is calibrated to nothing. Sensitivity is steep — at 0.012 the 5% budget
   becomes infeasible entirely (`benchmarks/results.md` §5).
3. **Derive the severity thresholds from cost.** 0.5% tolerance, 2% major and 10%
   critical are defensible defaults, not an expected-cost decision against the real cost
   of a held container versus a customs penalty.
4. **Learn the confidence weights.** They are constants in `pipeline.py`, tuned once on
   the calibration corpus and frozen. A fitted calibrator would likely beat them, at the
   cost of the explanation an operator gets today.
5. **Widen layout diversity.** Three variants is far below real intake; rule coverage
   will degrade faster than these numbers suggest.

## Takeaways

- **(Skill 1)** Field accuracy is not decision-relevant. 97.9% per field and 78.7% per
  document are the same system; only the second number tells you how much human work
  remains.
- **(Skill 1)** A confidence threshold at a fixed error budget is a governance mechanism,
  not a model score — and reporting "2% is infeasible" is a more useful output than a
  tuned number that hides it.
- **(Skill 2)** More model calls is a testable hypothesis, not an improvement: 7.33
  calls/doc scored 0.9 points of STP *worse* than 0.53.
- **(Skill 2)** Suppressing alerts you cannot substantiate is worth losing recall for —
  precision 0.37 to 0.76 is the difference between a queue analysts use and one they stop
  opening.
- **(Skill 3)** Agents are trustworthy in proportion to the verifiers you give them. The
  binding constraint was "the README must be written from real output", not the prompt.

## How to explore this repo

1. `docs/design.md` — why the repository generates its own documents, and the three
   separated noise mechanisms.
2. `benchmarks/results.md` — every number in this document, regenerable with
   `PYTHONPATH=src python benchmarks/run_benchmarks.py`.
3. `src/tradedoc/sweep.py` — the STP curve and the operating-point rule.
4. `src/tradedoc/pipeline.py` — the confidence weights and the `confirm` policy.
5. `tests/test_extraction.py` — the absence-is-an-answer invariant.
6. `examples/broker_intake.py` — one shipment end to end, offline.
