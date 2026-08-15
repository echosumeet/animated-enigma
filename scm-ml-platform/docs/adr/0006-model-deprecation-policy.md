# ADR-0006: Time-boxed deprecation with a forced rollback drill

Status: Accepted

## Context

Models accumulate. Old versions stay registered because someone might need them, stale
feature specs stay computed because something might read them, and the fleet grows a
long tail of artefacts nobody owns. The cost is not storage; it is that every contract
change, every dependency bump and every incident review has to consider models that
nobody has looked at in a year.

The opposing risk is real: aggressive deletion removes the version you wanted to roll
back to, at the exact moment you want it. The rollback path is the most valuable thing
the registry provides and it is worthless if the target has been archived away.

The two are reconciled by separating stages. Rollback needs the immediately preceding
production version and a few weeks of history. Audit needs the model card -- what shipped,
trained on what, approved by whom -- which is small and cheap. Neither needs a
two-year-old serialised artefact.

## Decision

- Every model version has a named owner. A version with no owner is deprecated at the
  next review, no exceptions.
- The current production version and the two preceding production versions are always
  retained and always loadable. That is the rollback set.
- Versions outside the rollback set move to `archived` after 90 days without traffic.
  Archived versions retain their model card and metrics permanently; the serialised
  artefact is deleted after 12 months.
- Deprecating a model requires 30 days' notice to named consumers and a stated
  replacement, or an explicit decision that no replacement is needed.
- A rollback drill runs quarterly against the production model. If rollback takes more
  than 15 minutes, the retention window is not the problem and the drill is the finding.

## Consequences

- The registry stays small enough to reason about, and the rollback path is exercised
  often enough to be trusted at 02:00.
- Model cards are permanent, so "what were we serving in March last year and why" is
  always answerable even when the artefact is gone.
- Deleting artefacts after 12 months makes exact reproduction of an old prediction
  impossible. Accepted: the card records the data contract, feature specs and training
  window, which is enough to rebuild an equivalent model but not a bit-identical one.
- The 30-day notice period slows deprecation and will occasionally hold a cleanup for a
  consumer who was not really using the model. That is the price of not breaking one who was.
- Quarterly drills cost engineering time on a schedule, which is the only way they
  actually happen.
