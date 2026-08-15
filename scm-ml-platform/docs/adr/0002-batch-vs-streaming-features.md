# ADR-0002: Batch features on a daily cadence

Status: Accepted

## Context

Replenishment decisions are made on a daily or weekly review cycle. Streaming features
would let the forecast react within minutes to an inventory or price change. The
engineering cost is a second computation path with different failure modes, different
correctness semantics for late-arriving data, and a materially harder point-in-time
story: a streamed feature's value at a timestamp depends on what had arrived by then,
which is not reproducible from the historical table unless arrival times are stored.

The counter-argument is real. Some signals -- a price change, a promotion going live,
a stockout -- do move demand within hours, and a daily batch sees them a day late.

The question is whether that day matters given what happens downstream. It does not:
the order is placed on a review cycle measured in days, and the lead time is measured
in weeks. A forecast that is four hours fresher changes no decision.

## Decision

Batch only, daily. Features are computed once per day against a snapshot with a stated
cutoff, and the cutoff is part of the feature spec. Intraday signals are admitted only
as of the previous cutoff.

Revisit when a decision cycle shorter than 24 hours exists downstream -- same-day
fulfilment allocation is the likely first case.

## Consequences

- One computation path, so training and serving can share code and the truncation check
  is meaningful.
- Late-arriving upstream data is handled by re-running a day rather than by reasoning
  about watermarks.
- Genuine intraday signal is lost. A promotion that goes live at 09:00 is invisible to
  the model until the next cutoff, which will show up as error on promotion days.
- The platform cannot serve any use case with a sub-daily decision loop without a
  second architecture. That constraint should be stated to stakeholders early rather
  than discovered by one.
- Backfills and reruns are cheap and idempotent, which makes incident recovery simple.
