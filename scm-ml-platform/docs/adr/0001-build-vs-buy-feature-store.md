# ADR-0001: Build point-in-time feature specs rather than buy a feature store

Status: Accepted

## Context

Feature reuse across forecasting models is worth having, and training/serving skew is
the bug that most often survives review. Both are the stated value proposition of a
managed feature store. A procurement path exists.

Two things make buying a poor fit at our current size. First, the failure we actually
experience is not "the same feature was computed twice"; it is "the feature was computed
correctly and read data that did not exist at decision time". A feature store guarantees
the former and, in most products, only guarantees the latter if the team defines its
event timestamps correctly -- which is the hard part, and is not outsourced by the
purchase. Second, we forecast on a daily cadence with roughly a hundred features. The
online serving layer, which is most of the cost and nearly all of the operational
surface of a feature store, would be idle.

The cost of building is real: point-in-time joins, backfills and a spec registry are
weeks of work and permanently ours to maintain.

## Decision

Build. Feature definitions are declarative Python specs with an explicit lag and an
explicit knowledge delay per source column. Correctness is enforced by two automated
checks in CI -- a declarative knowledge-time check and an empirical truncation check --
rather than by a runtime system. Serving reads the same specs; skew is detected by
comparing the two paths on shared keys rather than prevented by construction.

Revisit if any of the following becomes true: sub-daily serving is required, feature
count passes roughly five hundred, or more than three teams are maintaining specs.

## Consequences

- The point-in-time semantics are visible in a file a reviewer can read, and violations
  fail a test rather than degrading a metric.
- We own backfill performance. A wide re-computation over several years of history is a
  batch job we have to make fast ourselves.
- No online store means no low-latency serving path. If a real-time use case appears,
  this decision blocks it and must be revisited rather than worked around.
- Skew is detected, not prevented. A serving-path change can ship broken and be caught
  after the fact by the skew report, which is weaker than a shared runtime would be.
- We avoid a vendor dependency in the critical path of every forecast.
