# PRD: Demand forecasting platform

Status: draft for review
Owner: product
Last updated: 2026

## 1. Problem

Demand forecasting teams do not fail at modelling. They fail at everything around the
model. A forecast that was accurate in a notebook degrades in production for reasons
that have nothing to do with the algorithm: an upstream feed changed a column type, a
feature was computed from data that did not exist at decision time, the serving path
fills nulls differently from the training path, or the item mix shifted and nobody
noticed because the aggregate metric held.

The consequence is not "the model is a bit worse". It is that planners stop trusting
the number, revert to manual overrides, and the organisation pays twice: once for the
platform, once for the spreadsheets that replaced it. Recovering that trust takes far
longer than losing it, because the evidence that the forecast is now fine is exactly
the evidence nobody kept when it was broken.

This platform makes the failure modes detectable before they reach a planner: contracts
on the inputs, point-in-time guarantees on the features, a release gate that is scored
in business cost as well as accuracy, a registry that can answer "what is serving and
what was serving before", and monitoring that reports by slice.

## 2. Users and jobs to be done

**Demand planner.** *When* the system forecast disagrees with what I know about my
category, *I want* to see which inputs moved and how the forecast performed on my slice
last month, *so that* I can decide whether to override rather than overriding by reflex.

**ML engineer.** *When* I propose a model change, *I want* the pipeline to tell me
whether it is safe to ship on the evidence, *so that* I am not arguing about a 40 bps
wMAPE difference in a review meeting.

**Data engineer / feed owner.** *When* I change a column upstream, *I want* to know
before I merge which downstream consumers break, *so that* I am not paged at 06:00 by
someone whose model silently started predicting zeros.

**Planning manager.** *When* I am asked whether the forecast is working, *I want* one
number in cost terms and its trend, *so that* I can answer without a data pull.

**On-call engineer.** *When* a model degrades at 02:00, *I want* a decision tree ending
in either "roll back" or "hold and investigate", *so that* I am not making a judgement
call about someone else's model with no context. See `docs/runbook.md`.

## 3. Non-goals

- **Not a feature store.** No online key-value serving layer, no materialisation
  service. See ADR-0001; the point-in-time semantics matter, the infrastructure does not.
- **Not an orchestrator.** No DAG runner. The platform assumes something already
  schedules jobs and is deliberately agnostic about what.
- **Not an experiment tracker.** The registry records what shipped and why, not every
  run of every sweep.
- **Not a model zoo.** One reference model exists to make the gate testable. Teams
  bring their own.
- **Not a planner UI.** Outputs are frames and JSON. A UI is downstream of this and out
  of scope for the first two phases.
- **Not real-time.** Daily and weekly cadence only, per ADR-0002.

## 4. Success metrics

Baselines are measured on the incumbent process during a two-week instrumentation
period before any rollout; every target is stated against that measured baseline rather
than an absolute.

| Metric | Definition | Baseline (measured pre-launch) | Target | Horizon |
| --- | --- | --- | --- | --- |
| Forecast-attributable inventory cost | Realised cost minus irreducible cost, per unit of demand | Instrumented baseline | -20% | 2 quarters |
| Incidents caused by input changes | Sev-2+ caused by an upstream schema or semantics change | Instrumented baseline | -75% | 1 quarter |
| Leakage escapes | Feature sets with a point-in-time violation reaching production | Unknown, assumed non-zero | 0 | Immediate |
| Time to detect degradation | Hours from onset to a page | Instrumented baseline | < 24h for PSI > 0.25 | 2 quarters |
| Time to roll back | Minutes from decision to previous version serving | Instrumented baseline | < 15 min | 1 quarter |
| Planner override rate | Share of forecast lines manually overridden | Instrumented baseline | -30% | 3 quarters |
| Worst-slice ratio | Worst slice wMAPE / overall wMAPE | 1.78 (benchmark panel) | < 1.4 | 3 quarters |

The override rate is the trust metric and the slowest to move. It is deliberately last;
treating it as a first-quarter target produces pressure to suppress overrides rather
than earn their absence, which is the wrong outcome. See ADR-0005.

## 5. Risks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Contracts are written once and never maintained | High | Medium | Contract diffing runs in the producer's CI, so a change is blocked at source rather than reported downstream |
| The cost gate is tuned until everything passes | Medium | High | Gate thresholds live in version control and changing them requires the same review as changing the model |
| Cost parameters are wrong, so the cost metric is confidently wrong | Medium | High | Report accuracy and cost side by side and publish the parameters; sensitivity to the shortage multiple is a standing review item |
| Slice monitoring produces so many alerts it is ignored | High | Medium | Slices are flagged only above a minimum row count and a ratio threshold; alert volume is itself reviewed monthly |
| Planners lose override capability and disengage | Medium | High | Overrides are never blocked, only recorded and attributed (ADR-0005) |
| Teams route around the registry to ship faster | Medium | High | Registration is the only path that produces a rollback, which is the thing they want at 02:00 |

## 6. Phased rollout

**Phase 0 -- instrument (weeks 1-4).** Run contracts and monitoring in report-only mode
against the existing forecast. Nothing is blocked. Output is the measured baseline for
every metric in section 4. Exit criterion: baselines agreed by the planning manager.

**Phase 1 -- guard the inputs (weeks 5-10).** Contract validation becomes blocking on
the two highest-volume feeds. Point-in-time audit becomes blocking in the ML repo's CI.
Exit criterion: zero leakage findings on the production feature set, and one real
upstream break caught before merge.

**Phase 2 -- guard the release (weeks 11-18).** Backtest gate blocks promotion.
Registry becomes the only path to production, with rollback drilled at least once.
Exit criterion: a rollback executed end to end in under 15 minutes during a drill.

**Phase 3 -- close the loop (quarter 2).** Decision-quality reporting published weekly
to planning. Slice monitoring drives a retraining backlog rather than ad-hoc requests.
Exit criterion: the cost metric is cited in a planning review without prompting.

**Phase 4 -- extend (quarter 3+).** Second forecast family (promotions or new-item
launch) onto the same contracts and gate. Deprecation policy (ADR-0006) enforced.

## 7. Open questions

1. Who owns cost parameters -- finance, planning, or the ML team? The metric is only
   credible if the owner is not the team being measured by it.
2. What is the right gate granularity? A single global gate is easy to reason about and
   too blunt for a portfolio with a genuinely hard long tail; per-segment gates are
   correct and invite gaming.
3. Should the gate block on a fixed threshold or on non-inferiority to the incumbent?
   Fixed thresholds age badly; relative gates let quality drift down one release at a time.
4. How long is the retention window for model cards and predictions? Rollback needs
   weeks; audit and post-incident review argue for years.
5. Does the platform own the override workflow, or only observe it? Observing is
   cheaper and leaves the trust problem with whoever owns the planning tool.
6. What triggers a scheduled retrain versus a drift-triggered one, and who approves it?
