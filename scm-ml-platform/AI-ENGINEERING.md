# AI Engineering skills map: `scm-ml-platform`

This repository is the platform layer around a demand forecasting model rather than the
model itself: data contracts with breaking-change diffing, point-in-time feature specs
with a two-check leakage audit, a release gate scored on accuracy *and* on the inventory
cost of the resulting order decisions, a file-backed registry with one-call rollback,
and PSI/KS drift plus slice-level monitoring. A `HistGradientBoostingRegressor` sits
inside it, so a learned model is genuinely in the running product — but no LLM is.

Against Andrew Ng's four skills, this repository is strongest on **skill 4, shaping the
build**: the [PRD](docs/prd/forecasting-platform.md) and [six ADRs](docs/adr/) are the
primary artifact, not decoration. **Skill 1** is the measurement apparatus — a gate with
a business-cost arm, drift monitoring, and a seeded leakage bug the tests catch. **Skill
2** is in the module boundaries and test design. **Skill 3** is in how the repository
was produced.

## Disclosure

The code here was written by Claude subagents on 2026-08-15, in a single session, under
human direction. The human (Sumeet) set the goal, the constraints and the portfolio
composition, made the scope call described below, and declined a proposal to backdate
commits. The subagents wrote the design, implementation, tests, benchmark harness,
figure and README. This document maps both the product and that build process against
the skills map. The suite is **35 tests, all passing** (`PYTHONPATH=src python -m
unittest discover -s tests`, 1.8s). Nothing here has run in production, at a company, or
at any scale beyond the 21,600-row synthetic panel it generates for itself.

## The four skills, as they actually appeared

### 1. Building and deploying AI applications

**The gate has a cost arm, and the cost arm is what fires.** `Gate` in
`src/scmplatform/backtest.py` sets wMAPE ≤ 0.25, |bias| ≤ 0.08, cost/unit ≤ 0.25. The
benchmark runs a challenger that is the champion with predictions scaled by 0.85 — the
classic way to buy a volume-weighted error metric by under-forecasting.

| metric | champion | challenger |
| --- | --- | --- |
| wMAPE | 0.1865 | 0.2223 |
| bias | -0.0020 | -0.1517 |
| cost per unit | 0.1995 | 0.7038 |
| regret | 12,019.84 | 51,314.24 |
| fill rate | 0.9957 | 0.9815 |

On accuracy the gap is 3.6 points of wMAPE, which two reviewers can argue about. On
realised inventory cost it is 3.5× per unit, which they cannot. The gate returns
`FAIL gate for challenger: |bias| 0.1517 > 0.0800; cost/unit 0.7038 > 0.2500`, decision
HOLD. Costs come from pushing forecasts through a periodic-review order-up-to policy
with a newsvendor critical ratio (`src/scmplatform/decisions.py`), reported as regret
against the cost that same policy incurs under a perfect point forecast: 15,548.46
realised against 3,528.62 irreducible, 4.41×. The 12,019.84 gap is the only part a
better model could recover.

**Error analysis is by slice, on every backtest.** Aggregate wMAPE of 0.1865 conceals a
long-tail slice at 0.3322 — 1.78× the headline on 5,582 of 77,926 units — against
`seasonal` 0.1774 and `core` 0.1736. A volume-weighted headline structurally hides the
slice its planner cares most about, so `slice_performance`
(`src/scmplatform/monitoring.py`) runs unconditionally.

**The leakage bug is seeded, and the tests catch it.** `leaky_specs()` in
`src/scmplatform/features.py` adds two specs to the clean seven: `returns_lag_1`, reading
a column with a 7-day knowledge delay at lag 1, and `sku_mean_units`, a per-SKU mean over
the full sample. Two independent checks catch one each — declarative knowledge-time, and
empirical truncation (recompute from history truncated at the label date; 120 rows moved,
max absolute difference 11.8230).
`tests/test_features.py::test_audit_reports_both_bugs_with_distinct_checks` asserts the
findings come from *different* check types, which is what breaks if someone makes one
check subsume the other. PSI and KS are reported side by side and disagree: KS returns
p < 0.001 on six of seven features at panel scale, while PSI puts only `on_hand_lag_1`
(0.1628) above the 0.10 investigate band.

### 2. Software engineering fundamentals

1,488 lines of source and 435 of tests across six modules wired only by pandas frames
and JSON on disk: `datagen → contracts → features → backtest (+decisions) → registry →
monitoring`. No service, database or orchestrator. `docs/design.md` states the tradeoff:
what actually fails in a forecasting platform is the contracts between stages, and those
are easiest to inspect when the plumbing is dumb. The registry is a directory of JSON
files for the same reason — at 02:00 the questions are "what is serving, what was
serving, can I put the old one back", and every answer should be readable with `cat`.

The build container had no PyPI access, so gradient boosting is
`sklearn.ensemble.HistGradientBoostingRegressor` rather than lightgbm and tests are
stdlib `unittest`, which pytest still collects. Test design carries intent:
`test_wmape_and_bias_on_a_hand_worked_example` pins metrics against hand arithmetic,
`test_deliberate_under_forecasting_is_rejected_on_cost` encodes the gate's reason for
existing, `tests/test_contracts.py` separates breaking from non-breaking schema changes,
and `test_rollback_restores_the_previous_production_version` is the drill ADR-0006
requires.

### 3. Using coding agents

One subagent per repository under a workflow orchestrator, with a shared
`CONVENTIONS.md` as the context contract — content rules, environment constraints,
required layout, README spec, definition of done. That file, not per-repo prompting, is
why ten independently written repositories look like one portfolio. Two cores meant
effective concurrency of about two agents, not ten.

The agent was not trusted on assertion. Done required actually running the suite,
`benchmarks/run_benchmarks.py` and `examples/01_release_pipeline.py` and writing the
README from real output; fabricating a benchmark number was prohibited and figures had
to be generated by code. Every number in `benchmarks/results.md` — including the 4.4s
total runtime — is a captured run. A separate audit agent that had written none of the
code re-ran everything across all ten repositories, checked authorship and dates, and
swept for employer references; it found one README number in another repository that
contradicted its own benchmark output and corrected it to the measured value.

### 4. Shaping the build

The PRD states five roles as jobs-to-be-done, six explicit non-goals (not a feature
store, orchestrator, experiment tracker, model zoo, planner UI, or real-time), a
success-metric table where every target is stated against a baseline measured in a
two-week instrumentation period rather than an absolute, six risks with mitigations, a
five-phase rollout with exit criteria, and six unresolved open questions. Override rate
is named the trust metric and placed last deliberately: making it a first-quarter target
produces pressure to suppress overrides rather than earn their absence.

| ADR | What it decides |
| --- | --- |
| [0001](docs/adr/0001-build-vs-buy-feature-store.md) | Build declarative point-in-time specs, do not buy a feature store; skew is detected by comparing paths, not prevented by construction. |
| [0002](docs/adr/0002-batch-vs-streaming-features.md) | Batch only, daily, cutoff part of the spec; revisit when a sub-24h decision cycle exists downstream. |
| [0003](docs/adr/0003-centralized-vs-embedded-model-ownership.md) | Platform owns contracts, registry, gate, monitoring; domains own models and thresholds. Platform cannot promote for a domain, nor block a green promotion. |
| [0004](docs/adr/0004-accuracy-metric-selection.md) | wMAPE plus signed bias as gate arms; MAPE and RMSE rejected on long-tail and scale grounds; MASE computed, not gated. |
| [0005](docs/adr/0005-human-override-policy.md) | Overrides never blocked, always recorded with planner, magnitude and reason code; reported organisationally, not as individual performance. |
| [0006](docs/adr/0006-model-deprecation-policy.md) | Every version has a named owner; current plus two preceding production versions always loadable; others archived after 90 days idle. |

The PRD's worst-slice target (< 1.4) is stated against the 1.78 this repository actually
measures — the link that makes a product document checkable.

## Build narrative for this repository

Spec received as `CONVENTIONS.md` plus a repository brief; design settled on the
five-stage pipeline in `docs/design.md`; implementation, tests, benchmarks and figure
followed; verification was the three required commands, run for real.

The correction is the honest part. Against a 1,200–2,500 line target the first
repositories came in near 6,000 lines each, and on two cores the run projected to about
three hours. The human called a scope cut: the run stopped after four repos had
committed, and the remaining six — including this one — were relaunched with a hard
budget (900–1,400 lines, 20–35 tests, one figure, one example) plus an explicit protect
list. **The PRD and the six ADRs here were on that list.** That is skill 4 constraining
skill 3: decide what carries signal, then bound what the agent may spend.

## AI during development vs AI in the product

| AI during development | AI in the product |
| --- | --- |
| Claude subagents wrote all source, tests, benchmarks, figure and docs on 2026-08-15 | `HistGradientBoostingRegressor`, refit per backtest fold |
| Orchestrated fan-out, ~2 concurrent on 2 cores | Everything around it deterministic: contracts, specs, gate arithmetic, PSI/KS, registry |
| Verifiers: unittest suite, benchmark harness, example script, independent audit agent | No LLM, no generative component, no non-determinism at serving time |

Both columns are non-empty here, unlike the five classical repositories in this portfolio.

## What I would do differently

- **Point forecasts only.** Safety stock comes from one residual standard deviation,
  understating uncertainty exactly on the long tail already at 1.78× headline error.
  Quantile forecasts with a pinball-loss gate arm are the fix.
- **The truncation check is sampled, not exhaustive.** A leak confined to a narrow window
  could pass; exhaustive checking is O(rows × specs) recomputations and would not fit CI.
- **Cost parameters are assumed, not estimated.** The 0.1995 cost per unit is only as
  credible as the holding rate and shortage multiple behind it; the PRD's first open
  question — who owns them — is unresolved for that reason.
- **Skew is detected after the fact**, so a divergence has already shipped. Prevention
  needs the serving runtime ADR-0001 declines to buy.
- **ADR-0005 has no code behind it.** Nothing captures override records or measures
  value-add, the missing input to the PRD's trust metric.

## Takeaways

1. **A gate on one metric selects for gaming that metric** (skill 1). Falsifiable: the
   challenger loses 3.6 wMAPE points and 3.5× on cost per unit, so any accuracy-only
   gate has a threshold band in which it promotes that model.
2. **Two leakage checks are needed, not one** (skill 1). Remove either from
   `features.py` and exactly one of the two seeded bugs ships clean.
3. **The document is the deliverable when the product is a platform** (skill 4). The PRD
   and ADRs survived a scope cut that deleted working code elsewhere in the portfolio.
4. **Agents close their own loop only if handed verifiers** (skill 3). The one README
   number that drifted from its benchmark here was caught by an independent auditing
   agent, not by the agent that wrote it.
5. **Constraints improve architecture more often than they damage it** (skill 2). No
   PyPI access forced sklearn over lightgbm and JSON files over a registry service;
   neither substitution costs this system anything it needed.

## How to explore this repo

Start with [`docs/prd/forecasting-platform.md`](docs/prd/forecasting-platform.md) and
[`docs/adr/README.md`](docs/adr/README.md) — they are the artifact. Then
[`benchmarks/results.md`](benchmarks/results.md) for every number above and
[`docs/design.md`](docs/design.md) for the module shape. In code:
`src/scmplatform/features.py` (the two leakage checks and `leaky_specs()`),
`src/scmplatform/backtest.py` (`Gate`, `check_gate`, `champion_challenger`) and
`src/scmplatform/decisions.py` (the cost model). Then run
`python examples/01_release_pipeline.py` end to end.
